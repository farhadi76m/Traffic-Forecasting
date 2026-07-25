#!/usr/bin/env python3
"""Run 24-hour SUMO mesoscopic simulations for each OD scenario.

Pipeline per scenario: od2trips (OD matrix -> trips using the TAZ) ->
duarouter (shortest-path routes) -> sumo --mesosim with hourly edgeData
output (one <interval> per hour, per edge).

Example:
    python scripts/run_sim.py --net sumo/tehran_2026_area.net.xml \
        --taz output/taz/taz_grid9.xml --od-dir output/od --out-dir output/sim
"""
import argparse
import glob
import gzip
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", required=True)
    p.add_argument("--taz", required=True)
    p.add_argument("--od-dir", required=True, help="directory of tazRelation OD files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--end", type=int, default=86400, help="sim end time in s")
    p.add_argument("--period", type=int, default=3600, help="edgeData aggregation in s")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-meso", action="store_true", help="use microscopic sim")
    p.add_argument("--jobs", type=int, default=1,
                   help="scenarios to simulate in parallel (SUMO is single-threaded)")
    p.add_argument("--skip-done", action="store_true",
                   help="skip scenarios whose edgedata already exists")
    p.add_argument("--keep-routes", action="store_true",
                   help="keep trips/routes XML. They are ~160MB per congested "
                        "scenario and are not needed downstream, so by default "
                        "only the (gzipped) edgedata survives.")
    return p.parse_args()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(" ".join(cmd) + "\n" + r.stderr[-3000:])
    return r.stderr  # sumo tools report stats on stderr


def simulate(job):
    """One scenario: od2trips -> duarouter -> sumo. Returns a status line.

    A failure is reported, not fatal: with hundreds of scenarios one bad draw
    should not throw away the whole batch.
    """
    try:
        return _simulate(job)
    except Exception as e:  # noqa: BLE001 - report and keep the batch alive
        return f"FAILED {os.path.basename(job[0])}: {e}"


def _simulate(job):
    od, args, k = job
    name = os.path.splitext(os.path.basename(od))[0]
    wdir = os.path.join(args.out_dir, name)
    os.makedirs(wdir, exist_ok=True)
    trips = os.path.join(wdir, "trips.xml")
    routes = os.path.join(wdir, "routes.xml")
    edgedata = os.path.join(wdir, "edgedata.xml")
    seed = str(args.seed + k)

    if args.skip_done and os.path.exists(edgedata + ".gz"):
        return f"{name}: skipped (already done)"

    run(["od2trips", "--taz-files", args.taz, "--tazrelation-files", od,
         "-o", trips, "--seed", seed, "--prefix", name + "_",
         "--ignore-vehicle-type",
         "--departpos", "random", "--arrivalpos", "random"])

    run(["duarouter", "-n", args.net, "--route-files", trips, "-o", routes,
         "--ignore-errors", "--no-warnings", "--repair", "--seed", seed])

    add = os.path.join(wdir, "edgedata_add.xml")
    with open(add, "w") as f:
        f.write(f'<additional>\n  <edgeData id="ed" file="{os.path.abspath(edgedata)}" '
                f'period="{args.period}" excludeEmpty="true"/>\n</additional>\n')

    cmd = ["sumo", "-n", args.net, "-r", routes, "--additional-files", add,
           "--begin", "0", "--end", str(args.end), "--seed", seed,
           "--no-step-log", "--no-warnings", "--ignore-route-errors",
           "--duration-log.statistics"]
    if not args.no_meso:
        cmd.append("--mesosim")
    log = run(cmd)

    # edgedata is 19MB of XML per congested scenario; gzip is ~15x smaller and
    # ElementTree reads it back through gzip.open just as happily.
    with open(edgedata, "rb") as fi, gzip.open(edgedata + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(edgedata)
    if not args.keep_routes:
        for f in (trips, routes, routes.replace(".xml", ".alt.xml")):
            if os.path.exists(f):
                os.remove(f)

    stats = [l.strip() for l in log.splitlines() if "Inserted:" in l]
    return f"{name}: {stats[0] if stats else 'done'}"


def main():
    args = parse_args()
    od_files = sorted(glob.glob(os.path.join(args.od_dir, "od_*.xml")))
    if not od_files:
        sys.exit(f"no od_*.xml files in {args.od_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = [(od, args, k) for k, od in enumerate(od_files)]
    failed = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        bar = tqdm(ex.map(simulate, jobs), total=len(jobs), desc="sumo")
        for msg in bar:
            if msg.startswith("FAILED"):
                failed.append(msg)
            bar.set_postfix_str(msg.split(": ", 1)[-1][:40])

    for msg in failed:
        print(msg, file=sys.stderr)
    print(f"{len(od_files) - len(failed)}/{len(od_files)} scenarios finished -> {args.out_dir}")


if __name__ == "__main__":
    main()
