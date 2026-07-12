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
import os
import subprocess
import sys
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
    return p.parse_args()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(" ".join(cmd), file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit(1)
    return r.stderr  # sumo tools report stats on stderr


def main():
    args = parse_args()
    od_files = sorted(glob.glob(os.path.join(args.od_dir, "od_*.xml")))
    if not od_files:
        sys.exit(f"no od_*.xml files in {args.od_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    for k, od in tqdm(enumerate(od_files)):
        name = os.path.splitext(os.path.basename(od))[0]
        wdir = os.path.join(args.out_dir, name)
        os.makedirs(wdir, exist_ok=True)
        trips = os.path.join(wdir, "trips.xml")
        routes = os.path.join(wdir, "routes.xml")
        edgedata = os.path.join(wdir, "edgedata.xml")
        seed = str(args.seed + k)

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
        stats = [l.strip() for l in log.splitlines()
                 if any(s in l for s in ("Inserted:", "Running:", "Statistics"))]
        print(f"{name}: done  {' | '.join(stats[:2])}")

    print(f"all {len(od_files)} scenarios finished -> {args.out_dir}")


if __name__ == "__main__":
    main()
