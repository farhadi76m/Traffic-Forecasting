"""The SUMO tool chain: od2trips -> duarouter -> sumo, with hourly edgeData."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from tqdm import tqdm

from ..xmlio import XML_HEADER

DAY_SECONDS = 86400


def run_tool(command: list[str]) -> str:
    """Run a SUMO CLI tool and return everything it said.

    sumo reports its run statistics on stdout and its warnings on stderr, so
    both streams are kept -- callers grep the result for what they need.
    """
    result = subprocess.run([str(c) for c in command], capture_output=True,
                            text=True)
    if result.returncode != 0:
        raise RuntimeError(" ".join(map(str, command)) + "\n"
                           + (result.stderr or result.stdout)[-3000:])
    return result.stdout + result.stderr


@dataclass
class SumoRunner:
    """One simulation of one OD matrix.

    Intermediates are written gzipped and deleted afterwards: a congested
    city-scale scenario is gigabytes of trips and routes, and only the
    edgeData survives downstream.
    """

    net: str
    taz: str
    meso: bool = True
    period: int = 3600
    seed: int = 7
    threads: int = 1
    keep_intermediates: bool = False
    extra_sumo_args: tuple[str, ...] = field(default_factory=tuple)
    last_log: str = field(default="", repr=False)

    def run(self, od_file: str, work_dir: str, *, begin: int = 0,
            end: int = DAY_SECONDS, seed: int | None = None,
            prefix: str | None = None) -> str:
        os.makedirs(work_dir, exist_ok=True)
        seed = str(self.seed if seed is None else seed)
        trips = os.path.join(work_dir, "trips.xml.gz")
        routes = os.path.join(work_dir, "routes.xml.gz")
        edgedata = os.path.join(work_dir, "edgedata.xml.gz")

        od2trips = ["od2trips", "--taz-files", self.taz,
                    "--tazrelation-files", od_file, "-o", trips,
                    "--seed", seed, "--ignore-vehicle-type",
                    "--begin", begin, "--end", end,
                    "--departpos", "random", "--arrivalpos", "random"]
        if prefix:
            od2trips += ["--prefix", prefix]
        run_tool(od2trips)

        run_tool(["duarouter", "-n", self.net, "--route-files", trips,
                  "-o", routes, "--begin", begin, "--end", end,
                  # routing dominates a city-scale run and is embarrassingly
                  # parallel; the routes themselves do not depend on it
                  "--routing-threads", self.threads,
                  # the alternatives file is as large as the routes and nothing
                  # downstream reads it
                  "--alternatives-output", os.devnull,
                  "--ignore-errors", "--no-warnings", "--repair", "--seed", seed])

        log = run_tool(self._sumo_command(routes, work_dir, edgedata, begin, end, seed))
        if not self.keep_intermediates:
            for path in (trips, routes):
                if os.path.exists(path):
                    os.remove(path)
        self.last_log = log
        return edgedata

    def _sumo_command(self, routes: str, work_dir: str, edgedata: str,
                      begin: int, end: int, seed: str) -> list[str]:
        additional = os.path.join(work_dir, "edgedata.add.xml")
        with open(additional, "w", encoding="utf-8") as handle:
            handle.write(XML_HEADER + "<additional>\n"
                         f'    <edgeData id="ed" file="{os.path.abspath(edgedata)}"'
                         f' period="{self.period}" excludeEmpty="true"/>\n'
                         "</additional>\n")
        command = ["sumo", "-n", self.net, "-r", routes,
                   "--additional-files", additional,
                   "--begin", begin, "--end", end, "--seed", seed,
                   "--no-step-log", "--no-warnings", "--ignore-route-errors"]
        if self.meso:
            command.append("--mesosim")
        return command + list(self.extra_sumo_args)


def _simulate(job) -> str:
    """A failure is reported, not fatal: one bad draw must not kill the batch."""
    runner, od_file, out_dir, seed = job
    name = os.path.splitext(os.path.basename(od_file))[0]
    work_dir = os.path.join(out_dir, name)
    try:
        runner.run(od_file, work_dir, seed=seed, prefix=f"{name}_")
    except Exception as exc:
        return f"FAILED {name}: {exc}"
    inserted = [line.strip() for line in runner.last_log.splitlines()
                if "Inserted:" in line]
    return f"{name}: {inserted[0] if inserted else 'done'}"


class ScenarioBatch:
    """Simulate a directory of OD files, optionally several at a time.

    SUMO itself is single-threaded, so parallelism is across scenarios.
    """

    def __init__(self, runner: SumoRunner, od_files: list[str], out_dir: str,
                 jobs: int = 1, skip_done: bool = False):
        self.runner = runner
        self.od_files = od_files
        self.out_dir = out_dir
        self.jobs = jobs
        self.skip_done = skip_done

    def _pending(self) -> list:
        jobs = []
        for index, od_file in enumerate(self.od_files):
            name = os.path.splitext(os.path.basename(od_file))[0]
            done = os.path.exists(os.path.join(self.out_dir, name,
                                               "edgedata.xml.gz"))
            if self.skip_done and done:
                continue
            jobs.append((self.runner, od_file, self.out_dir,
                         self.runner.seed + index))
        return jobs

    def run(self) -> list[str]:
        os.makedirs(self.out_dir, exist_ok=True)
        jobs = self._pending()
        skipped = len(self.od_files) - len(jobs)
        if skipped:
            print(f"skipping {skipped} scenarios that are already done")
        failures = []
        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            bar = tqdm(pool.map(_simulate, jobs), total=len(jobs), desc="sumo")
            for message in bar:
                if message.startswith("FAILED"):
                    failures.append(message)
                bar.set_postfix_str(message.split(": ", 1)[-1][:40])
        for message in failures:
            print(message, file=sys.stderr)
        print(f"{len(jobs) - len(failures)}/{len(jobs)} scenarios finished "
              f"-> {self.out_dir}")
        return failures
