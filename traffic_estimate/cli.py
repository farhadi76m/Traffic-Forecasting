"""Argparse plumbing shared by scripts/, so no script repeats an option group."""
from __future__ import annotations

import argparse
import os
import sys

from .network import RoadNetwork


def make_parser(doc: str, **kwargs) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter,
        **kwargs)


def add_network(parser, required: bool = True, help_text: str = "SUMO network (.net.xml)"):
    parser.add_argument("--net", required=required, help=help_text)
    return parser


def add_taz(parser, required: bool = True):
    parser.add_argument("--taz", required=required, help="TAZ file with the zones")
    return parser


def add_out_dir(parser, default=None):
    parser.add_argument("--out-dir", default=default, required=default is None)
    return parser


def add_seed(parser, default: int = 42):
    parser.add_argument("--seed", type=int, default=default)
    return parser


def add_scenarios(parser, num: int = 20, daily_trips: int = 6000):
    parser.add_argument("--num", type=int, default=num,
                        help="number of scenario days")
    parser.add_argument("--daily-trips", type=int, default=daily_trips,
                        help="average total trips per day")
    return parser


def add_polling(parser, out_default: str, probes_help: str, probes: int = 20):
    parser.add_argument("--out", default=out_default)
    parser.add_argument("--api-key", default=None,
                        help="defaults to the provider's environment variable")
    parser.add_argument("--probes", type=int, default=probes, help=probes_help)
    parser.add_argument("--once", action="store_true",
                        help="single snapshot, then exit")
    parser.add_argument("--poll", type=int, default=3600,
                        help="seconds between rounds")
    parser.add_argument("--hours", type=float, default=24,
                        help="how long to keep polling")
    return parser


def load_network(path: str, *, vclass: str = "passenger",
                 with_internal: bool = False, announce: bool = True
                 ) -> RoadNetwork:
    """Read a net once. City-sized nets cost minutes, so say so while it happens."""
    require_files(path)
    if announce:
        print(f"loading {os.path.basename(path)} ...", flush=True)
    return RoadNetwork(path, vclass=vclass, with_internal=with_internal)


def lazy_network(path: str, **kwargs):
    """Defer reading the net until something actually asks for it."""
    cache: list[RoadNetwork] = []

    def get() -> RoadNetwork:
        if not cache:
            cache.append(load_network(path, **kwargs))
        return cache[0]

    return get


def require_files(*paths: str) -> None:
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f"no such file: {path}")


def parse_hour(value: str) -> int:
    """'7' or '07:30' -> 7."""
    hour = int(str(value).split(":")[0])
    if not 0 <= hour <= 23:
        raise SystemExit(f"hour must be 0-23, got {value}")
    return hour


def hour_set(spec: str) -> set[int]:
    return {int(h) for h in spec.split(",")}
