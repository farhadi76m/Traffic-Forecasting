"""Datasets built from simulation output."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..edgedata import HOURS, EdgeData, find_edgedata, scenario_dirs
from ..network import STATIC_FEATURES, RoadNetwork

TRAFFIC_FILE = "traffic.csv.gz"
EDGES_FILE = "edges.csv"


@dataclass
class ForecastDataset:
    """(scenario, edge, hour, count) rows plus static per-edge features.

    This is the forecasting view: the OD is deliberately thrown away, because
    the forecaster never sees it at inference time.
    """

    frame: pd.DataFrame
    static: pd.DataFrame

    @property
    def edges(self) -> list[str]:
        return self.static["edge"].tolist()

    @property
    def scenarios(self) -> list[int]:
        return sorted(self.frame["scenario"].unique())

    @classmethod
    def build(cls, network: RoadNetwork, sim_dir: str) -> "ForecastDataset":
        edges = network.sorted_ids(connected_only=False)
        index = {edge: i for i, edge in enumerate(edges)}
        features = network.static_features(edges)
        static = pd.DataFrame({"edge": edges,
                               **{name: features[:, k]
                                  for k, name in enumerate(STATIC_FEATURES)}})

        directories = scenario_dirs(sim_dir)
        if not directories:
            raise SystemExit(f"no edgedata found under {sim_dir}")

        frames = []
        for scenario, directory in enumerate(tqdm(directories, desc="scenarios")):
            data = EdgeData.read(find_edgedata(directory), index)
            frames.append(pd.DataFrame({
                "scenario": scenario,
                "edge": np.repeat(edges, HOURS),
                "hour": np.tile(np.arange(HOURS), len(edges)),
                "count": data.counts.astype(np.int32).reshape(-1),
            }))
            tqdm.write(f"scenario {scenario} ({os.path.basename(directory)}): "
                       f"{data.total_entries} vehicle-entries, "
                       f"{data.active_edges} active edges")
        return cls(pd.concat(frames, ignore_index=True), static)

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        self.static.to_csv(os.path.join(out_dir, EDGES_FILE), index=False)
        self.frame.to_csv(os.path.join(out_dir, TRAFFIC_FILE), index=False)
        print(f"wrote {out_dir}: {len(self.frame)} rows, "
              f"{len(self.scenarios)} scenarios, {len(self.edges)} edges")
        return out_dir

    @classmethod
    def load(cls, data_dir: str) -> "ForecastDataset":
        return cls(pd.read_csv(os.path.join(data_dir, TRAFFIC_FILE)),
                   pd.read_csv(os.path.join(data_dir, EDGES_FILE)))

    def split(self, n_test: int, n_val: int) -> np.ndarray:
        """Per-row partition: 0 train, 1 val, 2 test. Held-out days, not rows."""
        scenarios = self.scenarios
        test = scenarios[len(scenarios) - n_test:]
        val = scenarios[len(scenarios) - n_test - n_val:len(scenarios) - n_test]
        print(f"scenarios: train {scenarios[:len(scenarios) - n_test - n_val]}, "
              f"val {val}, test {test}")
        return np.where(self.frame["scenario"].isin(test), 2,
                        np.where(self.frame["scenario"].isin(val), 1, 0))


@dataclass
class SurrogateDataset:
    """Both sides of the map: OD in, counts and travel times out."""

    od: np.ndarray            # (scenarios, 24, zones^2)
    counts: np.ndarray        # (scenarios, edges, 24)
    traveltime: np.ndarray    # (scenarios, edges, 24), seconds
    free_flow: np.ndarray     # (edges,)
    static: np.ndarray        # (edges, 4)
    edges: np.ndarray
    scenarios: np.ndarray

    @property
    def n_scenarios(self) -> int:
        return self.counts.shape[0]

    @property
    def n_edges(self) -> int:
        return self.counts.shape[1]

    @property
    def n_od(self) -> int:
        return self.od.shape[-1]

    @property
    def congestion_ratio(self) -> np.ndarray:
        return self.traveltime / self.free_flow[None, :, None]

    @classmethod
    def build(cls, network: RoadNetwork, od_dir: str,
              sim_dir: str) -> "SurrogateDataset":
        edges = network.sorted_ids(connected_only=False)
        index = {edge: i for i, edge in enumerate(edges)}
        static = network.static_features(edges)
        free_flow = network.free_flow_time(edges)

        od_all = np.load(os.path.join(od_dir, "od.npy"))
        kept, counts, travel = [], [], []
        for scenario in tqdm(range(len(od_all)), desc="scenarios"):
            path = find_edgedata(os.path.join(sim_dir, f"od_{scenario:03d}"))
            if path is None:
                continue  # scenario failed in SUMO; drop it from both sides
            data = EdgeData.read(path, index)
            kept.append(scenario)
            counts.append(data.counts)
            travel.append(data.traveltime_filled(free_flow))

        return cls(od_all[kept], np.stack(counts), np.stack(travel), free_flow,
                   static, np.array(edges), np.array(kept))

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "data.npz")
        np.savez_compressed(path, od=self.od, counts=self.counts,
                            traveltime=self.traveltime, free_flow=self.free_flow,
                            static=self.static, edges=self.edges,
                            scenarios=self.scenarios)
        return path

    @classmethod
    def load(cls, path: str) -> "SurrogateDataset":
        blob = np.load(path, allow_pickle=True)
        return cls(blob["od"], blob["counts"], blob["traveltime"],
                   blob["free_flow"], blob["static"], blob["edges"],
                   blob["scenarios"])

    def summary(self) -> str:
        ratio = self.congestion_ratio
        peak = ratio[:, :, 7:10]
        return (f"  {self.n_scenarios} scenarios, {self.n_edges} edges, "
                f"OD dim {self.n_od} x 24\n"
                f"  counts     : mean {self.counts.mean():.1f}  "
                f"max {self.counts.max():.0f}\n"
                f"  tt/freeflow: mean {ratio.mean():.2f}  "
                f"p90 {np.percentile(ratio, 90):.2f}  max {ratio.max():.1f}\n"
                f"  at peak 07-09: mean {peak.mean():.2f}  share of edge-hours "
                f">1.2x free-flow {100 * (peak > 1.2).mean():.1f}%")
