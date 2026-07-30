"""Zone maps: the OD demand map and the TAZ partition snapshot."""
from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon

from ..demand.landuse import LandUseWeights
from ..network import RoadNetwork
from ..odmatrix import ODMatrix
from ..taz import TazSet
from . import theme
from .theme import (BASELINE, DEMAND, DIVERGING, GRID, INK, INK2, ISLAND,
                    MUTED, ROAD_MAJOR, ROAD_MINOR, SURFACE, ZONE_COLORS)

ARTERIAL_MS = 15.0  # m/s; above this an edge is drawn as a main road


def shorten(text: str, limit: int = 14) -> str:
    """District names are long; keep labels legible."""
    return text if len(text) <= limit else text[:limit - 1] + "…"


class ODMap:
    """Land-use balance beside the demand it produces, with a provenance banner.

    Left is measured (OSM land use), right is modelled (trip counts). Saying
    which is which on the figure itself is the point.
    """

    def __init__(self, taz: TazSet, weights: LandUseWeights, od: ODMatrix, *,
                 hour: int = 7, network: RoadNetwork | None = None,
                 islands: set[str] | None = None, label_zones: int = 15,
                 top_flows: int = 150):
        self.taz = taz
        self.weights = weights
        self.od = od
        self.hour = hour
        self.network = network
        self.islands = islands or set()
        self.label_zones = label_zones
        self.top_flows = top_flows
        # metres -> km keeps the tick labels readable
        self.polygons = {z.id: z.shape / 1000.0 for z in taz}
        self.centroids = {z.id: z.centroid / 1000.0 for z in taz}
        self.productions = dict(zip(weights.zones, weights.productions))
        self.attractions = dict(zip(weights.zones, weights.attractions))

    def flows(self) -> dict:
        flows = self.od.flows(self.hour)
        # 343 zones means ~117k arrows: an unreadable hairball that also takes
        # minutes to draw. Keep the heaviest, which is what the map is for.
        if self.top_flows and len(flows) > self.top_flows:
            print(f"drawing the {self.top_flows} heaviest of {len(flows)} "
                  "zone-pair flows")
            flows = dict(sorted(flows.items(), key=lambda kv: -kv[1])[:self.top_flows])
        return flows

    def _road_layers(self):
        major, minor = [], []
        for edge in self.network.drivable:
            points = np.asarray(edge.getShape()) / 1000.0
            (major if edge.getSpeed() >= ARTERIAL_MS else minor).append(points)
        return major, minor

    def render(self, path: str, title: str, subtitle: str,
               provenance: str) -> str:
        fig, (ax_land, ax_flow) = plt.subplots(1, 2, figsize=(15, 7.2),
                                               facecolor=SURFACE)
        fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.12)
        if self.network is not None:
            self._draw_roads(ax_land, ax_flow)
        self._draw_land_use(fig, ax_land)
        self._draw_flows(fig, ax_flow, provenance)
        self._frame(ax_land, ax_flow)

        fig.suptitle(title, x=0.045, y=0.955, ha="left", fontsize=16, color=INK,
                     fontweight="bold")
        fig.text(0.045, 0.885, subtitle, ha="left", fontsize=10.5, color=INK2)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"{path}  ({len(self.polygons)} zones, hour {self.hour})")
        return path

    def _draw_roads(self, ax_land, ax_flow) -> None:
        major, minor = self._road_layers()
        # left: white roads etched into the choropleth
        ax_land.add_collection(LineCollection(minor, colors=SURFACE,
                                              linewidths=0.3, alpha=0.4, zorder=2))
        ax_land.add_collection(LineCollection(major, colors=SURFACE,
                                              linewidths=1.0, alpha=0.7, zorder=2))
        # right: grey basemap under the flow arrows
        ax_flow.add_collection(LineCollection(minor, colors=GRID, linewidths=0.3,
                                              zorder=0.4))
        ax_flow.add_collection(LineCollection(major, colors=BASELINE,
                                              linewidths=0.8, alpha=0.8, zorder=0.5))

    def _draw_land_use(self, fig, ax) -> None:
        ratios = {z: float(np.log2(self.attractions[z] / self.productions[z]))
                  for z in self.weights.zones}
        limit = max(abs(v) for v in ratios.values())
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        # with 343 zones every label overprints into a smear, so name the
        # busiest and let the choropleth carry the rest
        ranked = sorted(ratios, key=lambda z: -(self.productions[z]
                                                + self.attractions[z]))
        labelled = set(ranked[:self.label_zones]) if self.label_zones else set(ratios)
        dense = len(ratios) > 40

        for zone_id, points in self.polygons.items():
            if zone_id not in ratios:
                continue
            ax.add_patch(Polygon(points, closed=True,
                                 facecolor=DIVERGING(norm(ratios[zone_id])),
                                 edgecolor=SURFACE,
                                 linewidth=0.3 if dense else 2))
            if zone_id not in labelled:
                continue
            cx, cy = self.centroids[zone_id]
            ax.text(cx, cy + 0.18, shorten(zone_id), ha="center", va="center",
                    fontsize=7 if dense else 9, color=INK, fontweight="bold",
                    zorder=5)
            if not dense:
                ax.text(cx, cy - 0.24,
                        f"{int(self.productions[zone_id])}h / "
                        f"{int(self.attractions[zone_id])}j",
                        ha="center", va="center", fontsize=8, color=INK2)

        bar = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING, norm=norm),
                           ax=ax, orientation="horizontal", pad=0.07, shrink=0.75)
        bar.set_label("← more homes      jobs / homes (log₂)      more jobs →",
                      fontsize=9, color=INK2)
        bar.outline.set_edgecolor(BASELINE)
        theme.title(ax, "Land use — MEASURED (OpenStreetMap)", size=12)

    def _draw_flows(self, fig, ax, provenance: str) -> None:
        flows = self.flows()
        heaviest = max(flows.values()) if flows else 1
        for (origin, destination), trips in sorted(flows.items(),
                                                   key=lambda kv: kv[1]):
            if origin not in self.centroids or destination not in self.centroids:
                continue
            (x1, y1), (x2, y2) = self.centroids[origin], self.centroids[destination]
            share = trips / heaviest
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=DEMAND(share),
                                        linewidth=0.4 + 4.6 * share, alpha=0.85,
                                        shrinkA=9, shrinkB=11,
                                        connectionstyle="arc3,rad=0.12"))
        for zone_id, points in self.polygons.items():
            island = zone_id in self.islands
            ax.add_patch(Polygon(points, closed=True, facecolor="none",
                                 edgecolor=ISLAND if island else GRID,
                                 linewidth=1.6 if island else 1.0,
                                 linestyle=(0, (3, 2)) if island else "-", zorder=0))
            cx, cy = self.centroids[zone_id]
            ax.scatter([cx], [cy], s=90, facecolor=SURFACE,
                       edgecolor=ISLAND if island else BASELINE, linewidth=1.4,
                       zorder=3)
            ax.text(cx, cy + 0.3, shorten(zone_id), ha="center", va="bottom",
                    fontsize=8, color=ISLAND if island else INK, zorder=4,
                    fontweight="bold" if island else "normal")

        bar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=DEMAND, norm=Normalize(0, heaviest)),
            ax=ax, orientation="horizontal", pad=0.07, shrink=0.75)
        bar.set_label(f"trips in hour {self.hour:02d}:00–{self.hour + 1:02d}:00",
                      fontsize=9, color=INK2)
        bar.outline.set_edgecolor(BASELINE)
        theme.title(ax, f"Demand flows, {self.hour:02d}:00 peak — {provenance}",
                    size=12)
        if self.islands:
            ax.plot([], [], color=ISLAND, linestyle=(0, (3, 2)), linewidth=1.6,
                    label=f"unroutable island ({len(self.islands)})")
            ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)

    def _frame(self, *axes) -> None:
        # add_patch does not grow the data limits, so autoscale would silently
        # crop zones out of the frame
        points = np.vstack(list(self.polygons.values()))
        (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
        pad = 0.06 * max(x1 - x0, y1 - y0)
        for ax in axes:
            ax.set_facecolor(SURFACE)
            ax.set_xlim(x0 - pad, x1 + pad)
            ax.set_ylim(y0 - pad, y1 + 2 * pad)  # headroom for the legend
            ax.set_aspect("equal")
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_xlabel("km east of net origin", fontsize=9, color=MUTED)
        axes[0].set_ylabel("km north", fontsize=9, color=MUTED)


def find_one(folder: str, suffix: str) -> str:
    hits = glob.glob(os.path.join(folder, f"*{suffix}"))
    if not hits:
        raise SystemExit(f"no *{suffix} in {folder}")
    return hits[0]


def taz_snapshot(folder: str, out: str, title: str | None = None,
                 max_edges: int = 120_000) -> tuple[int, int]:
    """Render the *_districts / *_network GeoJSON pair from one output folder."""
    with open(find_one(folder, "_districts.geojson"), encoding="utf-8") as handle:
        districts = json.load(handle)
    with open(find_one(folder, "_network.geojson"), encoding="utf-8") as handle:
        network = json.load(handle)

    features = network["features"]
    if len(features) > max_edges:  # keep huge networks renderable
        features = features[::len(features) // max_edges + 1]
    major, minor = [], []
    for feature in features:
        points = np.asarray(feature["geometry"]["coordinates"])
        target = (major if feature["properties"].get("speed_kmh", 0) >= 50
                  else minor)
        target.append(points)

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    ax.add_collection(LineCollection(minor, colors=ROAD_MINOR, linewidths=0.25,
                                     alpha=0.6, zorder=1))
    ax.add_collection(LineCollection(major, colors=ROAD_MAJOR, linewidths=0.7,
                                     alpha=0.85, zorder=2))

    xs, ys = [], []
    for i, feature in enumerate(districts["features"]):
        geometry = feature["geometry"]
        rings = (geometry["coordinates"] if geometry["type"] == "Polygon"
                 else [r[0] for r in geometry["coordinates"]])
        for ring in rings:
            points = np.asarray(ring)
            xs.extend(points[:, 0])
            ys.extend(points[:, 1])
            ax.add_patch(Polygon(points, closed=True,
                                 facecolor=ZONE_COLORS[i % len(ZONE_COLORS)],
                                 alpha=0.45, edgecolor="white", linewidth=0.8,
                                 zorder=3))

    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(ys))))
    pad = 0.04 * max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    theme.bare(ax)

    zones = len(districts["features"])
    theme.title(ax, title or f"{os.path.basename(folder)} — {zones} TAZ zones")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({zones} zones, {len(features)} edges drawn)")
    return zones, len(features)
