"""One palette and one matplotlib style for every figure in the project."""
from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# neutral surface and ink
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

# series
BLUE, AQUA = "#2a78d6", "#1baf7a"      # predicted, simulated
ISLAND = "#eb6834"                      # unroutable zones
LIGHT_REGIME, CONGESTED_REGIME = "#4C78A8", "#E45756"
ROAD_MAJOR, ROAD_MINOR = "#5b5a56", "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]

SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
DEMAND = LinearSegmentedColormap.from_list(
    "demand", ["#cde2fb", "#5598e7", "#256abf", "#0d366b"])
# blue <-> grey <-> red, for quantities with a natural neutral midpoint
DIVERGING = LinearSegmentedColormap.from_list(
    "homejob", ["#184f95", "#86b6ef", "#f0efec", "#e88b8a", "#c22b2a"])

# distinct zone fills for TAZ maps
ZONE_COLORS = ["#e76c5a", "#2e86c1", "#f1c40f", "#27ae60", "#9b59b6", "#e67e22",
               "#16a085", "#34495e", "#c0392b", "#1a5276", "#d35400", "#7f8c8d",
               "#48c9b0", "#af7ac5", "#f39c12", "#5dade2"]

RC_PARAMS = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
}


def use_theme(headless: bool = False) -> None:
    """Apply the palette. `headless` also forces the Agg backend for CLI use."""
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC_PARAMS)


def bare(ax, keep_frame: bool = False) -> None:
    """Strip a map axis down to its geometry."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    if not keep_frame:
        for spine in ax.spines.values():
            spine.set_visible(False)


def title(ax, text: str, size: float = 13) -> None:
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", color=INK)
