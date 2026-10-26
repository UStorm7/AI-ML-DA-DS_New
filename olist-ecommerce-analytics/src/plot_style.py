"""Shared, minimal matplotlib style so every figure reads as one set."""
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def apply() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
        "lines.linewidth": 2, "legend.frameon": False,
    })


def save(fig: plt.Figure, path) -> None:
    fig.savefig(path)
    plt.close(fig)
