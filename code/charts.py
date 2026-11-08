"""Shared matplotlib style for the episode figures (dark, matches the video frames)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#0b1220"
TEXT = "#eceff4"
TEXT2 = "#a8b3c4"
GRID = "#1e293b"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
GOOD, BAD = "#4ade80", "#f87171"


def setup(w=19.2, h=10.8):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "Segoe UI",
        "font.size": 20, "axes.titlesize": 30, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 24, "legend.frameon": False, "legend.fontsize": 19,
    })
    return plt.subplots(figsize=(w, h), dpi=100)


def save(fig, path):
    fig.tight_layout(pad=2.2)
    fig.savefig(path, dpi=100)
    plt.close(fig)
