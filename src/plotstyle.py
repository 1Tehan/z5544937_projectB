"""FundX design system for all report figures.

One consistent, original visual language across every exhibit (Part B rubric:
'custom figure and design system rather than the provided style'). All figures in
results/figures/ and the Streamlit app share this palette so the report and the
product look like one brand. It evolves the house style I started from in Part A
(equity navy #1f4e79 / crypto orange #c55a11 in run_part_a.py) into a full named
FundX palette.

Usage:
    from src.plotstyle import fundx_theme, PALETTE, save_fig
    fundx_theme()
    fig, ax = plt.subplots(); ...; save_fig(fig, "results/figures/name.png")
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Brand palette (FundX): deep navy base, teal + gold accents, coral for risk.
PALETTE = {
    "navy":  "#12283F",
    "teal":  "#1F8A8C",
    "gold":  "#E3A82B",
    "coral": "#E2593B",
    "sky":   "#5FA8D3",
    "moss":  "#6B9080",
    "plum":  "#8D5A97",
    "steel": "#64748B",
    "sand":  "#C8B08A",
    "wine":  "#9E2B25",
    "ink":   "#1E293B",
    "paper": "#FBFAF7",
    "grid":  "#E5E1D8",
}
# Ordered cycle used for multi-series charts (10 distinct hues for 10 sectors).
CYCLE = [PALETTE[k] for k in
         ("navy", "teal", "gold", "coral", "sky", "moss", "plum", "steel",
          "sand", "wine")]


def fundx_theme() -> None:
    """Apply the FundX matplotlib theme globally."""
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["paper"],
        "axes.facecolor": PALETTE["paper"],
        "savefig.facecolor": PALETTE["paper"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["navy"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 110,
        "lines.linewidth": 1.7,
    })


def brand_footer(fig, note: str = "") -> None:
    """Small brand strip in the figure footer (period + source note)."""
    txt = "FundX  |  FINS3645 FinTech Project 2026"
    if note:
        txt += f"  |  {note}"
    fig.text(0.995, 0.005, txt, ha="right", va="bottom",
             fontsize=7, color=PALETTE["steel"])


def save_fig(fig, path: str, note: str = "") -> None:
    brand_footer(fig, note)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
