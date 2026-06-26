"""
viz.py
======
Shared palette, themes, and plotting helpers for the Nonlinear Dynamics on
Networks repository.

Palette and theme system ported verbatim from the Erdos-Renyi-Contagion repo
(src/utils.py) to ensure visual consistency across the two-repo research arc.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# Consistent color palette
NAVY = "#1A3A5C"
TEAL = "#0E7490"
RED = "#DC2626"
GOLD = "#D97706"
SLATE = "#64748B"
LIGHT = "#F1F5F9"

# Extended palette
PURPLE = "#7C3AED"
GREEN = "#059669"
ROSE = "#BE185D"
ORANGE = "#EA580C"
CYAN = "#06B6D4"
INDIGO = "#4F46E5"

# Dark theme palette (Manim + dark figures)
BG = "#020617"
CARD = "#0F172A"
DIM = "#334155"

# Ordered color cycle for multi-series plots
CYCLE = [TEAL, RED, GOLD, PURPLE, GREEN, ROSE, ORANGE, NAVY, CYAN, INDIGO]


# PATH HELPERS

def _repo_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_media_dir(subdir: str = "figures") -> str:
    """
    Return the absolute path to media/<subdir>/, creating it if needed.

    Parameters
    ----------
    subdir : str
        Subdirectory under media/ (default: 'figures').
    """
    path = os.path.join(_repo_root(), "media", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig: "plt.Figure", name: str, subdir: str = "figures",
                dpi: int = 150) -> str:
    """
    Save a figure to media/<subdir>/<name>.png with tight bbox.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to save.
    name : str
        Filename (without extension).
    subdir : str
        Subdirectory under media/.
    dpi : int
        Output resolution.

    Returns
    -------
    path : str
        Absolute path to the saved file.
    """
    path = os.path.join(get_media_dir(subdir), f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {path}")
    return path


# MATPLOTLIB THEME SETUP

def setup_dark_theme() -> None:
    """Configure matplotlib for dark-themed plots."""
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": CARD,
        "axes.edgecolor": DIM,
        "axes.labelcolor": LIGHT,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": LIGHT,
        "grid.color": DIM,
        "grid.linewidth": 0.5,
        "legend.facecolor": CARD,
        "legend.edgecolor": DIM,
    })


def setup_light_theme() -> None:
    """Configure matplotlib for the shared light theme."""
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor": "#F8FAFC",
        "axes.facecolor": LIGHT,
        "axes.edgecolor": SLATE,
        "axes.labelcolor": NAVY,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": NAVY,
        "grid.color": "white",
        "grid.linewidth": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": SLATE,
    })


# AXES STYLING

def despine(ax: "plt.Axes", keep: tuple[str, ...] = ("left", "bottom")) -> None:
    """
    Remove spines from a matplotlib axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to modify.
    keep : tuple[str, ...]
        Spines to keep (default: left and bottom).
    """
    for spine_name in ax.spines:
        if spine_name not in keep:
            ax.spines[spine_name].set_visible(False)
    for spine_name in keep:
        ax.spines[spine_name].set_color(DIM)


def apply_axes_style(ax: "plt.Axes", grid: bool = True,
                     alpha: float = 0.25) -> None:
    """
    Apply the shared styling to a matplotlib axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to style.
    grid : bool
        Whether to show grid.
    alpha : float
        Grid transparency.
    """
    ax.set_facecolor(LIGHT)
    despine(ax)
    if grid:
        ax.grid(True, color="white", linewidth=0.9, alpha=alpha, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
    ax.tick_params(colors=SLATE, labelsize=10)


def set_pub_style() -> None:
    """Set the shared figure defaults for all plots."""
    import matplotlib
    setup_light_theme()
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
