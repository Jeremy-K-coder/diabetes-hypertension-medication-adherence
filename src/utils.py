"""
src/utils.py
============
Reproducibility helpers, figure saving, and display utilities.
Imported at the top of every notebook.

Usage:
    from src.utils import set_seeds, save_figure, section
"""

import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np

# ── Project-wide constants ────────────────────────────────────────────────────
SEED        = 42
FIGSIZE_SM  = (6, 4)
FIGSIZE_MD  = (9, 5)
FIGSIZE_LG  = (12, 6)
FIGSIZE_SQ  = (6, 6)
DPI         = 150

# Colour palette (consistent across all notebooks)
PALETTE = {
    "adherent":     "#2E7D32",   # green
    "non_adherent": "#C62828",   # red
    "neutral":      "#1565C0",   # blue
    "highlight":    "#E65100",   # orange
    "light_grey":   "#ECEFF1",
    "dark_grey":    "#455A64",
}

# Feature group labels (used for plot titles and table headers)
GROUP_LABELS = {
    "A": "Socioeconomic Proxies",
    "B": "Clinical Consumption",
    "C": "Combined",
}


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seeds(seed: int = SEED) -> None:
    """
    Set random seeds for Python, NumPy, and (if available) scikit-learn
    so that all results are reproducible across notebook runs.

    Call this as the very first line after imports in every notebook.

    Parameters
    ----------
    seed : int
        The random seed to use everywhere. Default is the project-wide SEED = 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # scikit-learn uses numpy's global state, so setting numpy seed is enough,
    # but we also store it so pipelines can pass it explicitly where needed.
    print(f"[utils] Seeds set to {seed}.")


# ── Figure helpers ────────────────────────────────────────────────────────────
def save_figure(
    fig: plt.Figure,
    filename: str,
    subfolder: str = "",
    dpi: int = DPI,
    close: bool = True,
) -> str:
    """
    Save a matplotlib figure to the figures/ directory.

    Parameters
    ----------
    fig       : matplotlib Figure object to save.
    filename  : File name including extension, e.g. 'class_balance.png'.
    subfolder : Optional subdirectory inside figures/, e.g. 'eda' or 'calibration'.
    dpi       : Resolution. Default 150 — clear without being huge.
    close     : Whether to close the figure after saving (frees memory). Default True.

    Returns
    -------
    str : The full path where the figure was saved.

    Example
    -------
    fig, ax = plt.subplots()
    ax.bar(...)
    save_figure(fig, 'class_balance.png', subfolder='eda')
    """
    # Build path relative to the project root regardless of where the
    # notebook is opened from.
    root = _project_root()
    folder = os.path.join(root, "figures", subfolder) if subfolder else os.path.join(root, "figures")
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[utils] Figure saved → {path}")

    if close:
        plt.close(fig)

    return path


# ── Notebook display helpers ──────────────────────────────────────────────────
def section(title: str, level: int = 1) -> None:
    """
    Print a clearly visible section divider to notebook stdout.
    Helps navigate long notebooks without scrolling through cell outputs.

    Parameters
    ----------
    title : Section title text.
    level : 1 = major section (═══), 2 = subsection (───).

    Example
    -------
    section("1. Data Loading")
    section("1.1 Shape and dtypes", level=2)
    """
    width = 72
    if level == 1:
        print("\n" + "═" * width)
        print(f"  {title}")
        print("═" * width)
    else:
        print(f"\n── {title} " + "─" * max(0, width - len(title) - 4))


def print_df_summary(df, name: str = "DataFrame") -> None:
    """
    Print a concise summary of a DataFrame: shape, dtypes, missing values,
    and duplicate rows. Useful at the start of EDA.

    Parameters
    ----------
    df   : pandas DataFrame to summarise.
    name : Label printed in the header.
    """
    import pandas as pd

    section(f"Summary: {name}", level=2)
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Duplicate rows : {df.duplicated().sum():,}")
    print()

    # dtype + missing table
    summary = pd.DataFrame({
        "dtype":   df.dtypes,
        "missing": df.isna().sum(),
        "missing%": (df.isna().mean() * 100).round(2),
        "unique":  df.nunique(),
    })
    print(summary.to_string())
    print()


# ── Internal helpers ──────────────────────────────────────────────────────────
def _project_root() -> str:
    """
    Return the absolute path to the project root directory.
    Works whether the notebook is run from the project root or from
    inside the notebooks/ subdirectory (e.g. on Colab).
    """
    # Walk up from this file's location until we find README.md
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(4):  # maximum 4 levels up
        if os.path.exists(os.path.join(current, "README.md")):
            return current
        current = os.path.dirname(current)

    # Fallback: current working directory (works on Colab when cwd = project root)
    return os.getcwd()
