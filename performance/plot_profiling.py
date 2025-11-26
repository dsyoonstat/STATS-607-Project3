#!/usr/bin/env python3
# plot_profiling.py
#
# Plot runtime profiling breakdowns from aggregated profiling CSVs.
#
# Inputs (per variant):
#   timings/tables/single_normal_profiling_{variant}.csv
#   timings/tables/single_t_profiling_{variant}.csv
#   timings/tables/multi_normal_profiling_{variant}.csv
#   timings/tables/multi_t_profiling_{variant}.csv
#   timings/tables/convergence_profiling_{variant}.csv
#
# Outputs (per variant):
#   timings/figures/single_normal_profiling_{variant}.pdf / .svg
#   timings/figures/single_t_profiling_{variant}.pdf / .svg
#   timings/figures/multi_normal_profiling_{variant}.pdf / .svg
#   timings/figures/multi_t_profiling_{variant}.pdf / .svg
#   timings/figures/convergence_profiling_{variant}.pdf / .svg

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------- configuration ----------------------

VARIANTS = {
    "baseline": {
        "pretty": "Baseline",
    },
    "cholesky": {
        "pretty": "Cholesky",
    },
    "cholesky+parallelization": {
        "pretty": "Cholesky + Parallelization",
    },
}


# ---------------------- aesthetics ----------------------
def _set_pub_style():
    """Set matplotlib parameters for a clean, publication-quality style."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.constrained_layout.use": True,
    })


def _ensure_dirs():
    """Ensure timing figure output directory exists."""
    base = Path("timings")
    (base / "figures").mkdir(exist_ok=True, parents=True)


def _save(fig, stem: str):
    """Save figure as PDF and SVG into timings/figures."""
    out_pdf = Path("timings/figures") / f"{stem}.pdf"
    out_svg = Path("timings/figures") / f"{stem}.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    print(f"Saved: {out_pdf}\n       {out_svg}")


# ---------------------- loading helpers ----------------------
def _load_runtime_table(stem: str, variant_key: str) -> pd.DataFrame:
    """
    Load aggregated profiling table for a given stem and variant.

    Example:
      stem='single_normal_profiling', variant='baseline'
      -> timings/tables/single_normal_profiling_baseline.csv
    """
    path = Path("timings/tables") / f"{stem}_{variant_key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    return pd.read_csv(path)


# ---------------------- plotting helpers ----------------------
def _plot_three_buckets(df: pd.DataFrame, title: str, stem: str):
    """
    Plot bar chart with three buckets:
      - data_generation
      - estimator_computation
      - metric_computation
    over p.
    """
    labels = ["Data generation", "Estimator computation", "Metric computation"]
    keys   = ["data_generation", "estimator_computation", "metric_computation"]

    x = np.arange(len(df["p"]))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.9))

    for idx, key in enumerate(keys):
        ax.bar(
            x + (idx - 1) * width,
            df[key].to_numpy(),
            width,
            label=labels[idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["p"].astype(int))
    ax.set_xlabel(r"$p$")
    ax.set_ylabel("Total seconds")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=1)

    _save(fig, stem)
    plt.close(fig)


def _plot_variant(variant_key: str, pretty: str):
    """
    For a given variant, plot five profiling figures:
      - single_normal_profiling
      - single_t_profiling
      - multi_normal_profiling
      - multi_t_profiling
      - convergence_profiling
    """
    stems = [
        ("single_normal_profiling", "Single simulation runtime breakdown (Normal)"),
        ("single_t_profiling",      "Single simulation runtime breakdown (Student-t)"),
        ("multi_normal_profiling",  "Multi simulation runtime breakdown (Normal)"),
        ("multi_t_profiling",       "Multi simulation runtime breakdown (Student-t)"),
        ("convergence_profiling",   "Convergence simulation runtime breakdown (all SNRs)"),
    ]

    for stem, title in stems:
        df = _load_runtime_table(stem, variant_key)
        full_title = f"{title} – {pretty}"
        out_stem   = f"{stem}_{variant_key}"
        _plot_three_buckets(df, full_title, out_stem)


# ---------------------- main ----------------------
def main():
    _ensure_dirs()
    _set_pub_style()

    for variant_key, info in VARIANTS.items():
        pretty = info["pretty"]
        print(f"\n=== Plotting profiling figures: {variant_key} ({pretty}) ===")
        _plot_variant(variant_key, pretty)

    print("\nDone generating profiling plots.")


if __name__ == "__main__":
    main()