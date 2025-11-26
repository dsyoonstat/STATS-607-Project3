#!/usr/bin/env python3
# plot_complexity.py
#
# Plot complexity exponents from timings/tables/complexity.csv.
#
# For each step (data_generation, estimator_computation, metric_computation),
# we create one figure:
#   - x-axis: methods (baseline, cholesky, cholesky+parallelization)
#   - within each method group: 5 bars for experiments
#       (single_normal, single_t, multi_normal, multi_t, convergence)
#   - y-axis: estimated alpha in O(p^alpha)
#
# Outputs:
#   timings/figures/complexity_data_generation.(pdf|svg)
#   timings/figures/complexity_estimator_computation.(pdf|svg)
#   timings/figures/complexity_metric_computation.(pdf|svg)

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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
        "legend.fontsize": 9,
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
    """Ensure timings/tables and timings/figures directories exist."""
    base = Path("timings")
    (base / "tables").mkdir(exist_ok=True, parents=True)
    (base / "figures").mkdir(exist_ok=True, parents=True)


def _save(fig: mpl.figure.Figure, stem: str):
    """Save figure as PDF and SVG into timings/figures."""
    out_pdf = Path("timings/figures") / f"{stem}.pdf"
    out_svg = Path("timings/figures") / f"{stem}.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    print(f"Saved: {out_pdf}\n       {out_svg}")


# ---------------------- configuration ----------------------

METHOD_ORDER = [
    "baseline",
    "cholesky",
    "cholesky+parallelization",
]

METHOD_PRETTY = {
    "baseline": "Baseline",
    "cholesky": "Cholesky",
    "cholesky+parallelization": "Cholesky + Parallelization",
}

EXPERIMENT_ORDER = [
    "single_normal",
    "single_t",
    "multi_normal",
    "multi_t",
    "convergence",
]

EXPERIMENT_PRETTY = {
    "single_normal": "single / normal",
    "single_t": "single / t",
    "multi_normal": "multi / normal",
    "multi_t": "multi / t",
    "convergence": "convergence",
}

STEP_ORDER = [
    "data_generation",
    "estimator_computation",
    "metric_computation",
]

STEP_TITLE = {
    "data_generation": "Data generation",
    "estimator_computation": "Estimator computation",
    "metric_computation": "Metric computation",
}


# ---------------------- plotting ----------------------
def _plot_step(df: pd.DataFrame, step: str):
    """
    Plot one figure for a given step.

    df: full complexity dataframe filtered later by step.
    step: one of STEP_ORDER.
    """

    # Theoretical complexities
    if step == "data_generation":
        theoretical_alpha = 3.0
    else:  # estimator_computation, metric_computation
        theoretical_alpha = 1.0

    df_step = df[df["step"] == step].copy()
    if df_step.empty:
        raise ValueError(f"No rows found for step='{step}' in complexity.csv")

    # Sanity check: ensure we have all combinations
    for m in METHOD_ORDER:
        for exp in EXPERIMENT_ORDER:
            mask = (df_step["method"] == m) & (df_step["experiment"] == exp)
            if not mask.any():
                raise ValueError(
                    f"Missing combination in complexity.csv: method={m}, experiment={exp}, step={step}"
                )

    # Build matrix of shape (n_methods, n_experiments)
    n_methods = len(METHOD_ORDER)
    n_exps = len(EXPERIMENT_ORDER)

    alpha_mat = np.zeros((n_methods, n_exps), dtype=float)

    for i, m in enumerate(METHOD_ORDER):
        for j, exp in enumerate(EXPERIMENT_ORDER):
            row = df_step[(df_step["method"] == m) & (df_step["experiment"] == exp)]
            alpha_mat[i, j] = float(row["alpha"].iloc[0])

    # X positions: groups = methods
    x = np.arange(n_methods)
    width = 0.12  # bar width per experiment
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", None)

    for j, exp in enumerate(EXPERIMENT_ORDER):
        offset = (j - (n_exps - 1) / 2) * width
        y = alpha_mat[:, j]
        color = colors[j % len(colors)] if colors is not None else None
        ax.bar(
            x + offset,
            y,
            width=width,
            label=EXPERIMENT_PRETTY.get(exp, exp),
            align="center",
            edgecolor="black",
            linewidth=0.4,
            color=color,
        )

    # ←← 여기 추가된 부분: theoretical 수평선
    ax.axhline(
        theoretical_alpha,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Theoretical ($\\alpha$={theoretical_alpha})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_PRETTY[m] for m in METHOD_ORDER])
    ax.set_xlabel("Method")
    ax.set_ylabel(r"Estimated exponent $\alpha$ in $\mathcal{O}(p^{\alpha})$")
    ax.set_title(f"Complexity exponents by method and experiment\n({STEP_TITLE.get(step, step)})")

    ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15)
    )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=max(0.0, ymin - 0.1 * (ymax - ymin)))

    stem = f"complexity_{step}"
    _save(fig, stem)
    plt.close(fig)


def main():
    _ensure_dirs()
    _set_pub_style()

    complexity_path = Path("timings/tables") / "complexity.csv"
    if not complexity_path.exists():
        raise FileNotFoundError(f"Cannot find complexity.csv at {complexity_path}")

    df = pd.read_csv(complexity_path)

    required_cols = {"experiment", "method", "step", "alpha", "r2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"complexity.csv missing required columns: {missing}")

    for step in STEP_ORDER:
        print(f"Plotting step: {step}")
        _plot_step(df, step)


if __name__ == "__main__":
    main()