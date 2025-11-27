#!/usr/bin/env python3
# plot_profiling.py
#
# Plot runtime profiling breakdowns from aggregated profiling CSVs, and
# summarize overall speedups across simulation variants.
#
# Inputs (per variant for profiling breakdown):
#   timings/tables/single_normal_profiling_{variant}.csv
#   timings/tables/single_t_profiling_{variant}.csv
#   timings/tables/multi_normal_profiling_{variant}.csv
#   timings/tables/multi_t_profiling_{variant}.csv
#   timings/tables/convergence_profiling_{variant}.csv
#
# Inputs (overall timings for speedup plot):
#   timings/tables/timings_baseline.csv
#   timings/tables/timings_cholesky.csv
#   timings/tables/timings_cholesky+parallelization.csv
#
# Outputs (per variant, profiling breakdown):
#   timings/figures/single_normal_profiling_{variant}.pdf / .svg
#   timings/figures/single_t_profiling_{variant}.pdf / .svg
#   timings/figures/multi_normal_profiling_{variant}.pdf / .svg
#   timings/figures/multi_t_profiling_{variant}.pdf / .svg
#   timings/figures/convergence_profiling_{variant}.pdf / .svg
#
# Output (overall speedup summary):
#   timings/figures/overall_speedup.pdf / .svg

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


def _load_overall_timings():
    """
    Load overall simulation timings for baseline and optimized variants.

    Expects:
      timings/tables/timings_baseline.csv
      timings/tables/timings_cholesky.csv
      timings/tables/timings_cholesky+parallelization.csv

    Each CSV must have columns:
      - 'task'
      - 'seconds'
    """
    tables_dir = Path("timings/tables")

    df_base = pd.read_csv(tables_dir / "timings_baseline.csv")
    df_chol = pd.read_csv(tables_dir / "timings_cholesky.csv")
    df_par  = pd.read_csv(tables_dir / "timings_cholesky+parallelization.csv")

    # Merge on 'task' to align rows
    df = (
        df_base.rename(columns={"seconds": "baseline"})
        .merge(df_chol.rename(columns={"seconds": "cholesky"}), on="task", how="inner")
        .merge(
            df_par.rename(columns={"seconds": "cholesky+parallelization"}),
            on="task",
            how="inner",
        )
    )

    return df


# ---------------------- profiling breakdown plotting ----------------------
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


# ---------------------- overall speedup plotting ----------------------
def _pretty_task_label(task: str) -> str:
    """
    Turn a raw task name such as 'run_simulation_convergence_rate'
    into a shorter, more readable label.
    """
    if task.startswith("run_simulation_"):
        core = task[len("run_simulation_"):]
    else:
        core = task

    mapping = {
        "single": "Single",
        "multi": "Multi",
        "convergence_rate": "Convergence",
    }
    return mapping.get(core, core.replace("_", " ").title())


def _plot_overall_speedups():
    """
    Plot speedup of Cholesky and Cholesky+parallelization relative to baseline
    for each high-level task (single / multi / convergence).
    """
    df = _load_overall_timings()

    tasks = df["task"].tolist()
    baseline = df["baseline"].to_numpy(float)
    chol     = df["cholesky"].to_numpy(float)
    par      = df["cholesky+parallelization"].to_numpy(float)

    # Speedup = baseline time / optimized time
    speedup_chol = baseline / chol
    speedup_par  = baseline / par

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    ax.bar(x - width / 2, speedup_chol, width, label="Cholesky")
    ax.bar(x + width / 2, speedup_par,  width, label="Cholesky + Parallelization")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_task_label(t) for t in tasks])
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Overall speedup relative to Baseline")
    ax.legend(frameon=False, ncol=2, loc="upper center")

    _save(fig, "speedup_overall")   # ← 여기 변경됨
    plt.close(fig)


# ---------------------- main ----------------------
def main():
    _ensure_dirs()
    _set_pub_style()

    # Per-variant profiling breakdowns
    for variant_key, info in VARIANTS.items():
        pretty = info["pretty"]
        print(f"\n=== Plotting profiling figures: {variant_key} ({pretty}) ===")
        _plot_variant(variant_key, pretty)

    # Overall speedup summary
    print("\n=== Plotting overall speedup summary ===")
    _plot_overall_speedups()

    print("\nDone generating profiling and speedup plots.")


if __name__ == "__main__":
    main()