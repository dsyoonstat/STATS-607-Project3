#!/usr/bin/env python3
# plot_runtime_profiling.py
# - Single-spike, single-reference runtime breakdown
# - Multi-spike, multi-reference runtime breakdown
# - Convergence-rate simulation runtime breakdown
#   Buckets: Data generation / Estimator computation / Evaluation

from pathlib import Path
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
    """Ensure expected folders exist."""
    base = Path("results")
    (base / "tables").mkdir(exist_ok=True, parents=True)
    (base / "figures").mkdir(exist_ok=True, parents=True)


def _save(fig: mpl.figure.Figure, stem: str):
    """Save figure as PDF and SVG into results/figures."""
    out_pdf = Path("results/figures") / f"{stem}.pdf"
    out_svg = Path("results/figures") / f"{stem}.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    print(f"Saved: {out_pdf}\n       {out_svg}")


# ---------------------- generic aggregation helper ----------------------
def _aggregate_buckets(df: pd.DataFrame,
                       mask_data,
                       mask_sub,
                       mask_ang) -> pd.DataFrame:
    """
    Aggregate into three buckets for each p:

        data_gen            = sum seconds_total over mask_data
        computing_estimator = sum seconds_total over mask_sub
        principal_angles    = sum seconds_total over mask_ang

    Returns DataFrame with columns:
        p, data_gen, computing_estimator, principal_angles
    """
    g1 = (
        df.loc[mask_data]
          .groupby("p", as_index=False)["seconds_total"]
          .sum()
          .rename(columns={"seconds_total": "data_gen"})
    )

    g2 = (
        df.loc[mask_sub]
          .groupby("p", as_index=False)["seconds_total"]
          .sum()
          .rename(columns={"seconds_total": "computing_estimator"})
    )

    g3 = (
        df.loc[mask_ang]
          .groupby("p", as_index=False)["seconds_total"]
          .sum()
          .rename(columns={"seconds_total": "principal_angles"})
    )

    merged = (
        g1.merge(g2, on="p", how="outer")
          .merge(g3, on="p", how="outer")
          .fillna(0.0)
          .sort_values("p")
    )
    return merged


# ---------------------- single: build tables ----------------------
def _load_single_step_timings() -> pd.DataFrame:
    """Load single_step_timings.csv from results/tables."""
    csv_path = Path("results/tables/single_step_timings.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"p", "step", "seconds_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns in single_step_timings: {missing}")
    return df


def _build_single_runtime_tables() -> None:
    """
    From single_step_timings.csv, build and save:

      results/tables/single_normal_timings.csv
      results/tables/single_t_timings.csv
    """
    df = _load_single_step_timings()

    # Normal masks (single)
    is_data_gen_normal = df["step"].str.startswith("data_gen_normal")
    is_arg_normal_sub  = df["step"].str.startswith("ARG_normal_compute_ARG_PC_subspace")
    is_base_normal_sub = df["step"].str.startswith("baseline_normal_compute_PC_subspace")
    is_arg_normal_ang  = df["step"].str.startswith("ARG_normal_compute_principal_angles")
    is_base_normal_ang = df["step"].str.startswith("baseline_normal_compute_principal_angles")

    # t masks (single)
    is_data_gen_t      = df["step"].str.startswith("data_gen_t")
    is_arg_t_sub       = df["step"].str.startswith("ARG_t_compute_ARG_PC_subspace")
    is_base_t_sub      = df["step"].str.startswith("baseline_t_compute_PC_subspace")
    is_arg_t_ang       = df["step"].str.startswith("ARG_t_compute_principal_angles")
    is_base_t_ang      = df["step"].str.startswith("baseline_t_compute_principal_angles")

    normal = _aggregate_buckets(
        df,
        mask_data=is_data_gen_normal,
        mask_sub=is_arg_normal_sub | is_base_normal_sub,
        mask_ang=is_arg_normal_ang | is_base_normal_ang,
    )

    tdf = _aggregate_buckets(
        df,
        mask_data=is_data_gen_t,
        mask_sub=is_arg_t_sub | is_base_t_sub,
        mask_ang=is_arg_t_ang | is_base_t_ang,
    )

    tables_dir = Path("results/tables")
    normal_path = tables_dir / "single_normal_timings.csv"
    t_path      = tables_dir / "single_t_timings.csv"
    normal.to_csv(normal_path, index=False)
    tdf.to_csv(t_path, index=False)
    print(f"Saved single runtime tables:\n  {normal_path}\n  {t_path}")


# ---------------------- multi: build tables ----------------------
def _load_multi_step_timings() -> pd.DataFrame:
    """Load multi_step_timings.csv from results/tables."""
    csv_path = Path("results/tables/multi_step_timings.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"p", "step", "seconds_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns in multi_step_timings: {missing}")
    return df


def _build_multi_runtime_tables() -> None:
    """
    From multi_step_timings.csv, build and save:

      results/tables/multi_normal_timings.csv
      results/tables/multi_t_timings.csv
    """
    df = _load_multi_step_timings()

    # Normal masks (multi)
    is_data_gen_normal = df["step"].str.startswith("data_gen_normal")
    is_arg_normal_sub  = df["step"].str.startswith("ARG_normal_compute_ARG_PC_subspace")
    is_pca_normal_sub  = df["step"].str.startswith("PCA_normal_compute_PC_subspace")
    is_arg_normal_ang  = df["step"].str.startswith("ARG_normal_compute_principal_angles")
    is_pca_normal_ang  = df["step"].str.startswith("PCA_normal_compute_principal_angles")

    # t masks (multi)
    is_data_gen_t      = df["step"].str.startswith("data_gen_t")
    is_arg_t_sub       = df["step"].str.startswith("ARG_t_compute_ARG_PC_subspace")
    is_pca_t_sub       = df["step"].str.startswith("PCA_t_compute_PC_subspace")
    is_arg_t_ang       = df["step"].str.startswith("ARG_t_compute_principal_angles")
    is_pca_t_ang       = df["step"].str.startswith("PCA_t_compute_principal_angles")

    normal = _aggregate_buckets(
        df,
        mask_data=is_data_gen_normal,
        mask_sub=is_arg_normal_sub | is_pca_normal_sub,
        mask_ang=is_arg_normal_ang | is_pca_normal_ang,
    )

    tdf = _aggregate_buckets(
        df,
        mask_data=is_data_gen_t,
        mask_sub=is_arg_t_sub | is_pca_t_sub,
        mask_ang=is_arg_t_ang | is_pca_t_ang,
    )

    tables_dir = Path("results/tables")
    normal_path = tables_dir / "multi_normal_timings.csv"
    t_path      = tables_dir / "multi_t_timings.csv"
    normal.to_csv(normal_path, index=False)
    tdf.to_csv(t_path, index=False)
    print(f"Saved multi runtime tables:\n  {normal_path}\n  {t_path}")


# ---------------------- convergence: build table ----------------------
def _load_convergence_step_timings() -> pd.DataFrame:
    """Load convergence_step_timings.csv from results/tables."""
    csv_path = Path("results/tables/convergence_step_timings.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"p", "step", "seconds_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns in convergence_step_timings: {missing}")
    return df


def _build_convergence_runtime_table() -> None:
    """
    From convergence_step_timings.csv, build and save:

      results/tables/convergence_timings.csv

    Buckets:
      Data generation     : steps starting with 'data_gen'
      Compute discriminant: steps containing 'compute_discriminant'
      Evaluation          : steps starting with 'alpha_accumulate' or 'inner_product'

    All SNRs are summed: group only by p.
    """
    df = _load_convergence_step_timings()

    # Data generation
    is_data_gen = df["step"].str.startswith("data_gen")

    # Compute discriminant
    is_disc = df["step"].str.contains("compute_discriminant", regex=False)

    # Evaluation: alpha_accumulate + inner_product
    is_alpha = df["step"].str.startswith("alpha_accumulate")
    is_ip    = df["step"].str.startswith("inner_product")
    is_eval  = is_alpha | is_ip

    conv = _aggregate_buckets(
        df,
        mask_data=is_data_gen,
        mask_sub=is_disc,
        mask_ang=is_eval,
    )

    tables_dir = Path("results/tables")
    out_path = tables_dir / "convergence_timings.csv"
    conv.to_csv(out_path, index=False)
    print(f"Saved convergence runtime table:\n  {out_path}")


# ---------------------- plotting ----------------------
def _load_runtime_table(stem: str) -> pd.DataFrame:
    """
    Load aggregated runtime table from results/tables/{stem}.csv
    stem ∈ {
      "single_normal_timings", "single_t_timings",
      "multi_normal_timings", "multi_t_timings",
      "convergence_timings"
    }
    """
    path = Path("results/tables") / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. "
                                f"Run aggregation step first.")
    df = pd.read_csv(path)
    return df


def _plot_three_buckets(df: pd.DataFrame, title: str, stem: str):
    """
    Grouped bar chart:
      x-axis: p
      bars  : [Data generation, Estimator computation, Evaluation]
    """
    labels = ["Data generation", "Estimator computation", "Evaluation"]
    keys   = ["data_gen", "computing_estimator", "principal_angles"]

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
    ax.set_ylabel("Total seconds (per $p$)")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=1)

    _save(fig, stem)
    plt.close(fig)


def _plot_from_tables():
    """
    Read aggregated runtime tables from results/tables and plot:

      single_normal_timings.(pdf|svg)
      single_t_timings.(pdf|svg)
      multi_normal_timings.(pdf|svg)
      multi_t_timings.(pdf|svg)
      convergence_timings.(pdf|svg)
    """
    # Single
    single_normal = _load_runtime_table("single_normal_timings")
    single_t      = _load_runtime_table("single_t_timings")

    _plot_three_buckets(
        single_normal,
        "Single simulation runtime breakdown (Normal)",
        "single_normal_timings",
    )

    _plot_three_buckets(
        single_t,
        "Single simulation runtime breakdown (Student-t)",
        "single_t_timings",
    )

    # Multi
    multi_normal = _load_runtime_table("multi_normal_timings")
    multi_t      = _load_runtime_table("multi_t_timings")

    _plot_three_buckets(
        multi_normal,
        "Multi simulation runtime breakdown (Normal)",
        "multi_normal_timings",
    )

    _plot_three_buckets(
        multi_t,
        "Multi simulation runtime breakdown (Student-t)",
        "multi_t_timings",
    )

    # Convergence (all SNRs aggregated)
    conv = _load_runtime_table("convergence_timings")
    _plot_three_buckets(
        conv,
        "Convergence simulation runtime breakdown (all SNRs)",
        "convergence_timings",
    )


# ---------------------- main ----------------------
def main():
    _ensure_dirs()
    _set_pub_style()

    # 1) single_step_timings.csv → single_normal/t_timings.csv (tables/)
    _build_single_runtime_tables()

    # 2) multi_step_timings.csv → multi_normal/t_timings.csv (tables/)
    _build_multi_runtime_tables()

    # 3) convergence_step_timings.csv → convergence_timings.csv (tables/)
    _build_convergence_runtime_table()

    # 4) read those tables and make figures (figures/)
    _plot_from_tables()


if __name__ == "__main__":
    main()