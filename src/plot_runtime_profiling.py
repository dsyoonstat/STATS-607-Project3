#!/usr/bin/env python3
# plot_runtime_profiling_variants.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------- configuration ----------------------

# variant_key: 파일명에 그대로 사용됨
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
    base = Path("results")
    (base / "tables").mkdir(exist_ok=True, parents=True)
    (base / "figures").mkdir(exist_ok=True, parents=True)


def _save(fig, stem: str):
    out_pdf = Path("results/figures") / f"{stem}.pdf"
    out_svg = Path("results/figures") / f"{stem}.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    print(f"Saved: {out_pdf}\n       {out_svg}")


# ---------------------- generic aggregation helper ----------------------
def _aggregate_buckets(df, mask_data, mask_sub, mask_ang):
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


# ---------------------- loading helpers ----------------------
def _load_step_timings(kind: str, variant_key: str):
    csv_path = Path("results/tables") / f"{kind}_step_timings_{variant_key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"p", "step", "seconds_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def _load_convergence_step_timings(variant_key: str):
    csv_path = Path("results/tables") / f"convergence_step_timings_{variant_key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"p", "step", "seconds_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


# ---------------------- single simulation ----------------------
def _build_single_runtime_tables(variant_key: str):
    df = _load_step_timings("single", variant_key)

    is_data_gen_normal = df["step"].str.startswith("data_gen_normal")
    is_arg_normal_sub  = df["step"].str.startswith("ARG_normal_compute_ARG_PC_subspace")
    is_base_normal_sub = df["step"].str.startswith("baseline_normal_compute_PC_subspace")
    is_arg_normal_ang  = df["step"].str.startswith("ARG_normal_compute_principal_angles")
    is_base_normal_ang = df["step"].str.startswith("baseline_normal_compute_principal_angles")

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

    base = Path("results/tables")
    normal.to_csv(base / f"single_normal_timings_{variant_key}.csv", index=False)
    tdf.to_csv(base / f"single_t_timings_{variant_key}.csv", index=False)


# ---------------------- multi simulation ----------------------
def _build_multi_runtime_tables(variant_key: str):
    df = _load_step_timings("multi", variant_key)

    is_data_gen_normal = df["step"].str.startswith("data_gen_normal")
    is_arg_normal_sub  = df["step"].str.startswith("ARG_normal_compute_ARG_PC_subspace")
    is_pca_normal_sub  = df["step"].str.startswith("PCA_normal_compute_PC_subspace")
    is_arg_normal_ang  = df["step"].str.startswith("ARG_normal_compute_principal_angles")
    is_pca_normal_ang  = df["step"].str.startswith("PCA_normal_compute_principal_angles")

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

    base = Path("results/tables")
    normal.to_csv(base / f"multi_normal_timings_{variant_key}.csv", index=False)
    tdf.to_csv(base / f"multi_t_timings_{variant_key}.csv", index=False)


# ---------------------- convergence simulation ----------------------
def _build_convergence_runtime_table(variant_key: str):
    df = _load_convergence_step_timings(variant_key)

    is_data_gen = df["step"].str.startswith("data_gen")
    is_disc     = df["step"].str.contains("compute_discriminant")
    is_alpha    = df["step"].str.startswith("alpha_accumulate")
    is_ip       = df["step"].str.startswith("inner_product")

    conv = _aggregate_buckets(df,
                              mask_data=is_data_gen,
                              mask_sub=is_disc,
                              mask_ang=is_alpha | is_ip)

    Path("results/tables") \
        .joinpath(f"convergence_timings_{variant_key}.csv") \
        .write_text(conv.to_csv(index=False))


# ---------------------- plotting ----------------------
def _load_runtime_table(stem: str, variant_key: str):
    path = Path("results/tables") / f"{stem}_{variant_key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    return pd.read_csv(path)


def _plot_three_buckets(df, title: str, stem: str):
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


def _plot_variant(variant_key: str, pretty: str):
    stems = [
        ("single_normal_timings", "Single simulation runtime breakdown (Normal)"),
        ("single_t_timings",      "Single simulation runtime breakdown (Student-t)"),
        ("multi_normal_timings",  "Multi simulation runtime breakdown (Normal)"),
        ("multi_t_timings",       "Multi simulation runtime breakdown (Student-t)"),
        ("convergence_timings",   "Convergence simulation runtime breakdown (all SNRs)"),
    ]

    for stem, title in stems:
        df = _load_runtime_table(stem, variant_key)
        _plot_three_buckets(
            df,
            f"{title} – {pretty}",
            f"{stem}_{variant_key}"
        )


# ---------------------- main ----------------------
def main():
    _ensure_dirs()
    _set_pub_style()

    for variant_key, info in VARIANTS.items():
        pretty = info["pretty"]
        print(f"\n=== Processing: {variant_key} ===")

        _build_single_runtime_tables(variant_key)
        _build_multi_runtime_tables(variant_key)
        _build_convergence_runtime_table(variant_key)
        _plot_variant(variant_key, pretty)


if __name__ == "__main__":
    main()