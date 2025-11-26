# profiling.py
#
# Build aggregated runtime profiling tables from step-level timing CSVs.
#
# Inputs (per variant):
#   timings/tables/single_step_timings_{variant}.csv
#   timings/tables/multi_step_timings_{variant}.csv
#   timings/tables/convergence_step_timings_{variant}.csv
#
# Outputs (per variant):
#   timings/tables/single_normal_profiling_{variant}.csv
#   timings/tables/single_t_profiling_{variant}.csv
#   timings/tables/multi_normal_profiling_{variant}.csv
#   timings/tables/multi_t_profiling_{variant}.csv
#   timings/tables/convergence_profiling_{variant}.csv

from pathlib import Path
import pandas as pd

N_JOBS_SINGLE = 10
N_JOBS_MULTI = 10
N_JOBS_CONVERGENCE = 6

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


# ---------------------- generic aggregation helper ----------------------
def _aggregate_buckets(df, mask_data, mask_sub, mask_ang):
    """
    Aggregate raw step timings into three coarse buckets:
      - data_generation
      - estimator_computation
      - metric_computation
    """
    g1 = (
        df.loc[mask_data]
          .groupby("p", as_index=False)["seconds_total"]
          .sum()
          .rename(columns={"seconds_total": "data_generation"})
    )

    g2 = (
        df.loc[mask_sub]
          .groupby("p", as_index=False)["seconds_total"]
          .sum()
          .rename(columns={"seconds_total": "estimator_computation"})
    )

    g3 = (
        df.loc[mask_ang]
          .groupby("p", as_index=False)["seconds_total"]
          .sum()
          .rename(columns={"seconds_total": "metric_computation"})
    )

    merged = (
        g1.merge(g2, on="p", how="outer")
          .merge(g3, on="p", how="outer")
          .fillna(0.0)
          .sort_values("p")
    )
    return merged


# ---------------------- loading helpers ----------------------
def _load_step_timings(kind: str, variant_key: str) -> pd.DataFrame:
    """
    Load step-level timing CSV for a given simulation kind and variant.

    kind ∈ {"single", "multi"}
    variant_key ∈ VARIANTS.keys()
    """
    csv_path = Path("timings/tables") / f"{kind}_step_timings_{variant_key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"p", "step", "seconds_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def _load_convergence_step_timings(variant_key: str) -> pd.DataFrame:
    """
    Load step-level timing CSV for convergence-rate simulation.
    """
    csv_path = Path("timings/tables") / f"convergence_step_timings_{variant_key}.csv"
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
    """
    Build aggregated runtime tables for the single-spike simulations
    (Normal and Student-t) for a given variant, and save them under
    timings/tables/*_profiling_{variant}.csv.
    """
    df = _load_step_timings("single", variant_key)

    # Normal
    is_data_gen_normal = df["step"].str.startswith("data_gen_normal")
    is_arg_normal_sub  = df["step"].str.startswith("ARG_normal_compute_ARG_PC_subspace")
    is_base_normal_sub = df["step"].str.startswith("baseline_normal_compute_PC_subspace")
    is_arg_normal_ang  = df["step"].str.startswith("ARG_normal_compute_principal_angles")
    is_base_normal_ang = df["step"].str.startswith("baseline_normal_compute_principal_angles")

    # Student-t
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

    # If this variant used trial-level parallelization, convert summed CPU time
    # into an approximate wall-clock time by dividing by the number of jobs.
    if variant_key == "cholesky+parallelization":
        for df_bucket in (normal, tdf):
            df_bucket[["data_generation",
                       "estimator_computation",
                       "metric_computation"]] /= float(N_JOBS_SINGLE)

    base = Path("timings/tables")
    base.mkdir(parents=True, exist_ok=True)

    normal.to_csv(base / f"single_normal_profiling_{variant_key}.csv", index=False)
    tdf.to_csv(base / f"single_t_profiling_{variant_key}.csv", index=False)


# ---------------------- multi simulation ----------------------
def _build_multi_runtime_tables(variant_key: str):
    """
    Build aggregated runtime tables for the multi-spike simulations
    (Normal and Student-t) for a given variant, and save them under
    timings/tables/*_profiling_{variant}.csv.
    """
    df = _load_step_timings("multi", variant_key)

    # Normal
    is_data_gen_normal = df["step"].str.startswith("data_gen_normal")
    is_arg_normal_sub  = df["step"].str.startswith("ARG_normal_compute_ARG_PC_subspace")
    is_pca_normal_sub  = df["step"].str.startswith("PCA_normal_compute_PC_subspace")
    is_arg_normal_ang  = df["step"].str.startswith("ARG_normal_compute_principal_angles")
    is_pca_normal_ang  = df["step"].str.startswith("PCA_normal_compute_principal_angles")

    # Student-t
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

    if variant_key == "cholesky+parallelization":
        for df_bucket in (normal, tdf):
            df_bucket[["data_generation",
                       "estimator_computation",
                       "metric_computation"]] /= float(N_JOBS_MULTI)

    base = Path("timings/tables")
    base.mkdir(parents=True, exist_ok=True)

    normal.to_csv(base / f"multi_normal_profiling_{variant_key}.csv", index=False)
    tdf.to_csv(base / f"multi_t_profiling_{variant_key}.csv", index=False)


# ---------------------- convergence simulation ----------------------
def _build_convergence_runtime_table(variant_key: str):
    """
    Build aggregated runtime table for the convergence-rate simulation
    for a given variant, and save it under
    timings/tables/convergence_profiling_{variant}.csv.
    """
    df = _load_convergence_step_timings(variant_key)

    is_data_gen = df["step"].str.startswith("data_gen")
    is_disc     = df["step"].str.contains("compute_discriminant")
    is_alpha    = df["step"].str.startswith("alpha_accumulate")
    is_ip       = df["step"].str.startswith("inner_product")

    conv = _aggregate_buckets(
        df,
        mask_data=is_data_gen,
        mask_sub=is_disc,
        mask_ang=is_alpha | is_ip,
    )

    if variant_key == "cholesky+parallelization":
        conv[["data_generation",
              "estimator_computation",
              "metric_computation"]] /= float(N_JOBS_CONVERGENCE)

    base = Path("timings/tables")
    base.mkdir(parents=True, exist_ok=True)

    conv.to_csv(base / f"convergence_profiling_{variant_key}.csv", index=False)


# ---------------------- main ----------------------
def main():
    base = Path("timings/tables")
    base.mkdir(parents=True, exist_ok=True)

    for variant_key, info in VARIANTS.items():
        pretty = info["pretty"]
        print(f"\n=== Building profiling tables: {variant_key} ({pretty}) ===")

        _build_single_runtime_tables(variant_key)
        _build_multi_runtime_tables(variant_key)
        _build_convergence_runtime_table(variant_key)

    print("\nDone building profiling CSVs.")


if __name__ == "__main__":
    main()