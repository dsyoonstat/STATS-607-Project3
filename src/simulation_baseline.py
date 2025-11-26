# simulation_baseline.py
# - Single-spike, single-reference (Normal + t)
# - Multi-spike, multi-reference (Normal + t)
# - Convergence-rate study for Theorem 4 (Normal)
#
# Outputs:
#   results/tables/baseline/*.csv (summary tables)
#   timings/tables/*.csv          (runtime profiling)

from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict
from contextlib import contextmanager

# === import from files ===
from dgps import (
    generate_basis,
    generate_reference_vectors,
    sigma_single_spike,
    sigma_multi_spike,
    sample_normal,
    sample_t,
)
from methods import (
    compute_PC_subspace,
    compute_ARG_PC_subspace,
    compute_negative_ridge_discriminants,
)
from metrics import compute_principal_angles

# Tools for runtime profiling
class StepTimer:
    """Accumulates elapsed seconds per named step with a context manager."""
    def __init__(self):
        self.acc = defaultdict(float)

    @contextmanager
    def section(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.acc[key] += time.perf_counter() - t0

    def snapshot_and_reset(self):
        """Return dict of accumulated seconds and reset the storage."""
        d = dict(self.acc)
        self.acc.clear()
        return d

@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"⏱️ {label} took {dt:.2f} s")

# ------------------------------------------------------------
# Single-spike, single-reference simulation (Normal + t)
# ------------------------------------------------------------

def run_simulation_single(
    p_list: List[int],
    a_list: List[float],
    n: int,
    nu: int,
    n_trials: int,
    sigma_coef: Tuple[float, float],
    master_seed: int = 725,
) -> None:
    # --- paths ---
    results_base = Path("results")
    results_tables_dir = results_base / "tables" / "baseline"
    results_tables_dir.mkdir(exist_ok=True, parents=True)

    timings_base = Path("timings")
    timings_tables_dir = timings_base / "tables"
    timings_tables_dir.mkdir(exist_ok=True, parents=True)

    # Column labels (baseline + a^2 columns)
    col_labels = ["baseline"] + [f"a^2={a**2:.2g}" for a in a_list]

    mean_normal = np.zeros((len(p_list), len(col_labels)))
    mean_t      = np.zeros_like(mean_normal)
    std_normal  = np.zeros_like(mean_normal)
    std_t       = np.zeros_like(mean_normal)

    master_rng = np.random.default_rng(master_seed)

    # rows to write per-p step timings
    per_p_rows = []

    for pi, p in enumerate(tqdm(p_list, desc="single: p sweep", leave=False)):
        step = StepTimer()  # reset per p

        # Σ = c1 * p * e1 e1^T + c2 I_p, u1 = e1
        Sigma, u1 = sigma_single_spike(p=p, coef=sigma_coef)
        if u1.ndim == 1:
            u1 = u1[:, None]  # (p,1)

        E = generate_basis(p)  # (p,4)
        mu = np.zeros(p)

        normal_trials, t_trials = [], []
        # Build trials
        for _ in tqdm(range(n_trials), desc=f"single: build trials (p={p})", leave=False):
            seed_n = int(master_rng.integers(0, 2**31 - 1))
            seed_t = int(master_rng.integers(0, 2**31 - 1))
            with step.section("data_gen_normal"):
                normal_trials.append(sample_normal(Sigma=Sigma, n=n, mu=mu, seed=seed_n))
            with step.section("data_gen_t"):
                t_trials.append(sample_t(Sigma=Sigma, nu=nu, n=n, mu=mu, seed=seed_t))

        # Baseline (PCA top-1)
        baseline_normal_angles = np.zeros(n_trials)
        baseline_t_angles      = np.zeros(n_trials)

        for i in tqdm(range(n_trials), desc=f"single: baseline eval (p={p})", leave=False):
            with step.section("baseline_normal_compute_PC_subspace"):
                U_pc = compute_PC_subspace(samples=normal_trials[i], spike_number=1)
            with step.section("baseline_normal_compute_principal_angles"):
                baseline_normal_angles[i] = compute_principal_angles(u1, U_pc)[0]

            with step.section("baseline_t_compute_PC_subspace"):
                U_pc_t = compute_PC_subspace(samples=t_trials[i], spike_number=1)
            with step.section("baseline_t_compute_principal_angles"):
                baseline_t_angles[i] = compute_principal_angles(u1, U_pc_t)[0]

        mean_normal[pi, 0] = baseline_normal_angles.mean()
        mean_t[pi, 0]      = baseline_t_angles.mean()
        std_normal[pi, 0]  = baseline_normal_angles.std(ddof=0)
        std_t[pi, 0]       = baseline_t_angles.std(ddof=0)

        # ARG for each a
        for aj, a in enumerate(tqdm(a_list, desc=f"single: ARG loop (p={p})", leave=False)):
            A = np.array([[a, np.sqrt(max(0.0, 1.0 - a*a)), 0.0, 0.0]])
            V = generate_reference_vectors(E, A)  # (p,1)
            a2_tag = f"{a**2:.2g}"

            arg_normal_all = np.zeros(n_trials)
            arg_t_all      = np.zeros(n_trials)

            for i in range(n_trials):
                with step.section(f"ARG_normal_compute_ARG_PC_subspace[a2={a2_tag}]"):
                    U_ARG = compute_ARG_PC_subspace(
                        samples=normal_trials[i], reference_vectors=V, spike_number=1, orthonormal=True
                    )
                with step.section(f"ARG_normal_compute_principal_angles[a2={a2_tag}]"):
                    arg_normal_all[i] = compute_principal_angles(u1, U_ARG)[0]

                with step.section(f"ARG_t_compute_ARG_PC_subspace[a2={a2_tag}]"):
                    U_ARG_t = compute_ARG_PC_subspace(
                        samples=t_trials[i], reference_vectors=V, spike_number=1, orthonormal=True
                    )
                with step.section(f"ARG_t_compute_principal_angles[a2={a2_tag}]"):
                    arg_t_all[i] = compute_principal_angles(u1, U_ARG_t)[0]

            mean_normal[pi, 1+aj] = arg_normal_all.mean()
            mean_t[pi, 1+aj]      = arg_t_all.mean()
            std_normal[pi, 1+aj]  = arg_normal_all.std(ddof=0)
            std_t[pi, 1+aj]       = arg_t_all.std(ddof=0)

        # --- collect per-p timings
        snap = step.snapshot_and_reset()
        for k, sec in sorted(snap.items()):
            per_p_rows.append({
                "p": p,
                "step": k,
                "seconds_total": sec,
                "seconds_per_trial": sec / n_trials if n_trials > 0 else np.nan,
            })

    # Save summaries to results/tables/baseline
    idx = p_list
    pd.DataFrame(mean_normal, index=idx, columns=col_labels).to_csv(
        results_tables_dir / "single_normal_mean.csv"
    )
    pd.DataFrame(std_normal, index=idx, columns=col_labels).to_csv(
        results_tables_dir / "single_normal_std.csv"
    )
    pd.DataFrame(mean_t, index=idx, columns=col_labels).to_csv(
        results_tables_dir / "single_t_mean.csv"
    )
    pd.DataFrame(std_t, index=idx, columns=col_labels).to_csv(
        results_tables_dir / "single_t_std.csv"
    )

    # Save per-step timings (long format) to timings/tables
    timing_df = pd.DataFrame(per_p_rows)
    timing_df.to_csv(timings_tables_dir / "single_step_timings_baseline.csv", index=False)
    print("🕒 Wrote step timings to timings/tables/single_step_timings_baseline.csv")
    print("✅ Single simulation complete. Saved in results/tables/baseline")

# ------------------------------------------------------------
# Multi-spike, multi-reference simulation (Normal + t)
# ------------------------------------------------------------

def run_simulation_multi(
    p_list: List[int],
    n: int,
    nu: int,
    n_trials: int,
    sigma_coef: Tuple[float, float, float],
    master_seed: int = 725,
) -> None:
    """
    Run the multi-spike / multi-reference simulation under Normal and t sampling,
    write summary tables (means/stds) to results/tables/baseline, and record per-p
    step timings to timings/tables/multi_step_timings_baseline.csv.
    """
    # --- paths ---
    results_base = Path("results")
    results_tables_dir = results_base / "tables" / "baseline"
    results_tables_dir.mkdir(exist_ok=True, parents=True)

    timings_base = Path("timings")
    timings_tables_dir = timings_base / "tables"
    timings_tables_dir.mkdir(exist_ok=True, parents=True)

    col_labels = ["ARG1", "PCA1", "ARG2", "PCA2"]
    mean_normal = np.zeros((len(p_list), len(col_labels)))
    std_normal  = np.zeros_like(mean_normal)
    mean_t      = np.zeros_like(mean_normal)
    std_t       = np.zeros_like(mean_normal)

    master_rng = np.random.default_rng(master_seed)

    # step timings (long format)
    per_p_rows = []

    for pi, p in enumerate(tqdm(p_list, desc="multi: p sweep", leave=False)):
        step = StepTimer()  # reset per p

        Sigma, U_m = sigma_multi_spike(p=p, coef=sigma_coef)
        E = generate_basis(p)
        mu = np.zeros(p)

        A = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1/np.sqrt(2), 0.0, -1/np.sqrt(2), 0.0],
        ])
        V = generate_reference_vectors(E, A)  # (p,2)

        angle_normal = np.zeros((n_trials, 4))  # [ARG1, PCA1, ARG2, PCA2]
        angle_t      = np.zeros((n_trials, 4))

        for i in tqdm(range(n_trials), desc=f"multi: trials (p={p})", leave=False):
            seed_n = int(master_rng.integers(0, 2**31 - 1))
            seed_t = int(master_rng.integers(0, 2**31 - 1))

            with step.section("data_gen_normal"):
                Xn = sample_normal(Sigma=Sigma, n=n, mu=mu, seed=seed_n)
            with step.section("data_gen_t"):
                Xt = sample_t(Sigma=Sigma, nu=nu, n=n, mu=mu, seed=seed_t)

            # Normal
            with step.section("ARG_normal_compute_ARG_PC_subspace(k=2)"):
                U_ARG = compute_ARG_PC_subspace(Xn, V, spike_number=2, orthonormal=True)
            with step.section("PCA_normal_compute_PC_subspace(k=2)"):
                U_PCA = compute_PC_subspace(Xn, spike_number=2)
            with step.section("ARG_normal_compute_principal_angles"):
                th_ARG = compute_principal_angles(U_ARG, U_m)
            with step.section("PCA_normal_compute_principal_angles"):
                th_PCA = compute_principal_angles(U_PCA, U_m)
            angle_normal[i, :] = [th_ARG[0], th_PCA[0], th_ARG[1], th_PCA[1]]

            # t
            with step.section("ARG_t_compute_ARG_PC_subspace(k=2)"):
                U_ARG_t = compute_ARG_PC_subspace(Xt, V, spike_number=2, orthonormal=True)
            with step.section("PCA_t_compute_PC_subspace(k=2)"):
                U_PCA_t = compute_PC_subspace(Xt, spike_number=2)
            with step.section("ARG_t_compute_principal_angles"):
                th_ARG_t = compute_principal_angles(U_ARG_t, U_m)
            with step.section("PCA_t_compute_principal_angles"):
                th_PCA_t = compute_principal_angles(U_PCA_t, U_m)
            angle_t[i, :] = [th_ARG_t[0], th_PCA_t[0], th_ARG_t[1], th_PCA_t[1]]

        mean_normal[pi, :] = angle_normal.mean(axis=0)
        mean_t[pi, :]      = angle_t.mean(axis=0)
        std_normal[pi, :]  = angle_normal.std(axis=0, ddof=0)
        std_t[pi, :]       = angle_t.std(axis=0, ddof=0)

        # collect per-p timings
        snap = step.snapshot_and_reset()
        for k, sec in sorted(snap.items()):
            per_p_rows.append({
                "p": p,
                "step": k,
                "seconds_total": sec,
                "seconds_per_trial": sec / n_trials if n_trials > 0 else np.nan,
            })

    # Save summaries to results/tables/baseline
    idx = p_list
    pd.DataFrame(mean_normal, index=idx, columns=col_labels).to_csv(
        results_tables_dir / "multi_normal_mean.csv"
    )
    pd.DataFrame(std_normal, index=idx, columns=col_labels).to_csv(
        results_tables_dir / "multi_normal_std.csv"
    )
    pd.DataFrame(mean_t, index=idx, columns=col_labels).to_csv(
        results_tables_dir / "multi_t_mean.csv"
    )
    pd.DataFrame(std_t, index=idx, columns=col_labels).to_csv(
        results_tables_dir / "multi_t_std.csv"
    )

    # Save step timings to timings/tables
    timing_df = pd.DataFrame(per_p_rows)
    timing_df.to_csv(timings_tables_dir / "multi_step_timings_baseline.csv", index=False)
    print("🕒 Wrote step timings to timings/tables/multi_step_timings_baseline.csv")
    print("✅ Multi simulation complete. Saved in results/tables/baseline")

# ------------------------------------------------------------
# Convergence-rate simulation for Theorem 4
# ------------------------------------------------------------

def run_simulation_convergence_rate(
    p_list: List[int],
    n: int,
    n_trials: int,
    powers: List[float],
    snr_list: List[float],
    master_seed: int = 725,
) -> None:
    """
    Estimate convergence-rate curves and record per-(snr,p) step timings to
    timings/tables/convergence_step_timings_baseline.csv.
    """
    # --- paths ---
    results_base = Path("results")
    results_tables_dir = results_base / "tables" / "baseline"
    results_tables_dir.mkdir(exist_ok=True, parents=True)

    timings_base = Path("timings")
    timings_tables_dir = timings_base / "tables"
    timings_tables_dir.mkdir(exist_ok=True, parents=True)

    def _snr_tag(snr: float) -> str:
        return f"{snr:g}"

    # step timings (long format)
    per_rows = []

    # Outer loop over SNR settings
    for snr in tqdm(snr_list, desc="convergence: SNR sweep", leave=False):
        if snr <= 0:
            raise ValueError(f"SNR must be positive; got {snr}")
        c1 = 1.0
        c2 = c1 / snr

        mean_mat = np.zeros((len(p_list), len(powers)))
        std_mat  = np.zeros_like(mean_mat)   # kept in case

        master_rng = np.random.default_rng(master_seed)

        for pi, p in enumerate(tqdm(p_list, desc=f"  p sweep (snr={_snr_tag(snr)})", leave=False)):
            step = StepTimer()  # reset per (snr,p)

            # Σ = c1 * p * e1 e1^T + c2 I_p, u1 = e1
            Sigma, u1 = sigma_single_spike(p=p, coef=(c1, c2))
            if u1.ndim == 1:
                u1 = u1[:, None]  # (p,1)

            # v1 = (e1 + e2)/sqrt(2)
            E = generate_basis(p)
            A = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0]])  # (1,4)
            V = generate_reference_vectors(E, A)                     # (p,1)

            mu = np.zeros(p)
            vals = np.zeros((n_trials, len(powers)))  # trial × power

            for t in range(n_trials):
                seed_t = int(master_rng.integers(0, 2**31 - 1))
                with step.section("data_gen_normal"):
                    X = sample_normal(Sigma=Sigma, n=n, mu=mu, seed=seed_t)  # (n,p)

                with step.section("compute_discriminant"):
                    D = compute_negative_ridge_discriminants(
                        samples=X,
                        reference_vectors=V,
                        spike_number=1,
                        normalize=True,
                    )  # (p,1)
                    d1 = D[:, 0]

                with step.section("inner_product"):
                    ip = float(np.abs(u1[:, 0].T @ d1))  # |u1^T d1|

                with step.section("alpha_accumulate"):
                    # p^α * |u1^T d1| for each alpha
                    for j, alpha in enumerate(powers):
                        vals[t, j] = (p ** alpha) * ip

            # Aggregate at fixed p
            mean_mat[pi, :] = vals.mean(axis=0)
            std_mat[pi, :]  = vals.std(axis=0, ddof=0)

            # collect per-(snr,p) timings
            snap = step.snapshot_and_reset()
            for k, sec in sorted(snap.items()):
                per_rows.append({
                    "snr": snr,
                    "p": p,
                    "step": k,
                    "seconds_total": sec,
                    "seconds_per_trial": sec / n_trials if n_trials > 0 else np.nan,
                })

        # Normalize using the first row as baseline
        baseline = np.where(mean_mat[0, :] == 0.0, 1.0, mean_mat[0, :])
        mean_norm = mean_mat / baseline[None, :]

        # Save normalized table for this SNR to results/tables/baseline
        col_labels = [f"alpha={a:g}" for a in powers]
        idx = p_list
        out_csv = results_tables_dir / f"convergence_rate_snr{_snr_tag(snr)}.csv"
        pd.DataFrame(mean_norm, index=idx, columns=col_labels).to_csv(out_csv)

    # Save step timings to timings/tables
    timing_df = pd.DataFrame(per_rows)
    timing_df.to_csv(timings_tables_dir / "convergence_step_timings_baseline.csv", index=False)
    print("🕒 Wrote step timings to timings/tables/convergence_step_timings_baseline.csv")
    print("✅ Convergence-rate simulation complete. Saved in results/tables/baseline")

# ------------------------------------------------------------------
# Run simulation
# ------------------------------------------------------------------
if __name__ == "__main__":
    # overall timings go under timings/tables
    timings_root = Path("timings")
    tables_dir = timings_root / "tables"
    tables_dir.mkdir(exist_ok=True, parents=True)

    timings = []

    with timer("run_simulation_single"):
        t0 = time.perf_counter()
        run_simulation_single(
            p_list=[100, 200, 500, 1000, 2000],
            a_list=[0.0, 0.5, 1/np.sqrt(2), np.sqrt(3)/2, 1.0],
            n=40,
            nu=5,
            n_trials=100,
            sigma_coef=(1.0, 40.0),
            master_seed=725,
        )
        timings.append(("run_simulation_single", time.perf_counter() - t0))

    with timer("run_simulation_multi"):
        t0 = time.perf_counter()
        run_simulation_multi(
            p_list=[100, 200, 500, 1000, 2000],
            n=40,
            nu=5,
            n_trials=100,
            sigma_coef=(2.0, 1.0, 40.0),
            master_seed=725,
        )
        timings.append(("run_simulation_multi", time.perf_counter() - t0))

    with timer("run_simulation_convergence_rate"):
        t0 = time.perf_counter()
        run_simulation_convergence_rate(
            p_list=[100, 200, 500, 1000, 2000],
            n=40,
            n_trials=100,
            powers=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            snr_list=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            master_seed=725,
        )
        timings.append(("run_simulation_convergence_rate", time.perf_counter() - t0))

    # Save overall simulation timings to timings/tables/timings_baseline.csv
    timing_df = pd.DataFrame(timings, columns=["task", "seconds"])
    timing_df["seconds"] = timing_df["seconds"].round(2)
    timing_df.to_csv(tables_dir / "timings_baseline.csv", index=False)

    print("🕒 Wrote total simulation timings to timings/tables/timings_baseline.csv")
    print(timing_df)