# simulation_cholesky+parallelization.py
# - Single-spike, single-reference (Normal + t)
# - Multi-spike, multi-reference (Normal + t)
# - Convergence-rate study for Theorem 4 (Normal)
#
# Outputs:
#   results/tables/cholesky+parallelization/*.csv (summary tables)
#   timings/tables/*.csv                           (runtime profiling)

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import math
import os

# === import from files ===
from dgps import (
    generate_basis,
    generate_reference_vectors,
    sigma_single_spike,
    sigma_multi_spike,
    sample_normal_fast,
    sample_t,
)
from methods import (
    compute_PC_subspace,
    compute_ARG_PC_subspace,
    compute_negative_ridge_discriminants,
)
from metrics import compute_principal_angles

N_JOBS_SINGLE = 10
N_JOBS_MULTI = 10
N_JOBS_CONVERGENCE = 6

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

def _worker_single_trial_chunk(
    p: int,
    a_list: List[float],
    n: int,
    nu: int,
    sigma_coef: Tuple[float, float],
    seeds_normal_chunk: np.ndarray,
    seeds_t_chunk: np.ndarray,
) -> Dict[str, Any]:
    """
    Worker function for run_simulation_single:
    For a fixed p, process a chunk of trials given by seeds_normal_chunk / seeds_t_chunk.
    """
    step = StepTimer()

    # Σ = c1 * p * e1 e1^T + c2 I_p, u1 = e1
    Sigma, u1 = sigma_single_spike(p=p, coef=sigma_coef)
    if u1.ndim == 1:
        u1 = u1[:, None]  # (p,1)

    E = generate_basis(p)  # (p,4)
    mu = np.zeros(p)

    n_chunk = len(seeds_normal_chunk)
    n_a = len(a_list)

    baseline_normal = np.zeros(n_chunk)
    baseline_t      = np.zeros(n_chunk)
    arg_normal      = np.zeros((n_chunk, n_a))
    arg_t           = np.zeros((n_chunk, n_a))

    # Precompute reference vectors for each a
    V_list = []
    a2_tags = []
    for a in a_list:
        A = np.array([[a, np.sqrt(max(0.0, 1.0 - a * a)), 0.0, 0.0]])
        V = generate_reference_vectors(E, A)  # (p,1)
        V_list.append(V)
        a2_tags.append(f"{a**2:.2g}")

    # Process each trial in this chunk
    for i, (seed_n, seed_t) in enumerate(zip(seeds_normal_chunk, seeds_t_chunk)):
        # Data generation
        with step.section("data_gen_normal"):
            Xn = sample_normal_fast(Sigma=Sigma, n=n, mu=mu, seed=int(seed_n))
        with step.section("data_gen_t"):
            Xt = sample_t(Sigma=Sigma, nu=nu, n=n, mu=mu, seed=int(seed_t))

        # Baseline (PCA top-1)
        with step.section("baseline_normal_compute_PC_subspace"):
            U_pc = compute_PC_subspace(samples=Xn, spike_number=1)
        with step.section("baseline_normal_compute_principal_angles"):
            baseline_normal[i] = compute_principal_angles(u1, U_pc)[0]

        with step.section("baseline_t_compute_PC_subspace"):
            U_pc_t = compute_PC_subspace(samples=Xt, spike_number=1)
        with step.section("baseline_t_compute_principal_angles"):
            baseline_t[i] = compute_principal_angles(u1, U_pc_t)[0]

        # ARG for each a
        for aj, (V, a2_tag) in enumerate(zip(V_list, a2_tags)):
            with step.section(f"ARG_normal_compute_ARG_PC_subspace[a2={a2_tag}]"):
                U_ARG = compute_ARG_PC_subspace(
                    samples=Xn,
                    reference_vectors=V,
                    spike_number=1,
                    orthonormal=True,
                )
            with step.section(f"ARG_normal_compute_principal_angles[a2={a2_tag}]"):
                arg_normal[i, aj] = compute_principal_angles(u1, U_ARG)[0]

            with step.section(f"ARG_t_compute_ARG_PC_subspace[a2={a2_tag}]"):
                U_ARG_t = compute_ARG_PC_subspace(
                    samples=Xt,
                    reference_vectors=V,
                    spike_number=1,
                    orthonormal=True,
                )
            with step.section(f"ARG_t_compute_principal_angles[a2={a2_tag}]"):
                arg_t[i, aj] = compute_principal_angles(u1, U_ARG_t)[0]

    timings = step.snapshot_and_reset()

    return {
        "baseline_normal": baseline_normal,
        "baseline_t": baseline_t,
        "arg_normal": arg_normal,
        "arg_t": arg_t,
        "timings": timings,
    }


def run_simulation_single(
    p_list: List[int],
    a_list: List[float],
    n: int,
    nu: int,
    n_trials: int,
    sigma_coef: Tuple[float, float],
    master_seed: int = 725,
    n_jobs: Optional[int] = None,
) -> None:
    """
    Single-spike, single-reference simulation (Normal + t),
    with trial-level parallelization.
    """
    # --- paths ---
    results_base = Path("results")
    results_tables_dir = results_base / "tables" / "cholesky+parallelization"
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

    # Master RNG for reproducible seeds across all (p, trial)
    master_rng = np.random.default_rng(master_seed)

    # rows to write per-p step timings
    per_p_rows: List[dict] = []

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    for pi, p in enumerate(tqdm(p_list, desc="single: p sweep", leave=False)):
        print(f"▶ single: p={p} (trial-parallel, n_jobs={n_jobs})")

        # Generate seeds for this p
        seeds_normal = np.empty(n_trials, dtype=np.int64)
        seeds_t      = np.empty(n_trials, dtype=np.int64)
        for i in range(n_trials):
            seeds_normal[i] = int(master_rng.integers(0, 2**31 - 1))
            seeds_t[i]      = int(master_rng.integers(0, 2**31 - 1))

        # Split trials into chunks for workers
        n_jobs_p = min(n_jobs, n_trials)  # cannot have more workers than trials
        chunk_size = math.ceil(n_trials / n_jobs_p)

        args_list = []
        for start in range(0, n_trials, chunk_size):
            end = min(start + chunk_size, n_trials)
            args_list.append(
                (
                    p,
                    a_list,
                    n,
                    nu,
                    sigma_coef,
                    seeds_normal[start:end],
                    seeds_t[start:end],
                )
            )

        baseline_normal_all = []
        baseline_t_all = []
        arg_normal_all = []
        arg_t_all = []
        total_timings = defaultdict(float)

        # Parallel execution over trial chunks
        with ProcessPoolExecutor(max_workers=n_jobs_p) as ex:
            futures = [ex.submit(_worker_single_trial_chunk, *args) for args in args_list]
            for fut in futures:
                res = fut.result()
                baseline_normal_all.append(res["baseline_normal"])
                baseline_t_all.append(res["baseline_t"])
                arg_normal_all.append(res["arg_normal"])
                arg_t_all.append(res["arg_t"])
                for k, sec in res["timings"].items():
                    total_timings[k] += sec

        # Concatenate results for all trials at this p
        baseline_normal_all = np.concatenate(baseline_normal_all)  # (n_trials,)
        baseline_t_all      = np.concatenate(baseline_t_all)       # (n_trials,)
        arg_normal_all      = np.vstack(arg_normal_all)            # (n_trials, len(a_list))
        arg_t_all           = np.vstack(arg_t_all)

        assert baseline_normal_all.shape[0] == n_trials
        assert arg_normal_all.shape[0] == n_trials

        # Aggregate means and stds as in the original serial version
        mean_normal[pi, 0] = baseline_normal_all.mean()
        mean_t[pi, 0]      = baseline_t_all.mean()
        std_normal[pi, 0]  = baseline_normal_all.std(ddof=0)
        std_t[pi, 0]       = baseline_t_all.std(ddof=0)

        for aj in range(len(a_list)):
            mean_normal[pi, 1 + aj] = arg_normal_all[:, aj].mean()
            mean_t[pi,      1 + aj] = arg_t_all[:, aj].mean()
            std_normal[pi,  1 + aj] = arg_normal_all[:, aj].std(ddof=0)
            std_t[pi,       1 + aj] = arg_t_all[:, aj].std(ddof=0)

        # Collect per-p timings (sum over workers)
        for k, sec in sorted(total_timings.items()):
            per_p_rows.append({
                "p": p,
                "step": k,
                "seconds_total": sec,
                "seconds_per_trial": sec / n_trials if n_trials > 0 else np.nan,
            })

    # Save summaries to results/tables/cholesky+parallelization
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
    timing_df.to_csv(
        timings_tables_dir / "single_step_timings_cholesky+parallelization.csv",
        index=False,
    )
    print("🕒 Wrote step timings to timings/tables/single_step_timings_cholesky+parallelization.csv")
    print("✅ Single simulation (trial-parallel) complete. Saved in results/tables/cholesky+parallelization")



# ------------------------------------------------------------
# Multi-spike, multi-reference simulation (Normal + t)
# ------------------------------------------------------------

def _worker_multi_trial_chunk(
    p: int,
    n: int,
    nu: int,
    sigma_coef: Tuple[float, float, float],
    seeds_normal_chunk: np.ndarray,
    seeds_t_chunk: np.ndarray,
) -> Dict[str, Any]:
    """
    Worker function for run_simulation_multi:
    For a fixed p, process a chunk of trials.
    """
    step = StepTimer()

    Sigma, U_m = sigma_multi_spike(p=p, coef=sigma_coef)
    E = generate_basis(p)
    mu = np.zeros(p)

    A = np.array([
        [0.5, 0.5, 0.5, 0.5],
        [1/np.sqrt(2), 0.0, -1/np.sqrt(2), 0.0],
    ])
    V = generate_reference_vectors(E, A)  # (p,2)

    n_chunk = len(seeds_normal_chunk)
    angle_normal = np.zeros((n_chunk, 4))  # [ARG1, PCA1, ARG2, PCA2]
    angle_t      = np.zeros((n_chunk, 4))

    for i, (seed_n, seed_t) in enumerate(zip(seeds_normal_chunk, seeds_t_chunk)):
        with step.section("data_gen_normal"):
            Xn = sample_normal_fast(Sigma=Sigma, n=n, mu=mu, seed=int(seed_n))
        with step.section("data_gen_t"):
            Xt = sample_t(Sigma=Sigma, nu=nu, n=n, mu=mu, seed=int(seed_t))

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

    timings = step.snapshot_and_reset()
    return {
        "angle_normal": angle_normal,
        "angle_t": angle_t,
        "timings": timings,
    }


def run_simulation_multi(
    p_list: List[int],
    n: int,
    nu: int,
    n_trials: int,
    sigma_coef: Tuple[float, float, float],
    master_seed: int = 725,
    n_jobs: Optional[int] = None,
) -> None:
    """
    Run the multi-spike / multi-reference simulation under Normal and t sampling,
    with trial-level parallelization.
    """
    # --- paths ---
    results_base = Path("results")
    results_tables_dir = results_base / "tables" / "cholesky+parallelization"
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
    per_p_rows: List[dict] = []

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    for pi, p in enumerate(tqdm(p_list, desc="multi: p sweep", leave=False)):
        print(f"▶ multi: p={p} (trial-parallel, n_jobs={n_jobs})")

        # Seeds for this p
        seeds_normal = np.empty(n_trials, dtype=np.int64)
        seeds_t      = np.empty(n_trials, dtype=np.int64)
        for i in range(n_trials):
            seeds_normal[i] = int(master_rng.integers(0, 2**31 - 1))
            seeds_t[i]      = int(master_rng.integers(0, 2**31 - 1))

        # Split into chunks
        n_jobs_p = min(n_jobs, n_trials)
        chunk_size = math.ceil(n_trials / n_jobs_p)

        args_list = []
        for start in range(0, n_trials, chunk_size):
            end = min(start + chunk_size, n_trials)
            args_list.append(
                (
                    p,
                    n,
                    nu,
                    sigma_coef,
                    seeds_normal[start:end],
                    seeds_t[start:end],
                )
            )

        angle_normal_all = []
        angle_t_all = []
        total_timings = defaultdict(float)

        with ProcessPoolExecutor(max_workers=n_jobs_p) as ex:
            futures = [ex.submit(_worker_multi_trial_chunk, *args) for args in args_list]
            for fut in futures:
                res = fut.result()
                angle_normal_all.append(res["angle_normal"])
                angle_t_all.append(res["angle_t"])
                for k, sec in res["timings"].items():
                    total_timings[k] += sec

        angle_normal_all = np.vstack(angle_normal_all)  # (n_trials, 4)
        angle_t_all      = np.vstack(angle_t_all)

        assert angle_normal_all.shape[0] == n_trials
        assert angle_t_all.shape[0] == n_trials

        mean_normal[pi, :] = angle_normal_all.mean(axis=0)
        mean_t[pi, :]      = angle_t_all.mean(axis=0)
        std_normal[pi, :]  = angle_normal_all.std(axis=0, ddof=0)
        std_t[pi, :]       = angle_t_all.std(axis=0, ddof=0)

        # collect per-p timings
        for k, sec in sorted(total_timings.items()):
            per_p_rows.append({
                "p": p,
                "step": k,
                "seconds_total": sec,
                "seconds_per_trial": sec / n_trials if n_trials > 0 else np.nan,
            })

    # Save summaries to results/tables/cholesky+parallelization
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
    timing_df.to_csv(
        timings_tables_dir / "multi_step_timings_cholesky+parallelization.csv",
        index=False,
    )
    print("🕒 Wrote step timings to timings/tables/multi_step_timings_cholesky+parallelization.csv")
    print("✅ Multi simulation (trial-parallel) complete. Saved in results/tables/cholesky+parallelization")



# ------------------------------------------------------------
# Convergence-rate simulation for Theorem 4
#   - Parallelized over SNR (each worker handles one snr)
# ------------------------------------------------------------

def _worker_convergence_for_snr(
    snr: float,
    p_list: List[int],
    n: int,
    n_trials: int,
    powers: List[float],
    master_seed: int,
) -> Dict[str, Any]:
    """
    Worker for a single SNR value.
    For this snr, loops over all p in p_list and all trials serially,
    returning mean_mat and per-(snr,p) timings.
    """
    if snr <= 0:
        raise ValueError(f"SNR must be positive; got {snr}")
    c1 = 1.0
    c2 = c1 / snr

    mean_mat = np.zeros((len(p_list), len(powers)))
    std_mat  = np.zeros_like(mean_mat)   # kept in case
    per_rows: List[dict] = []

    # For this snr, use a fresh RNG as in the original
    master_rng = np.random.default_rng(master_seed)

    for pi, p in enumerate(p_list):
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
                X = sample_normal_fast(Sigma=Sigma, n=n, mu=mu, seed=seed_t)  # (n,p)

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

    return {
        "snr": snr,
        "mean_mat": mean_mat,
        "per_rows": per_rows,
    }


def run_simulation_convergence_rate(
    p_list: List[int],
    n: int,
    n_trials: int,
    powers: List[float],
    snr_list: List[float],
    master_seed: int = 725,
    n_jobs: Optional[int] = None,
) -> None:
    """
    Estimate convergence-rate curves and record per-(snr,p) step timings to
    timings/tables/convergence_step_timings_cholesky+parallelization.csv,
    parallelizing over SNR values. Each worker handles one snr
    (all p, all trials) serially.
    """
    # --- paths ---
    results_base = Path("results")
    results_tables_dir = results_base / "tables" / "cholesky+parallelization"
    results_tables_dir.mkdir(exist_ok=True, parents=True)

    timings_base = Path("timings")
    timings_tables_dir = timings_base / "tables"
    timings_tables_dir.mkdir(exist_ok=True, parents=True)

    def _snr_tag(snr: float) -> str:
        return f"{snr:g}"

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, len(snr_list))
    else:
        n_jobs = min(n_jobs, len(snr_list))

    # step timings (long format)
    all_per_rows: List[dict] = []

    print(f"▶ convergence: parallel over snr (n_jobs={n_jobs}, #snr={len(snr_list)})")

    # Parallel over snr_list
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(
                _worker_convergence_for_snr,
                snr,
                p_list,
                n,
                n_trials,
                powers,
                master_seed,
            )
            for snr in snr_list
        ]

        for fut in futures:
            res = fut.result()
            snr = res["snr"]
            mean_mat = res["mean_mat"]
            per_rows = res["per_rows"]

            # Normalize using the first row as baseline
            baseline = np.where(mean_mat[0, :] == 0.0, 1.0, mean_mat[0, :])
            mean_norm = mean_mat / baseline[None, :]

            # Save normalized table for this SNR to results/tables/cholesky+parallelization
            col_labels = [f"alpha={a:g}" for a in powers]
            idx = p_list
            out_csv = results_tables_dir / f"convergence_rate_snr{_snr_tag(snr)}.csv"
            pd.DataFrame(mean_norm, index=idx, columns=col_labels).to_csv(out_csv)

            # Accumulate timings
            all_per_rows.extend(per_rows)

    # Save step timings to timings/tables
    timing_df = pd.DataFrame(all_per_rows)
    timing_df.to_csv(
        timings_tables_dir / "convergence_step_timings_cholesky+parallelization.csv",
        index=False,
    )
    print("🕒 Wrote step timings to timings/tables/convergence_step_timings_cholesky+parallelization.csv")
    print("✅ Convergence-rate simulation (snr-parallel) complete. Saved in results/tables/cholesky+parallelization")



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
            n_jobs=N_JOBS_SINGLE,
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
            n_jobs=N_JOBS_MULTI,
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
            n_jobs=N_JOBS_CONVERGENCE,
        )
        timings.append(("run_simulation_convergence_rate", time.perf_counter() - t0))

    # Save overall simulation timings to timings/tables/timings_cholesky+parallelization.csv
    timing_df = pd.DataFrame(timings, columns=["task", "seconds"])
    timing_df["seconds"] = timing_df["seconds"].round(2)
    timing_df.to_csv(tables_dir / "timings_cholesky+parallelization.csv", index=False)

    print("🕒 Wrote total simulation timings to timings/tables/timings_cholesky+parallelization.csv")
    print(timing_df)