# complexity.py
#
# Estimate runtime complexity exponents via log-log regression
# using aggregated profiling CSVs.
#
# Inputs (per variant):
#   timings/tables/single_normal_profiling_{variant}.csv
#   timings/tables/single_t_profiling_{variant}.csv
#   timings/tables/multi_normal_profiling_{variant}.csv
#   timings/tables/multi_t_profiling_{variant}.csv
#   timings/tables/convergence_profiling_{variant}.csv
#
# Each CSV has columns:
#   p, data_generation, estimator_computation, metric_computation
#
# For each (experiment, method, step), we fit:
#   log(time) = a + alpha * log(p)
# and store (alpha, R^2).
#
# Output:
#   timings/tables/complexity.csv
#   with columns: experiment, method, step, alpha, r2

from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd


# ---------------------- configuration ----------------------

VARIANTS = [
    "baseline",
    "cholesky",
    "cholesky+parallelization",
]

# CSV stems before "_{variant}.csv"
EXPERIMENT_STEMS = [
    "single_normal_profiling",
    "single_t_profiling",
    "multi_normal_profiling",
    "multi_t_profiling",
    "convergence_profiling",
]

# Columns in profiling CSVs to use as "steps"
STEP_COLUMNS = [
    "data_generation",
    "estimator_computation",
    "metric_computation",
]


# ---------------------- helpers ----------------------
def _fit_log_log(p: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """
    Fit log-log regression: log(t) = a + alpha * log(p).

    Parameters
    ----------
    p : array-like
        Dimension sizes (must be positive).
    t : array-like
        Runtime values (must be positive).

    Returns
    -------
    alpha : float
        Estimated exponent (slope in log-log space).
    r2 : float
        Coefficient of determination in log-log space.
    """
    # Filter strictly positive values to avoid log problems
    mask = (p > 0) & (t > 0)
    p = p[mask]
    t = t[mask]

    if p.size < 2:
        raise ValueError("Need at least two positive points for log-log regression.")

    x = np.log(p)
    y = np.log(t)

    # Fit y = a + alpha * x using least squares
    # np.polyfit returns [slope, intercept] for deg=1
    alpha, intercept = np.polyfit(x, y, deg=1)

    # Compute R^2 in log-log space
    y_hat = intercept + alpha * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return float(alpha), float(r2)


def _stem_to_experiment_name(stem: str) -> str:
    """
    Convert CSV stem to experiment name string.

    Examples
    --------
    'single_normal_profiling' -> 'single_normal'
    'convergence_profiling'   -> 'convergence'
    """
    # Remove trailing '_profiling'
    if stem.endswith("_profiling"):
        return stem[:-len("_profiling")]
    return stem


# ---------------------- main logic ----------------------
def main():
    base_tables = Path("timings/tables")
    if not base_tables.exists():
        raise FileNotFoundError(f"Directory not found: {base_tables}")

    rows: List[Dict[str, object]] = []

    for variant in VARIANTS:
        method = variant  # 'baseline', 'cholesky', 'cholesky+parallelization'

        for stem in EXPERIMENT_STEMS:
            experiment = _stem_to_experiment_name(stem)
            csv_path = base_tables / f"{stem}_{variant}.csv"

            if not csv_path.exists():
                raise FileNotFoundError(f"Cannot find input CSV: {csv_path}")

            df = pd.read_csv(csv_path)

            if "p" not in df.columns:
                raise ValueError(f"Column 'p' not found in {csv_path}")

            p_vals = df["p"].to_numpy(dtype=float)

            for step in STEP_COLUMNS:
                if step not in df.columns:
                    raise ValueError(f"Column '{step}' not found in {csv_path}")

                t_vals = df[step].to_numpy(dtype=float)

                alpha, r2 = _fit_log_log(p_vals, t_vals)

                rows.append(
                    {
                        "experiment": experiment,
                        "method": method,
                        "step": step,
                        "alpha": alpha,
                        "r2": r2,
                    }
                )

    out_df = pd.DataFrame(rows, columns=["experiment", "method", "step", "alpha", "r2"])
    out_path = base_tables / "complexity.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Wrote complexity estimates to {out_path}")
    print(out_df)


if __name__ == "__main__":
    main()