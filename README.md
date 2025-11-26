# Optimization of Simulation for the Adaptive Reference-Guided Estimator

Umich STATS 607 Unit 3 Project, Fall 2025

This project optimizes the simulation of Unit 2 Project, which is based on [**Yoon and Jung (2025)**](https://onlinelibrary.wiley.com/doi/full/10.1002/sta4.70081).
It implements two optimization strategies:
1. **Cholesky Decomposition** for generating multivariate normal samples
2. **Parallelization**

For details, see the paper, `ADEMP.md` and `OPTIMIZATION.md`.

The simulation results are saved in `results/`, and runtime results are saved in `timings/`.

---

## Setup Instructions

1. **Clone or open the repository**, then create a Python environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   The following packages are needed:
   ```
   numpy>=1.24
   pandas>=2.0
   tqdm>=4.65
   matplotlib>=3.7
   pytest>=8.0
   ```
   (Python ≥ 3.9 recommended)

---

## How to Run the Complete Analysis

This project is fully automated via the included **Makefile**.

Run the full pipeline:
```bash
make all
```

This will:
1. Run all simulations (`src/simulation.py`) — saves CSV tables under `results/tables/`
2. Generate all figures (`src/plot.py`) — saves plots under `results/figures/`

You can also run individual steps:
```bash
make simulate     # Run simulations only
make figures      # Generate plots only
make clean        # Remove all generated files
make test         # Run test suite
```

---

## Estimated Runtime

- **Single-spike + Multi-spike + Convergence-rate simulations** (`make simulate`):  
  < 10 minutes total on a high-end laptop (Apple M4 pro).  
- Plot generation (`make figures`): < 10 seconds.

You can reduce runtime by lowering `n_trials` or `p_list` values inside `src/simulation.py`.

---

## Summary of Key Findings

 - The ARG estimator significantly outperformed the naive PCA-based estimator under multivariate normal data, yielding smaller principal angles to the true subspace. The performance gap widened as the reference vector became more aligned with the true principal component subspace. Under the multivariate $t$ distribution, where the assumptions of Yoon and Jung (2025) are not satisfied, the performance gain was smaller but still consistent — the ARG estimator continued to outperform the naive estimator.
 - However, the estimated convergence rates of $|u_1^{\top}d_1|$ varied substantially across different signal-to-noise ratios, and no consistent trend (e.g., increasing $\alpha$ with SNR) was observed. This irregularity appears to stem from the complex behavior of eigenvalues and eigenvectors.


---
