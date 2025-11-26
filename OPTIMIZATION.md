# 0. Runtime Results

| **Task**                        | **Baseline** | **Cholesky** | **Cholesky + Parallelization** |
| ------------------------------- | ------------ | ------------ | ------------------------------ |
| run_simulation_single           | 46.99        | 4.22         | 2.77                           |
| run_simulation_multi            | 47.40        | 3.45         | 2.70                           |
| run_simulation_convergence_rate | 275.60       | 9.19         | 4.08                           |
| **Total**                       | **369.99**   | **16.86**    | **9.55**                       |

Note that runtime changes for each run. Also, this runtime excludes runtime for plotting.


# 1. Implemented Optimization Strategies

## 1.1 Cholesky Decomposition for Generating Multivariate Normal Samples

### 1.1.1 Problem Identified

The major bottleneck of baseline was multivariate normal data generation using `sample_normal`, which implements `np.random.multivariate_normal` using Singular Value Decomposition internally.

### 1.1.2 Solution Implemented

To decrease the runtime, instead of using `np.random.multivariate_normal`, sampled non-multivariate normal samples using `np.random.normal` and multiplied the lower triangular matrix obtained by Cholesky Decomposition.

### 1.1.3 Code Comparison

```
def sample_normal(
    Sigma: np.ndarray,
    n: int,
    mu: Optional[np.ndarray] = None,
    seed: Optional[int] = 725
) -> np.ndarray:
    p = Sigma.shape[0]
    if mu is None:
        mu = np.zeros(p)
    rng = np.random.default_rng(seed)

    return rng.multivariate_normal(mu, Sigma, size=n)

def sample_normal_fast(
    Sigma: np.ndarray,
    n: int,
    mu: Optional[np.ndarray] = None,
    seed: Optional[int] = 725
) -> np.ndarray:
    p = Sigma.shape[0]
    if mu is None:
        mu = np.zeros(p)
    rng = np.random.default_rng(seed)

    # Cholesky
    L = cholesky(Sigma)
    Z = rng.normal(size=(n, p))

    return mu[None, :] + Z @ L.T
```

### 1.1.4 Performance Impact

The performance gain of implementing Cholesky Decomposition was significant. For `run_simulation_convergence_rate` where only multivariate normal samples were used, the Cholesky-optimized simulation was about **30 times faster** than the original one. For the other two simulations `run_simulation_single` and `run_simulation_multi`, these simulations consists of both generating multivariate normal samples and multivariate t samples with exactly same size and dimension. As Cholesky Decomposition was already implemented in `sample_t`, the performance gain was about 12~14 times.

### 1.1.5 Trade-Offs

Generally, Cholesky Decomposition has some trade-offs. If the covariance matrix $\Sigma$ is not a positive definite matrix or has very small ($\approx 0$) eigenvalues, then Cholesky Decomposition fails or is numerically unstable. However, for our case, due to the predefined covariance matrix $\Sigma = c_1 p e_1 e_1^{\top} + c_2 I_p$ for single-spike and $\Sigma = c_1 p e_1 e_1^{\top} + c_2 p e_2 e_2^{\top} + c_3 I_p$ for multi-spike, the covariance matrix is well-conditioned and there is absolutely no trade-off for using Cholesky Decomposition instead of Singular Value Decomposition.

## 1.2 Parallelization

### 1.2.1 Problem Identified

The simulations are embarrassingly parallelizable, since for `run_simulation_single` and `run_simulation_multi`, 100 repetitions are independent with each other. For `run_simulation_convergence_rate`, aside from repetitions, simulations for different signal-to-noise (SNR) values are also independent. Therefore, parallelization might give an extra boost.

### 1.2.2 Solution Implemented

My computer has 14 cores. For the first two simulations `run_simulation_single` and `run_simulation_multi`, 10 cores were used, so that each core can do 10 repetitions. For `run_simulation_convergen_rate`, 6 cores were used, since there are 6 SNR values.

### 1.2.3 Code Comparison

Parallelization was implemented. Please refer to `simulation_cholesky+parallelization.py`.

### 1.2.4 Performance Impact

The performance gain of implementing parallelization was not that significant as expected. There was a 40% reduction in runtime for the whole simulation pipeline, although the number of cores used were 10, 10, 6.

### 1.2.5 Trade-Offs

In terms of the simulation result itself, there was no trade-off. However, from a computational perspective, increases in memory consumption, coding difficulty and profiling difficulty are clear cons of parallelization.


# 2. Runtime Analysis

## 2.1 File Explanation

All runtime results are saved under `timings/tables` with relevant figures in `timings/figures`.
 - `(single/multi/convergence)_step_timings_(baseline/cholesky/cholesky+parallelization).csv`: initial profiling results
 - `timings_(baseline/cholesky/cholesky+parallelization).csv`: total runtime
 - `(single_normal/single_t/multi_normal/multi_t/convergence)_profiling_(baseline/cholesky/cholesky+parallelization).csv`: aggregated stepwise(data generation, estimator computation, metric computation) profiling results
 - `complexity.csv`: estimated complexity exponents using log-log regression.

## 2.2 


Since most of the computational resource was focused on the data generation step, we give plots on the ratio of runtime of data generation to runtime of the other two steps for three simulations. 





# 3. Lessons Learned

## 3.1 Unexpected Time Cost in Data Generation

We learned that the **data generation process** itself can be significantly time-consuming. For this project, even after targeted optimization efforts, the data generation step continued to account for the **majority of the total runtime**. Therefore, for future simulation studies, it is crucial to prioritize the **efficient and careful implementation** of the data generation mechanism as a prerequisite for overall performance.

## 3.2 The Critical Role of Constant Factors

We confirmed that for simulation studies involving relatively small or moderate data scales, the **constant factors** in the computational runtime are often more critical than the **asymptotic complexity** of the underlying algorithm. Specifically, we achieved a significant performance increase by reducing this constant factor through the substitution of **Cholesky Decomposition** for the **Singular Value Decomposition (SVD)** utilized in the standard `np.random.multivariate_normal` function.

## 3.3 Limitations of Parallelization Efficiency

The performance gains achieved through parallelization were less dramatic than initially anticipated across all simulation settings. Although the number of cores was optimized, the runtime was reduced by only approximately $30\%$ for the first two simulations and $50\%$ for the final simulation. Given the **relatively small number of trials (100 times)**, the **auxiliary time expenditure**—including the launching, managing, and synchronizing parallel processes, along with the overhead from potential repeated tasks and the final result integration—was a limiting factor on overall efficiency.

# 4. Consistency of the Result

To verify that the simulation results obtained under the optimization strategies remain consistent with the original implementation, we performed regression-based comparison tests in `tests/test_regression.py`. For each CSV file generated under `results/tables`, all numerical entries were compared with their corresponding baseline values. Consistency was assessed using two criteria: the mean-squared error (MSE) had to remain below 5%, and the maximum relative deviation was allowed up to 25%. If either threshold was exceeded, the test was marked as a failure. All CSV files passed these checks, confirming that the use of Cholesky decomposition and parallelization preserves the statistical correctness of the simulation outputs.
