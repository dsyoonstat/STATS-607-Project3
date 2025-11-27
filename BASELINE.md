# 0. Announcement

Note that runtime can vary for each simulation. The numbers in this documentation is based on the final experiment before submitting.

# 1. Total Runtime of Entire Simulation Study

| **Task**                        | **Seconds** |
| ------------------------------- | ----------- |
| run_simulation_single           | 46.35       |
| run_simulation_multi            | 45.03       |
| run_simulation_convergence_rate | 261.94      |
| **Total**                       | **353.32**  |

# 2. Runtime of each simulation

## 2.1 run_simulation_single

### 2.1.1 Multivariate Normal Case

| **p** | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| --- | --- | --- | --- |
| **100** | 0.0564 | 0.0911 | 0.0045 |
| **200** | 0.1761 | 0.0907 | 0.0043 |
| **500** | 1.1971 | 0.1023 | 0.0056 |
| **1000** | 5.4285 | 0.1118 | 0.0080 |
| **2000** | 36.7652 | 0.1262 | 0.0061 |

### 2.1.2 Multivariate t Case

| **p** | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| --- | --- | --- | --- |
| **100** | 0.0079 | 0.0874 | 0.0044 |
| **200** | 0.0151 | 0.0897 | 0.0044 |
| **500** | 0.0638 | 0.1015 | 0.0056 |
| **1000** | 0.2454 | 0.1108 | 0.0065 |
| **2000** | 1.2015 | 0.1256 | 0.0061 |

## 2.2 run_simulation_multi

### 2.2.1 Multivariate Normal Case

| **p** | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| --- | --- | --- | --- |
| **100** | 0.0547 | 0.0297 | 0.0017 |
| **200** | 0.1720 | 0.0312 | 0.0016 |
| **500** | 1.1451 | 0.0392 | 0.0021 |
| **1000** | 5.1310 | 0.0493 | 0.0033 |
| **2000** | 36.5534 | 0.0639 | 0.0039 |

### 2.2.2 Multivariate t Case

| **p** | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| --- | --- | --- | --- |
| **100** | 0.0069 | 0.0276 | 0.0016 |
| **200** | 0.0150 | 0.0294 | 0.0016 |
| **500** | 0.0651 | 0.0345 | 0.0017 |
| **1000** | 0.2478 | 0.0388 | 0.0018 |
| **2000** | 1.1821 | 0.0501 | 0.0026 |

## 2.3 run_simulation_convergence_rate

This simulation only considers multivariate normal case.

| **p** | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| --- | --- | --- | --- |
| **100** | 0.3122 | 0.0805 | 0.0019 |
| **200** | 1.0015 | 0.0836 | 0.0019 |
| **500** | 6.9220 | 0.1177 | 0.0030 |
| **1000** | 30.5764 | 0.1468 | 0.0054 |
| **2000** | 222.4254 | 0.1834 | 0.0060 |


# 3. Bottleneck Analysis

It is apparent that the main bottleneck is the data generation for multivariate normal distribution when the dimension $p$ is large. Note that data generation for multivariate t distribution is relatively fast. This is due to the fact that data generation for multivariate normal distribution used `np.random.multivariate_normal`, while that for multivariate t distribution used Cholesky decomposition of covariance matrix $\Sigma$. While doing **Unit 2 Project**, I did not notice that there is a huge difference in speed between those two methods. Therefore, by adopting Cholesky decomposition to generate multivariate normal data will give a huge boost in speed, although the data generation procedure still remains most time-consuming among data generation, estimator computation and metric computation, as seen in multivariate t cases.

# 4. Computational Complexity Analysis

Since High-Dimensional Low-Sample Size(HDLSS) asymptotics fix the sample size $n$ and let the dimension $p$ grows to $\infty$, we consider the computational complexity with respect to the $p$. 

## 4.1 Empirical Analysis

The estimated computational complexity using log-log regression is as follows:

| **Step** | **Metric** | **single/normal** | **single/t** | **multi/normal** | **multi/t** | **convergence** |
|------------------------|------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| **Data generation** | Complexity | $\mathrm{O}(p^{2.15})$   | $\mathrm{O}(p^{1.68})$   | $\mathrm{O}(p^{2.15})$   | $\mathrm{O}(p^{1.72})$   | $\mathrm{O}(p^{2.17})$   |
|                        | $R^2$      | $0.9928$                 | $0.9799$                 | $0.9919$                 | $0.9862$                 | $0.9923$                 |
| **Estimator computation** | Complexity | $\mathrm{O}(p^{0.11})$   | $\mathrm{O}(p^{0.12})$   | $\mathrm{O}(p^{0.26})$   | $\mathrm{O}(p^{0.19})$   | $\mathrm{O}(p^{0.29})$   |
|                        | $R^2$      | $0.9360$                 | $0.9677$                 | $0.9572$                 | $0.9483$                 | $0.9715$                 |
| **Metric computation** | Complexity | $\mathrm{O}(p^{0.17})$   | $\mathrm{O}(p^{0.14})$   | $\mathrm{O}(p^{0.32})$   | $\mathrm{O}(p^{0.15})$   | $\mathrm{O}(p^{0.44})$   |
|                        | $R^2$      | $0.6358$                 | $0.8473$                 | $0.9030$                 | $0.7404$                 | $0.9283$                 |

The high $R^2$ values across all three steps strongly suggest that the empirical computational complexity estimates are highly reliable. The results can be found in `timings/tables/complexity.csv`.

## 4.2 Theoretical Analysis

### 4.2.1 Data Generation

Since both Singular Value Decomposition used in `np.random.multivariate_normal` and Cholesky Decomposition has same computational complexity of $O(p^3)$, the theoretical computational complexity of data generation step is $O(p^3)$ as analyzed below.

| **Step**                          | **Complexity**             |
| --------------------------------- | -------------------------- |
| **Covariance Matrix Square Root** | $\mathrm{O}(p^3)$          |
| **Random Number Generation**      | $\mathrm{O}(p)$            |
| **Transformation**                | $\mathrm{O}(p^2)$          |
| **Overall**                       | $\mathbf{\mathrm{O}(p^3)}$ |

### 4.2.2 Estimator Computation

The theoretical computational complexity of estimator computation step is $O(p)$ thanks to Gram matrix trick which was adopted during **Unit 2 Project** to reduce runtime, as analyzed below.

| **Step**                    | **Complexity**           |
| --------------------------- | ------------------------ |
| **Gram Matrix Computation** | $\mathrm{O}(p)$          |
| **Eigenvector Recovery**    | $\mathrm{O}(p)$          |
| **Transformation**          | $\mathrm{O}(p)$          |
| **Overall**                 | $\mathbf{\mathrm{O}(p)}$ |

### 4.2.3 Metric Computation

The theoretical computational complexity of metric computation step is $O(p)$ as analyzed below.

| **Step**                         | **Complexity**           |
| -------------------------------- | ------------------------ |
| **Matrix Multiplication**        | $\mathrm{O}(p)$          |
| **Singular Value Decomposition** | $\mathrm{O}(1)$          |
| **Overall**                      | $\mathbf{\mathrm{O}(p)}$ |

## 4.3 Comparison between Empirical and Theoretical results

Although the theoretical computational complexities are $O(p^3)$, $O(p)$, and $O(p)$ respectively for data generation, estimator computation, and metric computation, the empirical results were lower. This is due to the fact that each step is composed of multiple sub-tasks with different complexity orders, and the observed runtime reflects a mixture of these costs rather than the dominant asymptotic term alone. Since the overall theoretical complexity is determined by the highest-order sub-task, the asymptotic behavior becomes evident only when p is extremely large. However, with the maximum dimension in our experiments being $p = 2000$, the dominant term does not fully overpower the lower-order components, leading the empirical log–log regressions to produce exponents smaller than their theoretical values.

# 5. Numerical Instability Analysis

During the simulation runs, NumPy generated several numerical warnings such as:
```
RuntimeWarning: divide by zero encountered in matmul
RuntimeWarning: overflow encountered in matmul
RuntimeWarning: invalid value encountered in matmul
```
The log file is saved as `log.txt`. These warnings occurred in **all three steps** of the pipeline—data generation, estimator computation, and metric computation—because certain intermediate BLAS operations briefly reached floating-point limits.

However, all simulated datasets, estimator outputs, and computed metrics contained only valid values (no NaN/Inf). Thus, although numerical warnings were observed, they did not compromise the stability or correctness of the final simulation outcomes.

# 6. Conclusion

To reduce runtime, we focus on reducing the runtime of data generation, as the other steps are relatively fast, especially for large $p$. Therefore, we plan to implement two strategies: (1) Algorithmic improvement on data generation using Cholesky decomposition and (2) Parallelization.