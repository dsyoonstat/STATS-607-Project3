# 1. Total Runtime of Entire Simulation Study

| **Task**                        | **Seconds** |
| ------------------------------- | ----------- |
| run_simulation_single           | 48.04       |
| run_simulation_multi            | 46.18       |
| run_simulation_convergence_rate | 272.47      |
| **Total**                       | **366.69**  |

# 2. Runtime of each simulation

### 2.1 run_simulation_single

##### 2.1.1 Multivariate Normal Case

| **p**    | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| -------- | ----------------------- | ----------------------------- | -------------------------- |
| **100**  | 0.0563                  | 0.0912                        | 0.00442                    |
| **200**  | 0.173                   | 0.0909                        | 0.00441                    |
| **500**  | 1.23                    | 0.102                         | 0.00576                    |
| **1000** | 5.41                    | 0.114                         | 0.00823                    |
| **2000** | 38.4                    | 0.127                         | 0.00618                    |

##### 2.1.2 Multivariate t Case

| **p**    | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| -------- | ----------------------- | ----------------------------- | -------------------------- |
| **100**  | 0.00789                 | 0.0868                        | 0.00438                    |
| **200**  | 0.0147                  | 0.0897                        | 0.00445                    |
| **500**  | 0.0676                  | 0.102                         | 0.00570                    |
| **1000** | 0.247                   | 0.112                         | 0.00666                    |
| **2000** | 1.24                    | 0.126                         | 0.00615                    |

### 2.2 run_simulation_multi

##### 2.2.1 Multivariate Normal Case

| **p**    | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| -------- | ----------------------- | ----------------------------- | -------------------------- |
| **100**  | 0.0550                  | 0.0301                        | 0.00167                    |
| **200**  | 0.173                   | 0.0315                        | 0.00159                    |
| **500**  | 1.14                    | 0.0398                        | 0.00222                    |
| **1000** | 5.12                    | 0.0499                        | 0.00342                    |
| **2000** | 37.7                    | 0.0687                        | 0.00407                    |
##### 2.2.2 Multivariate t Case

| **p**    | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| -------- | ----------------------- | ----------------------------- | -------------------------- |
| **100**  | 0.00704                 | 0.0286                        | 0.00159                    |
| **200**  | 0.0150                  | 0.0295                        | 0.00152                    |
| **500**  | 0.0652                  | 0.0345                        | 0.00164                    |
| **1000** | 0.248                   | 0.0392                        | 0.00185                    |
| **2000** | 1.21                    | 0.0525                        | 0.00266                    |
### 2.3 run_simulation_convergence_rate

This simulation only considers multivariate normal case.

| **p**    | **Data generation (s)** | **Estimator computation (s)** | **Metric computation (s)** |
| -------- | ----------------------- | ----------------------------- | -------------------------- |
| **100**  | 0.316                   | 0.0821                        | 0.00193                    |
| **200**  | 1.00                    | 0.0844                        | 0.00193                    |
| **500**  | 7.16                    | 0.136                         | 0.00326                    |
| **1000** | 31.1                    | 0.152                         | 0.00558                    |
| **2000** | 232                     | 0.199                         | 0.00648                    |

# 3. Bottleneck Analysis

It is apparent that the main bottleneck is the data generation for multivariate normal distribution when the dimension $p$ is large. Note that data generation for multivariate t distribution is relatively fast. This is due to the fact that data generation for multivariate normal distribution used `np.random.multivariate_normal`, while that for multivariate t distribution used Cholesky decomposition of covariance matrix $\Sigma$. While doing **Unit 2 Project**, I did not notice that there is a huge difference in speed. Therefore, by adopting Cholesky decomposition to generate multivariate normal data will give a huge boost in speed, although the data generation procedure still remains most time-consuming among data generation, estimator computation and metric computation.

# 4. Computational Complexity Analysis

Since High-Dimensional Low-Sample Size(HDLSS) asymptotics fix the sample size $n$ and let the dimension $p$ grows to $\infty$, we consider the computational complexity with respect to the $p$. 

### 4.1 Empirical Analysis

The estimated computational complexity using log-log regression is as follows:

| **Step**                | **Metric** | **single/normal**       | **single/t**            | **multi/normal**        | **multi/t**             | **convergence**         |
|------------------------|------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| **Data generation**    | Complexity | $\mathrm{O}(p^{2.92})$   | $\mathrm{O}(p^{2.86})$   | $\mathrm{O}(p^{2.16})$   | $\mathrm{O}(p^{1.72})$   | $\mathrm{O}(p^{2.18})$   |
|                        | $R^2$      | $0.9996$                 | $0.9997$                 | $0.9911$                 | $0.9850$                 | $0.9920$                 |
| **Estimator computation** | Complexity | $\mathrm{O}(p^{0.10})$   | $\mathrm{O}(p^{0.12})$   | $\mathrm{O}(p^{0.28})$   | $\mathrm{O}(p^{0.20})$   | $\mathrm{O}(p^{0.31})$   |
|                        | $R^2$      | $0.6974$                 | $0.7818$                 | $0.9440$                 | $0.9100$                 | $0.9566$                 |
| **Metric computation** | Complexity | $\mathrm{O}(p^{0.24})$   | $\mathrm{O}(p^{0.17})$   | $\mathrm{O}(p^{0.34})$   | $\mathrm{O}(p^{0.16})$   | $\mathrm{O}(p^{0.46})$   |
|                        | $R^2$      | $0.8170$                 | $0.8710$                 | $0.9147$                 | $0.7069$                 | $0.9399$                 |

The high $R^2$ values across all three steps strongly suggest that the empirical computational complexity estimates are highly reliable. It's worthy to note that $R^2$ for data generation is remarkably close to 1.0 (averaging 0.9935). This near-perfect fit occurs because the data generation step is dominated by a single, high-complexity operation—$\mathrm{O}(p^3)$ Singular Value Decomposition or Cholesky Decomposition—which serves as a clean, overriding bottleneck.

### 4.2 Theoretical Analysis

##### 4.2.1 Data Generation

Since both Singular Value Decomposition used in `np.random.multivariate_normal` and Cholesky Decomposition has same computational complexity of $O(p^3)$, the theoretical computational complexity of data generation step is $O(p^3)$ as analyzed below.

| **Step**                          | **Complexity**             |
| --------------------------------- | -------------------------- |
| **Covariance Matrix Square Root** | $\mathrm{O}(p^3)$          |
| **Random Number Generation**      | $\mathrm{O}(p)$            |
| **Transformation**                | $\mathrm{O}(p^2)$          |
| **Overall**                       | $\mathbf{\mathrm{O}(p^3)}$ |

##### 4.2.2 Estimator Computation

The theoretical computational complexity of estimator computation step is $O(p)$ thanks to Gram matrix trick which was adopted during **Unit 2 Project** to reduce runtime, as analyzed below.

| **Step**                    | **Complexity**           |
| --------------------------- | ------------------------ |
| **Gram Matrix Computation** | $\mathrm{O}(p)$          |
| **Eigenvector Recovery**    | $\mathrm{O}(p)$          |
| **Transformation**          | $\mathrm{O}(p)$          |
| **Overall**                 | $\mathbf{\mathrm{O}(p)}$ |

##### 4.2.3 Metric Computation

The theoretical computational complexity of metric computation step is $O(p)$ as analyzed below.

| **Step**                         | **Complexity**           |
| -------------------------------- | ------------------------ |
| **Matrix Multiplication**        | $\mathrm{O}(p)$          |
| **Singular Value Decomposition** | $\mathrm{O}(1)$          |
| **Overall**                      | $\mathbf{\mathrm{O}(p)}$ |

# 5. Numerical Instability Analysis

During the simulation runs, NumPy generated several numerical warnings such as:
```
RuntimeWarning: divide by zero encountered in matmul
RuntimeWarning: overflow encountered in matmul
RuntimeWarning: invalid value encountered in matmul
```
These warnings occurred in **all three steps** of the pipeline—data generation, estimator computation, and metric computation—because certain intermediate BLAS operations briefly reached floating-point limits.

However, all simulated datasets, estimator outputs, and computed metrics contained only valid values (no NaN/Inf). Thus, although numerical warnings were observed, they did not compromise the stability or correctness of the final simulation outcomes.

# 6. Conclusion

To reduce runtime, we focus on reducing the runtime of data generation, as the other steps are relatively fast, especially for large $p$. Therefore, we plan to implement two strategies: (1) Algorithmic improvement on data generation and (2) Parallelization.