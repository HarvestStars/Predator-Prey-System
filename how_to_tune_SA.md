## Why Tune SA Parameters?

- **Avoiding Poor Convergence:**  
  Without proper tuning, the SA algorithm may converge prematurely to poor local optima, failing to find near-global solutions.

- **Balancing Exploration and Exploitation:**  
  SA parameters such as initial temperature and cooling rate determine how the algorithm transitions from a broad exploratory search to a more focused local improvement. Poor choices can either result in insufficient exploration or wasteful lingering at high temperatures.

- **Efficiency and Computation Time:**  
  An inadequately tuned SA algorithm might spend too long exploring regions of the parameter space without improvement, or cool down so fast that it barely improves its solution after the initial steps.

## Suggested Tuning Procedure

1. **Start with a Small Problem:**  
   Use a simplified dataset (fewer time points) to quickly test multiple configurations.  
   - Easier to run multiple experiments.  
   - Faster feedback on parameter changes.

2. **Use a Stable Objective Function (H):**  
   Begin tuning with the standard MSE objective:
   \[
   H(\theta) = \frac{1}{N}\sum_{i=1}^{N} \left[ (x_{\text{sim}}(t_i;\theta) - x_{\text{data}}(t_i))^2 + (y_{\text{sim}}(t_i;\theta) - y_{\text{data}}(t_i))^2 \right]
   \]
   This provides a clear and consistent baseline for comparing parameter sets.

3. **Set Initial State and Bounds:**  
   - **Initial State:** Choose a plausible starting point for \((a,b,c,d)\) based on problem knowledge.  
   - **Bounds:** Define meaningful upper and lower limits for each parameter to confine the search space.

4. **Proposal Function Tuning:**  
   - Commonly use a Gaussian step: \(\theta_{\text{new}} = \theta_{\text{current}} + \epsilon\), \(\epsilon \sim N(0,\sigma^2)\).  
   - Adjust \(\sigma\) to achieve a moderate acceptance rate (e.g., 30%-60%).  
   - If acceptance is too low, reduce \(\sigma\). If too high, increase \(\sigma\).

5. **Initial Temperature (T0) and Cooling Rate (alpha):**  
   - **Initial Temperature (T0):**  
     Start high enough to allow acceptance of worse solutions, promoting exploration. If the algorithm remains too random, lower T0.  
   - **Cooling Rate (alpha):**  
     Commonly geometric: \(T_{\text{new}} = \alpha T_{\text{old}}\).  
     Try \(\alpha \in [0.8, 0.99]\). If the solution space is large or complex, a slower cooling (closer to 1) might be beneficial.  
     If convergence is too slow, lower \(\alpha\); if it gets stuck too quickly, raise \(\alpha\).

6. **Markov Chain Length (max_iter):**  
   - Set an initial max_iter (e.g., 1000 iterations).  
   - If solutions improve steadily with each iteration, consider increasing it; if improvement plateaus early, try shorter chains or adjust cooling.

7. **Multiple Runs for Statistical Confidence:**  
   - Run SA multiple times with the same parameter settings.  
   - Compute mean and variance of final \(H(\theta)\) values to assess stability and robustness.  
   - Stable parameters yield consistent results across runs.

8. **Scaling Up to Larger Problems:**  
   - Once parameters work well on a small scale, apply them to the full dataset.  
   - If performance degrades, readjust parameters: perhaps increase max_iter, slow down cooling, or modify \(\sigma\).
