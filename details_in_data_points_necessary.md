# Work Plan for the Last Question

## Background

The Last question asks us to investigate how the number and distribution of data points from each time-series (predator and prey populations) affect the accuracy of parameter inference in a Lotka-Volterra system. We already have:

- Two objective functions:
  1. **MSE-based objective (H)**: 
     \[
     H(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left[(x_{\text{sim}}(t_i;\theta) - x_{\text{data}}(t_i))^2 + (y_{\text{sim}}(t_i;\theta) - y_{\text{data}}(t_i))^2\right]
     \]
     where \(\theta = (a,b,c,d)\) are the parameters of the Lotka-Volterra model.

  2. **Weighted-error objective (H_w)**:
     \[
     H_w(\theta; W_x, W_y) = \frac{\sum_{i=1}^{N} w_{x,i}(x_{\text{sim}}(t_i;\theta) - x_{\text{data}}(t_i))^2 + w_{y,i}(y_{\text{sim}}(t_i;\theta) - y_{\text{data}}(t_i))^2}{\sum_{i=1}^{N}(w_{x,i} + w_{y,i})}
     \]
     Here, \(W_x = \{w_{x,i}\}\) and \(W_y = \{w_{y,i}\}\) are the weight matrices indicating the importance of each data point in the predator (x) and prey (y) time-series.

- Two optimization algorithms:
  1. **Global optimization**: Simulated Annealing (SA)
  2. **Local optimization**: Hill Climbing (HC)

We will use these tools to systematically remove or down-weight data points from one or both time-series and examine the impact on the accuracy and stability of the inferred parameters.

## Goals

1. **Critical Number of Data Points (Single-Series Reduction)**  
   Determine how many data points from the prey or predator series you can remove until the parameters \((a,b,c,d)\) can no longer be accurately inferred. One series is fixed, while the other is gradually reduced.

2. **Combined Reduction**  
   After identifying critical thresholds for each series individually, attempt reducing both simultaneously to see if parameters can still be inferred accurately.

3. **Identify Critical vs. Non-Critical Data Points**  
   By selectively removing or lowering the weight of specific data points, determine which points are "safe to remove" without significantly affecting inference quality, and which are "critical" points that cause a large deviation in accuracy when removed.

## Proposed Workflow

### Step 1: Baseline Parameter Estimation

1. Use the full dataset \(\{(t_i, x_{data}(t_i), y_{data}(t_i))\}_{i=1}^N\) with the standard MSE objective \(H\).
2. Apply the Simulated Annealing (SA) algorithm to find a near-global minimum:
   \[
   \theta_{\text{base}} = \arg\min_{\theta} H(\theta)
   \]
   This gives a benchmark solution \((a,b,c,d)\).

3. Store the baseline parameters \(\theta_{\text{base}}\) and the baseline MSE value \(H(\theta_{\text{base}})\).

### Step 2: Definition of the Importance of each series point

1. For each data point \( (t_i, x_{data}(t_i), y_{data}(t_i)) \):
   - Temporarily set its weight to zero for that point (in either \(W_x\) or \(W_y\) depending on the series).
   - Re-estimate parameters \(\theta_i\) and compute \(H(\theta_i)\).
   - If \(|H(\theta_i) - H(\theta_{\text{base}})|\) is large, the point is critical.
   - If \(|H(\theta_i) - H(\theta_{\text{base}})|\) is small, the point is not critical.

### Step 3: Single-Series Data Reduction Experiments

1. **Fix the predator data (x-series)** and remove one data point once a time from the prey data (y-series):
   - For each point \(i\) in the y-series, set \(w_{y,i}=0\) (or substantially lower than 1) and keep all x-points at full weight.
   - Optimize parameters \(\theta_i\) using SA (and optionally refine with HC) under the weighted objective \(H_w\).
   - Compute the resulting parameters \(\theta_i\) on the **original** MSE objective \(H(\theta_i)\) using **all data points and equal weights**.
   - Compare \(H(\theta_i)\) with the baseline \(H(\theta_{\text{base}})\). The difference:
     \[
     \Delta H_i = H(\theta_i) - H(\theta_{\text{base}})
     \]
     measures the importance of the removed point \(i\).
   - Repeat for each y-point to generate a vector \(\{\Delta H_1, \Delta H_2, \ldots\}\).

2. **Plotting single-point importance (Answer to Goal 3)**:
   - Create a plot with the x-axis as the data point index \(i\) and the y-axis as \(\Delta H_i\).
   - Points with large positive \(\Delta H_i\) are critical; points with small \(\Delta H_i\) are less important.
   - This provides a direct visualization of each data point's significance in ensuring accurate parameter inference.

3. **Gradual point removal based on importance (Answer to Goal 1)**:
   - Sort the points by their importance metric \(\Delta H_i\), for example, from least important to most important.
   - Start removing points in that order (from least important to more important) and each time re-estimate parameters, computing new \(H(\theta)\) values.
   - Observe the number of removed points at which parameter inference fails (i.e., when \(H(\theta)\) significantly deviates from the baseline), thus identifying the critical threshold for the y-series.

4. **Repeat the process by fixing the y-series** and removing points from the x-series to find the critical threshold for the x-series.

### Step 4: Combined Data Reduction

1. After identifying critical thresholds for each series separately, attempt reducing points simultaneously from both x and y series.
2. Construct a new weight matrix \(W_x, W_y\) that removes or down-weights selected points from both series.
3. Perform SA/HC optimization to find \(\theta_{xy}\).
4. Compute \(H(\theta_{xy})\) and compare with \(H(\theta_{\text{base}})\). If parameters remain stable and \(H(\theta_{xy})\) does not increase drastically, you have found a safe zone of combined reduction. Otherwise, you have identified points that must remain to preserve inference quality.

### Optional Metrics

- **Parameter Distance**:
  \[
  D(\theta, \theta_{\text{base}}) = \sqrt{(a - a_{\text{base}})^2 + (b - b_{\text{base}})^2 + (c - c_{\text{base}})^2 + (d - d_{\text{base}})^2}
  \]
  While the main criterion is the change in MSE, this parameter distance can provide additional insights into how much the parameters themselves shift when certain points are removed.

### Summary

By following this workflow:

- You establish a baseline model \(\theta_{\text{base}}\) using all data.
- You systematically remove or down-weight points from one or both time-series, using the weighted objective \(H_w\) to re-estimate parameters.
- You evaluate each new parameter set under the original MSE \(H\) to quantify the importance of each point (\(\Delta H_i\)) and identify thresholds for data reduction.
- Visualizations and sorted importance lists guide you in determining how many points can be removed while still maintaining inference accuracy and which specific points are critical to preserve.
