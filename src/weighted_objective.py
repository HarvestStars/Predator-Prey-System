import numpy as np
from scipy.integrate import odeint


def lotka_volterra(y, t, alpha, beta, delta, gamma):
    """
    Lotka-Volterra equations (predator-prey model)
    """
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def objective_weighted(params, t_data, x_data, y_data, zero_index_x, zero_index_y):
    """
    Weighted objective function for Lotka-Volterra fitting.

    Parameters:
    - params: [alpha, beta, delta, gamma], model parameters to optimize.
    - t_data: array, time data.
    - x_data: array, predator data.
    - y_data: array, prey data.
    - zero_index_x: list or set, indices in x_data where weights should be set to 0.
    - zero_index_y: list or set, indices in y_data where weights should be set to 0.

    Returns:
    - weighted_error: float, the weighted mean squared error.
    """
    alpha, beta, delta, gamma = params
    y0 = [x_data[0], y_data[0]]

    # Solve the Lotka-Volterra equations
    solution = odeint(lotka_volterra, y0, t_data, args=(alpha, beta, delta, gamma))
    x_sim, y_sim = solution[:, 0], solution[:, 1]

    # Calculate the weighted error
    error_squares = []
    for i in range(len(x_data)):
        # Determine weights dynamically
        w_x = 0 if i in zero_index_x else 1
        w_y = 0 if i in zero_index_y else 1

        # Compute squared error with weights
        error = w_x * (x_sim[i] - x_data[i])**2 + w_y * (y_sim[i] - y_data[i])**2
        error_squares.append(error)

    # Compute the mean of the weighted errors
    weighted_error = np.mean(error_squares)
    return weighted_error
