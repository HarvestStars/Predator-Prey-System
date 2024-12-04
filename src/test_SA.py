from scipy.integrate import odeint
import numpy as np
import SA as SA

# load data from path, csv file
def load_data():
    data = np.loadtxt("../observed_data/predator-prey-data.csv", delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2]

# Lotka-Volterra equations (predator-prey model)
def lotka_volterra(y, t, alpha, beta, delta, gamma):
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# symetric proposal function to generate a new state
def proposal_func(x):
    return x + np.random.normal(0, 0.1, size=x.shape)

# objective function to minimize (sum of squared errors)
def objective(params, t_data, x_data, y_data):
    alpha, beta, delta, gamma = params
    y0 = [x_data[0], y_data[0]]  # Initial condition
    solution = odeint(lotka_volterra, y0, t_data, args=(alpha, beta, delta, gamma))
    x_sim, y_sim = solution[:, 0], solution[:, 1]
    mse = np.mean((x_sim - x_data)**2 + (y_sim - y_data)**2)  # Mean Squared Error
    return mse

if __name__ == "__main__":
    # Load observed data
    t_data, x_data, y_data = load_data()

    # Initial parameter guesses and bounds
    initial_params = [1.0, 0.1, 0.1, 1.0]  # Initial guess for [alpha, beta, delta, gamma]
    bounds = [(0.01, 2.0), (0.01, 2.0), (0.01, 2.0), (0.01, 2.0)]  # Parameter bounds
    initial_temp = 10.0  # Starting temperature
    alpha = 0.9  # Cooling rate
    max_iter = 1000  # Maximum iterations

    # Run Simulated Annealing
    best_params, best_mse = SA.simulated_annealing(
        objective, t_data, x_data, y_data, initial_params, bounds, proposal_func, initial_temp, alpha, max_iter
    )

    print("Best Parameters (alpha, beta, delta, gamma):", best_params)
    print("Best Mean Squared Error:", best_mse)