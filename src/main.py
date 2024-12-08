from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import SA as SA
import HC as HC

def load_data():
    """
    Load data from path, csv file
    """
    data = np.loadtxt("../observed_data/predator-prey-data.csv", delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2]

def lotka_volterra(y, t, alpha, beta, delta, gamma):
    """
    Lotka-Volterra equations (predator-prey model)
    """
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def proposal_func(x):
    """
    Symetric proposal function to generate a new state
    """
    return x + np.random.normal(0, 0.1, size=x.shape)

def objective(params, t_data, x_data, y_data):
    """
    Objective function to minimize (sum of squared errors)
    """
    alpha, beta, delta, gamma = params
    y0 = [x_data[0], y_data[0]]  # Initial condition
    solution = odeint(lotka_volterra, y0, t_data, args=(alpha, beta, delta, gamma))
    x_sim, y_sim = solution[:, 0], solution[:, 1]
    mse = np.mean((x_sim - x_data)**2 + (y_sim - y_data)**2)  # Mean Squared Error
    return mse

def compute_model(t_data, x_data, y_data, alpha, beta, delta, gamma):
    """
    Compute the Lotka-Volterra model using the best parameters
    """
    y0 = [x_data[0], y_data[0]]
    solution = odeint(lotka_volterra, y0, t_data, args=(alpha, beta, delta, gamma))
    prey = solution[:, 0]
    predator = solution[:, 1]
    return prey, predator

def plot_model(t_data, x_data, y_data, sim_data):
    """
    Plot the observed data and the best fit
    """
    prey, predator = sim_data
    
    fig, axis = plt.subplots(1, 2, figsize=(12, 6))
    axis[0].plot(t_data, x_data, 'o')
    axis[0].plot(t_data, y_data, 'o')
    axis[0].set_title("Observed Data")

    axis[1].plot(t_data, prey, label='Prey')
    axis[1].plot(t_data, predator, label='Predator')
    axis[1].set_title("Lotka-Volterra Model")

    fig.suptitle(f"Lotka-Volterra Model\n given by $\\alpha = 0.57104693$, $\\beta = 0.47572982$, $\\delta = 0.93209038$, $\\gamma = 1.1220162$")
    fig.supxlabel("Time")
    fig.supylabel("Population")
    fig.legend()

    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.savefig("visualization/lotka_volterra.png")
    plt.show()

def bootstrapping(data_points):
    pass

if __name__ == "__main__":
    true_data = load_data()

    initial_params = np.array([1.0, 0.1, 0.1, 1.0])  # Initial guess for [alpha, beta, delta, gamma]
    bounds = [(0.01, 2.0), (0.01, 2.0), (0.01, 2.0), (0.01, 2.0)]  # Parameter bounds
    initial_temp = 10.0  # Starting temperature
    alpha = 0.9  # Cooling rate
    max_iter = 1000  # Maximum iterations
    tol = 10**-4
    NOISE_STD = 0.01

    # best_params, best_mse = SA.simulated_annealing(
    #     objective, true_data, initial_params, bounds, proposal_func, initial_temp, alpha, max_iter
    # )

    best_params, best_mse = HC.hill_climbing(
        objective, proposal_func, true_data, initial_params, max_iter, tol
    )

    print("Best Parameters (alpha, beta, delta, gamma):", best_params)
    print("Best Mean Squared Error:", best_mse)

    prey, predator = compute_model(*true_data, *best_params)
    # plot_model(*true_data, (prey, predator))