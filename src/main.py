from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import SA
import HC


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
    mse = np.mean((x_sim - x_data) ** 2 + (y_sim - y_data) ** 2)  # Mean Squared Error
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
    axis[0].plot(t_data, x_data, "o")
    axis[0].plot(t_data, y_data, "o")
    axis[0].set_title("Observed Data")

    axis[1].plot(t_data, prey, label="Prey")
    axis[1].plot(t_data, predator, label="Predator")
    axis[1].set_title("Lotka-Volterra Model")

    fig.suptitle(
        "Lotka-Volterra Model\n given by $\\alpha = 0.57104693$, $\\beta = 0.47572982$, $\\delta = 0.93209038$, $\\gamma = 1.1220162$"
    )
    fig.supxlabel("Time")
    fig.supylabel("Population")
    fig.legend()

    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.savefig("visualization/lotka_volterra.png")
    plt.show()


def plot_mse(N_boot, max_boot, objective, initial_params, proposal_func, max_iter, tol):
    boot_range = np.array(range(0, max_boot))
    mse = np.zeros(max_boot)

    for i in boot_range:
        mean_mse = 0
        for _ in range(N_boot):
            boot_data = bootstrapping(i)
            _, best_mse = HC.hill_climbing(
                objective, proposal_func, boot_data, initial_params, max_iter, tol
            )
            mean_mse += best_mse
        mse[i] = mean_mse / N_boot

    plt.plot(boot_range, mse)
    plt.xlabel("Number of Data Points Removed")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Squared Error vs. Number of Data Points")
    plt.savefig("../visualization/mse_vs_data_points.png")
    plt.show()


def bootstrapping(num_data_points):
    """
    Perform bootstrapping to generate new data
    """
    true_data = load_data()
    length = len(true_data[0])

    indices = np.random.choice(length, num_data_points, replace=False)
    result_data = tuple(np.delete(data, indices) for data in true_data)
    return result_data


def param_estimation(
    steps,
    alpha_bounds,
    temperature_bounds,
    N_boots,
    data,
    params,
    bounds,
    max_iter,
):
    alpha_list = np.linspace(*alpha_bounds, steps)
    temperature_list = np.linspace(*temperature_bounds, steps)

    rows, cols = len(alpha_list), len(temperature_list)
    mse_array = np.zeros((rows, cols), dtype=float)

    total_best_params = None
    total_best_mse = float("inf")
    with open("../data/param_estimation.csv", "a") as file:
        if file.tell() == 0:
            file.write("Alpha,Temperature,MSE\n")

        for i, alpha in enumerate(alpha_list):
            for j, temperature in enumerate(temperature_list):
                print(f"Alpha: {alpha}, Temperature: {temperature}")
                total_mse = 0
                for _ in range(N_boots):
                    best_params, best_mse = SA.simulated_annealing(
                        objective,
                        data,
                        params,
                        bounds,
                        proposal_func,
                        temperature,
                        alpha,
                        max_iter,
                    )
                    if best_mse < total_best_mse:
                        total_best_params = best_params
                        total_best_mse = best_mse
                    total_mse += best_mse

                mse_array[i, j] = total_mse / N_boots
                file.write(f"{alpha},{temperature},{mse_array[i, j]}\n")

    return total_best_params, total_best_mse

    # # Plot heatmap
    # plt.imshow(mse_array, cmap="hot", interpolation="nearest")
    # plt.colorbar()
    # plt.xlabel("Temperature")
    # plt.ylabel("Alpha")
    # plt.title("Mean Squared Error Heatmap")
    # plt.savefig("../visualization/mse_heatmap.png")
    # plt.show()


if __name__ == "__main__":
    true_data = load_data()
    partualy_data = tuple(tuple(arr[::2]) for arr in true_data)

    initial_params = np.array(
        [1.0, 0.1, 0.1, 1.0]
    )  # Initial guess for [alpha, beta, delta, gamma]
    bounds = np.array(
        [
            (0.01, 3.0),
            (0.01, 3.0),
            (0.01, 3.0),
            (0.01, 3.0),
        ]
    )  # Parameter bounds
    param_estimation(
        2, (0.8, 0.99), (10, 100), 10, partualy_data, initial_params, bounds, 1000
    )

#     initial_params = np.array(
#         [1.0, 0.1, 0.1, 1.0]
#     )  # Initial guess for [alpha, beta, delta, gamma]
#     bounds = [
#         (0.01, 3.0),
#         (0.01, 3.0),
#         (0.01, 3.0),
#         (0.01, 3.0),
#     ]  # Parameter bounds
#     initial_temp = 10.0  # Starting temperature
#     alpha = 0.9  # Cooling rate
#     max_iter = 1000  # Maximum iterations
#     tol = 10**-4

#     best_params, best_mse = SA.simulated_annealing(
#         objective,
#         partualy_data,
#         initial_params,
#         bounds,
#         proposal_func,
#         initial_temp,
#         alpha,
#         max_iter,
#     )

#     # best_params, best_mse = HC.hill_climbing(
#     #     objective, proposal_func, bootstrap_data, initial_params, max_iter, tol
#     # )

#     print("Best Parameters (alpha, beta, delta, gamma):", best_params)
#     print("Best Mean Squared Error:", best_mse)

#     prey, predator = compute_model(*true_data, *best_params)
#     # plot_model(*true_data, (prey, predator))
