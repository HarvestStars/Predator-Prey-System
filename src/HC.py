import numpy as np

def hill_climbing(objective, proposal_func, true_data, init_params, max_iter, tol):
    """
    Hill Climbing Algorithm.

    Parameters:
    ------------
    - objective: callable, the objective function to minimize.
    - proposal_func: callable, generates a new state given the current state.
    - init_params: array-like, the initial state.
    - max_iter: int, maximum number of iterations.
    - tol: float, tolerance for stopping criterion.
    - noise: bool, whether to add noise to the data.

    Returns:
    ------------
    - best_params: array-like, the best state found.
    - best_error: float, the value of the objective function at the best state.
    """
    current_params = init_params
    best_params = current_params
    best_error = objective(current_params, *true_data)

    for i in range(max_iter):
        new_params = proposal_func(current_params)
        new_error = objective(new_params, *true_data)

        if new_error < best_error:
            best_error = new_error
            best_params = new_params
            current_params = new_params
        
        if np.abs(best_error - new_error) < tol:
            break

        # if i % 100 == 0:
        #     print(f"Iteration {i}, Best Value: {best_params}")

    return best_params, best_error
 