import numpy as np
from SA import simulated_annealing

def generate_proposal_function(step_size):
    """
    Given a step size, returns a proposal function that perturbs the current state.
    """
    def proposal_func(current_state):
        return current_state + np.random.normal(0, step_size, size=len(current_state))
    return proposal_func

def run_single_sa_experiment(objective_func, t_data, x_data, y_data, initial_state, 
                             param_bounds, proposal_func, initial_temp, alpha, max_iter):
    """
    Runs a single SA optimization and returns the best final value.
    
    Parameters
    ----------
    - objective_func: callable
        objective_func(params, t_data, x_data, y_data) -> float (the value of objective)
    - t_data, x_data, y_data: arrays
        The data sets.
    - initial_state: array-like
        Initial guess for (a,b,c,d).
    - param_bounds: tuple of tuples
        Bounds for each parameter.
    - proposal_func: callable
        Function to generate new candidate states.
    - initial_temp, alpha: floats
        SA parameters for initial temperature and cooling rate.
    - max_iter: int
        Maximum number of iterations for SA.
    
    Should return:
    - best_state
    - best_value
    - iteration_history: a list (or array) of best_values recorded at each iteration (or at intervals), like [value_at_iter_1, value_at_iter_2, ..., value_at_iter_max]
    """

    best_state, best_value, iteration_history = simulated_annealing(
        objective_h=lambda params: objective_func(params, t_data, x_data, y_data),
        t_data=t_data,
        x_data=x_data,
        y_data=y_data,
        initial_state=initial_state,
        bounds=param_bounds,
        proposal_func=proposal_func,
        initial_temp=initial_temp,
        alpha=alpha,
        max_iter=max_iter
    )
    return best_state, best_value, iteration_history

def evaluate_parameter_config(objective_func, t_data, x_data, y_data, 
                              param_bounds, initial_state_guess, 
                              initial_temp, alpha, step_size, max_iter, 
                              num_runs):
    """
    Runs multiple SA experiments with the given parameter configuration 
    and returns statistical metrics:
      - mean/std of final objective values (error margins)
      - mean/std convergence curves (for studying convergence behavior)
    """

    proposal_func = generate_proposal_function(step_size)
    
    final_values = []
    final_states = []  # To store best states from each run
    all_histories = []  # To store iteration_history from each run
    
    for _ in range(num_runs):
        best_state, best_val, iteration_history = run_single_sa_experiment(
            objective_func=objective_func,
            t_data=t_data,
            x_data=x_data,
            y_data=y_data,
            initial_state=initial_state_guess,
            param_bounds=param_bounds,
            proposal_func=proposal_func,
            initial_temp=initial_temp,
            alpha=alpha,
            max_iter=max_iter
        )
        final_values.append(best_val)
        final_states.append(best_state)
        all_histories.append(iteration_history)

    # Compute error margins for final values
    mean_val = float(np.mean(final_values))
    mean_state = np.mean(final_states, axis=0)
    std_val = float(np.std(final_values))
    
    # Analyze convergence behavior
    # Assume all_histories have the same length = max_iter
    # Convert to an array of shape (num_runs, max_iter)
    all_histories_array = np.array(all_histories)
    # Mean convergence curve
    mean_curve = np.mean(all_histories_array, axis=0)
    # Std convergence curve
    std_curve = np.std(all_histories_array, axis=0)

    # Here, you can define a metric for "convergence speed", for instance:
    # The iteration at which improvement falls below a threshold
    # Example: find the first iteration where the difference between 
    # consecutive means is less than a small epsilon
    epsilon = 1e-4
    convergence_iteration = None
    for i in range(1, len(mean_curve)):
        if abs(mean_curve[i] - mean_curve[i-1]) < epsilon:
            convergence_iteration = i
            break

    return {
        'mean_final_objective': mean_val,
        'mean_final_state': mean_state,
        'std_final_objective': std_val,
        'final_values': final_values,
        'mean_convergence_curve': mean_curve,
        'std_convergence_curve': std_curve,
        'convergence_iteration': convergence_iteration
    }

def tune_sa_parameters(objective_func, t_data, x_data, y_data, param_bounds, 
                       initial_state_guess, 
                       initial_temp_candidates, alpha_candidates, 
                       step_size_candidates, max_iter_candidates, 
                       num_runs=10):
    """
    Tunes SA parameters by trying all combinations of initial_temp, alpha, step_size, and max_iter.
    For each combination, runs multiple experiments to gather statistics and store the results.

    Parameters
    ----------
    - objective_func: callable
        The objective function: objective_func(params, t_data, x_data, y_data) -> float
    - t_data, x_data, y_data: arrays
        The datasets of length N.
    - param_bounds: tuple of tuples
        Bounds for parameters (e.g. ((a_min,a_max),(b_min,b_max),(c_min,c_max),(d_min,d_max))).
    - initial_state_guess: array-like
        Initial guess of parameters (a,b,c,d).
    - initial_temp_candidates: list
        Candidate initial temperatures.
    - alpha_candidates: list
        Candidate cooling rates.
    - step_size_candidates: list
        Candidate step sizes for proposal distribution.
    - max_iter_candidates: list
        Candidate maximum iterations (or chain lengths).
    - num_runs: int
        Number of runs per configuration to gather statistics.

    Returns
    -------
    results: dict
        Keys: (initial_temp, alpha, step_size, max_iter) tuples
        Values: dict with keys 'mean_final_objective', 'std_final_objective', 'all_final_values'
    """
    results = {}
    # Iterate over all candidate parameters
    for T0 in initial_temp_candidates:
        for alpha in alpha_candidates:
            for step_size in step_size_candidates:
                for max_iter in max_iter_candidates:
                    stats = evaluate_parameter_config(
                        objective_func=objective_func,
                        t_data=t_data,
                        x_data=x_data,
                        y_data=y_data,
                        param_bounds=param_bounds,
                        initial_state_guess=initial_state_guess,
                        initial_temp=T0,
                        alpha=alpha,
                        step_size=step_size,
                        max_iter=max_iter,
                        num_runs=num_runs
                    )
                    config_key = (T0, alpha, step_size, max_iter)
                    results[config_key] = stats

    return results

def evaluate_generalization_with_incremental_data(
    objective_func,
    initial_t_data, initial_x_data, initial_y_data,
    full_t_data, full_x_data, full_y_data,
    best_state,            # best state from initial dataset
    increment_list,        # increments of data to add
    mse_increase_threshold=0.3
):
    """
    Evaluate generalization by incrementally adding samples. 
    With no randomness, one evaluation per dataset is enough.

    Parameters
    ----------
    - objective_func: callable
        objective_func(params, t_data, x_data, y_data) -> float (deterministic)
    - initial_t_data, initial_x_data, initial_y_data: arrays
        Initial smaller dataset used to find best_state.
    - full_t_data, full_x_data, full_y_data: arrays
        Full dataset to increment from.
    - best_state: array-like
        Best parameters found on the initial dataset.
    - increment_list: list of int
        Increments of data points to add at each step.
    - mse_increase_threshold: float
        Relative threshold for MSE increase indicating failure.

    Returns
    -------
    results: dict
        {
            'initial_mse': float,
            'increments_results': [
                {
                    'new_size': int,
                    'expanded_mse': float,
                    'relative_increase': float,
                    'generalize_success': bool
                },
                ...
            ],
            'final_state': best_state,
            'final_size': int
        }

    If generalization fails at some increment, stops and returns immediately.
    """

    # Compute initial MSE once
    initial_mse = objective_func(best_state, initial_t_data, initial_x_data, initial_y_data)

    results = {
        'initial_mse': initial_mse,
        'increments_results': [],
        'final_state': best_state,
        'final_size': len(initial_t_data)
    }

    current_size = len(initial_t_data)

    for inc in increment_list:
        new_size = current_size + inc
        if new_size > len(full_t_data):
            # Not enough data
            break

        t_sub = full_t_data[:new_size]
        x_sub = full_x_data[:new_size]
        y_sub = full_y_data[:new_size]

        expanded_mse = objective_func(best_state, t_sub, x_sub, y_sub)
        rel_increase = (expanded_mse - initial_mse) / initial_mse

        generalize_success = True
        if rel_increase > mse_increase_threshold:
            generalize_success = False
            results['increments_results'].append({
                'new_size': new_size,
                'expanded_mse': expanded_mse,
                'relative_increase': rel_increase,
                'generalize_success': generalize_success
            })
            # Stop here, let outer logic handle it
            results['final_size'] = new_size
            return results

        # Update results and continue
        results['increments_results'].append({
            'new_size': new_size,
            'expanded_mse': expanded_mse,
            'relative_increase': rel_increase,
            'generalize_success': generalize_success
        })
        results['final_size'] = new_size
        current_size = new_size

    return results
