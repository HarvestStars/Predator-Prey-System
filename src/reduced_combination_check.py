import numpy as np
import importantce_check as ic
import SA as sa
import weighted_objective
import matplotlib.pyplot as plt

def reduced_combination_check(x_data, y_data, t_data, importance_X, importance_Y, weighted_objective, global_SA_param, global_objective, global_ODE_param, repeat=10):
    # Sort indices by importance (ascending order)
    sorted_indices_X = np.argsort(importance_X)
    sorted_indices_Y = np.argsort(importance_Y)
    removal_impact_combination = []
    initial_state, bounds, proposal_func, initial_temp, alpha, max_iter = global_SA_param

    for r in range(repeat):
        removal_impact_combination_once = []
        # Remove Y points and measure impact while keeping X fixed
        for remove_count in range(len(sorted_indices_Y) - 10):
            zero_index_y = set(sorted_indices_Y[:remove_count + 1])
            zero_index_x = set(sorted_indices_X[:remove_count + 1])
            trained_data = (x_data, y_data, t_data, zero_index_x, zero_index_y) # Data to pass to SA

            weighted_params, weighted_mse = ic.simulated_annealing_for_importance_check(weighted_objective, trained_data, initial_state, bounds, proposal_func, initial_temp, alpha, max_iter)
            mse_global = global_objective(global_ODE_param, t_data, x_data, y_data)
            mse_weighted = global_objective(weighted_params, t_data, x_data, y_data)
            removal_impact_combination_once.append(abs(mse_global - mse_weighted))
    
        # Append impact values for this repeat
        removal_impact_combination.append(removal_impact_combination_once)

    return removal_impact_combination

if __name__ == "__main__":
    
    import_x = [1.54874547, 0.28063917, 0.21834619, 0.14315223, 3.07697937, 3.79186496,
                0.28779032, 1.20023047, 0.36561821, 0.21744707, 1.10002108, 0.1803091,
                1.46810452, 3.4810551, 0.37391491, 2.48770886, 0.54674331, 1.03172059,
                0.95112705, 0.0527759,  0.12858293, 0.53713594, 3.0736799,  2.36828881,
                0.22847406, 4.06715214, 0.35751135, 0.30062834, 3.06294998, 2.83825915,
                1.61343006, 0.75274751, 1.83161782, 0.32025698, 1.23511914, 0.32655349,
                0.46622688, 0.38787294, 3.23762695, 0.25370015, 0.28886548, 0.96204958,
                2.93325523, 1.37136002, 0.16919331, 2.15845852, 2.75198123, 0.47867078,
                0.93277966, 3.31128425, 0.19147175, 0.32857252, 1.13010573, 1.70339563,
                0.60634642, 0.13413685, 0.2107082,  0.06454633, 0.96018865, 0.3165444,
                0.32013658, 2.46260301, 0.14570762, 0.37893408, 0.48045148, 0.18067808,
                0.3865201,  0.29426471, 2.97275151, 3.21133395, 0.28999335, 0.31070495,
                0.35806467, 4.59345143, 0.48085805, 0.17225277, 3.07698398, 0.68501835,
                0.24335508, 0.15765475, 2.94938302, 0.17596425, 5.35839094, 3.44223957,
                1.04691421, 0.33846544, 0.75886921, 1.0085601,  0.4196205,  1.34494881,
                0.47690837, 0.20995651, 3.14201656, 0.45674085, 3.91906743, 0.62953774,
                0.36239812, 2.69185341, 0.3901205,  2.98414503]
    
    import_y = [0.4822787,  3.50832488 ,3.13082731, 3.04700913, 0.27148133, 0.38336931,
                1.2400508,  0.63046768 ,0.22203841, 0.93635141, 3.26929045, 3.37810165,
                2.00672756, 0.2355101,  1.09567314, 0.26696692, 1.35280984, 0.80722457,
                3.28819581, 0.30731465, 4.67630961, 3.10451521, 0.68862765, 2.93142854,
                2.57328477, 0.19297597, 3.24649678, 0.13249698, 0.04127156, 0.87732716,
                1.9644626 , 0.0725236 , 0.84969553, 1.30496586, 2.80777128, 0.27984079,
                0.25465052 ,2.93738421, 0.57293466, 1.57956828, 1.76861542, 0.3939043,
                1.46874115, 0.35396074, 0.23218226, 2.81879057, 2.90712134, 2.74323701,
                2.93590495, 2.89565169, 2.75625323, 0.2418387,  0.76987993, 0.16396167,
                1.09096962, 0.35099753, 0.28570669, 0.90647254, 0.33435617, 1.04845313,
                0.05928693, 0.01631674, 0.05502707, 0.78489617, 0.36306418, 0.2173876,
                0.20324152 ,1.18699169, 0.98903525, 0.26057699, 0.35209972, 1.21124182,
                3.53314268, 5.97398392, 0.19803516, 0.94917666, 2.20049586, 1.7708009,
                0.20471975, 0.97240295, 0.13209019, 3.05693036, 0.25888486, 2.64991315,
                0.48658588, 1.91018993, 1.41558382, 0.73770469, 0.25332984, 3.19109363,
                1.54453874 ,0.62274009, 0.20306884, 0.38021365, 0.40550473, 0.94836306,
                0.36680388 ,0.1435744 , 0.3137448,  0.16673496]

    def load_data():
        data = np.loadtxt("../observed_data/predator-prey-data.csv", delimiter=",")
        return data[:, 0], data[:, 1], data[:, 2]
    
    raw_data = load_data()
    t_data = raw_data[0]
    x_data = raw_data[1]
    y_data = raw_data[2]

    # a guess of global_SA_param and global_ODE_param
    global_SA_param = [
        np.array([1.0, 0.1, 0.1, 1.0]),  # initial_state
        [(0.1, 10), (0.01, 1), (0.01, 1), (0.1, 10)],  # bounds
        lambda x: x + np.random.normal(0, 0.1, size=len(x)),  # proposal_func
        200,  # initial_temp
        0.9,  # alpha
        500,  # max_iter
    ]

    # global_ODE_param = [0.8, 0.99, 10, 100]
    global_ODE_param, best_mse = sa.simulated_annealing(    
        ic.global_objective,
        raw_data,
        np.array([1.0, 0.1, 0.1, 1.0]),
        [(0.1, 10), (0.01, 1), (0.01, 1), (0.1, 10)],
        lambda x: x + np.random.normal(0, 0.1, size=len(x)),
        200,
        0.9,
        1000
    )
    print(f"Global ODE Parameters: {global_ODE_param}")
    print(f"Global Objective Value: {best_mse}")

    combination_effect = reduced_combination_check(x_data, y_data, t_data, import_x, import_y, weighted_objective.objective_weighted, global_SA_param, ic.global_objective, global_ODE_param, repeat=2)

    print(f"Importance of Removed X: {np.mean(combination_effect, axis=0)}")
    plt.plot(np.mean(combination_effect, axis=0), label="combination of reducing some X and Y")
    plt.legend()
    plt.show()