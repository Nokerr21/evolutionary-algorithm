"""
Author: Wojciech Kondracki 310941
Date: 21.03.2024
"""

import numpy as np
from cec2017.functions import f2, f13
import cec2017
from logic import evolutionary_classic

if __name__ == "__main__":
    MAX_X = 100  # Boundary limit for the solution
    # DIMENSIONALITY must be equal to 10 for the f2 and f13 functions
    DIMENSIONALITY = 10

    RUNS = 500   # Number of runs for the optimization algorithm
    SIGMA = 1    # Mutation strength
    U = 10       # Population size
    FES = 50000  # Number of objective function evaluations

    res = []

    for i in range(RUNS):
        p0 = []
        for j in range(U):
            x = np.random.uniform(-MAX_X, MAX_X, DIMENSIONALITY)
            p0.append(x)

        t_max = FES / U  # Maximum number of generations (iterations)
        o, x = evolutionary_classic(f2, p0, U, SIGMA, t_max, MAX_X)
        res.append(o)

    print(
        f"min: {np.min(res):.3f}, max: {np.max(res):.3f}, avg: {np.mean(res):.3f}, std: {np.std(res):.3f}"
    )
