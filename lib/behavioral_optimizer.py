import numpy as np
import pandas as pd
import cvxpy as cp
import time

epsilon = 1e-2


def solve_linear(K, N, pos, Y):
    """
    Solves a behaviorally consistent optimization problem.
    @param K: (int) number of zones
    @param N: (int) number of available drivers
    @param pos: (np.array) a numpy array consisting of drivers' locations, each of which a number
    @param Y: (np.ndarray) 2D array of zone-to-zone optimal relocations
    Example:
    K = 70
    N = 1500
    pos = np.random.randint(0, K + 1, N);
    Y = np.random.randn(K, K) * 2 + np.random.randint(0, 20);
    """
    start = time.time()
    Z = cp.Variable((K, N))
    #######
    cost = 0
    for k in range(K):
        for i in range(K):
            cost += cp.abs(Y[i, k] - cp.sum(cp.multiply(Z[k], np.where(pos == k, 1, 0))))
    constraints = [
        Z >= epsilon,
        cp.sum(Z, axis=0, keepdims=True) == 1
    ]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(verbose=True, solver=cp.MOSEK, warm_start=True)
    print("The optimal value for K = {} and N = {}  is {}".format(
        K, N, problem.value))
    print("Solved the problem in {} seconds".format(time.time() - start))
    return problem


def retrieve_si(problem):
    optimal_values = problem.variables()[0].value
    return np.apply_along_axis(lambda col: np.log(col / col[np.argmin(col)]),
                               0, optimal_values)
