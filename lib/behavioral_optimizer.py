import numpy as np
import pandas as pd
# import cvxpy as cp
import time
import gurobipy as gb
from gurobipy import GRB
from lib.Constants import ZONE_IDS

epsilon = 1e-5


# 1. compute attractiveness of each destination, for each origin
# we need Q_k, f_k, and c_ik

def solve_for_one_zone(origin_id, one_t_ahead_move, dest_attraction, supply_count, t_fifteen):
    """
    https://support.gurobi.com/hc/en-us/articles/360043111231-How-do-I-use-multiprocessing-in-Python-with-Gurobi-
    # TODO: fix distance (rebalancing) cost
    see the Edit in https://or.stackexchange.com/questions/5040/linearizing-a-program-with-multinomial-logit-in-the-objective
    in my case, because I have no intercept, t will be the lower bound value, i.e. 0. in which case, I can't recover
    v. so we arbitrarily set t > 0.
    @param origin_id:
    @param one_t_ahead_move:
    @param dest_attraction:
    @param supply_count:
    @param t_fifteen:
    @return:
    """
    # print(f'solving for zone {origin_id}')
    max_adj_budget = 10
    m_behavioral = gb.Model("Behavioral")
    # get the drivers in zone z
    drivers_zone_z = list(range(supply_count))
    # print(len(drivers_zone_z), 'len(drivers_zone,z)')

    w = m_behavioral.addVars(ZONE_IDS, drivers_zone_z, name="weights", vtype=GRB.CONTINUOUS, lb=0.0)
    t = m_behavioral.addVars(drivers_zone_z, name="ts", vtype=GRB.CONTINUOUS, lb=1)
    x_helper = m_behavioral.addVars(ZONE_IDS, ZONE_IDS, name="x_helper")

    m_behavioral.update()
    obj = x_helper.sum()

    # absolute values constraints, from Dimitris's book
    m_behavioral.addConstrs((one_t_ahead_move[(origin_id, dest, t_fifteen)] - gb.quicksum(
        w[(dest, d)] * dest_attraction[dest] for d in drivers_zone_z) <= x_helper[(origin_id, dest)]
                             for dest in ZONE_IDS
                             ))
    m_behavioral.addConstrs((one_t_ahead_move[(origin_id, dest, t_fifteen)] - gb.quicksum(
        w[(dest, d)] * dest_attraction[dest] for d in drivers_zone_z) >= -x_helper[(origin_id, dest)]
                             for dest in ZONE_IDS
                             ))

    # m_behavioral.addConstrs((w[(dest, d)] <= t[d] * (max_adj_budget / (dest_attraction[dest] + epsilon) + 1)
    #                          for d in drivers_zone_z for dest in ZONE_IDS), "adj budget pos")

    m_behavioral.addConstrs((gb.quicksum(w[(dest, d)] * dest_attraction[dest] for dest in ZONE_IDS) == 1
                             for d in drivers_zone_z), "driver_util_prob")

    # m_behavioral.addConstrs((gb.quicksum(w[(dest, d)] * dest_attraction[dest] - epsilon * t[d] for dest in ZONE_IDS) >= 0
    #                          for d in drivers_zone_z), "positive denominator")
    m_behavioral.setParam('OutputFlag', 0)
    # m_behavioral.update()

    # print(obj.size())
    m_behavioral.setObjective(obj, GRB.MINIMIZE)
    m_behavioral.update()
    # print(m_behavioral)
    m_behavioral.optimize()

    if m_behavioral.status == 2:
        print(f"obj value is {m_behavioral.objVal}")
        optimal_ws = m_behavioral.getAttr("x", w)
    else:
        print(f"status was {m_behavioral.status}")
        m_behavioral.computeIIS()
        m_behavioral.write("model.ilp")
        return None
    # because I set t = 1, w is equal to v.
    optimal_si = {}
    for dest in ZONE_IDS:
        for d in drivers_zone_z:
            optimal_si[dest, d] = np.log(optimal_ws[dest, d] * dest_attraction[dest] + epsilon) / (
                        dest_attraction[dest] + epsilon)
    return optimal_si


def solve_linear(K, N, pos, Y):
    """
    Solves a behaviorally consistent optimization problem.
    it only solves it for one step ahead.
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
            cost += cp.abs(Y[i, k] - cp.sum(
                cp.multiply(Z[k], np.where(pos == k, 1, 0))))  # so this is ignoring the attractiveness of each zone
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
