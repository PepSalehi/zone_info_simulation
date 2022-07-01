import pickle

import numpy as np
import pandas as pd
# import cvxpy as cp
import time
import gurobipy as gb
from gurobipy import GRB
from lib.Constants import ZONE_IDS

epsilon = 1e-5
c_counter = 0


# 1. compute attractiveness of each destination, for each origin
# we need Q_k, f_k, and c_ik
# https://support.gurobi.com/hc/en-us/community/posts/360071928312-Stopping-the-program-if-best-objective-has-not-changed-after-a-while
def cb(model, where):
    if where == GRB.Callback.MIPNODE:
        # Get model objective
        obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

        # Has objective changed?
        if abs(obj - model._cur_obj) > 1e-2:
            # If so, update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # Terminate if objective has not improved in 20s
    if time.time() - model._time > 20:
        model.terminate()


# from joblib import wrap_non_picklable_objects
# from joblib.externals.loky import set_loky_pickler
# from joblib import parallel_backend
# from joblib import Parallel, delayed
# @delayed
# @wrap_non_picklable_objects

def solve_for_personalized_drivers(origin_id, one_t_ahead_move, dest_attraction, supply_count, t_fifteen,
                                   LOWER_BOUND_SI=0.5, UPPER_BOUND_SI=2):
    '''
    we assume we will give a different map to each individual driver, i.e. personalize the map
    If n_drivers >= total_drivers:
        for demand d in each zone, solve the problem for one driver, and assign it to d drivers
    else:
        start from one zone, assign, then move to next one. order is arbitary

    @param origin_id:
    @param one_t_ahead_move:
    @param dest_attraction:
    @param supply_count:
    @param t_fifteen:
    @return:
    '''
    # print("zone id ", origin_id)
    total_demand = np.sum([one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS])
    if total_demand < 1:
        return None

    normalized_dest_attraction = {}
    max_util = np.max(list(dest_attraction.values()))
    if max_util <= 0: max_util = 1
    for k, v in dest_attraction.copy().items():
        if v < 1:
            v = 0
        normalized_dest_attraction[k] = v / max_util

    demand_vector = np.array([one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS])
    demand_zones = [dest for dest in ZONE_IDS if one_t_ahead_move[(origin_id, dest, t_fifteen)] > 0]
    demand_hotspots = {zone: one_t_ahead_move[(origin_id, zone, t_fifteen)] for zone in demand_zones}
    zero_attraction_zoneids = [k for k, v in normalized_dest_attraction.items() if v == 0]
    candidate_zones = [zone for zone in zero_attraction_zoneids if zone not in demand_zones]
    nonzero_attractions = {k: v for k, v in normalized_dest_attraction.items() if v > 0}
    nonezero_zoneids = [k for k, v in normalized_dest_attraction.items() if v > 0]
    leave_alone_zoneid = [zone for zone in nonezero_zoneids if zone not in demand_zones]

    epsilon = 1e-5
    all_d_values = []
    optimal_si = {}
    start_t = time.time()

    # get the drivers in zone z
    drivers_zone_z = list(range(supply_count))  # and if its length is zero, i.e., no drivers there?
    start_d_index = 0

    if supply_count >= total_demand:
        for zone, demand in demand_hotspots.items():
            demand = int(demand)
            # solve for one
            TARGET_DEST = zone
            m_behavioral = gb.Model("Behavioral")
            si = m_behavioral.addVars(ZONE_IDS, name="si", vtype=GRB.CONTINUOUS, lb=LOWER_BOUND_SI, ub=UPPER_BOUND_SI)
            p = m_behavioral.addVars(ZONE_IDS, name="prob", vtype=GRB.CONTINUOUS, lb=0, ub=1.0)
            # y = log(p)
            lgP = m_behavioral.addVars(ZONE_IDS, name="logP", vtype=GRB.CONTINUOUS,
                                       lb=-GRB.INFINITY)  # if p is zero,then log(p) is inf.lb=np.floor(np.log(epsilon))
            # y = m_behavioral.addVars(nonezero_zoneids, name="y", vtype=GRB.BINARY)
            obj = 1 - p[TARGET_DEST]
            m_behavioral.update()
            m_behavioral.addConstr(gb.quicksum(p[dest] for dest in ZONE_IDS) == 1, "driver_util_prob")
            # define ln(P)
            for dest in nonezero_zoneids:
                m_behavioral.addGenConstrLog(p[dest], lgP[dest], options="FuncPieces=2")
            for k in ZONE_IDS:
                for j in ZONE_IDS:
                    if k != j:
                        m_behavioral.addConstr((lgP[k] - lgP[j] - (
                                normalized_dest_attraction[k] * si[k] - normalized_dest_attraction[j] * si[
                            j]) <= epsilon))
            m_behavioral.Params.MIPFocus = 2
            m_behavioral.Params.LogToConsole = 0
            m_behavioral.setObjective(obj, GRB.MINIMIZE)
            m_behavioral.optimize()
            # print(m_behavioral.status)
            # print('m_behavioral.SolCount ', m_behavioral.SolCount)

            if m_behavioral.status == 2:
                si_solution = m_behavioral.getAttr("x", si)
                for d in drivers_zone_z[start_d_index: (start_d_index + demand)]:
                    for k, v in si_solution.items():
                        optimal_si[(k, d)] = v

                    # if len(zero_attraction_zoneids) > 0:
                    #     for k in zero_attraction_zoneids:
                    #         optimal_si[(k, d)] = 0
                start_d_index = start_d_index + demand

        # for extra drivers, don't do anything
        for d in drivers_zone_z[start_d_index:]:
            for k in ZONE_IDS:
                optimal_si[(k, d)] = 1

    else:
        # now solve it for one zone
        num_available_drivers = supply_count
        for zone, demand in demand_hotspots.items():
            demand = int(demand)
            if num_available_drivers >= demand:
                # solve as usual
                x = demand
                TARGET_DEST = zone
                m_behavioral = gb.Model("Behavioral")
                si = m_behavioral.addVars(ZONE_IDS, name="si", vtype=GRB.CONTINUOUS, lb=LOWER_BOUND_SI,
                                          ub=UPPER_BOUND_SI)
                p = m_behavioral.addVars(ZONE_IDS, name="prob", vtype=GRB.CONTINUOUS, lb=0, ub=1.0)
                # y = log(p)
                lgP = m_behavioral.addVars(ZONE_IDS, name="logP", vtype=GRB.CONTINUOUS,
                                           lb=-GRB.INFINITY)  # if p is zero,then log(p) is inf.lb=np.floor(np.log(epsilon))
                # y = m_behavioral.addVars(nonezero_zoneids, name="y", vtype=GRB.BINARY)
                obj = 1 - p[TARGET_DEST]
                m_behavioral.update()
                m_behavioral.addConstr(gb.quicksum(p[dest] for dest in ZONE_IDS) == 1, "driver_util_prob")
                # define ln(P)
                for dest in ZONE_IDS:
                    m_behavioral.addGenConstrLog(p[dest], lgP[dest], options="FuncPieces=2")
                for k in ZONE_IDS:
                    for j in ZONE_IDS:
                        if k != j:
                            m_behavioral.addConstr((lgP[k] - lgP[j] - (
                                    normalized_dest_attraction[k] * si[k] - normalized_dest_attraction[j] * si[
                                j]) <= epsilon))
                m_behavioral.Params.MIPFocus = 2
                m_behavioral.Params.LogToConsole = 0
                m_behavioral.setObjective(obj, GRB.MINIMIZE)
                m_behavioral.optimize()
                # print(m_behavioral.status)
                # print('m_behavioral.SolCount ', m_behavioral.SolCount)

                if m_behavioral.status == 2:
                    si_solution = m_behavioral.getAttr("x", si)
                    for d in drivers_zone_z[start_d_index: (start_d_index + demand)]:
                        for k, v in si_solution.items():
                            optimal_si[(k, d)] = v

                        # if len(zero_attraction_zoneids) > 0:
                        #     for k in zero_attraction_zoneids:
                        #         optimal_si[(k, d)] = 0
                    start_d_index = start_d_index + demand

                num_available_drivers = num_available_drivers - demand

            elif num_available_drivers > 0:
                TARGET_DEST = zone
                m_behavioral = gb.Model("Behavioral")
                si = m_behavioral.addVars(ZONE_IDS, name="si", vtype=GRB.CONTINUOUS, lb=LOWER_BOUND_SI,
                                          ub=UPPER_BOUND_SI)
                p = m_behavioral.addVars(ZONE_IDS, name="prob", vtype=GRB.CONTINUOUS, lb=0, ub=1.0)
                # y = log(p)
                lgP = m_behavioral.addVars(ZONE_IDS, name="logP", vtype=GRB.CONTINUOUS,
                                           lb=-GRB.INFINITY)  # if p is zero,then log(p) is inf.lb=np.floor(np.log(epsilon))
                # y = m_behavioral.addVars(nonezero_zoneids, name="y", vtype=GRB.BINARY)
                obj = 1 - p[TARGET_DEST]
                m_behavioral.update()
                m_behavioral.addConstr(gb.quicksum(p[dest] for dest in nonezero_zoneids) == 1, "driver_util_prob")
                # define ln(P)
                for dest in ZONE_IDS:
                    m_behavioral.addGenConstrLog(p[dest], lgP[dest], options="FuncPieces=2")
                for k in ZONE_IDS:
                    for j in ZONE_IDS:
                        if k != j:
                            m_behavioral.addConstr((lgP[k] - lgP[j] - (
                                    normalized_dest_attraction[k] * si[k] - normalized_dest_attraction[j] * si[
                                j]) <= epsilon))
                m_behavioral.Params.MIPFocus = 2
                m_behavioral.Params.LogToConsole = 0
                m_behavioral.setObjective(obj, GRB.MINIMIZE)
                m_behavioral.optimize()
                # print(m_behavioral.status)
                # print('m_behavioral.SolCount ', m_behavioral.SolCount)

                if m_behavioral.status == 2:
                    si_solution = m_behavioral.getAttr("x", si)
                    for d in drivers_zone_z[start_d_index: (start_d_index + demand)]:
                        for k, v in si_solution.items():
                            optimal_si[(k, d)] = v

                        # if len(zero_attraction_zoneids) > 0:
                        #     for k in zero_attraction_zoneids:
                        #         optimal_si[(k, d)] = 0
                    start_d_index = start_d_index + demand

                num_available_drivers = 0

            else:
                # no drivers left
                break

    return optimal_si


def solve_for_area_wide_drivers(origin_id, one_t_ahead_move, dest_attraction, supply_count, t_fifteen,
                                LOWER_BOUND_SI=0.5, UPPER_BOUND_SI=2):
    '''
    give the same map to all drivers in a given zone

    @param origin_id:
    @param one_t_ahead_move:
    @param dest_attraction:
    @param supply_count:
    @param t_fifteen:
    @return:
    '''
    # print("zone id ", origin_id)
    total_demand = np.sum([one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS])
    if total_demand < 1:
        return None

    normalized_dest_attraction = {}
    max_util = np.max(list(dest_attraction.values()))
    if max_util <= 0: max_util = 1
    for k, v in dest_attraction.copy().items():
        if v < 1:
            v = 0
        normalized_dest_attraction[k] = v / max_util

    demand_vector = np.array([one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS])
    demand_zones = [dest for dest in ZONE_IDS if one_t_ahead_move[(origin_id, dest, t_fifteen)] > 0]
    demand_hotspots = {zone: one_t_ahead_move[(origin_id, zone, t_fifteen)] for zone in demand_zones}
    zero_attraction_zoneids = [k for k, v in normalized_dest_attraction.items() if v == 0]
    candidate_zones = [zone for zone in zero_attraction_zoneids if zone not in demand_zones]
    nonzero_attractions = {k: v for k, v in normalized_dest_attraction.items() if v > 0}
    nonezero_zoneids = [k for k, v in normalized_dest_attraction.items() if v > 0]
    leave_alone_zoneid = [zone for zone in nonezero_zoneids if zone not in demand_zones]

    epsilon = 1e-5
    all_d_values = []
    optimal_si = {}
    start_t = time.time()

    # get the drivers in zone z
    drivers_zone_z = list(range(supply_count))  # and if its length is zero, i.e., no drivers there?
    start_d_index = 0
    # solve for one
    m_behavioral = gb.Model("Behavioral")
    si = m_behavioral.addVars(ZONE_IDS, name="si", vtype=GRB.CONTINUOUS, lb=LOWER_BOUND_SI, ub=UPPER_BOUND_SI)
    p = m_behavioral.addVars(ZONE_IDS, name="prob", vtype=GRB.CONTINUOUS, lb=0, ub=1.0)
    # y = log(p)
    lgP = m_behavioral.addVars(ZONE_IDS, name="logP", vtype=GRB.CONTINUOUS,
                               lb=-GRB.INFINITY)  # if p is zero,then log(p) is inf.lb=np.floor(np.log(epsilon))
    obj = gb.quicksum((1 - p[dest]) * one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS)

    m_behavioral.update()
    m_behavioral.addConstr(gb.quicksum(p[dest] for dest in ZONE_IDS) == 1, "driver_util_prob")
    # define ln(P)
    for dest in nonezero_zoneids:
        m_behavioral.addGenConstrLog(p[dest], lgP[dest], options="FuncPieces=2")
    for k in ZONE_IDS:
        for j in ZONE_IDS:
            if k != j:
                m_behavioral.addConstr((lgP[k] - lgP[j] - (
                        normalized_dest_attraction[k] * si[k] - normalized_dest_attraction[j] * si[j]) <= epsilon))
    m_behavioral.Params.MIPFocus = 2
    m_behavioral.Params.LogToConsole = 0
    m_behavioral.setObjective(obj, GRB.MINIMIZE)
    m_behavioral.optimize()

    if m_behavioral.status == 2:
        si_solution = m_behavioral.getAttr("x", si)
        for d in drivers_zone_z:
            for k, v in si_solution.items():
                optimal_si[(k, d)] = v
    else:
        return  None
    return optimal_si




def alternative_solve_for_one_zone(origin_id, one_t_ahead_move, dest_attraction, supply_count, t_fifteen):
    # supply_count: int. number of drivers in the zone
    # one_t_ahead_move: how many should relocate to where
    # if works, it probably pays off to run this in parallel.
    print("zone id ", origin_id)
    if np.sum([one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS]) < 1:
        return None

    normalized_dest_attraction = {}
    max_util = np.max(list(dest_attraction.values()))
    if max_util <= 0: max_util = 1
    for k, v in dest_attraction.copy().items():
        if v < 1:
            v = 0
        normalized_dest_attraction[k] = v / max_util

    demand_vector = [one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS]
    demand_zones = [dest for dest in ZONE_IDS if one_t_ahead_move[(origin_id, dest, t_fifteen)] > 0]
    zero_attraction_zoneids = [k for k, v in normalized_dest_attraction.items() if v == 0]
    nonzero_attractions = {k: v for k, v in normalized_dest_attraction.items() if v > 0}
    nonezero_zoneids = [k for k, v in normalized_dest_attraction.items() if v > 0]

    epsilon = 1e-5
    all_d_values = []
    start_t = time.time()

    # get the drivers in zone z
    drivers_zone_z = list(range(supply_count))  # and if its length is zero, i.e., no drivers there?
    # if len(drivers_zone_z) <= 10:
    #     return {}, 0

    m_behavioral = gb.Model("Behavioral")
    print('initiating the model for zone ', origin_id)
    si = m_behavioral.addVars(nonezero_zoneids, drivers_zone_z, name="si", vtype=GRB.CONTINUOUS, lb=0, ub=500)
    p = m_behavioral.addVars(nonezero_zoneids, drivers_zone_z, name="prob", vtype=GRB.CONTINUOUS, lb=epsilon)
    # y = log(p)
    lgP = m_behavioral.addVars(nonezero_zoneids, drivers_zone_z, name="logP", vtype=GRB.CONTINUOUS,
                               lb=-GRB.INFINITY)  # if p is zero,then log(p) is inf.lb=np.floor(np.log(epsilon))

    t = m_behavioral.addVars(nonezero_zoneids, name="x_helper")

    m_behavioral.update()
    obj = t.sum()

    # reference zone
    # for d in drivers_zone_z:
    #     si[ZONE_IDS[0], d].ub = 1.00
    #     si[ZONE_IDS[0], d].lb = 1.00

    # Abs value in the objective function
    m_behavioral.addConstrs(
        (one_t_ahead_move[(origin_id, dest, t_fifteen)] - gb.quicksum(p[(dest, d)] for d in drivers_zone_z) <= t[dest]
         for dest in nonezero_zoneids
         ))
    m_behavioral.addConstrs(
        (one_t_ahead_move[(origin_id, dest, t_fifteen)] - gb.quicksum(p[(dest, d)] for d in drivers_zone_z) >= -t[dest]
         for dest in nonezero_zoneids
         ))

    # m_behavioral.addConstrs((gb.quicksum(p[(dest, d)] for dest in ZONE_IDS) == 1 for d in drivers_zone_z),
    #                         "driver_util_prob")
    m_behavioral.addConstrs(
        (gb.quicksum(p[(dest, d)] for dest in nonezero_zoneids) >= 1 - epsilon for d in drivers_zone_z),
        "driver_util_prob")
    m_behavioral.addConstrs(
        (gb.quicksum(p[(dest, d)] for dest in nonezero_zoneids) <= 1 + epsilon for d in drivers_zone_z),
        "driver_util_prob2")
    # define ln(P)
    for dest in nonezero_zoneids:
        for d in drivers_zone_z:
            m_behavioral.addGenConstrLog(p[dest, d], lgP[dest, d], options="FuncPieces=2")

    # m_behavioral.update()
    print('setting up the rest of the model for zone ', origin_id)
    for d in drivers_zone_z:
        for k in nonezero_zoneids:
            for j in nonezero_zoneids:
                if k != j:
                    # what if instead of equality we add a difference threshold?
                    m_behavioral.addConstr((lgP[k, d] - lgP[j, d] - (
                            normalized_dest_attraction[k] * si[k, d] - normalized_dest_attraction[j] * si[j, d]) <= 10))
                    m_behavioral.addConstr((lgP[k, d] - lgP[j, d] - (
                            normalized_dest_attraction[k] * si[k, d] - normalized_dest_attraction[j] * si[
                        j, d]) >= -10))
                    # m_behavioral.addConstr((lgP[k, d] - lgP[j, d] ==
                    #                         dest_attraction[k] * si[k, d] - dest_attraction[j] * si[j, d]))

    # m_behavioral.update()

    m_behavioral.Params.MIPFocus = 2
    # m_behavioral.Params.Threads = 1
    #     m_behavioral.Params.Presolve = 0
    m_behavioral.Params.LogToConsole = 0

    # m_behavioral.update()
    # print(obj.size())
    m_behavioral.setObjective(obj, GRB.MINIMIZE)
    m_behavioral.update()
    # print(m_behavioral)
    # Last updated objective and time
    m_behavioral._cur_obj = float('inf')
    m_behavioral._time = time.time()
    print('ready to optimize for zone id ', origin_id)
    m_behavioral.optimize()  # callback=cb
    # print(m_behavioral.status)
    # print('m_behavioral.SolCount ', m_behavioral.SolCount)
    # if m_behavioral.SolCount > 0:
    #     # print(m_behavioral.ObjVal)
    # else:
    # print("############")
    # print("no solution")
    # print("#####")
    # print(m_behavioral.ObjBoundC)

    end = time.time()
    print('time in seconds', (end - start_t), 'for zone ', origin_id)

    if m_behavioral.status == 2:
        print("DONE with zone id ", origin_id)
        print(f"obj value is {m_behavioral.objVal}")

        optimal_si = m_behavioral.getAttr("x", si)
        if len(zero_attraction_zoneids) > 0 and len(drivers_zone_z) > 0:
            for k in zero_attraction_zoneids:
                for d in drivers_zone_z:
                    optimal_si[(k, d)] = 0
        # print(f"SOLVED")
        # print(f"number of drivers {len(drivers_zone_z)}")
        # print(f"demand was {[one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS]}")
        # print(f"Total demand was {np.sum([one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS])}")
    else:
        # print(f"status was {m_behavioral.status}")
        # print(f"number of drivers {len(drivers_zone_z)}")
        # print(f"demand was {[one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS]}")
        # print(f"Total demand was {np.sum([one_t_ahead_move[(origin_id, dest, t_fifteen)] for dest in ZONE_IDS])}")

        # global c_counter
        # data = {'one_t_ahead_move': one_t_ahead_move,
        #         'dest_attraction': dest_attraction,
        #         'supply_count': supply_count,
        #         't_fifteen': t_fifteen,
        #         'origin_id': origin_id
        #         }
        # with open(f'opt_data_{c_counter}.pickle', 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(f'source_data_{c_counter}.pickle', 'wb') as handle:
        #     pickle.dump(source_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # c_counter += 1
        # there is something fishy going on with the demand one_t_ahead_move. or rather, the rebalancing thing. it is all zero bar one.
        # go to debug?
        # m_behavioral.computeIIS()
        # m_behavioral.write("model.ilp")
        return None  # , 1
    # because I set t = 1, w is equal to v.
    # optimal_si = {}
    # for dest in ZONE_IDS:
    #     for d in drivers_zone_z:
    #         optimal_si[dest, d] = np.log(optimal_ws[dest, d] * dest_attraction[dest] + epsilon) / (
    #                     dest_attraction[dest] + epsilon) # I'm not sure what this is ...
    return optimal_si  # , m_behavioral.objVal


def solve_for_one_zone(origin_id, one_t_ahead_move, dest_attraction, supply_count, t_fifteen):
    """
    https://support.gurobi.com/hc/en-us/articles/360043111231-How-do-I-use-multiprocessing-in-Python-with-Gurobi-
    # TODO: fix distance (rebalancing) cost
    see the Edit in https://or.stackexchange.com/questions/5040/linearizing-a-program-with-multinomial-logit-in-the-objective
    in my case, because I have no intercept, t will be the lower bound value, i.e. 0. in which case, I can't recover
    v. so we arbitrarily set t > 0.
    @param origin_id:
    @param one_t_ahead_move:
    @param dest_attraction: {zone_id: profit} (it is calculated for each zone)
    @param supply_count:
    @param t_fifteen:
    @return:
    """
    # print(f'solving for zone {origin_id}')
    max_adj_budget = 10
    m_behavioral = gb.Model("Behavioral")
    # get the drivers in zone z
    drivers_zone_z = list(range(supply_count))  # and if its length is zero, i.e., no drivers there?
    if len(drivers_zone_z) <= 10:
        return {}, 0
    # print(len(drivers_zone_z), 'len(drivers_zone,z)')

    w = m_behavioral.addVars(ZONE_IDS, drivers_zone_z, name="weights", vtype=GRB.CONTINUOUS, lb=0.0)
    t = m_behavioral.addVars(drivers_zone_z, name="ts", vtype=GRB.CONTINUOUS, lb=1)  # and this is ?
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
        # print(f"obj value is {m_behavioral.objVal}")
        optimal_ws = m_behavioral.getAttr("x", w)
    else:
        print(f"status was {m_behavioral.status}")
        # m_behavioral.computeIIS()
        # m_behavioral.write("model.ilp")
        return None
    # because I set t = 1, w is equal to v.
    optimal_si = {}
    for dest in ZONE_IDS:
        for d in drivers_zone_z:
            optimal_si[dest, d] = np.log(optimal_ws[dest, d] * dest_attraction[dest] + epsilon) / (
                    dest_attraction[dest] + epsilon)
    return optimal_si, m_behavioral.objVal


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
