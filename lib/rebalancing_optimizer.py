import numpy as np
import pandas as pd
import copy
from lib.Constants import ZONE_IDS, my_dist_class, my_travel_time_class, convert_seconds_to_15_min, FUEL_COST
import gurobipy as gb
from gurobipy import GRB

import logging
import pickle

c_counter = 0


class RebalancingOpt:
    def __init__(self, output_path):
        # self.od_pairs = ((x, y) for x in ZONE_IDS for y in ZONE_IDS)
        self.ODs = gb.tuplelist(
            [(x, y) for x in ZONE_IDS for y in ZONE_IDS]
        )
        # self.dist = dist
        self.ongoing_pickups = []
        self.ongoing_rebalancing = []

        self.rebalancing_cost = - 0.33 / 1000  # per meter
        self.denied_cost = -10
        self.pickup_revenue = 6  # these could be also the avg fare per origin.

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(output_path + 'MPC optimizer.log', mode='w')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def MPC(self, prediction_times, predicted_demand, current_supply, incoming_supply):
        """
        I get horrible results. The experiments in "rebalacing_experiment" suggest that it is because of limited supply.
        Which manifests itself in supply variable. Have to verify that

        @param prediction_times:
        @param predicted_demand:
        @param current_supply:
        @param incoming_supply:
        @return:
        """
        # save the data for experimentation

        source_data = {'prediction_times': prediction_times,
                       'predicted_demand': predicted_demand,
                       'current_supply': current_supply,
                       'incoming_supply': incoming_supply}
        # # save as pickle
        # global c_counter
        # with open(f'data_{c_counter}.pickle', 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # c_counter += 1

        print("Running the Gurobi code")
        reb_cost = gb.tupledict()
        pic_cost = gb.tupledict()
        den_cost = gb.tupledict()
        # TODO Ideally these should be generate within the class definition, but how to pass the prediction times?
        for i in ZONE_IDS:
            for j in ZONE_IDS:
                for t in prediction_times:
                    ds = my_dist_class.return_distance(i, j)
                    reb_cost[(i, j, t)] = self.rebalancing_cost * my_dist_class.return_distance(i,
                                                                                                j)  # ds * FUEL_COST # # distance * fuel
                    pic_cost[(i, j, t)] = self.pickup_revenue
                    den_cost[(i, j, t)] = self.denied_cost

        if 'm' in globals():
            del m
        m = gb.Model("rebalancer")
        rebal = m.addVars(self.ODs, prediction_times, name="rebalancing_flow", vtype=GRB.INTEGER)
        pickup = m.addVars(self.ODs, prediction_times, name="pickup_flow", vtype=GRB.INTEGER)
        denied = m.addVars(self.ODs, prediction_times, name="denied_flow", vtype=GRB.INTEGER)
        # define the objective function
        obj = gb.LinExpr()
        obj.add(rebal.prod(reb_cost))
        obj.add(pickup.prod(pic_cost))
        obj.add(denied.prod(den_cost))

        m.addConstrs(
            (pickup[(i, j, t)] + denied[(i, j, t)] == predicted_demand[(i, j, t)]
             for i, j, t in rebal.keys()), "conservation of pax"
        )

        supply = gb.tupledict()
        for i in ZONE_IDS:
            for idx, t in enumerate(prediction_times):
                if idx == 0:
                    supply[(i, t)] = current_supply[i] + incoming_supply[(i, t)]
                else:
                    supply[(i, t)] = incoming_supply[(i, t)]
        # print("max slack value is ", np.max([v for k, v in supply.items()]))
        # print("min slack value is ", np.min([v for k, v in supply.items()]))
        # construct veh_to_be_available list
        pickup_to_be_avail = {}
        for t_end in prediction_times:
            for zone in ZONE_IDS:
                add_ct = False
                pickup_to_be_avail[(zone, t_end)] = 0  # initialize
                ct = gb.LinExpr()
                for origin, destination, pickup_time in pickup.keys():
                    if (pickup_time + (
                            my_travel_time_class.return_travel_time_15_min_bin(origin, destination)) == t_end) \
                            and (destination == zone):
                        # bingo
                        ct.add(pickup[(origin, destination, pickup_time)])
                        add_ct = True
                #                 ct.add(rebal[(j, i, t)])
                for origin, destination, move_time in rebal.keys():
                    if (move_time + (my_travel_time_class.return_travel_time_15_min_bin(origin, destination)) == t_end) \
                            and (destination == zone):
                        # bingo
                        ct.add(rebal[(origin, destination, move_time)])
                        add_ct = True
                if add_ct:
                    pickup_to_be_avail[(zone, t_end)] = ct

        m.addConstrs(
            (pickup.sum(i, '*', t) + rebal.sum(i, '*', t) - pickup_to_be_avail[(i, t)] == supply[(i, t)]
             for i in ZONE_IDS for t in prediction_times), "conservation of incoming flows")

        # self.logger.info(f"total demand is {sum(predicted_demand.values())}")
        # self.logger.info(f"total supply is {np.sum(supply.values())}")
        # self.logger.info(f"current supply is {sum(current_supply.values())}")
        # self.logger.info(f"incoming supply is {sum(incoming_supply.values())}")

        m.setParam('OutputFlag', 0)  # Also dual_subproblem.params.outputflag = 0
        # print(obj.size())
        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # print(m)
        m.optimize()
        if m.status == 2:
            print(f"obj value is {m.objVal}")
            # self.logger.info(f"obj value is {m.objVal}")

            sol_p = m.getAttr("x", pickup)
            sol_d = m.getAttr("x", denied)
            sol_r = m.getAttr("x", rebal)
            # print("total non-empty assignment solutions: ", len([v for k, v in sol_p.items() if v > 0]))
            # print("total non-empty denied solution: ", len([v for k, v in sol_d.items() if v > 0]))
            # print("total non-empty rebal solution: ", len([v for k, v in sol_r.items() if v > 0]))
            # print("total assignment revenue: ", np.sum([v * self.pickup_revenue for k, v in sol_p.items() if v > 0]))
            # print("total rebal loss: ", np.sum([v * self.rebalancing_cost for k, v in sol_r.items() if v > 0]))
            # print("total denied loss: ", np.sum([v * self.denied_cost for k, v in sol_d.items() if v > 0]))
            #
            # self.logger.info(f"total assignment revenue: {np.sum([v * self.pickup_revenue for k, v in sol_p.items() if v > 0])}")
            # self.logger.info(f"total rebal loss:  {np.sum([v * self.rebalancing_cost for k, v in sol_r.items() if v > 0])}")
            # self.logger.info(f"total denied loss:  {np.sum([v * self.denied_cost for k, v in sol_d.items() if v > 0])}")
            # self.logger.info("*" * 10)
            return sol_p, sol_d, sol_r, m.objVal, source_data
        else:
            print("Gurobi's status is NOT 2, instead is ", m.status)
            return None, None, None, None, None
