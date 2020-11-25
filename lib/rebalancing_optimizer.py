import numpy as np
from gurobipy import *
import pandas as pd
import copy
from lib.Constants import ZONE_IDS, my_dist_class


class RebalancingOpt:
    def __init__(self):
        od_dict = {(x, y): 0 for x in ZONE_IDS for y in ZONE_IDS}
        ODs = tupledict(od).keys()
        # self.dist = dist
        self.ongoing_pickups = []
        self.ongoing_rebalancing = []

        self.rebalancing_cost = 5
        self.denied_cost = 10
        self.pickup_revenue = 10 # how to do this?



    def MPC(T_forward, slack, demand):
        """
        standard foirmat shtinx
        FUNCTION:
            finds optimal rebalancing flows among zones using predicted demand

        INPUT:
            T_Forward : a list of time periods
            zones: a list of zones
            slack : a dictionary with keys (i,t) for i in zones and t in T_forward
            reb_cost, pic_cost, den_cost: dictionaries with keys (i,j,t) for ij in OD and t in T representing rebalancing, pick up and denial costs repsectively
            demand :
        OUTPUTS:
            optimal rebalancing flows for (i,j) in ODs and for t in T_forward
        """
        pic_cost_sub, den_cost_sub, reb_cost_sub = {}, {}, {}
        for i in ZONE_IDS:
            for j in ZONE_IDS:
                for t in T_forward:
                    reb_cost_sub[i, j, t] = reb_cost[i, j, t]
                    pic_cost_sub[i, j, t] = pic_cost[i, j, t]
                    den_cost_sub[i, j, t] = den_cost[i, j, t]

        m = Model("rebalancer")
        # isnt it better to model them as continuous variables?

        reb = m.addVars(ODs, T_forward, name="reb_flow", vtype=GRB.INTEGER)
        pic = m.addVars(ODs, T_forward, name="pic_flow", vtype=GRB.INTEGER)
        den = m.addVars(ODs, T_forward, name="den_flow", vtype=GRB.INTEGER)

        m.addConstrs((pic[(i, j, t)] + den[(i, j, t)] == demand[(i, j, t)]
                      for (i, j) in ODs for t in T_forward), "conservation of pax")

        for i in ZONE_IDS:
            for t in T_forward:
                temp = LinExpr()
                for j in ZONE_IDS:

                    temp.add(pic[i, j, t], 1.0)
                    temp.add(reb[i, j, t], 1.0)
                    # these variable should be received as input. It should not be the same "pic" and "reb"
                    if (t - my_dist_class.return_distance(j, i)) >= T_forward[0]:
                        temp.add(pic[j, i, (t - my_dist_class.return_distance(j, i))], -1.0)
                        temp.add(reb[j, i, (t - my_dist_class.return_distance(j, i))], -1.0)

                print(temp.size())
                m.addConstr((temp == slack[(i, t)]), "conservation of veh")

        # slack constraints
        slack = {}
        for i in ZONE_IDS:
            for t in T_forward:
                if t == 0:
                    slack[i, t] = int(np.random.uniform(1, 5))
                else:
                    slack[i,t] = 0

        obj = LinExpr()
        obj.add(reb.prod(reb_cost_sub))
        obj.add(pic.prod(pic_cost_sub))
        obj.add(den.prod(den_cost_sub))
        print("#####################")
        print(obj.size())

        m.setObjective(obj, GRB.MINIMIZE)
        m.update()
        print(m)
        m.optimize()

        if m.status == 2:
            sol_p = m.getAttr("x", pic)
            sol_d = m.getAttr("x", den)
            sol_r = m.getAttr("x", reb)
            return sol_p, sol_d, sol_r
        else:
            print("Gurobi's status is NOT 2, instead is ", m.status)
            return 0
