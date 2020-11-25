import pickle
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

from lib.Constants import (
    ZONE_IDS,
    # DEMAND_SOURCE,
    INT_ASSIGN,
    FLEET_SIZE,
    PRO_SHARE,
    SURGE_MULTIPLIER,
    BONUS,
    PERCENT_FALSE_DEMAND,
    FUEL_COST
)
# from lib.Constants import PERCE_KNOW, CONST_FARE, AV_SHARE
from lib.Operator import Operator
from lib.Vehicles import Veh
from lib.Zones import Zone
from lib.ProfessionalDriver import ProfessionalDriver
from lib.NaiveDriver import NaiveDriver
from lib.InexperiencedDriver import InexperiencedDriver



class Model:
    """ 
    Encompassing object for simulation. Creates all the zones, initializes daily demand.
    Zone objects then create Requests from the matrix row by row.
    """

    def __init__(
            self,
            data_obj,
            configs,
            beta
    ):
        """
        @param zone_ids:
        @param daily_demand:
        @param warmup_time_hour:
        @param analysis_time_hour :
        @param fleet_size (int):
        @param pro_share (float): percent of fleet with pro drivers
        @param surge_multiplier (float)
        @param bonus (float): dollar bonus awarded to drivers
        @param percent_false_demand (float)
        @param percentage_know_fare (float): percent of drivers that know the fare
        @param av_share (float): percent of fleet that's AVs
        @param beta (float): TODO: what's this?

        @rtype: Model object
        """
        print("calling init function of Model")
        seed1 = 1100  # np.random.randint(0,1000000)
        self.rs1 = np.random.RandomState(seed1)  # random state
        self.data_obj = data_obj
        self.zone_ids = ZONE_IDS
        self.zones = []
        self.daily_OD_demand = data_obj.DEMAND_SOURCE  # data_obj.BINNED_DEMAND
        self.daily_pickup_demand = data_obj.BINNED_DEMAND
        self._create_zones()
        print("instantiated zones")

        self.set_analysis_time(data_obj.WARMUP_TIME_HOUR * 3600)
        print("generated demand")

        self.WARMUP_TIME_SECONDS = data_obj.WARMUP_TIME_HOUR / 3600
        self.WARMUP_PHASE = True
        self.ANALYSIS_TIME_HOUR = data_obj.ANALYSIS_TIME_HOUR
        self.ANALYSIS_TIME_SECONDS = data_obj.ANALYSIS_TIME_HOUR * 3600
        self.PRO_SHARE = data_obj.PRO_SHARE
        self.AV_SHARE = data_obj.AV_SHARE
        self.SURGE_MULTIPLIER = data_obj.SURGE_MULTIPLIER
        self.FLEET_SIZE = data_obj.FLEET_SIZE  # FLEET_SIZE
        self.BONUS_POLICY = data_obj.BONUS_POLICY
        self.budget = data_obj.BUDGET
        _s = str(self.SURGE_MULTIPLIER).split(".")
        _s = "".join(_s)

        # get expected number of drivers per zone
        if self.PRO_SHARE > 0:
            m = pickle.load(
                open(
                    "Outputs/model for fleet size {f} surge {s}fdemand 0.0perc_k 0pro_s 0 repl0.p".format(
                        f=self.FLEET_SIZE, s=_s
                    ),
                    "rb",
                )
            )
            report = m.get_service_rate_per_zone()
            report = self.__calc_s(report)
        else:
            report = None

        # Create operator object
        self.operator = Operator(report, bonus=data_obj.BONUS,
                                 surge_multiplier=data_obj.SURGE_MULTIPLIER,
                                 scenario = configs["scenario"],
                                 bonus_policy=self.BONUS_POLICY, budget=self.budget,
                                 which_day_numerical=self.data_obj.day_of_run)

        # Pro drivers know fares, so perc_know must take this into account
        self.fleet_pro_size = int(self.PRO_SHARE * self.FLEET_SIZE)
        self.percent_false_demand = data_obj.PERCENT_FALSE_DEMAND
        self.fleet_deceived_size = int(self.percent_false_demand * self.FLEET_SIZE)
        self.percentage_know_fare = np.minimum(self.PRO_SHARE + data_obj.PERCE_KNOW, 1)
        self.fleet_know_fare = int(data_obj.PERCE_KNOW * self.FLEET_SIZE)
        self.fleet_DONT_know_fare = int(
            (1 - self.percentage_know_fare) * self.FLEET_SIZE
        )
        self.fleet_AV = int(self.AV_SHARE * self.FLEET_SIZE)
        print("fleet AV", self.fleet_AV)
        self._create_vehicles(beta)

        print(Counter(v.ozone for v in self.vehicles))

        # TODO: debug, delete this
        self.targets = []
        self.performance_results = {}

    def __calc_s(self, df):
        """
        Computes average number of drivers, clips values between 0 and 1.
        Calculates probability of finding a match.

        @param df (pd dataframe):
        @return: pd dataframe with zone ids and corresponding probabilities
        """
        df.loc[:, "avg_num_drivers"] = df.idle + df.incoming
        s = df.total / df.avg_num_drivers  # df.total := amount of demand
        s[s > 1] = 1
        s[np.isnan(s)] = 0.0001
        s[np.isinf(s)] = 1

        df.loc[:, "prob_of_s"] = s
        df = df[["zone_id", "prob_of_s"]]
        return df

    def _create_zones(self):
        """
        Make the zones, and initiates their demand matrix.
        Updates the self.zones attribute in-place.
        """
        for z_id in self.zone_ids:
            Z = Zone(z_id,  rs=self.rs1)
            Z.read_daily_demand(self.daily_OD_demand) # , self.daily_pickup_demand
            self.zones.append(Z)

    def _create_vehicles(self, beta=1):
        """
        Creates list of Vehicles and assigns random sets of them to be
        pro, deceived, fare-aware, and AV.

        @param beta: TODO: what's this?
        @return: None; modifies self.vehicles in place.
        """
        # self.vehicles = [
        #     Veh(self.rs1, self.operator, beta) for i in range(self.FLEET_SIZE)
        # ]
        self.vehicles = [
            ProfessionalDriver(self.rs1, self.operator, beta) for _ in range(self.FLEET_SIZE)
        ]
        self.vehicles.extend(
            [NaiveDriver(self.rs1, self.operator, beta) for _ in range(100)] )
        self.vehicles.extend(
            [InexperiencedDriver(self.rs1, self.operator, beta) for _ in range(300)])
        # # None of the below ones are mutually exclusive; a vehicle might be pro and don't know fare (which is wrong)
        # if self.fleet_pro_size > 0:
        #     print("fleet pro size", self.fleet_pro_size)
        #     # vs = random.choices(self.vehicles, k=self.fleet_pro_size)
        #     vs = np.random.choice(self.vehicles, self.fleet_pro_size, replace=False)
        #     for v in vs:
        #         v.professional = True
        #         v.know_fare = True
        #     remaining_veh = list(set(self.vehicles) - set(vs))
        #
        # if self.fleet_deceived_size > 0:
        #     if not ("remaining_veh" in locals()):
        #         remaining_veh = self.vehicles
        #     # vs = random.choices(remaining_veh, k=self.fleet_deceived_size)
        #     vs = np.random.choice(
        #         remaining_veh, self.fleet_deceived_size, replace=False
        #     )
        #     for v in vs:
        #         v.true_demand = False
        #
        #     remaining_veh = list(set(remaining_veh) - set(vs))
        #
        # if self.fleet_know_fare > 0:
        #     print("fleet know fare", self.fleet_know_fare)
        #     if not ("remaining_veh" in locals()):
        #         remaining_veh = self.vehicles
        #
        #     vs = np.random.choice(remaining_veh, self.fleet_know_fare, replace=False)
        #     for v in vs:
        #         v.know_fare = True
        #
        #     remaining_veh = list(set(remaining_veh) - set(vs))
        #
        # if self.fleet_AV > 0:
        #     if not ("remaining_veh" in locals()):
        #         remaining_veh = self.vehicles
        #     # vs = random.choices(remaining_veh, k=self.AV_SHARE)
        #     self.av_vehs = np.random.choice(remaining_veh, self.fleet_AV, replace=False)
        #
        #     for v in self.av_vehs:
        #         v.is_AV = True
        #         # v.RL_engine = self.RL_engine
        #
        #     remaining_veh = list(set(remaining_veh) - set(self.av_vehs))

    def set_analysis_time(self, t):
        """
        Sets the demand rate per zone given an hour of day.
        @param t: seconds
        @return: None; sets values in-place.
        """
        for z in self.zones:
            z.set_demand_rate_per_t(t)

    def generate_zonal_demand(self, t):
        """
        First checks to see if should change the demand rates,
        then generates demand for each zone.
        this is called every ASSIGN seconds. Need not generate demand each time (?)
        @param t: time of day (seconds)
        @return: None; sets values in place
        """
        t_hour = (t / 3600)
        t_15_min = np.floor(t / 900)

        # this needs to be double checked
        if self.WARMUP_PHASE and t >= self.ANALYSIS_TIME_SECONDS:
            # print("changing demand rate from warm up to analysis")
            # self.set_analysis_time(self.ANALYSIS_TIME_HOUR)
            self.WARMUP_PHASE = False
            # update drivers infor/expectations

        # generate demand only every 15 minutes
        if t % 900 == 0:
            for z in self.zones:
                z.generate_requests_to_time(t)  # t is seconds

    def move_fleet(self, t, warmup_phase, action):
        """
        Applies action to each vehicle.
        An improvement could be to filter out veh based on their status, then run those that have to move in parallel

        @param t: time of day
        @param warmup_phase (bool): whether we are in the warmup phase
        @param action: (unused)
        @return: None
        """

        for veh in self.vehicles:
                    _ = veh.act(t, self.zones, warmup_phase)

            # if not veh.is_AV:  # AV is already being moved by the engine
            #     _ = veh.act(t, self.zones, WARMUP_PHASE)
            # if veh.is_AV:
            #     # if veh.should_move(): this causes errors, since move is not just moving, but also rebalancing, waiting, etc.
            #     veh.act(t, self.zones, WARMUP_PHASE, action)

    def assign_zone_veh(self, t, warmup_phase, penalty, operator):
        """
        Assigns zone to each vehicle.this_t_demand

        @param t: time of day
        @param warmup_phase (bool): whether we are in the warmup phase
        @param penalty (float): penalty amount
        @return: None
        """
        for z in self.zones:
            z.assign(self.zones, t, warmup_phase, penalty, operator)

    def get_service_rate_per_zone(self):
        """
        @return (df): observations for each zone, including demand, number served, total, etc.
        """
        performance_results = {}
        for z in self.zones:
            print( "req lengths",len(z.reqs))
            w = len(z.demand)
            served = len(z.served_demand)
            los = served / (served + w) if (served + w) > 0 else 0

            r = {
                "zone_id": z.id,
                "w": w,
                "served": served,
                "total": w + served,
                "LOS": los,
                "idle": len(z.idle_vehicles),
                "incoming": len(z.incoming_vehicles),
                "times_surged": z.num_surge,
            }
            performance_results[z.id] = r

        performance_results = pd.DataFrame.from_dict(
            performance_results, orient="index"
        )
        performance_results = performance_results.sort_values("LOS", ascending=False)

        return performance_results

    # @profile
    def dispatch_at_time(self, t, penalty=-10, action=None):
        """
        Dispatches the AMoD system: move vehicles, generate requests, assign, reoptimize and rebalance.
        @param t: time of day (seconds)
        @param penalty (float)
        @param action
        @return: None
        """
        self.generate_zonal_demand(t)
        self.operator.update_zonal_info(t)
        self.operator.update_zone_policy(t, self.zones, self.WARMUP_PHASE)
        self.assign_zone_veh(t, self.WARMUP_PHASE, penalty, self.operator)
        self.move_fleet(t, self.WARMUP_PHASE, action)

        if t % 500 == 0:
            print("time is {time}".format(time=t))

    @lru_cache(maxsize=None)
    def _get_demand_per_zone(self, t):
        """
        Gets demand per zone.

        @param t: time of day
        @return (df): Dataframe with zone_id as index and demand column
        """
        a = {z.id: len(z.demand) for z in self.zones}
        demand_df = pd.DataFrame.from_dict(a, orient="index", columns=["demand"])
        # normalize it 
        demand_df["demand"] = demand_df["demand"] / (demand_df["demand"].max() + 1)
        # print ("normalized demand ", demand_df)
        return demand_df

    @lru_cache(maxsize=None)
    def _get_supply_per_zone(self, t):
        """
        Gets supply per zone.

        @param t: time of day
        @returns (df): Dataframe with zone_id as index and supply column
        """
        b = {z.id: len(z.idle_vehicles) for z in self.zones}
        supply_df = pd.DataFrame.from_dict(b, orient="index", columns=["supply"])

        # normalize it 
        supply_df["supply"] = supply_df["supply"] / (supply_df["supply"].max() + 1)
        return supply_df

    @lru_cache(maxsize=None)
    def _calc_rebl_cost(self, ozone, max_cost=7):
        """
        This should be based on the value of time and time it took to get to the destination

        @param ozone (int): original pickup location ID
        @param max_cost (float): 7 is just a preliminary attempt at normalizing the costs

        @return (df): distance to all zones with costs
        """

        dist = Veh._get_dist_to_all_zones(ozone)[["DOLocationID", "trip_distance_meter"]]
        # dist = veh._get_dist_to_all_zones(veh.ozone)[["DOLocationID", "trip_distance_meter"]]
        # this is the costliest operation! 
        dist["costs"] = ((dist.trip_distance_meter * self.data_obj.FUEL_COST).apply(
            lambda x: np.around(x, 1))) / max_cost
        # dist["costs"] = dist["costs"].apply(lambda x: np.around(x, 1))
        # dist["costs"] /= max_cost

        return dist

    @lru_cache(maxsize=None)
    def _get_both_supply_and_demand_per_zone(self, t):
        """
        @param t: time of day
        @return (df): merged dataframe of demand and supply for all zones
        """
        demand_df = self._get_demand_per_zone(t)
        supply_df = self._get_supply_per_zone(t)
        return pd.merge(demand_df, supply_df, left_index=True, right_index=True)

    @lru_cache(maxsize=None)
    def _get_demand_supply_costs_df(self, ozone, t):
        """
        @param ozone (int): original pickup location id
        @param t: time of day
        @return: df with demand, supply, and costs for all zones
        """
        dist = self._calc_rebl_cost(ozone)
        d_s = self._get_both_supply_and_demand_per_zone(t)
        # d_s_c = pd.merge(d_s, dist, left_index=True, right_on="DOLocationID")[["demand", "supply", "costs"]].values
        # d_s_c = d_s_c[["demand", "supply", "costs"]]
        # d_s_c = d_s_c.values
        # return pd.merge(d_s, dist.set_index('DOLocationID'), left_index=True, right_index=True)[["demand", "supply", "costs"]].values
        return pd.merge(d_s, dist, left_index=True, right_on="DOLocationID")[["demand", "supply", "costs"]].values

    def get_state(self, veh, t):
        """
        Gets the model state.

        @param veh: an object
        @return : matrix of size (#zones * 3), where each row is  (u_i, v_i, c_ij)
        """
        return self._get_demand_supply_costs_df(veh.ozone, t)


if __name__ == "__main__":
    pass
