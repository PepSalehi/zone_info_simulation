import pickle
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import time
from lib.Constants import (
    ZONE_IDS,
    MyTravelTime,
    convert_seconds_to_15_min,
    # DEMAND_SOURCE,
    INT_ASSIGN,
    FLEET_SIZE,
    PRO_SHARE,
    SURGE_MULTIPLIER,
    BONUS,
    PERCENT_FALSE_DEMAND,
    FUEL_COST,
    PHI)
# from lib.Constants import PERCE_KNOW, CONST_FARE, AV_SHARE
from lib.Operator import Operator
from lib.Vehicles import Veh, DriverType, _convect_time_to_peak_string
from lib.Zones import Zone
from lib.ProfessionalDriver import ProfessionalDriver
from lib.NaiveDriver import NaiveDriver
from lib.AvDriver import AvDriver
from lib.InexperiencedDriver import InexperiencedDriver
from lib.rebalancing_optimizer import RebalancingOpt
from lib.behavioral_optimizer import solve_for_one_zone

_get_dist_to_all_zones = Veh._get_dist_to_all_zones
_get_dist_to_all_zones_as_a_dict = Veh._get_dist_to_all_zones_as_a_dict
import multiprocessing as mp

import gurobipy as gb
from gurobipy import GRB


def _compute_expected_attractiveness(fare_df, dist, unit_rebalancing_cost=FUEL_COST):
    """
    returns {z_id : Q * U }
    @param fare_df:
    @param dist:
    @return:
    """
    expected_income = (1 - PHI) * fare_df["avg_fare"].values * fare_df["surge"].values + \
                      fare_df["bonus"].values
    # 5) compute the expected cost
    expected_cost = (
            dist["trip_distance_meter"].values * unit_rebalancing_cost)
    # 6) compute the expected profit
    difference = expected_income - expected_cost
    # 7) weight by total demand
    difference = difference * fare_df["total_pickup"].values
    # difference = difference.to_numpy()
    pulocations = fare_df["PULocationID"].values
    # expected_revenue = {z_id: difference.iloc[i] for i, z_id in enumerate(fare_df["PULocationID"].values)}
    expected_revenue = {z_id: difference[i] for i, z_id in enumerate(pulocations)}

    return expected_revenue


class Model:
    """ 
    Encompassing object for simulation. Creates all the zones, initializes daily demand.
    Zone objects then create Requests from the matrix row by row.
    """

    def __init__(
            self,
            data_obj,
            configs,
            beta,
            output_path
    ):
        """
        @param zone_ids:
        @param daily_OD_demand:
        @param warmup_time_hour:
        @param analysis_time_hour :
        @param fleet_size (int):
        @param pro_share (float): percent of fleet with pro drivers
        @param surge_multiplier (float)
        @param bonus (float): dollar bonus awarded to drivers
        @param percent_false_demand (float)
        @param percentage_know_fare (float): percent of drivers that know the fare
        @param av_share (float): percent of fleet that's AVs
        @param beta (float):

        @rtype: Model object
        """
        # print("Starting day ", self.day_number)
        print("calling init function of Model")
        seed1 = 1100  # np.random.randint(0,1000000)
        self.rs1 = np.random.RandomState(seed1)  # random state
        self.data_obj = data_obj
        self.zone_ids = ZONE_IDS
        self.zones = []
        self.daily_OD_demand = data_obj.DEMAND_SOURCE  # data_obj.BINNED_DEMAND
        # self.daily_pickup_demand = data_obj.BINNED_DEMAND
        self.output_path = output_path
        #

        self.set_analysis_time(data_obj.WARMUP_TIME_HOUR * 3600)
        # print("generated demand")

        self.rebalancing_engine = RebalancingOpt(self.output_path)

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

        self.sol_p = self.sol_r = self.sol_d = None

        # Create operator object
        self.operator = Operator(bonus=data_obj.BONUS,
                                 surge_multiplier=data_obj.SURGE_MULTIPLIER,
                                 scenario=configs["scenario"],
                                 bonus_policy=self.BONUS_POLICY,
                                 budget=self.budget,
                                 which_day_numerical=self.data_obj.day_of_run,
                                 which_month = self.data_obj.MONTH,
                                 do_behavioral_opt=data_obj.do_behavioral_opt,
                                 do_surge_pricing=data_obj.do_surge_pricing,
                                 output_path=self.output_path)

        self._create_zones(self.output_path)
        print("instantiated zones")
        self._create_vehicles(data_obj, self.output_path, beta)

        print("initial distribution of drivers", Counter(v.ozone for v in self.vehicles))

        # TODO: debug, delete this
        self.targets = []
        self.performance_results = None

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

    def _create_zones(self, output_path):
        """
        Make the zones, and initiates their demand matrix.
        Updates the self.zones attribute in-place.
        """
        for z_id in self.zone_ids:
            Z = Zone(z_id, self.operator, output_path , rs=self.rs1)
            Z.read_daily_demand(self.daily_OD_demand)  # , self.daily_pickup_demand
            self.zones.append(Z)

    def _create_vehicles(self, data_obj, output_path, beta=1):
        """
        Creates list of Vehicles and assigns random sets of them to be
        pro, deceived, fare-aware, and AV.
        https://www1.nyc.gov/site/tlc/businesses/yellow-cab.page#:~:text=Taxicabs%20are%20the%20only%20vehicles,a%20medallion%20affixed%20to%20it.
        There are a total of 13587 Yellow Cab taxis in NYC
        @param beta: TODO: what's this?
        @return: None; modifies self.vehicles in place.
        """
        # self.vehicles = [
        #     Veh(self.rs1, self.operator, beta) for i in range(self.FLEET_SIZE)
        # ]
        # self.vehicles = [
        #     AvDriver(self.rs1, self.operator, beta) for _ in range(2500)
        # ]

        self.vehicles = [
            NaiveDriver(self.rs1, self.operator, day_of_run=self.data_obj.day_of_run, output_path=output_path) for _ in
            range(data_obj.NAIVE_FLEET_SIZE)
        ]
        self.vehicles.extend(
            [ProfessionalDriver(self.rs1, self.operator, day_of_run=self.data_obj.day_of_run, output_path=output_path) for _ in
             range(data_obj.PRO_FLEET_SIZE)])

        self.vehicles.extend( # need to add day of run here
            [AvDriver(self.rs1, self.operator, self.data_obj.day_of_run, output_path, beta) for _ in range(data_obj.AV_FLEET_SIZE)])

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
        t_15_min = convert_seconds_to_15_min(t)

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

    def get_operators_revenue(self):
        return self.operator.revenues

    def get_service_rate_per_zone(self, day, month):
        """
        @return (df): observations for each zone, including demand, number served, total, etc.
        """
        performance_results = {}
        for z in self.zones:
            print("req lengths", len(z.reqs))
            w = len(z.demand)
            served = len(z.served_demand)
            denied = len(z.denied_requests)
            los = served / (served + denied) if (served + denied) > 0 else 0

            r = {
                "month": month,
                "day": day,
                "zone_id": z.id,
                "waiting": w,
                "denied": denied,
                "served": served,
                "total": denied + served,
                "LOS": los,
                "idle": len(z.idle_vehicles),
                "incoming": len(z.incoming_vehicles),
                "times_surged": z.num_surge,
            }
            performance_results[z.id] = r

        performance_results = pd.DataFrame.from_dict(
            performance_results, orient="index"
        )
        performance_results = performance_results.sort_values("total", ascending=False)
        performance_results = pd.DataFrame(data=performance_results)
        if self.performance_results is None:
            self.performance_results = performance_results
        else:
            self.performance_results = pd.concat([self.performance_results, performance_results],
                                                 ignore_index=True)
        # return performance_results

    def report_final_performance(self):
        return self.performance_results

    # @profile
    def dispatch_at_time(self, t, day_idx, penalty=-10, action=None):
        """
        Dispatches the AMoD system: move vehicles, generate requests, assign, reoptimize and rebalance.
        @param day_idx:
        @param t: time of day (seconds)
        @param penalty (float)
        @param action
        @return: None
        """
        self.generate_zonal_demand(t)
        self.operator.update_zonal_info(t)
        if self.operator.do_behavioral_opt:
            self.optimize_rebalancing_flows(t)
        self.update_pax_status(t)
        if self.operator.do_surge_pricing:
            self.operator.update_zone_policy(t, self.zones, self.WARMUP_PHASE)
        self.assign_zone_veh(t, self.WARMUP_PHASE, penalty, self.operator)
        if self.data_obj.AV_FLEET_SIZE > 0:
            self.compute_driver_instructions(t, self.zones)  # set_action of the AV driver
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
        # I shouldn't have cast it as df. dicts are much more efficient
        # normalize it 
        supply_df["supply"] = supply_df["supply"] / (supply_df["supply"].max() + 1)
        return supply_df

    def optimize_information_design(self, sol_p, sol_d, sol_r, current_supply, incoming_supply):
        """

        @param incoming_supply:
        @param current_supply:
        @param sol_p:
        @param sol_d:
        @param sol_r:
        @return:
        """
        print("inside behavioral optimization unit")
        # Only consider the next time step
        t_fifteen = np.min([t for i, j, t in sol_p.keys()])
        one_t_ahead_sol_p = {(i, j, t): sol_p[(i, j, t)] for i, j, t in sol_p.keys() if t == t_fifteen}
        one_t_ahead_sol_r = {(i, j, t): sol_r[(i, j, t)] for i, j, t in sol_r.keys() if t == t_fifteen}
        # total assignment + rebalancing
        # one_t_ahead_move = {}
        # for k, v in one_t_ahead_sol_p.items():
        #     one_t_ahead_move[k] = v + one_t_ahead_sol_r[k]
        # Maybe it should be just the rebalancing ones
        one_t_ahead_move = one_t_ahead_sol_r
        # get attractiveness of each destination, per origin
        dest_attraction = defaultdict(lambda: defaultdict)  # origin : {dest: att}, these are C_k^d
        # get the expected number of drivers in the origin zone (i.e., current + incoming)
        supply = gb.tupledict()
        demand_df = self.operator.get_zonal_info_for_general()
        for z_id in self.zone_ids:
            dist = _get_dist_to_all_zones(z_id)
            # get demand
            assert dist.shape[0] == demand_df.shape[0]
            # {z.id : utility}
            expected_attractiveness = _compute_expected_attractiveness(demand_df, dist, unit_rebalancing_cost=FUEL_COST)
            dest_attraction[z_id] = expected_attractiveness
            supply[(z_id, t_fifteen)] = current_supply[z_id] + incoming_supply[(z_id, t_fifteen)]
        optimal_si = {}
        start_t = time.time()
        # can I parallelize this?
        for origin_id in self.zone_ids:
            # some optimal si might be empty, maybe because there were no drivers there to begin with
            # print('calling LP model')
            optimal_si[origin_id] = solve_for_one_zone(origin_id, one_t_ahead_move, dest_attraction[origin_id],
                                                       supply[(origin_id, t_fifteen)], t_fifteen)

        print(f"it took {(time.time() - start_t) / 60} minutes to optimize all zones")

        # pass the optimal si to the operator
        self.operator.get_optimal_si(optimal_si)

        # manipulate demand information, per driver (how??)
        # each zone should have a counter, keeping track of how many times information has been
        # requested for that time interval. specifically, no need to ask for driver id or anything
        # every time a new information request comes it, if there are still variables left in the optimal_si,
        # give the manipulated information. otherwise, give the raw one

    def optimize_rebalancing_flows(self, t):
        """
        @param t: seconds
        @return:
        """
        if (t >= 6 * 3600) and (t % 900 == 0):
            print("Finding the optimal rebalancing")
            t_fifteen = convert_seconds_to_15_min(t)
            prediction_horizon = 2  # 2 steps (i.e. 15 mins) ahead
            prediction_times = list(range(t_fifteen, t_fifteen + prediction_horizon + 1))
            # get the current supply (i.e. idle vehicles) per zone
            # at the very start,
            current_supply = {z.id: len(z.idle_vehicles) for z in self.zones}  # is able to handle zero, right?
            # get the incoming flows, i.e., the currently busy drivers that are to become available
            # this time should be converted to 15-min index
            incoming_supply = {(z, t): 0 for z in self.zone_ids for t in prediction_times}
            for z in self.zones:
                for veh in z.incoming_vehicles:
                    tba = convert_seconds_to_15_min(veh.time_to_be_available)
                    tba = int(tba + t_fifteen)
                    if tba in prediction_times:
                        try:
                            incoming_supply[(z.id, tba)] += 1
                        except KeyError:
                            print("key ", (z.id, tba))
                            print("t_fifteen ", t_fifteen)
                            print("prediction_times ", prediction_times)
                            raise KeyError
            # get the current unserved pax, OD
            # {(origin, dest, t_of_analysis not request} : # of reqs
            current_demand = {(o, d, t): 0 for o in self.zone_ids for d in self.zone_ids for t in prediction_times}
            for z in self.zones:
                for req in z.demand:
                    current_demand[(req.ozone, req.dzone, t_fifteen)] += 1

            df1 = self.daily_OD_demand[self.daily_OD_demand.time_of_day_index_15m.isin(prediction_times)][
                ["PULocationID", "DOLocationID", "time_of_day_index_15m"]]
            d = df1.groupby(['PULocationID', 'DOLocationID', 'time_of_day_index_15m']).size().reset_index(
                name='demand').sort_values(by='demand')
            predicted_demand = {(row.PULocationID, row.DOLocationID, row.time_of_day_index_15m): row.demand for row in
                                d.itertuples()}
            # combine current and predicted demand
            for key, value in current_demand.items():
                if key in predicted_demand.keys():
                    predicted_demand[key] += value
                else:
                    predicted_demand[key] = value

            sol_p, sol_d, sol_r = self.rebalancing_engine.MPC(prediction_times, predicted_demand, current_supply,
                                                              incoming_supply)
            self.sol_p = sol_p
            self.sol_r = sol_r
            self.sol_d = sol_d
            self.optimize_information_design(sol_p, sol_d, sol_r, current_supply, incoming_supply)

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

    def reset_after_one_day_of_operation(self, stop_month, stop_day):
        """
        -1. change the data_obj
        0. need to change the code, so that every function that uses caching takes day index as well.
        Otherwise, it won't update to using a new day
        1. Removes demand from zones, drivers.
        2. Sets drivers' locations randomly
        3. Proceeds to the next day
        4. if entering test days, starts recording
        @return:
        """
        status = self.data_obj.start_the_next_day()
        if status is None:
            # already simulated all days
            return
        if stop_month == self.data_obj.MONTH and stop_day == self.data_obj.day_of_run:
            print('stopped after reaching {} days of month {}'.format(stop_day, stop_month))
            return
        print("reset_after_one_day_of_operation")

        self.daily_OD_demand = self.data_obj.DEMAND_SOURCE
        # removes caches, except for rebal cost function, which is always fixed
        self._get_supply_per_zone.cache_clear()
        self._get_demand_per_zone.cache_clear()
        self._get_both_supply_and_demand_per_zone.cache_clear()
        self._get_demand_supply_costs_df.cache_clear()
        self._get_both_supply_and_demand_per_zone.cache_clear()
        # this is the demand file operator uses to inform zones
        # self.operator.demand_fare_stats_of_the_day = pd.read_csv(
        #     "./Data/Daily_stats/stats_for_day_{}.csv".format(self.data_obj.day_of_run)
        # )
        if self.data_obj.MONTH != self.operator.month:
            print('data month is ', self.data_obj.MONTH)
            print('operator month is ', self.operator.month)
            print('switching')
            self.operator.month = self.data_obj.MONTH
            self.operator.demand_fare_stats_of_the_month = pd.read_csv('./Data/stats_for_{}_18.csv'.format(self.operator.month))
            self.operator.demand_fare_stats_of_the_day = self.operator.demand_fare_stats_of_the_month.query(
                'Day=={}'.format(self.data_obj.day_of_run))
        else:
            self.operator.demand_fare_stats_of_the_day = self.operator.demand_fare_stats_of_the_month.query(
                'Day=={}'.format(self.data_obj.day_of_run))

        vs = self.operator.demand_fare_stats_of_the_day.time_of_day_index_15m.values * 15 * 60
        vs = np.vectorize(_convect_time_to_peak_string)(vs)
        self.operator.demand_fare_stats_of_the_day["time_of_day_label"] = vs # this throws the pandas warning
        ports = pd.get_dummies(self.operator.demand_fare_stats_of_the_day.time_of_day_label)
        self.operator.demand_fare_stats_of_the_day = self.operator.demand_fare_stats_of_the_day.join(ports)
        #TODO: self.daily_OD_demand is wrong, and in addition is not updated
        for v in self.vehicles:
            v.reset(self.data_obj.day_of_run, self.data_obj.MONTH)
        for z in self.zones:
            z.reset(self.daily_OD_demand, self.data_obj.WARMUP_TIME_HOUR * 3600)

        self.operator.revenues = []

    def compute_driver_instructions(self, t, zones):
        """
        process sol_p and sol_r, pass it to each zone
        @param t:
        @param zones:
        @return:
        """
        t_fifteen = np.min([t for i, j, t in self.sol_p.keys()])
        if t == 25200:
            # initialization step
            for veh in self.vehicles:
                if veh.driver_type == DriverType.AV:
                    veh.rebalance(zones, veh.ozone)
        # this should be a memoized function
        for zone in zones:
            one_t_ahead_sol_p = {(i, j, t): self.sol_p[(i, j, t)] for i, j, t in self.sol_p.keys() if (t == t_fifteen
                                                                                                       and i == zone.id)}
            one_t_ahead_sol_r = {(i, j, t): self.sol_r[(i, j, t)] for i, j, t in self.sol_r.keys() if (t == t_fifteen
                                                                                                       and i == zone.id)}
            one_t_ahead_sol_d = {(i, j, t): self.sol_d[(i, j, t)] for i, j, t in self.sol_r.keys() if (t == t_fifteen
                                                                                                       and i == zone.id)}
            # what happens when it's empty???
            # print(f"instruct drivers in zone {zone.id}")
            zone.instruct_drivers(t, zones, one_t_ahead_sol_p, one_t_ahead_sol_r, one_t_ahead_sol_d)

    def update_pax_status(self, t_seconds):
        """
        check if pax have been waiting too long
        @param t_seconds:
        @return:
        """
        for zone in self.zones:
            zone.check_pax_denied(t_seconds)

    def get_drivers_earnings_for_one_day(self, d_idx, month):
        for veh in self.vehicles:
            veh.bookkeep_one_days_earnings(d_idx, month)

    def get_operators_earnings_for_one_day(self, d_idx, month):
        self.operator.bookkeep_one_days_revenue(d_idx, month)


if __name__ == "__main__":
    pass
