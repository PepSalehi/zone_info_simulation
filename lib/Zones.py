import numpy as np
import pandas as pd

# import geopandas as gpd
from collections import deque
# from functools import lru_cache
from lib.Requests import Req
from lib.Constants import WARMUP_TIME_SECONDS, BONUS
from lib.Vehicles import VehState


class Zone:
    """
    Attributes:
        rs1: a seeded random generator for requests
        id: zone_id
        DD : daily demand (every trip)
        M: demand matrix
        D: demand volume (trips/hour)
        V: number of vehicles
        K: capacity of vehicles
        vehs: the list of vehicles
        N: number of requests

        mid: row number of the demand file
        reqs: the list of requests
        rejs: the list of rejected requests
        distance_rejs: the list of requests rejected because the distance from O to D
            was below the distance threshold (not included in rejs)
        queue: requests in the queue
        assign: assignment method
    """

    def __init__(self, ID, rs=None):
        """
        Initializes a zone object.
        @param ID: (int) zone id
        @param rs: random seeder
        """
        if rs is None:
            seed1 = 10
            self.rs1 = np.random.RandomState(seed1)
        else:
            self.rs1 = rs
        self.id = ID
        self.demand = deque([])  # demand maybe should be a time-based dictionary?
        self.served_demand = []
        self.idle_vehicles = list()
        self.busy_vehicles = list()
        self.incoming_vehicles = list()
        self.undecided_vehicles = list()
        self.fare = None
        self.reqs = []
        self.N = 0
        self.M = None
        self.DD = None
        self.D = None
        self.pickup_binned = None
        self.mid = 0
        self.surge = 1
        self.bonus = 0
        self.num_surge = 0  # number of times there was a surge pricing in the zone
        self.DEMAND_ELASTICITY = -0.6084  # https://www.nber.org/papers/w22627.pdf
        self.adjusted_demand_rate = None
        self._n_matched = 0
        self.revenue_generated = 0
        self._demand_history = []
        self._served_demand_history = []
        self._supply_history = []
        self._incoming_supply_history = []
        # for debugging
        self._time_demand = []

    def read_daily_demand(self, demand_df):
        """
        Updates the daily OD demand of this zone.
        @param pickup_df: df pick ups
        @param demand_df: df describing OD demand for all zones.
        @return: None
        """
        # self.DD = demand_df.query("PULocationID == {zone_id}".format(zone_id=self.id))
        self.DD = demand_df[demand_df["PULocationID"] == self.id]  ## OD data
        # self.pickup_binned = pickup_df[pickup_df["PULocationID"] == self.id]

    def calculate_demand_function(self, demand, surge):
        """
        Calculates demand as a function of current demand, elasticity, and surge.

        This should be a decreasing function of price 
        use elasticities instead 
        -0.6084 for NYC
        @param demand:
        @param surge (float): surge multiplier.
        @requires surge >= 1

        @return (float): new demand according to surge
        """
        base_demand = demand
        change = self.DEMAND_ELASTICITY * (
                surge - 1
        )  # percent change in price * elascticity
        new_demand = int((1 + change) * base_demand)  # since change is negative
        new_demand = np.max([0, new_demand])
        assert new_demand <= base_demand
        # print("surge was ", surge)
        # print("change in demand as calculated by elasticity ", change)

        return new_demand

    def join_incoming_vehicles(self, veh):
        """
        Adds incoming vehicle to list of incoming vehicles.
        @param veh (Vehicle)
        @return: None
        """
        try:
            assert veh not in self.incoming_vehicles
        except AssertionError:
            print(veh.locations)
            print(veh.zone.id)
            print(veh.ozone)
            print(veh.rebalancing)
            print(veh.time_to_be_available)

        self.incoming_vehicles.append(veh)

    def join_undecided_vehicles(self, veh):
        """
        Adds vehicle to list of undecided vehicles.
        @param veh (Vehicle)
        """
        try:
            assert veh not in self.undecided_vehicles
        except AssertionError:
            print(veh.locations)
            print(veh.zone.id)
            print(veh.ozone)
            print(veh.idle)
            print(veh.rebalancing)
            print(veh.time_to_be_available)

        self.undecided_vehicles.append(veh)

    def remove_veh_from_waiting_list(self, veh):
        """
        Removes vehicle from idle vehicles.
        @param veh (Vehicle)
        """
        if veh in self.idle_vehicles:
            self.idle_vehicles.remove(veh)

    def identify_idle_vehicles(self):
        """
        Updates the idle vehicles and incoming vehicle list.
        """
        for v in self.incoming_vehicles:
            # if v.time_to_be_available <= 0:  # not v.rebalancing and
            if v._state == VehState.IDLE:
                assert v not in self.idle_vehicles

                self.idle_vehicles.append(v)
                self.incoming_vehicles.remove(v)

    def match_veh_demand(self, Zones, t, WARMUP_PHASE, operator, penalty=-10):
        """
        Matches idle vehicles to requests via a queue.
        @param Zones:
        @param t: time
        @param WARMUP_PHASE (bool)
        @param penalty (float)
        @return: None
        """
        for v in self.idle_vehicles[:]:
            if len(self.demand) > 0:
                # check see if it's time
                if self.demand[0].Tr <= t:
                    req = self.demand.popleft()
                    status = v.match_w_req(req, Zones, t, WARMUP_PHASE)
                    if status:  # if matched, remove from the zone's idle list
                        #                    print("matched")
                        self._n_matched += 1
                        self.idle_vehicles.remove(v)
                        assert v.ozone == req.dzone
                        req.Tp = t
                        # if not WARMUP_PHASE:
                        self.served_demand.append(req)
                        self.revenue_generated += req.fare * self.surge
                        #
                        operator.budget -= self.bonus


                    else:
                        print("Not matched by zone ", self.id)
                        if v.is_AV:
                            "should be penalized"
                            # v.profits.append(penalty)

    # break

    def assign(self, Zones, t, WARMUP_PHASE, penalty, operator):
        """
        Identifies idle vehicles, then amends history and matches vehicle demand.

        @param Zones:
        @param t:
        @param WARMUP_PHASE:
        @param penalty:
        @return: None
        """
        self.identify_idle_vehicles()
        # bookkeeping
        self._demand_history.append(len(self.demand))
        self._served_demand_history.append(len(self.served_demand))
        self._supply_history.append(len(self.idle_vehicles))
        self._incoming_supply_history.append(len(self.incoming_vehicles))
        self.match_veh_demand(Zones, t, WARMUP_PHASE, operator, penalty)

    def set_demand_rate_per_t(self, t):
        """
        Sets the demand per time period.
        This should use self.demand as the (hourly) demand, and then generate demand according to a Poisson distribution
        @param t: seconds
        """
        t_15_min = np.floor(t / 900)
        # demand = self.DD.query("Hour == {T}".format(T=t))
        self.this_t_demand = self.DD[self.DD['time_of_day_index_15m'] == t_15_min]
        self.D = self.this_t_demand.shape[0]  # number of rows, i.e., transactions
        # print(self.D, "inside set demand")
        # self.D = self.pickup_binned[self.pickup_binned["total_seconds"] == t].shape[0]
        self.mid = 0

    def set_surge_multiplier(self, m):
        """
        Sets the surge multiplier.
        @param m: (float) desired surge multiplier
        """
        self.surge = m

    # @profile
    def _generate_request(self, d, t_15):
        """
        Generate one request, following exponential arrival interval.
        https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter13_stochastic/02_poisson.ipynb
        @param d: demand (number)
        @return: request
        when would it return None??
            1. where there is no demand
            2. when it cannot find the demand info in the df
        """
        # check if there is any demand first
        if self.D == 0:  # i.e., no demand
            print("no demand")
            return

        time_interval = 900.0  # 15 minutes. TODO: make sure this imports from constants.py
        t_15_start = t_15 * time_interval
        # print("t_15_start :", t_15_start)
        rate = d
        scale = time_interval / d
        # inter-arrival time is generated according to the exponential distribution
        # y is the arrival times, between 0-time_interval
        y = np.cumsum(self.rs1.exponential(scale,
                                           size=int(rate)))
        y += t_15_start
        # print("arrival times are :", y)
        self.__generate_exact_req_object2(y)

        # dt = np.int(self.rs1.exponential(scale))
        # try:
        #     # destination = self.DD.iloc[self.mid]["DOLocationID"]
        #     destination = self.DD.iloc[self.mid]["DOLocationID"]
        #     # distance = self.DD.iloc[self.mid]["trip_distance_meter"]
        #     fare = self.DD.iloc[self.mid]["fare_amount"] * self.surge + self.bonus
        # except:
        #     # if there are no more rows in the data
        #     print("couldn't find anymore rows")
        #     return None
        #
        # req = Req(
        #     id=0 if self.N == 0 else self.reqs[-1].id + 1,
        #     Tr=WARMUP_TIME_SECONDS + dt if self.N == 0 else self.reqs[-1].Tr + dt,
        #     ozone=self.id,
        #     dzone=destination,
        #     fare=fare,
        # )
        # #                    dist = distance)
        # self.mid += 1
        # return req

    def __generate_exact_req_object2(self, y):
        # self.mid = 0
        # print("mid", self.mid)
        # def __generate_exact_req_object(self, y):

        two_d_array = self.this_t_demand.iloc[0:len(y)][["DOLocationID", "fare_amount"]].values
        two_d_array[:, 1] = two_d_array[:, 1] * self.surge + self.bonus
        id_counter_start = 0 if self.N == 0 else self.reqs[-1].id + 1
        ids = [id_counter_start + i for i in range(y.shape[0])]
        reqs = [Req(
            id=ids[i],
            Tr=y[i],
            ozone=self.id,
            dzone=int(two_d_array[i][0]),
            fare=two_d_array[i][1]
        )
            for i in range(y.shape[0])]

        self.reqs.extend(reqs)
        self.demand.extend(reqs)

    # @profile
    def __generate_exact_req_object(self, y):
        self.mid = 0
        # print("mid", self.mid)
        # def __generate_exact_req_object(self, y):
        for arr_time in y:
            # print("mid", self.mid)
            # destination = self.DD.iloc[self.mid]["DOLocationID"]
            try:
                destination = self.this_t_demand.iloc[self.mid]["DOLocationID"]
            except IndexError:
                print("sss")
            # distance = self.DD.iloc[self.mid]["trip_distance_meter"]
            # try:
            fare = self.this_t_demand.iloc[self.mid]["fare_amount"] * self.surge + self.bonus
            # except (IndexError, TypeError):
            #     print("fare error")

            req = Req(
                id=0 if self.N == 0 else self.reqs[-1].id + 1,
                # Tr=WARMUP_TIME_SECONDS + arr_time if self.N == 0 else self.reqs[-1].Tr + arr_time,
                Tr=arr_time,
                ozone=self.id,
                dzone=destination,
                fare=fare
            )
            self.reqs.append(req)
            self.demand.append(req)
            # print("len of reqs", len(self.reqs))
            self.mid += 1
            # except:
            #     # if there are no more rows in the data
            #     print("couldn't find anymore rows")

    # @profile
    def generate_requests_to_time(self, t):
        """
        Generate requests up to time T, following Poisson process
        @param t: time (seconds)
        @return: None
        """
        t_15_min = np.floor(t / 900)

        self.set_demand_rate_per_t(t)
        # print("demand before possible surge in zone", self.id, self.D)
        demand = self.calculate_demand_function(self.D, self.surge)
        # print("demand after possible surge in zone", self.id, demand)

        before_demand = len(self.demand)
        # print("self.D", self.D)
        # print("demand after surge computation", demand)
        self._generate_request(self.D, t_15_min)

        # if there has not been a previous demand, just create one
        # if self.N == 0:
        #     req = self._generate_request()
        #     if req is not None:
        #         self.reqs.append(req)
        #         self.N += 1
        #
        # d = self.calculate_demand_function(self.D, self.surge)
        # before_demand = len(self.demand)
        # while d != 0 and self.reqs[-1].Tr <= t:
        #     # if there was a prior demand, check and see if the time was handled correctly
        #     req = self._generate_request(d)
        #     if req is not None:
        #         self.demand.append(req)
        #         self.reqs.append(req)
        #         self.N += 1
        #         # for debugging, number of requests per 5 minutes
        #         self._time_demand.append(
        #             np.floor((req.Tr - WARMUP_TIME_SECONDS) / (5 * 60))
        #         )
        #     else:
        #         break
        #
        # after_demand = len(self.demand)
        # if after_demand - before_demand > 100:
        #     print("Huge increase in the number of requests over 30 seconds!!")
        #     print("d ", d)
        #     print("T ", T)
        #     print("zone id ", self.id)

# TODO: df_hourly_stats_over_days is what a professional driver knows
# TODO: df_hourly_stats is stats per hour per day. Can be the true information provided by the operator (although how are they gonna know it in advance?)
