import numpy as np
import pandas as pd

# import geopandas as gpd
from collections import deque
# from functools import lru_cache
from lib.Requests import Req
from lib.Constants import WARMUP_TIME_SECONDS, BONUS, zones_neighbors
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
        self.mid = 0
        self.surge = 1
        self.bonus = 0
        self.num_surge = 0  # number of times there was a surge pricing in the zone
        self.DEMAND_ELASTICITY = -0.6084  # https://www.nber.org/papers/w22627.pdf
        self.adjusted_demand_rate = None
        self._n_matched = 0
        self.revenue_generated = 0
        self._demand_history = []
        self._serverd_demand_history = []
        self._supply_history = []
        self._incoming_supply_history = []
        # for debugging
        self._time_demand = []

    def read_daily_demand(self, demand_df):
        """
        Updates the daily demand of this zone.
        @param demand_df: df describing demand for all zones.
        @return: None
        """
        # self.DD = demand_df.query("PULocationID == {zone_id}".format(zone_id=self.id))
        self.DD = demand_df[demand_df["PULocationID"] == self.id]

    def calculate_demand_function(self, surge):
        """
        Calculates demand as a function of current demand, elasticity, and surge.

        This should be a decreasing function of price 
        use elasticities instead 
        -0.6084 for NYC
        @param surge (float): surge multiplier.
        @requires surge >= 1

        @return (float): new demand according to surge
        """
        base_demand = self.D
        change = self.DEMAND_ELASTICITY * (
            surge - 1
        )  # percent change in price * elascticity
        new_demand = int((1 + change) * base_demand)  # since change is negative
        new_demand = np.max([0, new_demand])
        assert new_demand <= base_demand
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

    def match_veh_demand(self, Zones, t, WARMUP_PHASE, penalty=-10):
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
                req = self.demand.popleft()
                status = v.match_w_req(req, Zones, WARMUP_PHASE)
                if status:  # if matched, remove from the zone's idle list
                    #                    print("matched")
                    self._n_matched += 1
                    self.idle_vehicles.remove(v)
                    assert v.ozone == req.dzone
                    req.Tp = t
                    # if not WARMUP_PHASE:
                    self.served_demand.append(req)
                    self.revenue_generated += req.fare * self.surge
                else:
                    print("Not mathced by zone ", self.id)
                    if v.is_AV:
                        "should be penalized"
                        # v.profits.append(penalty)

    # break

    def assign(self, Zones, t, WARMUP_PHASE, penalty):
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
        self._serverd_demand_history.append(len(self.served_demand))
        self._supply_history.append(len(self.idle_vehicles))
        self._incoming_supply_history.append(len(self.incoming_vehicles))
        self.match_veh_demand(Zones, t, WARMUP_PHASE, penalty)

    def set_demand_rate_per_t(self, t):
        """
        Sets the demand per time period.
        This should use self.demand as the (hourly) demand, and then generate demand according to a Poisson distribution
        @param t: hour of day
        """
        # print (self.DD.shape)
        demand = self.DD.query("Hour == {T}".format(T=t))
        # print (demand)
        self.D = demand.shape[0]  # number of rows, i.e., transactions
        self.mid = 0

    def set_surge_multiplier(self, m):
        """
        Sets the surge multiplier.
        @param m: (float) desired surge multiplier
        """
        self.surge = m

    # @profile 
    def _generate_request(self, d=None):
        """
        Generate one request, following exponential arrival interval.
        @param d: demand
        @return: request
        """
        # check if there is any demand first
        if self.D == 0:  # i.e., no demand
            return None
        if d is None:
            d = self.D

        scale = 3600.0 / d
        # inter-arrival time is generated according to the exponential distribution
        dt = np.int(self.rs1.exponential(scale))
        try:
            destination = self.DD.iloc[self.mid]["DOLocationID"]
            # distance = self.DD.iloc[self.mid]["trip_distance_meter"]
            fare = self.DD.iloc[self.mid]["fare_amount"] * self.surge + self.bonus
        except:
            # if there are no more rows in the data
            return None

        req = Req(
            id=0 if self.N == 0 else self.reqs[-1].id + 1,
            Tr=WARMUP_TIME_SECONDS + dt if self.N == 0 else self.reqs[-1].Tr + dt,
            ozone=self.id,
            dzone=destination,
            fare=fare,
        )
        #                    dist = distance)
        self.mid += 1
        return req


    # @profile
    def generate_requests_to_time(self, T):
        """
        Generate requests up to time T, following Poisson process
        @param T: time
        @return: None
        """
        if self.N == 0:
            req = self._generate_request()
            if req is not None:
                self.reqs.append(req)
                self.N += 1

        d = self.calculate_demand_function(self.surge)
        before_demand = len(self.demand)
        while d != 0 and self.reqs[-1].Tr <= T:  # self.N <= self.D:
            req = self._generate_request(d)
            if req is not None:
                # self.demand.append(self.reqs[-1]) # hmm, what?
                self.demand.append(req)
                self.reqs.append(req)
                self.N += 1
                # for debugging, number of requests per 5 minutes
                self._time_demand.append(
                    np.floor((req.Tr - WARMUP_TIME_SECONDS) / (5 * 60))
                )
            else:
                break
        after_demand = len(self.demand)
        if after_demand - before_demand > 100:
            print("Huge increase in the number of requests over 30 seconds!!")
            print("d ", d)
            print("T ", T)
            print("zone id ", self.id)

# TODO: df_hourly_stats_over_days is what a professional driver knows
# TODO: df_hourly_stats is stats per hour per day. Can be the true information provided by the operator (although how are they gonna know it in advance?)
