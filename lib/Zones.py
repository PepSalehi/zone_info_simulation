import numpy as np
import pandas as pd

# import geopandas as gpd
from collections import deque

from lib.Requests import Req
from lib.Constants import WARMUP_TIME_SECONDS, BONUS, zones_neighbors

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


class Zone:
    def __init__(self, ID, rs=None):
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
        # self.neighbors = self.get_neighboring_zone_ids()

    # def get_neighboring_zone_ids(self):
    #     neighbors_list = zones_neighbors[str(self.id)]
    #     return neighbors_list

    def read_daily_demand(self, demand_df):
        self.DD = demand_df.query("PULocationID == {zone_id}".format(zone_id=self.id))

    #        print (self.DD.shape) # there are zones with no daily demand
    # self._set_demand_rate()
    # self.mid = 0

    def calculate_demand_function(self, surge):
        """ 
        This should be a decreasing function of price 
        use elasticities instead 
        -0.6084 for NYC
        
        """
        base_demand = self.D
        change = self.DEMAND_ELASTICITY * (
            surge - 1
        )  # percent change in price * elascticity
        new_demand = int((1 + change) * base_demand)  # since change is negative
        new_demand = np.max([0, new_demand])
        return new_demand

    def join_incoming_vehicles(self, veh):
        try:
            assert veh not in self.incoming_vehicles
        except AssertionError:
            print(veh.locations)
            print(veh.zone.id)
            print(veh.ozone)
            print(veh.idle)
            print(veh.rebalancing)
            print(veh.time_to_be_available)
        #            return veh

        self.incoming_vehicles.append(veh)

    def remove_veh_from_waiting_list(self, veh):
        if veh in self.idle_vehicles:
            #            print("removed idle")
            self.idle_vehicles.remove(veh)

    def identify_idle_vehicles(self):
        for v in self.incoming_vehicles:
            if v.time_to_be_available < 5:  # not v.rebalancing and
                assert v not in self.idle_vehicles
                v.rebalancing = False
                v.idle = True
                self.idle_vehicles.append(v)
                self.incoming_vehicles.remove(v)

    def match_veh_demand(self, Zones, t, WARMUP_PHASE, penalty=-10):
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

    #                    break

    def assign(self, Zones, t, WARMUP_PHASE, penalty):
        self.identify_idle_vehicles()
        # bookkeeping
        self._demand_history.append(len(self.demand))
        self._serverd_demand_history.append(len(self.served_demand))
        self._supply_history.append(len(self.idle_vehicles))
        self._incoming_supply_history.append(len(self.incoming_vehicles))
        #
        self.match_veh_demand(Zones, t, WARMUP_PHASE, penalty)

    def set_demand_rate_per_t(self, t):
        """
        This should use self.demand as the (hourly) demand, and then generate demand accoring to a Poisson distribution
        time_period: hour of day
        """
        # print (self.DD.shape)
        demand = self.DD.query("Hour == {T}".format(T=t))
        # print (demand)
        self.D = demand.shape[0]  # number of rows, i.e., transactions
        self.mid = 0

    def set_surge_multiplier(self, m):
        self.surge = m

    # generate one request, following exponential arrival interval
    def _generate_request(self, d=None):
        # check if there is any demand first
        if self.D == 0:  # i.e., no demand
            return None
        if d is None:
            d = self.D

        scale = 3600.0 / d
        dt = np.int(self.rs1.exponential(scale))
        try:
            destination = self.DD.iloc[self.mid]["DOLocationID"]
            distance = self.DD.iloc[self.mid]["trip_distance_meter"]
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

    # generate requests up to time T, following Poisson process
    def generate_requests_to_time(self, T):
        if self.N == 0:
            req = self._generate_request()
            if req is not None:
                self.reqs.append(req)
                self.N += 1

        d = self.calculate_demand_function(self.surge)

        while d != 0 and self.reqs[-1].Tr <= T:  # self.N <= self.D:
            req = self._generate_request(d)
            if req is not None:
                self.demand.append(self.reqs[-1])
                self.reqs.append(req)
                self.N += 1
            else:
                break


#        assert self.N == len(self.reqs)


# if __name__ == "__main__":
#     z = Zone(4)
#     z.generate_demand()
#     z.generate_requests_to_time(3600)
#     len(z.demand)
#     z.demand[0].Tr
#     z.demand[-1].Tr
#     pass

# TODO: df_hourly_stats_over_days is what a professional driver knows
# TODO: df_hourly_stats is stats per hour per day. Can be the true information provided by the operator (although how are they gonna know it in advance?)

