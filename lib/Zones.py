import numpy as np
import pandas as pd

# import geopandas as gpd
from collections import deque
# from functools import lru_cache
from lib.Requests import Req
from lib.Constants import WARMUP_TIME_SECONDS, BONUS, convert_seconds_to_15_min
from lib.Vehicles import VehState, DriverType
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')



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
        demand: requests in the queue
        assign: assignment method
    """

    def __init__(self, ID, operator, output_path, rs=None):
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
        self.operator = operator
        self.output_path = output_path
        self.demand = deque([])  # demand maybe should be a time-based dictionary?
        self.denied_requests = []
        self.served_demand = []
        self.idle_vehicles = list()
        # self.busy_vehicles = list() # what is the difference btw busy and incoming?
        self.incoming_vehicles = list()
        self.undecided_vehicles = list()
        self.fare = None
        self.reqs = []
        self.N = 0
        self.M = None
        self.daily_destination_demand = None
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
        self.driver_information_count = 0  # how many times drivers, in a 15 min time interval, have asked for information
        self.available_AV_vehicles = []

        fh = logging.FileHandler(output_path + 'zones.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def give_info_to_drivers(self, t_seconds):
        """
        This should ask the operator for info, passing along its zone_id and driver_information count
        @return:
        """
        self.driver_information_count += 1  # this is wrong tho. it should be time-dependent count
        data = self.operator.get_zonal_info_for_veh(self.id, self.driver_information_count, t_seconds)
        return data

    def read_daily_demand(self, demand_df):
        """
        Updates the daily OD demand of this zone.
        @param pickup_df: df pick ups
        @param demand_df: df describing OD demand for all zones.
        @return: None
        """
        # self.daily_destination_demand = demand_df.query("PULocationID == {zone_id}".format(zone_id=self.id))
        self.daily_destination_demand = demand_df[demand_df["PULocationID"] == self.id]  ## OD data
        # self.pickup_binned = pickup_df[pickup_df["PULocationID"] == self.id]

    def calculate_demand_function(self, demand, surge):
        """
        Calculates demand as a function of current demand, elasticity, and surge.
        new_demand = elas * ((p_old * (surge-1)/p_old)) * old_demand + old_demand
        =
        new_demand = elas * ((surge-1)) * old_demand + old_demand
        https://www.khanacademy.org/economics-finance-domain/microeconomics/elasticity-tutorial/price-elasticity-tutorial/a/price-elasticity-of-demand-and-price-elasticity-of-supply-cnx#:~:text=Key%20points,the%20percentage%20change%20in%20price.
        https://www.nber.org/system/files/working_papers/w22627/w22627.pdf

        This should be a decreasing function of price 
        use elasticities instead 
        -0.6084 for NYC
        @param surge:
        @param demand:
        @requires surge >= 1

        @return (float): new demand according to surge
        """
        base_demand = demand
        new_demand = self.DEMAND_ELASTICITY * (surge-1) * base_demand + base_demand
        # change = self.DEMAND_ELASTICITY * (
        #         surge - 1
        # )  # percent change in price * elascticity
        # new_demand = int((1 + change) * base_demand)  # since change is negative
        new_demand = np.max([0, new_demand])
        assert new_demand <= base_demand
        # print("surge was ", surge)
        # print("change in demand as calculated by elasticity ", change)
        denied_pax = base_demand - new_demand
        # if int(denied_pax) > 0 :
        #     logger.info(f'base demand was {base_demand}, surge was {surge}, new demand is {new_demand}')
        #     logger.info(f'this means {base_demand - new_demand} denied requests')

        if int(denied_pax) > 0:
            for i in range(int(denied_pax)):
                self.denied_requests.append('denied_surge')

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
            # should raise an error

        self.incoming_vehicles.append(veh)

    def remove_veh_from_idle_list(self, veh):
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
            if v.driver_type != DriverType.AV:
                if v._state == VehState.IDLE:
                    assert v not in self.idle_vehicles
                    self.idle_vehicles.append(v)
                    self.incoming_vehicles.remove(v)

    def identify_available_AVs(self):
        """

        Updates the idle vehicles and incoming vehicle list.
        """
        for v in self.incoming_vehicles:
            # if v.time_to_be_available <= 0:  # not v.rebalancing and
            if v.driver_type == DriverType.AV:
                if v._state == VehState.IDLE or v._state == VehState.DECISION:  # after finishing a req, it's available again
                    assert v not in self.available_AV_vehicles
                    self.available_AV_vehicles.append(v)
                    self.incoming_vehicles.remove(v)

    def match_veh_demand(self, Zones, t, WARMUP_PHASE, operator, penalty=-10):
        """
        Matches idle vehicles to requests via a queue.
        @param operator:
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
        t_15_min = convert_seconds_to_15_min(t)
        # demand = self.daily_destination_demand.query("Hour == {T}".format(T=t))
        # TODO all these operations could've been simplified if I had used demand stats .csv instead of the actual demand
        self.this_t_demand = self.daily_destination_demand[
            self.daily_destination_demand['time_of_day_index_15m'] == t_15_min]
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
            # print("no demand")
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
            # destination = self.daily_destination_demand.iloc[self.mid]["DOLocationID"]
            try:
                destination = self.this_t_demand.iloc[self.mid]["DOLocationID"]
            except IndexError:
                print("sss")
            # distance = self.daily_destination_demand.iloc[self.mid]["trip_distance_meter"]
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
        self.driver_information_count = 0
        # print("self.D", self.D)
        # print("demand after surge computation", demand)
        self._generate_request(self.D, t_15_min)

    def instruct_drivers(self, t, zones, sol_p, sol_r, sol_d):
        """
        1. Gather all the drivers in state DECISION
        2. instruct them to move to a given zone by calling their set_action method
        @param zones:
        @param sol_d: denied
        @param sol_r: rebalnacing
        @param sol_p: assignment
        @param t:
        @return:
        """
        self.identify_available_AVs()
        # available_vehicles = [veh for veh in self.available_AV_vehicles] # if veh._state == VehState.DECISION
        # print(len(available_vehicles))
        if len(self.available_AV_vehicles) > 0:
            # for i,j,k in sol_r:
            t_fifteen = convert_seconds_to_15_min(t)
            # First, assign vehicles
            jk = 0
            # print(f"total demand for zone is {len(self.demand)}")
            for av in self.available_AV_vehicles[:]:
                if len(self.demand) > 0:
                    if self.demand[0].Tr <= t:
                        req = self.demand.popleft()
                        req_dest = req.dzone
                        # check and see if it's in the sol_p
                        if sol_p[(self.id, req_dest, t_fifteen)] > 0:
                            sol_p[(self.id, req_dest, t_fifteen)] -= 1
                        else:
                            if sol_d[(self.id, req_dest, t_fifteen)] > 0:
                                sol_d[(self.id, req_dest, t_fifteen)] -= 1
                                print("should deny this request")
                            else:
                                print('req dest not in optimal flow')

                        status = av.match_w_req(req, zones, t, False)
                        if status:  # if matched, remove from the zone's idle list
                            # print("assigning")
                            self._n_matched += 1
                            jk += 1
                            self.available_AV_vehicles.remove(av)
                            assert av.ozone == req.dzone
                            req.Tp = t
                            # if not WARMUP_PHASE:
                            self.served_demand.append(req)
                            self.revenue_generated += req.fare
                        else:
                            print("AV Not matched by zone ", self.id)

        # print(f"assigned {jk} out of {len(available_vehicles)}")
        # rebalance, if any left
        all_rebalancing_dests = [j for (i, j, t), v in sol_r.items() if v > 0]
        # if len(self.available_AV_vehicles)>0:
        #     print(f"there are still {len(self.available_AV_vehicles)} to rebalance")
        self.identify_available_AVs()
        ll = self.available_AV_vehicles[:]
        for av in ll:
            for dest in all_rebalancing_dests:
                if sol_r[(self.id, dest, t_fifteen)] > 0:
                    sol_r[(self.id, dest, t_fifteen)] -= 1
                    try:
                        self.available_AV_vehicles.remove(av)
                    except ValueError:
                        # how can av not be in the available vehicles?? I'm iterating over it...
                        print(f'av {av.id} is not in {[veh.id for veh in self.available_AV_vehicles]}')
                        raise ValueError
                    av.rebalance(zones, dest)
                break

    def check_pax_denied(self, t_seconds):
        """

        @param t_seconds:
        @return:
        """
        q = list(self.demand)
        for pax in q:
            if pax.has_waited_too_long(t_seconds):
                self.denied_requests.append(pax)
                self.demand.remove(pax)
                self.reqs.remove(pax)

    def reset(self, daily_OD_demand, t_seconds):
        """
        this is called by the Model after each day of operation
        cleans up demand and vehicles, ready to start a new day
        @return:
        """
        self.demand = deque([])  # demand maybe should be a time-based dictionary?
        self.denied_requests = []
        # self.served_demand = []
        # self.served_demand = []
        self.idle_vehicles = list()
        self.reqs = []
        # self.busy_vehicles = list()
        self.incoming_vehicles = list()
        # self.undecided_vehicles = list()
        self.fare = None
        self.revenue_generated = 0
        self.driver_information_count = 0
        self._n_matched = 0
        self.this_t_demand = None
        self.N = 0
        self.M = None
        self.daily_destination_demand = None
        self.D = None
        self.pickup_binned = None
        self.mid = 0
        self.surge = 1
        self.bonus = 0
        self.num_surge = 0  # number of times there was a surge pricing in the zone
        self.read_daily_demand(daily_OD_demand)
        self.set_demand_rate_per_t(t_seconds)
