import numpy as np
import pandas as pd
from collections import defaultdict
# all these should be moved to the data class. Or maybe shouldn't
from lib.Constants import (
    ZONE_IDS,
    PHI,
    DIST_MAT,
    CONSTANT_SPEED,
    INT_ASSIGN,
    MAX_IDLE,
    FUEL_COST,
    CONST_FARE,
    zones_neighbors,
    PENALTY,
    my_dist_class
)

from lib.Requests import Req
from lib.configs import configs
from functools import lru_cache
from enum import Enum, unique, auto
import pickle

import logging

logging.basicConfig(filename='drivers.log', level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(message)s')

# from lib.rl_policy import DQNAgent
driver_id = 0


# https://stackoverflow.com/a/57516323/2005352
class VehState(Enum):
    """
    Enum describing the state of a vehicle, where
    IDLE = waiting to be matched
    REBAL = travelling, but without a passenger. Upon arrival, should wait to be matched
    SERVING = currently saving demand. Should make a decision to move upon arrival at the req's destination
    DECISION = should make a decision
    """
    IDLE = auto()
    REBAL = auto()
    SERVING = auto()
    DECISION = auto()


class DriverType(Enum):
    '''
    There are 3 types:
    Pro:
    they are experienced drivers, and have accurate estimates of fare and matching probabilities.
    also different choice making behavior
    Naive:
    These are the most inexperienced ones: they have no idea of their own about fares/etc.
    importantly, they don't learn anything as they gather more experience.
    inexperienced:
    These start just like naives, but start to learn immediately
    '''
    PROFESSIONAL = auto()
    NAIVE = auto()
    INEXPERIENCED = auto()
    AV = auto()


# https://stackoverflow.com/questions/18622781/why-is-numpy-random-choice-so-slow
def _choice(options, probs):
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            break
    return options[i]


def compute_mae(y, yhat):
    return np.mean(np.abs(y - yhat))


def _convert_reporting_dict_to_df(dic):
    '''

    @param dic: this is the self.REPORTING_DICT
    @return: pandas dataframe, with columns being the keys of the dic
    '''
    return pd.DataFrame.from_dict(
        dic, orient="columns"
    )


def _convect_time_to_peak_string(t):
    """

    @param t: in seconds
    @return: morning_peak, evening_peak, off_peak
    """
    if (t >= 8 * 3600) and (t <= 10 * 3600):
        return "morning_peak"
    elif (t >= 17 * 3600) and (t <= 19 * 3600):
        return "evening_peak"
    else:
        return "off_peak"


class Veh:
    """
    Class encapsulating a vehicle.
    """

    def __init__(
            self,
            rs,
            operator,
            beta=1,
            true_demand=True,
            driver_type=None,
            ini_loc=None,
            know_fare=False,
            is_AV=False,
            dist_mat=DIST_MAT

    ):
        """
        Creates a Vehicle object.

        @param rs: # TODO: what is this
        @param operator (Operator): object describing operator-specific details (e.g. Uber)
        @param beta (float):
        @param true_demand (bool): # TODO: what is this
        @param driver_type (enum): whether the driver is professional, naive, or inexperienced
        @param ini_loc (int): initial location (zone) of the driver
        @param know_fare (bool): whether the driver knows the fare
        @param is_AV (bool)
        @param dist_mat:
        """
        global driver_id
        driver_id += 1
        self.id = driver_id
        # self.just_started = True
        self.is_AV = is_AV  # TODO: REMOVE
        self.rs = rs
        self.DIST_MAT = dist_mat
        self.operator = operator
        self._state = VehState.IDLE

        self.true_demand = true_demand  # TODO: REMOVE
        self.driver_type = driver_type
        self.know_fare = know_fare  # TODO: REMOVE

        self.prior_fare_dict = None
        self.initialize_prior_fare_info()

        self.locations = []
        self.req = None
        self.beta = beta  # TODO: REMOVE
        if ini_loc is None:
            self.ozone = rs.choice(ZONE_IDS)
            self.locations.append(self.ozone)
            self._state = VehState.DECISION

        self.IDLE_COST = 0  # should be based on the waiting time
        self.unit_rebalancing_cost = FUEL_COST  # should be based on dist to the destination zone
        self.profits = []
        self.time_idled = 0
        self.MAX_IDLE = MAX_IDLE  # 15 minutes

        # self.t_since_idle = None
        self.number_of_times_moved = 0
        self.number_of_times_overwaited = 0
        self.distance_travelled = 0
        self.time_to_be_available = 0

        self.tba = []
        self.total_waited = 0
        self.zone = None
        self.collected_fares = []
        self.collected_fare_per_zone = defaultdict(list)

        self.REPORTING_DICT = {'driver_id': [self.id],
                               'driver_type': [self.driver_type],
                               'starting_zone': [self.ozone],
                               'destination_zone': [self.zone],
                               'driver_state': [self._state]
                               }
        # debugging 
        self._times_chose_zone = []
        # to store (state, action, reward) for each vehicle 
        self._info_for_rl_agent = []
        self.reqs = []
        self.total_served = 0
        self.state_hist = []

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_dist_to_all_zones(ozone):
        """
        @param ozone (int): current zone
        @return: df of distance to each zone. (show an example of the output)
        """
        # return DIST_MAT.query("PULocationID=={o}".format(o=ozone))
        return my_dist_class.return_distance_from_origin_to_all(ozone)

    @lru_cache(maxsize=None)
    def _get_dist_to_only_neighboring_zones(self, ozone):
        """
        @param ozone (int): current zone
        @return: df of distances to neighboring zones
        """
        # neighbors_list = self._get_neighboring_zone_ids()
        dists = DIST_MAT[(DIST_MAT["PULocationID"] == self.ozone) & (
            DIST_MAT["DOLocationID"].isin(self._get_neighboring_zone_ids(ozone)))]
        # dists = DIST_MAT.query(
        #     "PULocationID=={o} & DOLocationID.isin({destinations})".format(
        #         o=self.ozone, destinations=neighbors_list
        #     )
        # )
        return dists

    @lru_cache(maxsize=None)
    def _get_time_to_destination(self, ozone, dest):
        """
        @param ozone (int): original zone
        @param dest (int): destination zone
        @return: time to destination
        """
        # dist = self._get_distance_to_destination(dest)
        # t = self._get_distance_to_destination(ozone, dest) / CONSTANT_SPEED
        return self._get_distance_to_destination(ozone, dest) / CONSTANT_SPEED

    @lru_cache(maxsize=None)
    def _get_distance_to_destination(self, ozone, dest):
        """
        @param ozone (int): original zone
        @param dest (int): destination zone
        @return: distance to destination
        """
        try:  # because of the Nans, etc.  just a hack
            dist = np.ceil(
                # DIST_MAT.loc[ozone, dest]["trip_distance_meter"].values
                my_dist_class.return_distance(ozone, dest)
            )
        except:
            dist = 1000
            print(
                "Couldn't find the distance btw {o} and {d}".format(o=self.ozone, d=dest)
            )

        return dist

    @lru_cache(maxsize=None)
    def _get_neighboring_zone_ids(self, ozone):
        """
        @param ozone (int): pickup zone
        @return: a list of ids (ints) of the neighboring zones
        """
        neighbors_list = zones_neighbors[str(ozone)]
        neighbors_list.append(self.ozone)
        return neighbors_list

    def _calc_matching_prob(self):
        """
        If the driver is not professional, it will be matched.
        @return: int if the car is not professional, else None. (currently just 1?)
        TODO: this is not good design
        """
        if not self.driver_type == DriverType.PROFESSIONAL:
            return 1

    @lru_cache(maxsize=None)
    def get_data_from_operator(self, t, true_demand):
        """
        @param t: time
        @param true_demand (bool)
        @return: dataframe with zonal info for vehicle
        """
        return self.operator.zonal_info_for_veh(true_demand)

    def cal_profit_per_zone_per_app(self, t):
        """
        @param t: time of day
        @return: None
        """
        #        df = self.get_data_from_operator(t)
        pass

    def waited_too_long(self):
        """
        Makes sure it's not idle nor rebalancing, then if it has been idle for too long returns True
        @return: (bool)
        """
        return self._state == VehState.IDLE and self.time_idled > self.MAX_IDLE
        # return self.idle and not self.rebalancing and self.time_idled > self.MAX_IDLE

    def update_rebalancing(self, WARMUP_PHASE):
        """
        Updates when the vehicle state is rebalancing.

        @param WARMUP_PHASE (bool)
        @return: None
        """
        assert self._state == VehState.REBAL
        self.time_to_be_available -= INT_ASSIGN  # delta t, this is a hack

        if self.time_to_be_available < 0:
            self._state = VehState.IDLE
            self.state_hist.append(self._state)
            self.time_idled = 0
            self.time_to_be_available = 0
            if not WARMUP_PHASE:
                self.number_of_times_moved += 1

    def keep_waiting(self):
        """
        Updates the waiting time of a vehicle.
        @return: None
        """
        self.time_idled += INT_ASSIGN
        self.total_waited += INT_ASSIGN

    def keep_serving(self):
        """
        # TODO: what's going on here?
        @return:
        """
        self.time_to_be_available -= INT_ASSIGN
        if self.time_to_be_available < 0:
            assert self._state == VehState.SERVING
            self._state = VehState.DECISION
            if self.is_AV:
                try:
                    assert len(self._info_for_rl_agent) == 3
                except AssertionError:
                    print(self.waited_too_long())
                    print(self.time_idled)
                    print(self._info_for_rl_agent)
                    print(len(self.reqs))
                    print([r.fare for r in self.reqs])
                    print("time_to_be_available", self.time_to_be_available)
                    print("total_served", self.total_served)
                    print("self.is_busy", self.is_busy)
                    print("ozone", self.ozone)
                    print(self._state)
                    print("locations", self.locations)
                    print("self.state_hist", self.state_hist)
                    print("veh id ", self.id)
                    pickle.dump(self, open("veh.p", "wb"))
                    raise AssertionError

    @property
    def is_busy(self):
        """
        @return: bool
        """
        # try:
        #     assert self.serving == (not self.idle and not self.rebalancing )

        return self._state == VehState.SERVING

    def set_action(self, action):
        """
        Use the RL agent to decide on the target.
        """
        assert self.is_AV
        assert action is not None
        self.action = int(action)
        # print("action is", action)

    @property
    def is_waiting_to_be_matched(self):
        """
        @return: (bool)
        """
        if self._state == VehState.IDLE and self.time_idled <= self.MAX_IDLE:
            return True
        else:
            return False
            # if self.idle and not self.rebalancing and self.time_idled <= self.MAX_IDLE :

    @property
    def is_rebalancing(self):
        """
        @return: (bool)
        """
        # True if self._state == VehState.REBAL else False 
        if self._state == VehState.REBAL:
            return True
        else:
            return False

    def should_move(self):
        """
        @return: bool, true if just started or has been idle for too long
        """
        return self.waited_too_long() or self._state == VehState.DECISION

    def act(self, t, Zones, WARMUP_PHASE, action=None):
        """ 
        1. just started or has been idle for too long -> choose zone 
        2. if it's rebalancing (i.e., on the way to the target zone) -> check whether or not it has gotten there 
        3. if it's idle, but the waiting time has not yet exceeded the threshold ->  keep waiting 
        4. if it's currently serving a demand -> update status 
        5. idling is also an option 
        action is the INDEX of the zone, needs to be converted to the actual zone_id 
        """

        def _make_a_decision(t):
            """

            @param t: is seconds
            @return:
            """
            # first, get out of the current zone's queue
            target_zone = None
            if self.zone is not None:
                self.zone.remove_veh_from_waiting_list(self)
            # then choose the destination
            if self.driver_type != DriverType.AV:
                target_zone = self.choose_target_zone(t)

            if self.driver_type == DriverType.AV:
                # assert self.action is not None
                target_zone = Zones[self.action].id

            dist = None
            for z in Zones:
                if z.id == target_zone:
                    self._state = VehState.REBAL
                    self.state_hist.append(self._state)
                    # self.rebalancing = True
                    # self.idle = False
                    # self.TIME_TO_MAKE_A_DECISION  = False 
                    self.time_to_be_available = self._get_time_to_destination(
                        self.ozone, target_zone
                    )
                    self.tba.append(self.time_to_be_available)
                    dist = self._get_distance_to_destination(self.ozone, target_zone)
                    z.join_incoming_vehicles(self)
                    self.zone = z

                    break
            if dist is None:
                print("z.id: ", z.id)
                print("target_zone", target_zone)
                print("origin: ", self.ozone)
                print([z.id for z in Zones])

                raise Exception('dist was None')

            self.ozone = (
                target_zone
            )  # debugging for now (so what is this comment? should I delete this line? WTF is this doing?)

            if not WARMUP_PHASE:
                self.distance_travelled += dist
                self.number_of_times_moved += 1
                self.locations.append(self.ozone)

            # self.time_idled = 0
            return target_zone

        if self.should_move():
            _ = _make_a_decision(t)
            # self.update_rebalancing(WARMUP_PHASE)
            # return 
        if self.is_busy:
            self.keep_serving()
            return
        if self.is_waiting_to_be_matched:
            # it's sitting somewhere
            self.keep_waiting()
            return
        if self.is_rebalancing:  # and not self.busy:
            self.update_rebalancing(WARMUP_PHASE)
            return

    def match_w_req(self, req, Zones, t, WARMUP_PHASE):
        """
        Matches with request if possible and returns an indicator whether the vehicle is matched.
        
        @param req: (Request)
        @param Zones: list of zones (ints)
        @param WARMUP_PHASE: bool
        @return: bool (matched or not)
        """
        # try:
        #     assert self._state == VehState.IDLE
        # except:
        #     print(self.is_AV)
        #     print(self._state)
        #     print(self.time_to_be_available)
        #     raise AssertionError 
        assert self._state == VehState.IDLE
        self.time_idled = 0
        dest = req.dzone
        matched = False
        for z in Zones:
            if z.id == dest:
                self._state = VehState.SERVING
                self.state_hist.append(self._state)
                self.time_to_be_available = self._get_time_to_destination(self.ozone, dest)
                dist = self._get_distance_to_destination(self.ozone, dest)

                self.ozone = dest
                self.zone = z

                matched = True
                # don't match incoming, rather join the undecided list. 
                # actually, don't join any list because in the next step, "act" will take care of it
                # z.join_incoming_vehicles(self)
                # z.join_undecided_vehicles(self)
                #
                # if not WARMUP_PHASE:

                self.collect_fare(req.fare)
                self.add_to_experienced_zonal_fares(req.fare, req.ozone, t)
                self.operator.collect_fare(req.fare)
                self.update_posterior_fare_info(req.fare, req.ozone, t)
                # if (self.driver_type != DriverType.PROFESSIONAL) and not self.is_AV:
                #     self.collect_fare(req.fare)
                #     self.operator.revenues.append(PHI * req.fare)
                #
                #     self.collected_fare_per_zone[req.ozone] += (1 - PHI) * req.fare
                #
                # elif self.driver_type == DriverType.PROFESSIONAL:
                #     self.collected_fares.append(req.fare)
                #     self.operator.collect_fare(req.fare)
                #     self.collected_fare_per_zone[req.ozone] += req.fare

                self.locations.append(dest)
                self.distance_travelled += dist
                self.profits.append(
                    req.fare
                )
                if self.driver_type == DriverType.AV:

                    self.reqs.append(req)
                    self.locations.append(dest)
                    self.total_served += 1

                    try:
                        assert len(self._info_for_rl_agent) == 2
                    except AssertionError:
                        print(self._state)
                        print(self.waited_too_long())
                        print(self.time_idled)
                        print(self._info_for_rl_agent)
                        print(len(self.reqs))
                        print([r.fare for r in self.reqs])
                        print(self.time_to_be_available)
                        print(self.total_served)
                        raise AssertionError

                    self._info_for_rl_agent.append(np.round(req.fare, 4))  # doesn't account for rebl cost yet
                    try:
                        assert len(self._info_for_rl_agent) == 3
                    except AssertionError:
                        print(self._state)
                        print(self.waited_too_long())
                        print(self.time_idled)
                        print(self._info_for_rl_agent)
                        print(len(self.reqs))
                        print([r.fare for r in self.reqs])
                        print(self.time_to_be_available)
                        print(self.total_served)
                        raise AssertionError

                self.req = req
                return True

        if not matched:
            print("zone {} does not exist ".format(dest))
        # why and when would it return False?
        return False

    def get_fare_based_on_prior(self, t):
        """

        @param t:
        @return:
        """
        t = _convect_time_to_peak_string(t)
        # returns {zone_id: avg_fare}
        return {k[0]: v[0] for k, v in self.prior_fare_dict.items() if k[1] == t}

    def estimate_matching_prob_based_on_prior_LR(self, t):
        """
        this should logistic regression to estimate the matching probability as conveyed by the app.
        params of the LR should be (Q_app, t, surge_mult, bonus). removed zone_id bc there might not be enough experience
        per zone.
        @param t:
        @return:
        """
        t = _convect_time_to_peak_string(t)
        # returns {zone_id: matching_prob}
        # return {k[0]: v[0] for k, v in self.prior_fare_dict.items() if k[1] == t}
        pass

    def compute_prior_expected_revenue(self, prior_fare_dict, prior_matching_dict):
        """

        @param prior_fare_dict: from self.get_fare_based_on_prior
        @return:
        """
        {z_id: (1 - PHI) * fare for z_id, fare in prior_fare_dict.items()}
        expected_revenue = (1 - PHI) * fare_to_use * df.surge.values * match_prob + df.bonus.values

    @lru_cache(maxsize=None)
    # @profile
    def _compute_attractiveness_of_zones(self, t, ozone, true_demand):
        """
        @param t: time
        @param ozone: (int) current zone
        @param true_demand: (bool)
        @return: (df) attractiveness to all zones and (df) probability to go to each zone
        """
        # 1)  get demand and distances
        dist = self._get_dist_to_all_zones(ozone)
        # 1.1) demand as told by the app
        df = (self.get_data_from_operator(t, true_demand))  # .set_index('Origin')
        assert dist.shape[0] == df.shape[0]
        # 1.2) demand as expected from experience
        # PRO: get estimates based on the prior
        # naive: solely base decision on the app's info
        # inexperienced: the first day, act naively. Then start to act like a PRO, with inferior estimates of course
        if self.driver_type == DriverType.PROFESSIONAL:
            df_prior = None

        # a = pd.merge(df, dist, left_on='PULocationID', right_on="DOLocationID", how="left")
        # neighbors_list = self._get_neighboring_zone_ids(ozone)
        # a = a[a["Origin"].isin(neighbors_list)]
        # a = a[[i in neighbors_list for i in a.index.get_level_values(1)]]

        if df.empty:
            print(
                "corner case: take zone 127. there and no demand is left, df and therefore a will be empty. "
                "in this situation, it should just move to one of its neighbors"
            )
            print("ozone", self.ozone)
            # print("destination", neighbors_list[0])
            neighbors_list = self._get_neighboring_zone_ids(ozone)
            return neighbors_list[0]

        # 2) get the fare
        # Does the app tell them about fares?
        if not self.know_fare:  # they don't know the average fare for an area, they use one for all
            fare_to_use = CONST_FARE
        else:
            fare_to_use = np.ceil(df.avg_fare.values)
        # 3 )get matching probabilities
        if not self.driver_type == DriverType.PROFESSIONAL:
            # if not professional, you don't consider supply in your decision making
            match_prob = 1
        else:
            try:
                match_prob = df.match_prob.values
            except:
                # replace all these with logging
                # https://docs.python.org/3.8/howto/logging.html#configuring-logging
                print(df)
                print("that was df")
                # print(a)
                # print("that was a")
                print(self.driver_type)
                raise NotImplementedError

        # a["relative_demand"] = a["total_pickup"] / a["total_pickup"].sum()
        # 4) compute the expected revenue
        expected_revenue = (1 - PHI) * fare_to_use * df.surge.values * match_prob + df.bonus.values
        # 5) compute the expected cost
        expected_cost = (
                dist.trip_distance_meter.values * self.unit_rebalancing_cost)  # doesn't take into account the distance travelled once the demand is picked up
        # 6) compute the expected profit
        # https://github.com/numpy/numpy/issues/14281
        prof = np.core.umath.clip((expected_revenue - expected_cost) * df["total_pickup"].values, 0, 10000)
        # prof = np.clip((expected_revenue - expected_cost) * df["total_pickup"].values, a_min=0, a_max=None)
        # 7) compute the probability of moving to each zone
        # http://cs231n.github.io/linear-classify/#softmax
        prob = prof / prof.sum()
        # return a.index.get_level_values(1).values, prob
        return df["PULocationID"], prob
        # return a, prob, a["Origin"]

    def choose_target_zone(self, t):
        """
        This has to be based on the information communicated by the app, as well as the prior experience
        It should have the option of not moving. Maybe include that in the neighbors list
        @param t: time of day
        @return
        """

        a, prob = self._compute_attractiveness_of_zones(t, self.ozone, self.true_demand)
        try:
            selected_destination = _choice(a.values, prob)
            while selected_destination not in ZONE_IDS:
                # this seems to happen for zone 202
                print("the selected destination was not in the list of zones: ", selected_destination)
                # print("this is happened {} times".format(count_occourance))
                # selected_destination = np.random.choice(a, size=1, p=prob, replace=True)[0]
                selected_destination = _choice(a.values, prob)

        except:
            raise Exception("selected was empty, here is a {}".format(a))

        return selected_destination

    def _calculate_fare(self, request, surge):
        """
        From Yellow Cab taxi webpage
        @param request (Request)
        @param surge (float): surge multiplier
        @return: (float) fare
        """
        distance_meters = request.dist
        p1 = 2.5
        p2 = 0.5 * 5 * 1609 * distance_meters
        f = p1 + surge * p2
        return f

    ###### fare collection and updating methods
    def collect_fare(self, fare):
        self.collected_fares.append((1 - PHI) * fare)

    def add_to_experienced_zonal_fares(self, fare, zone_id, t):
        t = _convect_time_to_peak_string(t)
        zone_id = int(zone_id)
        self.collected_fare_per_zone[(zone_id, t)].append(fare)

    ###### prior fare info
    def initialize_prior_fare_info(self):
        """
        Sets prior demand/fare info.
        @return (df): prior
        """
        self.prior_fare_dict = self.operator.expected_fare_total_demand_per_zone_over_days(self.driver_type)

    def get_prior_experienced_fare_info(self, zone_id, t):
        '''

        @param zone_id:
        @param fare:
        @return:
        '''
        # t = _convect_time_to_peak_string(t)
        zone_id = int(zone_id)
        try:
            experienced_fares = self.collected_fare_per_zone[(zone_id, t)]
        except KeyError:
            return None, None, None
        if len(experienced_fares) < 1:
            return None, None, None
        elif len(experienced_fares) == 1:
            return experienced_fares[0], 5, 1  # large std
        return np.mean(experienced_fares), np.std(experienced_fares), len(experienced_fares)

    def update_posterior_fare_info(self, zone_id, fare, t):
        """
        Bayesian update. After serving one zone and observing fare f
        @param zone_id:
        @param fare:
        @return:
        @param t: time (morning peak ,etc)
        @return (df): prior
        """
        t = _convect_time_to_peak_string(t)
        zone_id = int(zone_id)

        def bayesian_update(prior_mu, prior_sigma, data_mu, data_sigma, n):
            _post_mu = (
                    ((prior_mu / np.power(prior_sigma, 2)) + ((n * data_mu) / np.power(data_sigma, 2))) /
                    ((1 / np.power(prior_sigma, 2)) + (n / np.power(data_sigma, 2)))
            )
            _post_sigma = np.sqrt(1 / ((1 / np.power(prior_sigma, 2)) + (n / np.power(data_sigma, 2))))

            return _post_mu, _post_sigma

        try:
            p_mu, p_sigma = self.prior_fare_dict[(zone_id, t)]
        except KeyError:
            p_mu, p_sigma = 6, 5
            # print((zone_id, t), ' not in self.prior_fare_dict')

        should_update_bayesian_wise = True
        d_mu, d_sigma, n_trips = self.get_prior_experienced_fare_info(zone_id, t)
        if d_mu is None:
            # this is because there has not been a trip of this zone yet
            should_update_bayesian_wise = False

        if should_update_bayesian_wise:
            if self.driver_type == DriverType.PROFESSIONAL:
                logging.info('PROFESSIONAL DRIVER {} is updating their belief'.format(self.id))
            logging.info('driver {} is updating their belief'.format(self.id))
            logging.info('      p_mu = {} and p_sigma ={}'.format(p_mu, p_sigma))
            logging.info('      d_mu = {} , d_sigma ={} and n_trips={}'.format(d_mu, d_sigma, n_trips))
            post_mu, post_sigma = bayesian_update(p_mu, p_sigma, d_mu, d_sigma, n_trips)
            self.prior_fare_dict[(zone_id, t)] = (post_mu, post_sigma)
            logging.info('      post_mu = {} and post_sigma ={}'.format(self.prior_fare_dict[(zone_id, t)][0],
                                                                        self.prior_fare_dict[(zone_id, t)][1]))
