import numpy as np
import pandas as pd
# from numba import jit
from typing import Dict
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
from functools import lru_cache
from enum import Enum, unique, auto
import pickle

# import logging
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh3 = logging.FileHandler('driver.log', mode='a')
# fh3.setFormatter(formatter)
# logger.addHandler(fh3)

# from lib.rl_policy import DQNAgent
driver_id = 0


class VehState(Enum):
    """
    https://stackoverflow.com/a/57516323/2005352
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
    """
    There are 3 types:
    Pro:
    they are experienced drivers, and have accurate estimates of fare and matching probabilities.
    also different choice making behavior
    Naive:
    These are the most inexperienced ones: they have no idea of their own about fares/etc.
    importantly, they don't learn anything as they gather more experience.
    inexperienced:
    These start just like naives, but start to learn immediately
    """
    PROFESSIONAL = auto()
    NAIVE = auto()
    INEXPERIENCED = auto()
    AV = auto()


# @jit(nopython=True)
def _choice(options, probs):
    """
    chooses randomly with probability probs from options
    https://stackoverflow.com/questions/18622781/why-is-numpy-random-choice-so-slow

    @param options:
    @param probs:
    @return:
    """
    # rs = np.random.RandomState(10)
    # x = rs.rand()  # np.random.rand()
    # I think I should NOT use seeding here. bc then every driver will receive the same probability and make the same choice
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            break
    return options[i]


# @jit(nopython=True)
def compute_mae(y, yhat):
    return np.mean(np.abs(y - yhat))


def _convert_reporting_dict_to_df(dic):
    """
    Converts df into dictionary. Used for the distance matrix
    @param dic: this is the self.REPORTING_DICT
    @return: pandas dataframe, with columns being the keys of the dic
    """
    return pd.DataFrame.from_dict(
        dic, orient="columns"
    )


def _convect_time_to_peak_string(t):
    """
    "converts t in seconds to t in (off_peak, morning_peak, evening_peak)
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
            day_of_run,
            output_path,
            beta=1,
            driver_type=None,
            ini_loc=None,
            dist_mat=DIST_MAT

    ):
        """
        Creates a Vehicle object.

        @param rs:
        @param operator (Operator): object describing operator-specific details (e.g. Uber)
        @param beta (float):
        @param driver_type (enum): whether the driver is professional, naive, or inexperienced
        @param ini_loc (int): initial location (zone) of the driver
        @param know_fare (bool): whether the driver knows the fare
        @param dist_mat:
        """

        global driver_id
        driver_id += 1
        self.id = driver_id
        # self.just_started = True
        self.rs = rs
        self.output_path = output_path
        self.DIST_MAT = dist_mat
        self.operator = operator
        self._state = VehState.IDLE

        self.driver_type = driver_type
        self.day_of_run = day_of_run
        self.prior_fare_dict = None
        self.prior_m_dict = None

        self.locations = []
        self.req = None
        self.beta = beta  # TODO: REMOVE
        if ini_loc is None:
            self.ozone = rs.choice(ZONE_IDS)
            self.locations.append(self.ozone)
            self._state = VehState.DECISION

        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # fh = logging.FileHandler(self.output_path + 'driver_' + str(self.id) + '.log')
        # fh.setFormatter(formatter)
        # self.logger.addHandler(fh)
        ######
        # fh3 = logging.FileHandler(output_path + 'driver.log', mode='a')
        # fh3.setFormatter(formatter)
        # logger.addHandler(fh3)
        #
        # if self.id == 1:
        #     logger.info("starting out")
        #     logger.info(f"initial location is {self.ozone}")
        ######
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
        # self.total_waited = 0
        self.zone = None
        self.collected_fares = []
        self.collected_fare_per_zone = defaultdict(list)
        # self.observed_matching_per_zone = defaultdict(list) #{(zone-id, t) : m}

        self.REPORTING_DICT = {'driver_id': [self.id],
                               'driver_type': [self.driver_type],
                               'starting_zone': [self.ozone],
                               'destination_zone': [self.zone],
                               'driver_state': [self._state]
                               }
        self.earning_report_dict = {
            'driver_id': [self.id],
            'driver_type': [self.driver_type],
            'total_day_earning': None,
            'day': None,
            'month': None
        }
        # debugging 
        self._times_chose_zone = []
        # to store (state, action, reward) for each vehicle 
        self._info_for_rl_agent = []
        self.reqs = []
        self.total_served = 0
        self.state_hist = []

    def bookkeep_one_days_earnings(self, day, month):
        if self.earning_report_dict['total_day_earning'] is None:
            # first time
            self.earning_report_dict['total_day_earning'] = [np.sum(self.collected_fares)]
            self.earning_report_dict['day'] = [day]
            self.earning_report_dict['month'] = [month]
        else:
            self.earning_report_dict['total_day_earning'].extend([np.sum(self.collected_fares)])
            self.earning_report_dict['day'].extend([day])
            self.earning_report_dict['month'].extend([month])
            self.earning_report_dict['driver_type'].extend([self.driver_type])
            self.earning_report_dict['driver_id'].extend([self.id])

    def report_final_earnings(self):
        df = pd.DataFrame(data=self.earning_report_dict)
        return df

    def reset(self, d_idx, month):
        # print("reset of naive driver")
        self.day_of_run = d_idx
        self.month = month
        self._state = VehState.IDLE
        self.collected_fares = []
        self.req = None
        self.reqs = []
        self.tba = []
        # self.total_waited = 0
        self.zone = None
        self.number_of_times_moved = 0
        self.number_of_times_overwaited = 0
        self.distance_travelled = 0
        self.time_to_be_available = 0
        self.time_idled = 0
        # start from a random location again
        self.ozone = self.rs.choice(ZONE_IDS)
        self.locations.append(self.ozone)
        self._state = VehState.DECISION
        # self.get_data_from_operator.cache_clear()

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_dist_to_all_zones(ozone) -> pd.DataFrame:
        """
        @param ozone (int): current zone
        @return: df of distance to each zone. (show an example of the output)
        """
        # return DIST_MAT.query("PULocationID=={o}".format(o=ozone))
        return my_dist_class.return_distance_from_origin_to_all(ozone)

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_dist_to_all_zones_as_a_dict(ozone):
        """
        @param ozone (int): current zone
        @return: df of distance to each zone. (show an example of the output)
        """
        # return DIST_MAT.query("PULocationID=={o}".format(o=ozone))
        return my_dist_class.return_distance_from_origin_to_all_as_a_dict(ozone)

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

    # @lru_cache(maxsize=None)
    def get_data_from_operator(self, t_seconds):
        """
        @param t: time
        @return: dataframe with zonal info for vehicle
        """

        if self.zone is not None:
            return self.zone.give_info_to_drivers(t_seconds)

        data = self.operator.get_zonal_info_for_general()
        assert data is not None
        return data

    def cal_profit_per_zone_per_app(self, t):
        """
        @param t: time of day
        @return: None
        """
        #        df = self.get_data_from_operator(t)
        pass

    def waited_too_long(self, t):
        """
        Makes sure it's not idle nor rebalancing, then if it has been idle for too long returns True
        @return: (bool)
        """
        # if self._state == VehState.IDLE and self.time_idled > self.MAX_IDLE:
        #     logging.info("Driver {} of type {} waited too long at zone {} after waiting for {} seconds".format(
        #         self.id, self.driver_type, self.ozone, self.time_idled))

        return self._state == VehState.IDLE and self.time_idled > self.MAX_IDLE

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
            #####
            # if self.id == 1:
            #     logger.info(f"finished rebalancing. arrived at zone {self.ozone}")
            #####

    def keep_waiting(self):
        """
        Updates the waiting time of a vehicle.
        @return: None
        """
        self.time_idled += INT_ASSIGN
        # self.total_waited += INT_ASSIGN
        #
        #####
        # if self.id == 1:
        #     logger.info(f"waiting to find a match at zone {self.ozone}. Have waited {self.time_idled} so far")
        #####

    def keep_serving(self):
        """
        #
        @return:
        """
        self.time_to_be_available -= INT_ASSIGN
        if self.time_to_be_available < 0:
            # dropped off a passenger
            assert self._state == VehState.SERVING
            self._state = VehState.DECISION
            # wait a few mins to get a match
            # self._state = VehState.IDLE
            # self.time_idled = 0

            #####
            # if self.id == 1:
            #     logger.info(f"dropped off the passenger. arrived at zone {self.ozone}")
            #     logger.info(
            #         f"now going to await for a while to get a match. is_waiting_to_be_matched : {self.is_waiting_to_be_matched}")
            #####

    @property
    def is_busy(self):
        """
        @return: bool
        """
        # try:
        #     assert self.serving == (not self.idle and not self.rebalancing )

        return self._state == VehState.SERVING

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
        VehState.DECISION is set to true when 1) at the very beginning, 2) after is done serving a trip
        @return: bool, true if just started or has been idle for too long
        """
        return self._state == VehState.DECISION

    def act(self, t, Zones, WARMUP_PHASE):
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
                self.zone.remove_veh_from_idle_list(self)
            # then choose the destination
            if self.driver_type != DriverType.AV:
                target_zone = self.choose_target_zone(t)

            # if self.driver_type == DriverType.AV:
            #     # assert self.action is not None
            #     # target_zone = Zones[self.action].id
            #     target_zone = self.action
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
                    self.ozone = z.id  # added Nov 27 2020, I think it's correct but maybe I missed sth

                    #####
                    # if self.id == 1:
                    #     logger.info(f"decided to go to zone {self.ozone}")
                    #     logger.info(f"time to be available is  {self.time_to_be_available}")
                    #####

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

        if self.waited_too_long(t):
            _ = _make_a_decision(t)

        elif self.should_move():
            _ = _make_a_decision(t)
            # self.update_rebalancing(WARMUP_PHASE)
            # return 
        elif self.is_busy:
            self.keep_serving()
            return
        elif self.is_waiting_to_be_matched:
            # it's sitting somewhere
            self.keep_waiting()
            return
        elif self.is_rebalancing:  # and not self.busy:
            self.update_rebalancing(WARMUP_PHASE)
            return

    def match_w_req(self, req, Zones, t, WARMUP_PHASE=False):
        """
        Matches with request if possible and returns an indicator whether the vehicle is matched.
        
        @param req: (Request)
        @param Zones: list of zones (ints)
        @param WARMUP_PHASE: bool
        @return: bool (matched or not)
        """
        if self.driver_type != DriverType.AV:
            if self._state != VehState.IDLE:
                print('agh, state should be idle, but is ', self._state)
                # self.logger.info("agh, state should be idle")
                # self.logger.info(self._state)
                # self.logger.info(self.time_to_be_available)

            # assert self._state == VehState.IDLE
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
                self.collect_fare(req.fare)
                self.add_to_experienced_zonal_fares(req.fare, req.ozone, t)
                self.operator.collect_fare(req.fare)
                # self.update_posterior_fare_info(req.fare, req.ozone, t)

                self.locations.append(dest)
                self.distance_travelled += dist
                self.profits.append(
                    req.fare
                )
                self.req = req
                #####
                # if self.id == 1:
                #     logger.info(f"matched with a pax going to zone {dest}")
                #     logger.info(f"time to be available is  {self.time_to_be_available}")
                #     logger.info(f"request's fare was {req.fare}")
                #####

                return True

        if not matched:
            print("zone {} does not exist ".format(dest))
        # why and when would it return False?
        return False
        # if self.driver_type == DriverType.AV:
        #
        #     self.reqs.append(req)
        #     self.locations.append(dest)
        #     self.total_served += 1
        #
        #     try:
        #         assert len(self._info_for_rl_agent) == 2
        #     except AssertionError:
        #         print(self._state)
        #         print(self.waited_too_long())
        #         print(self.time_idled)
        #         print(self._info_for_rl_agent)
        #         print(len(self.reqs))
        #         print([r.fare for r in self.reqs])
        #         print(self.time_to_be_available)
        #         print(self.total_served)
        #         raise AssertionError
        #
        #     self._info_for_rl_agent.append(np.round(req.fare, 4))  # doesn't account for rebl cost yet
        #     try:
        #         assert len(self._info_for_rl_agent) == 3
        #     except AssertionError:
        #         print(self._state)
        #         print(self.waited_too_long())
        #         print(self.time_idled)
        #         print(self._info_for_rl_agent)
        #         print(len(self.reqs))
        #         print([r.fare for r in self.reqs])
        #         print(self.time_to_be_available)
        #         print(self.total_served)
        #         raise AssertionError

    def choose_target_zone(self, t):
        """
        This has to be based on the information communicated by the app, as well as the prior experience
        It should have the option of not moving. Maybe include that in the neighbors list
        @param t: time of day
        @return
        """
        # zone_ids and prob are numpy arrays
        zone_ids, prob, _profits = self._compute_attractiveness_of_zones(t, self.ozone)
        selected_destination = self._choose_based_on_prob(zone_ids, prob, _profits)
        # if int(selected_destination) in [120, 153]:
        #     if self.day_of_run == 2:
        #         logger = self.operator.data_obj.get_logger()
        #         logger.info(f"decided to go to zone {selected_destination}")
        #         logger.info(f"the zones were {list(zip(zone_ids, prob))}")
        #         logger.info(f"the profits were { list(zip(zone_ids, _profits))}")
                # print("###logging###")

            # logger.info(f"time to be available is  {self.time_to_be_available}")

        ###
        # TODO: limiting to just the neighboring zones requires changing the optimization model, info model, and info
        # dissemination. A big deal. commented out for now
        # Narrow it down only to the neighboring zones
        # neighboring_zones = self._get_neighboring_zone_ids(self.ozone)
        # mask = np.where(np.in1d(zone_ids, neighboring_zones))[0]
        # candidate_zones = zone_ids[mask]
        # candidate_probs = prob[mask]
        # # renormalize the probs to add up to one
        # if candidate_probs.sum() <= 0.0001:
        #     selected_destination = candidate_zones[0]
        # else:
        #     candidate_probs_normalized = candidate_probs / candidate_probs.sum()
        #     ###
        #     #
        #     selected_destination = self._choose_based_on_prob(candidate_zones, candidate_probs_normalized, info_dict)
        #     #####
        # if self.id == 1:
        #     # logger.info(f"Rebalancing: decided to go to zone {selected_destination} from {self.ozone} \n")
        #     # logger.info(f"the probabilities were {candidate_probs_normalized} \n")
        #     # ca_idx = np.where(candidate_zones == selected_destination)[0][0]
        #     # chosen_prob = candidate_probs_normalized[ca_idx]
        #     # logger.info(f"the probability of choosing the destination it did was {chosen_prob}")
        #     # logger.info(f"the max probability of a zone was {np.max(candidate_probs_normalized)}")
        #     # logger.info(f"the sum of all probabilities were {np.sum(candidate_probs_normalized)}")
        #
        #     logger.info(f"Rebalancing: decided to go to zone {selected_destination} from {self.ozone} \n")
        #     logger.info(f"the probabilities were {prob} \n")
        #     ca_idx = np.where(zone_ids == selected_destination)[0][0]
        #     chosen_prob = prob[ca_idx]
        #     logger.info(f"the probability of choosing the destination it did was {chosen_prob}")
        #     logger.info(f"the max probability of a zone was {np.max(prob)}")
        #     logger.info(f"the sum of all probabilities were {np.sum(prob)}")

        #####

        return selected_destination

    def _choose_based_on_prob(self, z_ids, probs, info_dict):
        """
        helper function. selects a zone based on its choice probability
        @param z_ids: list of zone ids
        @param probs: probability of choosing each of the zones
        @return:
        """
        try:
            selected_dest = _choice(z_ids, probs)
            while selected_dest not in ZONE_IDS:
                # this seems to happen for zone 202
                print("the selected destination was not in the list of zones: ", selected_dest)
                print("choosing another one")
                # print("this is happened {} times".format(count_occourance))
                # selected_dest = np.random.choice(a, size=1, p=prob, replace=True)[0]
                selected_dest = _choice(z_ids, probs)
        except:
            raise Exception(f"selected was empty, here are zones {z_ids} and their probs {probs}")
        return selected_dest

    ###### fare collection and updating methods
    def collect_fare(self, fare):
        # this ignores bonus/ surge
        self.collected_fares.append((1 - PHI) * fare)

    def add_to_experienced_zonal_fares(self, fare, zone_id, t):
        t = _convect_time_to_peak_string(t)
        zone_id = int(zone_id)
        self.collected_fare_per_zone[(zone_id, t)].append(fare)

    # def _calculate_fare(self, request, surge):
    #     """
    #     From Yellow Cab taxi webpage
    #     @param request (Request)
    #     @param surge (float): surge multiplier
    #     @return: (float) fare
    #     """
    #     distance_meters = request.dist
    #     p1 = 2.5
    #     p2 = 0.5 * 5 * 1609 * distance_meters
    #     f = p1 + surge * p2
    #     return f
    def _compute_attractiveness_of_zones(self, t, ozone):
        pass

    def update_experienced_matching(self, zone_id, t, param):
        pass
