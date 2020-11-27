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
from lib.Vehicles import Veh, DriverType, VehState, _convect_time_to_peak_string, _choice, \
    _convert_reporting_dict_to_df, compute_mae
from functools import lru_cache
from enum import Enum, unique, auto
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('drivers_pro.log', mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger_2 = logging.getLogger('d.e')
eh = logging.FileHandler('drivers_pro_experiences.log', mode='w')
eh.setLevel(logging.INFO)
eh.setFormatter(formatter)
logger_2.addHandler(eh)


def update_avg_incrementally(prev_est, prev_n, new_val):
    """
    To be used by the update match prob
    https://math.stackexchange.com/questions/106700/incremental-averageing
    https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time
    @param prev_est:
    @param prev_n:
    @param new_val:
    @return:
    """
    return prev_est + (new_val - prev_est) / (prev_n + 1)


class ProfessionalDriver(Veh):
    """
    Class encapsulating a vehicle.
    """

    def __init__(self, rs, operator, beta=1, true_demand=True, driver_type=DriverType.PROFESSIONAL,
                 ini_loc=None, know_fare=False,
                 is_AV=False, dist_mat=DIST_MAT):
        """
        Creates a Vehicle object.

        @param rs: #
        @param operator (Operator): object describing operator-specific details (e.g. Uber)
        @param beta (float):
        @param true_demand (bool): #
        @param driver_type (enum): whether the driver is professional, naive, or inexperienced
        @param ini_loc (int): initial location (zone) of the driver
        @param know_fare (bool): whether the driver knows the fare
        @param is_AV (bool)
        @param dist_mat:
        """
        super().__init__(rs, operator, beta, true_demand, driver_type, ini_loc, know_fare, is_AV, dist_mat)
        self.initialize_prior_fare_info()
        self.initialize_prior_matching_info()

        # keep track of fare and matching accuracies
        self.app_fare_history = []
        self.app_matching_history = []
        self.app_demand_history = []
        # and my own priors
        self.prior_fare_estimates_history = []
        self.prior_matching_estimates_history = []
        # and what actually was experienced
        # these should be closely related to self.prior_fare_dict
        self.experienced_fares_history = []
        self.experienced_matching_history = []
        self.clf = None

    @lru_cache(maxsize=None)
    # @profile
    def _compute_attractiveness_of_zones(self, t, ozone, true_demand):
        """
        @param t: time in seconds
        @param ozone: (int) current zone
        @param true_demand: (bool)
        @return: (df) attractiveness to all zones and (df) probability to go to each zone
        """

        # 1)  get demand and distances
        dist = self._get_dist_to_all_zones(ozone)
        # apps data
        # 2) demand as told by the app
        app_demand_df = (self.get_data_from_operator(t, true_demand))  # .set_index('Origin')
        assert dist.shape[0] == app_demand_df.shape[0]
        if app_demand_df.empty:
            print(
                "corner case: take zone 127. there and no demand is left, df and therefore a will be empty. "
                "in this situation, it should just move to one of its neighbors"
            )
            print("ozone", self.ozone)
            # print("destination", neighbors_list[0])
            neighbors_list = self._get_neighboring_zone_ids(ozone)
            return neighbors_list[0]
        # compute the expected revenue & matching prob according to the app 
        estimated_revenue_from_app_dict = self._compute_apps_expected_income(app_demand_df, dist)
        estimated_match_prob_from_app_dict = self._estimate_matching_prob_based_on_LR(t, app_demand_df)

        # 3) compute the expected revenue & matching prob based on experience
        prior_revenue_estimates_dict = self._estimate_income_based_on_prior_estimate(t, dist)
        prior_matching_prob_estimates_dict = self._get_prior_matching_prob(t)

        # 4) compute weighted fare estimates
        # 4.1) get fare reliability of the app and  prior experience
        w_fare_app, w_fare_prior = self.get_reliability_of_apps_and_prior_fare_info()

        weighted_income = self._get_weighted_estimate(w_fare_app, estimated_revenue_from_app_dict,
                                                      w_fare_prior, prior_revenue_estimates_dict)
        # 5) compute weighted match_prob estimates
        # 5.1) get m reliability of the app
        w_match_app = self.get_reliability_of_apps_matching_info()
        # 5.2) get m reliability of prior experience
        w_match_prior = self.get_reliability_of_prior_matching_info()

        weighted_matching_prob = self._get_weighted_matching_estimate(w_match_app, estimated_match_prob_from_app_dict,
                                                                      w_match_prior, prior_matching_prob_estimates_dict)
        # 6) compute the final utility function
        # Pro: revenue * match_prob
        predicted_utility = {z_id: np.exp(fare * weighted_matching_prob[z_id])
                             for z_id, fare in weighted_income.items()
                             }
        # https://stackoverflow.com/a/21870021/2005352
        location_ids, utilities = zip(*predicted_utility.items())
        utilities = np.array(utilities)
        utilities = np.clip(utilities, a_min=0, a_max=None)
        # 7) compute the probability of moving to each zone
        prob = utilities / utilities.sum()
        # return a.index.get_level_values(1).values, prob
        return location_ids, prob, {'estimated_revenue_from_app_dict': estimated_revenue_from_app_dict,
                                    'estimated_match_prob_from_app_dict': estimated_match_prob_from_app_dict,
                                    'prior_revenue_estimates_dict': prior_revenue_estimates_dict,
                                    'prior_matching_prob_estimates_dict': prior_matching_prob_estimates_dict,
                                    'app_demand_df': app_demand_df}

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
                # print("this is happened {} times".format(count_occourance))
                # selected_dest = np.random.choice(a, size=1, p=prob, replace=True)[0]
                selected_dest = _choice(z_ids, probs)
        except:
            raise Exception(f"selected was empty, here are zones {z_ids} and their probs {probs}")

        # bookkeeping for modeling the learning process
        assert info_dict is not None
        app_fare = info_dict['estimated_revenue_from_app_dict'][selected_dest]
        app_m = info_dict['estimated_match_prob_from_app_dict'][selected_dest]
        prior_fare = info_dict['prior_revenue_estimates_dict'][selected_dest]
        prior_m = info_dict['prior_matching_prob_estimates_dict'][selected_dest]
        app_demand_df = info_dict['app_demand_df']
        app_demand = app_demand_df[app_demand_df["PULocationID"] == selected_dest]["total_pickup"].values[0]

        # keep track of fare and matching accuracies
        self.app_fare_history.append(app_fare)
        self.app_matching_history.append(app_m)
        self.app_demand_history.append(app_demand)
        # and my own priors
        self.prior_fare_estimates_history.append(prior_fare)
        self.prior_matching_estimates_history.append(prior_m)

        logger_2.info(
            f'Driver {self.id} added app fare : {app_fare}, app m : {app_m}, app demand : {app_demand}, '
            f'prior_f : {prior_fare}, and prior m : {prior_m}, '
            f"total_trips : {len(self.app_matching_history)}"
        )

        return selected_dest

    def _estimate_income_based_on_prior_estimate(self, t, dist):
        """
        returns {zone_id: avg_fare}. This is therefore just one number
        @param t:
        @return:
        """
        t = _convect_time_to_peak_string(t)
        #
        #  dist.T.to_dict(): key=zone, value: dict['trip_distance_meter':dist]
        dist_dict = dist.T.to_dict()
        return {k[0]: v[0] -
                      dist_dict[k[0]]['trip_distance_meter'] * self.unit_rebalancing_cost
                for k, v in self.prior_fare_dict.items()
                if k[1] == t}
        # return {k[0]: v[0] for k, v in self.prior_fare_dict.items() if k[1] == t}

    def _get_prior_matching_prob(self, t):
        """
        simply looks up how often driver was matched. Should add surge/bonus/demand to it.
        @param t:
        @return:
        """
        t = _convect_time_to_peak_string(t)
        return {k[0]: v[0] for k, v in self.prior_m_dict.items() if k[1] == t}

    def _estimate_matching_prob_based_on_LR(self, t, app_demand_df):
        """
        this should logistic regression to estimate the matching probability using info conveyed by the app.
        params of the LR should be (Q_app, t, surge_mult, bonus). removed zone_id bc there might not be enough experience
        per zone.
        @param t:
        @return:
        """
        t = _convect_time_to_peak_string(t)
        # returns {zone_id: matching_prob}
        #
        if len(self.experienced_matching_history) > 10:
            # train LR
            logger_2.info(f'Driver {self.id} has started to train LR')
            # then predict based on the current demand
            m_preds_dict = self.predict_m_using_trained_LR(app_demand_df)
            logger_2.info(f'and the predicted dict is {m_preds_dict}')
            return m_preds_dict
        return {k[0]: 1 for k, v in self.prior_fare_dict.items() if k[1] == t}

    def _compute_apps_expected_income(self, fare_df, dist):
        """
        just computes \pi
        @param fare_df:
        @param dist:
        @return:
        """
        expected_income = (1 - PHI) * fare_df.avg_fare * fare_df.surge.values + fare_df.bonus.values
        # 5) compute the expected cost
        expected_cost = (
                dist.trip_distance_meter.values * self.unit_rebalancing_cost)
        # 6) compute the expected profit
        difference = expected_income - expected_cost
        expected_revenue = {z_id: difference.iloc[i] for i, z_id in enumerate(fare_df["PULocationID"].values)}
        return expected_revenue

    def _compute_prior_expected_utility(self, prior_fare_dict, prior_matching_dict, dist):
        """
        computes zone utilities based on the prior experience for fare, and the output of LR for matching prob
        NOTE: it does NOT include (1 - PHI)  in the formulation, since driver always collects fare after platform
        has taken away its share
        not a static method, bc later I might want to add user specific coefficients
        @param prior_matching_dict:
        @param prior_fare_dict: from self._estimate_income_based_on_prior_estimate
        @return: {zone_id:utility}
        """
        #  dist.T.to_dict(): key=zone, value: dict['trip_distance_meter':dist]
        # TODO: to_dict is slow. but I think calling iloc repeatedly would be slower, haven't tried it
        dist_dict = dist.T.to_dict()
        return {z_id: fare * prior_matching_dict[z_id] -
                      dist_dict[z_id]['trip_distance_meter'] * self.unit_rebalancing_cost
                for z_id, fare in prior_fare_dict.items()}

    def match_w_req(self, req, Zones, t, WARMUP_PHASE):
        """
        Matches with request if possible and returns an indicator whether the vehicle is matched.

        @param req: (Request)
        @param Zones: list of zones (ints)
        @param WARMUP_PHASE: bool
        @return: bool (matched or not)
        """
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
                self.collect_fare(req.fare)
                self.add_to_experienced_zonal_fares(req.fare, req.ozone, t)
                self.operator.collect_fare(req.fare)
                # update experienced fare history
                # note: this is effectively doing the same task as self.collect_fare. But added here for clarity
                # note that req.ozone should be used, ozone gets set ot destination not the current zone
                # TODO: this ignores surge/bonus
                self.experienced_fares_history.append((1 - PHI) * req.fare)
                self.experienced_matching_history.append(1)
                self.update_experienced_matching(req.ozone, t, 1)
                # update perception of fare
                self.update_posterior_fare_info(req.fare, req.ozone, t)

                # These were all moved to when a decision to move is being made
                # update perception of app's reliability
                # Here, driver did get a match.
                # self.update_reliability_of_apps_fare_info()
                # self.update_reliability_of_apps_matching_info()
                # self.update_reliability_of_prior_fare_info()
                # self.update_reliability_of_prior_matching_info()

                self.locations.append(dest)
                self.distance_travelled += dist
                self.profits.append(
                    req.fare
                )
                self.req = req
                return True

        if not matched:
            print("zone {} does not exist ".format(dest))
        # why and when would it return False?
        return False

    def waited_too_long(self, t):
        """
        Serves to record when a driver has not been matched after waiting for self.MAX_IDLE time
        Makes sure it's not idle nor rebalancing, then if it has been idle for too long returns True
        @return: (bool)
        """
        if self._state == VehState.IDLE and self.time_idled > self.MAX_IDLE:
            logger.info(
                "MAX_WAITING: Driver {} of type {} waited too long at zone {} after waiting for {} seconds".format(
                    self.id, self.driver_type, self.ozone, self.time_idled))
            # record the matching rate
            self.experienced_matching_history.append(0)
            self.update_experienced_matching(self.ozone, t, 0)
            self.experienced_fares_history.append('NM')
        return self._state == VehState.IDLE and self.time_idled > self.MAX_IDLE

    def initialize_prior_fare_info(self):
        """
        Sets prior demand/fare info.
        this is reading it from the operator. Can be tricky when working with multi day operations
        @return (df): prior {(z_id, t_string) : (avg_fare, std_fare))}
        """
        self.prior_fare_dict = self.operator.expected_fare_total_demand_per_zone_over_days(self.driver_type)

    def initialize_prior_matching_info(self):
        """
        Reads zones and their matching probs from history
        Perhaps, later, these should be read for each driver specifically. i.e. each driver outputs a file that
        will be read by itself again the next day

        @return: {(zone_id, t_string): (m, n_obs)}
        """
        self.prior_m_dict = self.operator.expected_matching_per_zone_over_days(self.driver_type)

    def update_experienced_matching(self, zone_id, t_sec, new_val):
        assert new_val in (1, 0)
        t_string = _convect_time_to_peak_string(t_sec)
        zone_id = int(zone_id)
        m_est, n = self.prior_m_dict[(zone_id, t_string)]
        new_m = update_avg_incrementally(m_est, n, new_val)
        self.prior_m_dict[(zone_id, t_string)] = (new_m, n + 1)

    def get_prior_experienced_fare_info_for_a_zone(self, zone_id, t_string):
        '''
        returns mean &std of fares for a zone, and number of trips to that zone
        @param t_string: "morning_peak" etc
        @param zone_id:
        @return:
        '''
        # t = _convect_time_to_peak_string(t)
        zone_id = int(zone_id)
        try:
            experienced_fares = self.collected_fare_per_zone[(zone_id, t_string)]
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
        t_string = _convect_time_to_peak_string(t)
        zone_id = int(zone_id)

        def bayesian_update(prior_mu, prior_sigma, data_mu, data_sigma, n):
            """
            this sometimes warns about division by zero.
            @param prior_mu:
            @param prior_sigma:
            @param data_mu:
            @param data_sigma:
            @param n:
            @return:
            """
            _post_mu = (
                    ((prior_mu / np.power(prior_sigma, 2)) + ((n * data_mu) / np.power(data_sigma, 2))) /
                    ((1 / np.power(prior_sigma, 2)) + (n / np.power(data_sigma, 2)))
            )
            _post_sigma = np.sqrt(1 / ((1 / np.power(prior_sigma, 2)) + (n / np.power(data_sigma, 2))))

            return _post_mu, _post_sigma

        try:
            p_mu, p_sigma = self.prior_fare_dict[(zone_id, t_string)]
        except KeyError:
            p_mu, p_sigma = 6, 5
            # print((zone_id, t), ' not in self.prior_fare_dict')

        should_update_bayesian_wise = True
        d_mu, d_sigma, n_trips = self.get_prior_experienced_fare_info_for_a_zone(zone_id, t_string)
        if d_mu is None:
            # this is because there has not been a trip of this zone yet
            should_update_bayesian_wise = False
        logger.info(f'Driver {self.id} did NOT update prior bc ntrips : {n_trips}')
        if should_update_bayesian_wise:
            logger.info('{} DRIVER {} is updating their belief'.format(self.driver_type, self.id))
            logger.info('driver {} is updating their belief'.format(self.id))
            logger.info('      p_mu = {} and p_sigma ={}'.format(p_mu, p_sigma))
            logger.info('      d_mu = {} , d_sigma ={} and n_trips={}'.format(d_mu, d_sigma, n_trips))
            post_mu, post_sigma = bayesian_update(p_mu, p_sigma, d_mu, d_sigma, n_trips)
            self.prior_fare_dict[(zone_id, t_string)] = (post_mu, post_sigma)
            logger.info('      post_mu = {} and post_sigma ={}'.format(self.prior_fare_dict[(zone_id, t_string)][0],
                                                                       self.prior_fare_dict[(zone_id, t_string)][1]))
        logger_2.info(f'They have visited {len(set(self.locations))} distinct zones after {len(self.locations)} trips,'
                      f'having been matched {np.sum(np.array(self.experienced_matching_history)) / len(self.experienced_matching_history)}, '
                      f'and this current zone {zone_id},'
                      f' at time {t_string} a total of {n_trips} times')

    def get_reliability_of_prior_matching_info(self):
        # first check and see 1) there has been a finished trip
        if len(self.experienced_fares_history) < 1:
            return 0.5
        prior_m = np.array(self.prior_matching_estimates_history)
        true_m = np.array(self.experienced_matching_history)
        assert prior_m.shape == true_m.shape
        acc = (np.sum(true_m == prior_m)) / len(self.experienced_matching_history)
        return acc

    def get_reliability_of_apps_matching_info(self):
        # first check and see 1) there has been a finished trip
        if len(self.experienced_fares_history) < 1:
            return 0.5
        app_m = np.array(self.app_matching_history)
        true_m = np.array(self.experienced_matching_history)
        assert app_m.shape == true_m.shape
        acc = (np.sum(true_m == app_m)) / len(self.experienced_matching_history)
        return acc

    def get_reliability_of_apps_and_prior_fare_info(self):
        """

        @return:
        """
        # first check and see 1) there has been a finished trip or 2) has at least matched once
        if len(self.experienced_fares_history) <= 1 or set(self.experienced_fares_history) == {'NM'}:
            return 0.5, 0.5
        true_f = np.array(self.experienced_fares_history).astype(str)
        app_f = np.array(self.app_fare_history)
        prior_f = np.array(self.prior_fare_estimates_history)
        # exclude non-matches
        app_f = app_f[np.where(true_f != 'NM')]
        prior_f = prior_f[np.where(true_f != 'NM')]
        true_f = true_f[np.where(true_f != 'NM')].astype(np.float)

        assert app_f.shape == true_f.shape == prior_f.shape
        app_mae = compute_mae(app_f, true_f)
        prior_mae = compute_mae(prior_f, true_f)

        def weighted_reliability(mae_1, mae_2):
            w_1 = (1 / mae_1) / ((1 / mae_1) + (1 / mae_2))
            w_2 = (1 / mae_2) / ((1 / mae_1) + (1 / mae_2))
            return w_1, w_2

        app_rel, prior_rel = weighted_reliability(app_mae, prior_mae)
        return app_rel, prior_rel

    def _get_weighted_estimate(self, w_app, estimated_from_app, w_prior,
                               prior_estimates_dict):
        """

        @param w_app: app reliability. A number
        @param estimated_from_app: dict(zone_id: revenue)
        @param w_prior: experience reliability. A number
        @param prior_estimates_dict:
        @return: dict(zone_id: fare)
        """
        weighted = {z_id: prior_estimates_dict[z_id] * w_prior +
                          app_fare * w_app
                    for z_id, app_fare in estimated_from_app.items()}
        return weighted

    def _get_weighted_matching_estimate(self, w_app, estimated_from_app, w_prior,
                                        prior_estimates_dict):
        """
        Because w might be 1, I need to explicitly divide by their sum
        @param w_app:
        @param estimated_from_app:
        @param w_prior:
        @param prior_estimates_dict:
        @return: {zone_id: m}
        """
        total_w = w_prior + w_app
        weighted = {z_id: prior_estimates_dict[z_id] * w_prior / total_w +
                          app_fare * w_app / total_w
                    for z_id, app_fare in estimated_from_app.items()}
        return weighted

    def _train_matching_prob_LR(self):
        """
        # TODO: all of these require t_string, which I don't record right now
        # if every experience is m=1, then don't train
        ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 1

        @return:
        """
        y_train = self.experienced_matching_history
        if len(set(y_train)) == 1: return None
        x_train = self.app_demand_history
        x_scaled = preprocessing.scale(x_train);
        x_scaled = x_scaled.reshape(-1, 1);
        self.clf = LogisticRegression(random_state=0).fit(x_scaled, y_train)
        return True

    def predict_m_using_trained_LR(self, demand_df):
        status = self._train_matching_prob_LR()
        z_ids = demand_df["PULocationID"].values.astype(int)
        if status is None:
            return {k: 1 for k in z_ids}

        assert self.clf is not None
        x = demand_df["total_pickup"].values
        x_scaled = preprocessing.scale(x);
        x_scaled = x_scaled.reshape(-1, 1);
        probs = self.clf.predict_proba(x_scaled)[:, 1]
        return {k: v for k, v in zip(z_ids, probs)}
