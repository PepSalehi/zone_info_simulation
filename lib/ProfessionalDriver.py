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
# import dill
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh = logging.FileHandler('drivers_pro.log', mode='w')
# fh.setFormatter(formatter)
# logger.addHandler(fh)
#
# logger_2 = logging.getLogger('d.e')


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


def CrossEntropy(yHat, y):
    epsilon = 1e-10;
    if y == 1:
        return -np.log(yHat + epsilon)
    else:
        return -np.log(1 - yHat + epsilon)


class ProfessionalDriver(Veh):
    """
    Class encapsulating a vehicle.
    """

    def __init__(self, rs, operator, day_of_run, output_path, beta=1, driver_type=DriverType.PROFESSIONAL,
                 ini_loc=None, training_days=5,
                 dist_mat=DIST_MAT):
        """
        Creates a Vehicle object.

        @param rs: #
        @param operator (Operator): object describing operator-specific details (e.g. Uber)
        @param beta (float):
        @param driver_type (enum): whether the driver is professional, naive, or inexperienced
        @param ini_loc (int): initial location (zone) of the driver
        @param know_fare (bool): whether the driver knows the fare
        @param dist_mat:
        """
        super().__init__(rs, operator, day_of_run, output_path, beta, driver_type, ini_loc, dist_mat)

        self.day_of_run = day_of_run
        self._skipped_recording = 0
        self._prior_fare_reliability_history = []
        self.initialize_prior_fare_info()
        self.initialize_prior_matching_info()
        self.training_days = training_days
        # keep track of fare and matching accuracies
        self.app_fare_history = []
        self.app_matching_history = []
        self.app_demand_history = []
        self.app_surge_history = []
        self.off_peak_history = []
        self.evening_peak_history = []
        self.morning_peak_history = []
        self.app_bonus_history = []
        self.app_advertised_fare_history = []
        self.visited_zones_history = []
        # and my own priors
        self.prior_fare_estimates_history = []
        self.prior_matching_estimates_history = []
        # and what actually was experienced
        # these should be closely related to self.prior_fare_dict
        self.experienced_fares_history = []
        self.experienced_matching_history = []
        self.clf = None

        self._app_matching_reliability_history = []
        self._app_fare_reliability_history = []
        self._prior_fare_reliability_history = []
        self._prior_matching_reliability_history = []

        # bookkeeping

        # def dd():
        #     # https: // stackoverflow.com / questions / 16439301 / cant - pickle - defaultdict
        #     return defaultdict(list)
        # {driver_id : {(zone_id, t_string) : [f,f,f,f,f]}}
        self._learning_day_to_day_fare_mean = defaultdict(lambda: defaultdict(list))
        self._learning_day_to_day_fare_sd = defaultdict(lambda: defaultdict(list))
        self._learning_day_to_day_m = defaultdict(lambda: defaultdict(list))
        self._learning_day_to_day_m_obs = defaultdict(lambda: defaultdict(list))

        # eh = logging.FileHandler(output_path + 'drivers_pro_experiences.log', mode='w')
        # eh.setLevel(logging.INFO)
        # eh.setFormatter(formatter)
        # logger_2.addHandler(eh)

    def report_learning_rates(self):
        def __convert_defaultdict_to_dict(a):
            a = dict(self._learning_day_to_day_fare_mean)
            for k, v in a.items(): a[k] = dict(v)
            return a

        fmean = __convert_defaultdict_to_dict(self._learning_day_to_day_fare_mean)
        f_sd = __convert_defaultdict_to_dict(self._learning_day_to_day_fare_sd)
        m = __convert_defaultdict_to_dict(self._learning_day_to_day_m)
        m_obs = __convert_defaultdict_to_dict(self._learning_day_to_day_m_obs)
        # want a df with columns : zone_id, time, fm, fsd, m, n
        data = defaultdict(list)
        for d_id, v in self._learning_day_to_day_fare_mean.items():
            for k, fms in v.items():  # k: (zone, time) v2: [fm]
                data["driver_id"].extend(np.repeat(self.id, len(fms)))
                data["z_ids"].extend(np.repeat(k[0], len(fms)))
                data["t_strings"].extend(np.repeat(k[1], len(fms)))
                data["fms"].extend(fms)
        df = pd.DataFrame(data=data)
        return df
        # merge all dfs at the end. Note: should remove lines with incomplete trips

    def report_surge_bonus_behavior(self):
        '''
        TODO: this is not doint what it's supposed to be doing
        TODO: I should also record when they don't get a match
        @return:
        '''
        if len(self.app_demand_history) > len(self.experienced_matching_history):
            # there was an unfinished trip
            self.app_demand_history.pop()
            self.app_fare_history.pop()
            self.app_matching_history.pop()
            self.prior_matching_estimates_history.pop()
            self.prior_fare_estimates_history.pop()
            #
            self.visited_zones_history.pop()
            self.app_advertised_fare_history.pop()
            self.app_bonus_history.pop()
            self.app_surge_history.pop()
            self.morning_peak_history.pop()
            self.evening_peak_history.pop()
            self.off_peak_history.pop()
        data = defaultdict(list)
        data["app_surge_history"] = [k[0] for k in (self.app_surge_history)]
        data["app_bonus_history"] = [k[0] for k in (self.app_bonus_history)]
        data["app_advertised_fare_history"] = [k[0] for k in (self.app_advertised_fare_history)]
        data["matched"] = (self.experienced_matching_history)
        data["driver_id"].extend(np.repeat(self.id, len(self.app_bonus_history)))
        data["app_demand"] = [k[0] for k in (self.app_demand_history)]
        data["morning_peak"] = [k[0] for k in (self.morning_peak_history)]
        data["off_peak"] = [k[0] for k in (self.off_peak_history)]
        df = pd.DataFrame(data=data)
        return df

    def report_fare_reliability_evolution(self):
        data = defaultdict(list)
        data["app_fare_reliability"] = self._app_fare_reliability_history
        data["prior_fare_reliability"] = self._prior_fare_reliability_history
        data["driver_id"].extend(np.repeat(self.id, len(self._app_fare_reliability_history)))
        df = pd.DataFrame(data=data)
        return df

    def report_matching_reliability_evolution(self):
        data = defaultdict(list)
        data["app_m_reliability"] = self._app_matching_reliability_history
        data["prior_m_reliability"] = self._prior_matching_reliability_history
        data["driver_id"].extend(np.repeat(self.id, len(self._app_matching_reliability_history)))
        df = pd.DataFrame(data=data)
        return df

    def reset(self, day_index, month):

        # np.savetxt(self.output_path + f'driver {self.id} for day {self.day_of_run} x_scaled', x_scaled, delimiter=",")
        # np.savetxt(self.output_path + f'driver {self.id} for day {self.day_of_run} x', x_train, delimiter=",")
        # np.savetxt(self.output_path + f'driver {self.id} for day {self.day_of_run} y', y_train, delimiter=",")

        # https://stackoverflow.com/questions/805066/how-to-call-a-parent-classs-method-from-child-class-in-python
        super(ProfessionalDriver, self).reset(day_index, month)
        self._compute_attractiveness_of_zones.cache_clear()
        # remove history of unfinished trips
        if len(self.app_demand_history) > len(self.experienced_matching_history):
            # there was an unfinished trip
            self.app_demand_history.pop()
            self.app_fare_history.pop()
            self.app_matching_history.pop()
            self.prior_matching_estimates_history.pop()
            self.prior_fare_estimates_history.pop()
            #
            self.visited_zones_history.pop()
            self.app_advertised_fare_history.pop()
            self.app_bonus_history.pop()
            self.app_surge_history.pop()
            self.morning_peak_history.pop()
            self.evening_peak_history.pop()
            self.off_peak_history.pop()


    @lru_cache(maxsize=None)
    def _compute_attractiveness_of_zones(self, t, ozone):
        """
        @param t: time in seconds
        @param ozone: (int) current zone
        @return: (df) attractiveness to all zones and (df) probability to go to each zone
        """

        # 1)  get demand and distances
        dist = self._get_dist_to_all_zones(ozone)
        dist_dict = self._get_dist_to_all_zones_as_a_dict(ozone)
        # apps data
        # 2) demand as told by the app
        app_demand_df = (self.get_data_from_operator(t))  # .set_index('Origin')
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
        prior_revenue_estimates_dict = self._estimate_income_based_on_prior_estimate(t, dist_dict)
        prior_matching_prob_estimates_dict = self._get_prior_matching_prob(t)

        # 4) compute weighted fare estimates
        # 4.1) get fare reliability of the app and  prior experience
        w_fare_app, w_fare_prior, status = self.get_reliability_of_apps_and_prior_fare_info()

        weighted_income = self._get_weighted_estimate(w_fare_app, estimated_revenue_from_app_dict,
                                                      w_fare_prior, prior_revenue_estimates_dict)
        # 5) compute weighted match_prob estimates
        # 5.1) get m reliability of the app
        w_match_a = self.get_reliability_of_apps_matching_info(status)
        # 5.2) get m reliability of prior experience
        w_match_p = self.get_reliability_of_prior_matching_info(status)
        epsilon = 1e-6
        w_match_sum = w_match_a + w_match_p
        # Note: it SHOULD be that w app uses w prior is numerator and vice versa.
        w_match_prior = w_match_a / w_match_sum
        w_match_app = w_match_p / w_match_sum

        self._prior_matching_reliability_history.append(w_match_prior)
        self._app_matching_reliability_history.append(w_match_app)

        weighted_matching_prob = self._get_weighted_matching_estimate(w_match_app, estimated_match_prob_from_app_dict,
                                                                      w_match_prior, prior_matching_prob_estimates_dict)
        # 6) compute the final utility function
        # Pro: revenue * match_prob
        predicted_utility = {}
        for z_id, fare in weighted_income.items():
            predicted_utility[z_id] = np.exp(fare * weighted_matching_prob[z_id])

        # predicted_utility = {z_id: np.exp(fare * weighted_matching_prob[z_id])
        #                      for z_id, fare in weighted_income.items()
        #                      }
        # https://stackoverflow.com/a/21870021/2005352
        location_ids, utilities = zip(*predicted_utility.items())
        utilities = np.array(utilities)
        utilities = np.clip(utilities, a_min=0, a_max=None)
        # 7) compute the probability of moving to each zone
        prob = utilities / utilities.sum()
        # return a.index.get_level_values(1).values, prob
        # 'app_demand_df': app_demand_df[['PULocationID', 'total_pickup']],
        return location_ids, prob, {'estimated_revenue_from_app_dict': estimated_revenue_from_app_dict,
                                    'estimated_match_prob_from_app_dict': estimated_match_prob_from_app_dict,
                                    'prior_revenue_estimates_dict': prior_revenue_estimates_dict,
                                    'prior_matching_prob_estimates_dict': prior_matching_prob_estimates_dict,
                                    'app_demand_locations': app_demand_df['PULocationID'].values,
                                    'app_demand_pickups': app_demand_df['total_pickup'].values,
                                    'app_fare': app_demand_df['avg_fare'].values,
                                    'app_bonus': app_demand_df['bonus'].values,
                                    'app_surge': app_demand_df['surge'].values,
                                    'morning_peak': app_demand_df['morning_peak'].values,
                                    'evening_peak': app_demand_df['evening_peak'].values,
                                    'off_peak': app_demand_df['off_peak'].values
                                    }

    # @profile
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
                print("choosing again")
                # print("this is happened {} times".format(count_occourance))
                # selected_dest = np.random.choice(a, size=1, p=prob, replace=True)[0]
                selected_dest = _choice(z_ids, probs)
                print(f"The new choice is {selected_dest}")
        except:
            raise Exception(f"selected was empty, here are zones {z_ids} and their probs {probs}")

        # bookkeeping for modeling the learning process
        assert info_dict is not None
        app_fare = info_dict['estimated_revenue_from_app_dict'][selected_dest]
        app_m = info_dict['estimated_match_prob_from_app_dict'][selected_dest]
        prior_fare = info_dict['prior_revenue_estimates_dict'][selected_dest]
        prior_m = info_dict['prior_matching_prob_estimates_dict'][selected_dest]
        locs = info_dict['app_demand_locations']
        pickups = info_dict['app_demand_pickups']
        app_demand = pickups[np.where(locs == selected_dest)]
        app_advertised_fare = info_dict['app_fare'][np.where(locs == selected_dest)]
        app_surge = info_dict['app_surge'][np.where(locs == selected_dest)]
        app_bonus = info_dict['app_bonus'][np.where(locs == selected_dest)]
        morning_peak = info_dict['morning_peak'][np.where(locs == selected_dest)]
        evening_peak = info_dict['evening_peak'][np.where(locs == selected_dest)]
        off_peak = info_dict['off_peak'][np.where(locs == selected_dest)]

        # app_demand_df = info_dict['app_demand_df']
        # cond = app_demand_df["PULocationID"].values == selected_dest
        # app_demand = app_demand_df[cond]["total_pickup"].values[0] # this is super slow

        # keep track of fare and matching accuracies
        self.app_fare_history.append(app_fare)
        self.app_matching_history.append(app_m)
        self.app_demand_history.append(app_demand)
        self.app_advertised_fare_history.append(app_advertised_fare)
        self.app_surge_history.append(app_surge)
        self.app_bonus_history.append(app_bonus)
        self.morning_peak_history.append(morning_peak)
        self.evening_peak_history.append(evening_peak)
        self.off_peak_history.append(off_peak)
        self.visited_zones_history.append(selected_dest)

        # and my own priors
        self.prior_fare_estimates_history.append(prior_fare)
        self.prior_matching_estimates_history.append(prior_m)

        # logger_2.info(
        #     f'Driver {self.id} added app fare : {app_fare}, app m : {app_m}, app demand : {app_demand}, '
        #     f'prior_f : {prior_fare}, and prior m : {prior_m}, '
        #     f"total_trips : {len(self.app_matching_history)}"
        # )

        return selected_dest

    def _estimate_income_based_on_prior_estimate(self, t, dist_dict):
        """
        returns {zone_id: avg_fare}. This is therefore just one number
        @param t:
        @return:
        """
        t_string = _convect_time_to_peak_string(t)
        #
        #  dist.T.to_dict(): key=zone, value: dict['trip_distance_meter':dist]

        return {k[0]: v[0] -
                      dist_dict[k[0]]['trip_distance_meter'] * self.unit_rebalancing_cost
                for k, v in self.prior_fare_dict.items()
                if k[1] == t_string}

        # return {k[0]: v[0] -
        #               dist.loc[k[0]].values[0] * self.unit_rebalancing_cost
        #         for k, v in self.prior_fare_dict.items()
        #         if k[1] == t_string}

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
        t_string = _convect_time_to_peak_string(t)
        # returns {zone_id: matching_prob}
        #
        if self.day_of_run >= self.training_days:
            # if len(self.experienced_matching_history) > 10:
            # train LR
            # logger_2.info(f'Driver {self.id} has started to train LR')
            # then predict based on the current demand
            m_preds_dict = self.predict_m_using_trained_LR(app_demand_df, t_string)
            # logger_2.info(f'and the predicted dict is {m_preds_dict}')
            return m_preds_dict
        return {k[0]: 1 for k, v in self.prior_fare_dict.items() if k[1] == t_string}

    # @profile
    def _compute_apps_expected_income(self, fare_df, dist, unit_rebalancing_cost=None):
        """
        just computes \pi
        @param fare_df:
        @param dist:
        @return:
        """
        if unit_rebalancing_cost is None: unit_rebalancing_cost = self.unit_rebalancing_cost
        expected_income = (1 - PHI) * fare_df["avg_fare"].values * fare_df["surge"].values + \
                          fare_df["bonus"].values
        # 5) compute the expected cost
        expected_cost = (
                dist["trip_distance_meter"].values * unit_rebalancing_cost)
        # 6) compute the expected profit
        difference = expected_income - expected_cost
        # difference = difference.to_numpy()
        pulocations = fare_df["PULocationID"].values
        # expected_revenue = {z_id: difference.iloc[i] for i, z_id in enumerate(fare_df["PULocationID"].values)}
        expected_revenue = {z_id: difference[i] for i, z_id in enumerate(pulocations)}

        return expected_revenue

    # def _compute_prior_expected_utility(self, prior_fare_dict, prior_matching_dict, dist):
    #     """
    #     computes zone utilities based on the prior experience for fare, and the output of LR for matching prob
    #     NOTE: it does NOT include (1 - PHI)  in the formulation, since driver always collects fare after platform
    #     has taken away its share
    #     not a static method, bc later I might want to add user specific coefficients
    #     @param prior_matching_dict:
    #     @param prior_fare_dict: from self._estimate_income_based_on_prior_estimate
    #     @return: {zone_id:utility}
    #     """
    #     #  dist.T.to_dict(): key=zone, value: dict['trip_distance_meter':dist]
    #     # TODO: to_dict is slow. calling loc is slower
    #
    #     dist_dict = dist.T.to_dict()
    #     return {z_id: fare * prior_matching_dict[z_id] -
    #                   dist_dict[z_id]['trip_distance_meter'] * self.unit_rebalancing_cost
    #             for z_id, fare in prior_fare_dict.items()}

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
                #
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
            # logger.info(
            #     "MAX_WAITING: Driver {} of type {} waited too long at zone {} after waiting for {} seconds".format(
            #         self.id, self.driver_type, self.ozone, self.time_idled))
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
        #TODO: make it start blank
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
        # bookkeeping
        self._learning_day_to_day_m[self.id][(zone_id, t_string)].append(new_m)
        self._learning_day_to_day_m_obs[self.id][(zone_id, t_string)].append(n + 1)

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

        # @jit(nopython=True)
        def bayesian_update(prior_mu, prior_sigma, data_mu, data_sigma, n):
            """
            this sometimes warns about division by zero.
            maybe because if there is only one obs, sd would be zero, causing issues
            @param prior_mu:
            @param prior_sigma:
            @param data_mu:
            @param data_sigma:
            @param n:
            @return:
            """
            if n <= 1: return prior_mu, prior_sigma
            if np.isclose(np.power(data_sigma, 2), 0):
                print(f"data_sigma was {data_sigma}")
                data_sigma = 1

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
        # logger.info(f'Driver {self.id} did NOT update prior bc ntrips : {n_trips}')
        if should_update_bayesian_wise:
            # logger.info('{} DRIVER {} is updating their belief'.format(self.driver_type, self.id))
            # logger.info('driver {} is updating their belief'.format(self.id))
            # logger.info('      p_mu = {} and p_sigma ={}'.format(p_mu, p_sigma))
            # logger.info('      d_mu = {} , d_sigma ={} and n_trips={}'.format(d_mu, d_sigma, n_trips))

            post_mu, post_sigma = bayesian_update(p_mu, p_sigma, d_mu, d_sigma, n_trips)
            self.prior_fare_dict[(zone_id, t_string)] = (post_mu, post_sigma)
            #
            # logger.info('      post_mu = {} and post_sigma ={}'.format(self.prior_fare_dict[(zone_id, t_string)][0],
            #                                                            self.prior_fare_dict[(zone_id, t_string)][1]))
            # keep track of fare belief evolution
            self._learning_day_to_day_fare_mean[self.id][(zone_id, t_string)].append(post_mu)
            self._learning_day_to_day_fare_sd[self.id][(zone_id, t_string)].append(post_sigma)

        # logger_2.info(f'They have visited {len(set(self.locations))} distinct zones after {len(self.locations)} trips,'
        #               f'having been matched {np.sum(np.array(self.experienced_matching_history)) / len(self.experienced_matching_history)}, '
        #               f'and this current zone {zone_id},'
        #               f' at time {t_string} a total of {n_trips} times')

    def get_reliability_of_prior_matching_info(self, skip_recording):
        # first check and see 1) there has been a finished trip
        if len(self.experienced_fares_history) < 10:
            return 0.5
        if self.day_of_run < self.training_days:
            return 0.5
        prior_m = np.array(self.prior_matching_estimates_history)
        true_m = np.array(self.experienced_matching_history)
        assert prior_m.shape == true_m.shape
        # CE is not btw 0-1. So, we find the max value, and normalize it that way
        CE = [CrossEntropy(yh, y) for yh, y in zip(prior_m, true_m)]
        acc = np.sum(CE)
        # acc = (np.sum(true_m == prior_m)) / len(self.experienced_matching_history)
        # if not skip_recording:
        # self._prior_matching_reliability_history.append(acc)
        # if np.isclose(acc, 0):
        #     logger_2.info("prior matching reliability is close to zero")
        #     logger_2.info("prior_m")
        #     logger_2.info(prior_m)
        #     logger_2.info("true_m")
        #     logger_2.info(true_m)
        return acc

    def get_reliability_of_apps_matching_info(self, skip_recording):
        # first check and see 1) there has been a finished trip
        if len(self.experienced_fares_history) < 10:
            return 0.5
        if self.day_of_run < self.training_days:
            return 0.5
        app_m = np.array(self.app_matching_history)
        true_m = np.array(self.experienced_matching_history)
        assert app_m.shape == true_m.shape
        # acc = (np.sum(true_m == app_m)) / len(self.experienced_matching_history)
        CE = [CrossEntropy(yh, y) for yh, y in zip(app_m, true_m)]
        acc = np.sum(CE)
        if not skip_recording:
            self._skipped_recording += 1
        # self._app_matching_reliability_history.append(acc)
        return acc

    def get_reliability_of_apps_and_prior_fare_info(self):
        """

        @return:
        """
        # first check and see 1) there has been a finished trip or 2) has at least matched once
        # in these cases, returns False so that we don't record matching reliabiltiy. otw different lengths
        if len(self.experienced_fares_history) < 1 or set(self.experienced_fares_history) == {'NM'}:
            return 0.5, 0.5, False
        if self.day_of_run < self.training_days:
            return 0.5, 0.5, False

        true_f = np.array(self.experienced_fares_history).astype(str)
        app_f = np.array(self.app_fare_history)
        prior_f = np.array(self.prior_fare_estimates_history)
        # exclude non-matches
        try:
            app_f = app_f[np.where(true_f != 'NM')]
        except IndexError:
            app_f
            raise IndexError
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
        self._app_fare_reliability_history.append(app_rel)
        self._prior_fare_reliability_history.append(prior_rel)
        return app_rel, prior_rel, True

    def _get_weighted_estimate(self, w_app, estimated_from_app, w_prior,
                               prior_estimates_dict):
        """

        @param w_app: app reliability. A number
        @param estimated_from_app: dict(zone_id: revenue)
        @param w_prior: experience reliability. A number
        @param prior_estimates_dict:
        @return: dict(zone_id: fare)
        """
        # weighted = {z_id: prior_estimates_dict[z_id] * w_prior +
        #                   app_fare * w_app
        #             for z_id, app_fare in estimated_from_app.items()}
        weighted = {}
        for z_id, app_fare in estimated_from_app.items():
            weighted[z_id] = prior_estimates_dict[z_id] * w_prior + app_fare * w_app

        return weighted

    @staticmethod
    def _get_weighted_matching_estimate(w_app, estimated_from_app, w_prior,
                                        prior_estimates_dict):
        """
        Because w might be 1, I need to explicitly divide by their sum
        @param w_app:
        @param estimated_from_app:
        @param w_prior:
        @param prior_estimates_dict:
        @return: {zone_id: m}
        """
        # logger_2.info("estimated_from_app_dict")
        # logger_2.info(estimated_from_app)
        weighted = {}
        # total_w = w_prior + w_app
        # if np.isclose(total_w, 0):
        #     # print(f"total w is close to zero, in fact {total_w}")
        #     # print(f"w_prior is {w_prior} and w_app is {w_app}")
        #     w_prior = 0.1
        #     w_app = 0.1
        #     total_w = w_prior + w_app

        weighted = {}
        for z_id, app_fare in estimated_from_app.items():
            weighted[z_id] = prior_estimates_dict[z_id] * w_prior + app_fare * w_app
        # for z_id, app_fare in estimated_from_app.items():
        #     weighted[z_id] = (prior_estimates_dict[z_id] * w_prior / total_w +
        #                       app_fare * w_app / total_w)
        # weighted = {z_id: (prior_estimates_dict[z_id] * w_prior / total_w ) +
        #                   (app_fare * w_app / total_w)
        #             for z_id, app_fare in estimated_from_app.items()}
        return weighted

    def _train_matching_prob_LR(self):
        """
        # TODO: should only be used after ~10 days and used after 15 days
        # if every experience is m=1, then don't train
        ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 1

        @return:
        """
        y_train = self.experienced_matching_history
        if len((y_train)) <= 5: return None
        x1 = np.array(self.app_demand_history)
        x2 = np.array(self.off_peak_history)
        x3 = np.array(self.evening_peak_history)
        x4 = np.array(self.morning_peak_history)
        x5 = np.array(self.app_bonus_history)
        x6 = np.array(self.app_surge_history)
        x7 = np.array(self.app_advertised_fare_history)
        x_train = np.concatenate((x1, x2, x3, x4, x5, x6, x7), axis=1)
        x_scaled = preprocessing.scale(x_train)
        # x_scaled = x_scaled.reshape(-1, 1)
        self.clf = LogisticRegression(random_state=0).fit(x_scaled, y_train)
        return True

    def predict_m_using_trained_LR(self, demand_df, t_string):
        """
        TODO: is this doing the right thing? it should predict each zones matching prob.
        @param demand_df:
        @param t_string:
        @return:
        """
        status = self._train_matching_prob_LR()
        z_ids = demand_df["PULocationID"].values.astype(int)
        if status is None:
            return {k: 1 for k in z_ids}
        assert self.clf is not None
        x = demand_df["total_pickup"].values
        x = demand_df[['total_pickup', 'off_peak', 'evening_peak', 'morning_peak', 'bonus', 'avg_fare', 'PULocationID']]
        x_scaled = preprocessing.scale(x);
        # x_scaled = x_scaled.reshape(-1, 1);
        probs = self.clf.predict_proba(x_scaled)[:, 1]
        return {k: v for k, v in zip(z_ids, probs)}
