from functools import lru_cache

import numpy as np
import pandas as pd
from lib.Constants import (
    ZONE_IDS,
    DEMAND_UPDATE_INTERVAL,
    POLICY_UPDATE_INTERVAL,
    MIN_DEMAND,
    MAX_BONUS,
    PHI)
import logging
from lib.Vehicles import DriverType, _convect_time_to_peak_string

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Operator:
    """
    Represents the actions of the company operating the ride-share vehicles.
    """

    def __init__(
            self,
            bonus_policy,
            bonus,
            surge_multiplier,
            budget,
            scenario,
            which_day_numerical,
            which_month,
            do_behavioral_opt,
            do_surge_pricing,
            output_path,
            data_obj=None,
            name="Uber",

    ):
        """
        Creates an operator instance.
        @param report: (df)
        @param which_day_numerical: (int)
        @param name: (str) name of operator
        @param BONUS: (float)
        @param SURGE_MULTIPLIER: (float)
        """
        self.do_surge_pricing = do_surge_pricing
        self.scenario = scenario
        self.name = name
        self.month = which_month
        self.do_behavioral_opt = do_behavioral_opt
        self.output_path = output_path
        self.data_obj = data_obj
        fh3 = logging.FileHandler(output_path + 'operators_log.log', mode='w')
        fh3.setFormatter(formatter)
        logger.addHandler(fh3)
        # self.demand_fare_stats_prior = pd.read_csv(
        #     "./Data/df_hourly_stats_over_days.csv"
        # )
        # self.demand_fare_stats_of_the_day = pd.read_csv(
        #     "./Data/df_hourly_stats.csv"
        # ).query("Day=={d}".format(d=which_day_numerical))
        self.demand_fare_stats_prior = pd.read_csv(
            "./Data/stats_over_all_days_w_st.csv"
        )
        self.demand_fare_stats_prior_naive = pd.read_csv(
            "./Data/stats_over_all_days_w_st_for_naive.csv"
        )
        self.demand_fare_stats_prior_peak_off_peak_dict = {
            tuple(row[['PULocationID', 'time_period_label']].values): (row.avg_fare, row.std_fare)
            for _, row in self.demand_fare_stats_prior.iterrows()}
        # basically all the same, and random, values. Made to unify the interface
        self.demand_fare_stats_prior_peak_off_peak_dict_naive = {
            tuple(row[['PULocationID', 'time_period_label']].values): (row.avg_fare, row.std_fare)
            for _, row in self.demand_fare_stats_prior_naive.iterrows()}

        self.demand_fare_stats_of_the_month = pd.read_csv('./Data/stats_for_{}_18.csv'.format(self.month))
        self.demand_fare_stats_of_the_day = self.demand_fare_stats_of_the_month.query(
            'Day=={}'.format(which_day_numerical))

        vs = self.demand_fare_stats_of_the_day.time_of_day_index_15m.values * 15 * 60
        vs = np.vectorize(_convect_time_to_peak_string)(vs)
        self.demand_fare_stats_of_the_day["time_of_day_label"] = vs
        ports = pd.get_dummies(self.demand_fare_stats_of_the_day.time_of_day_label)
        self.demand_fare_stats_of_the_day = self.demand_fare_stats_of_the_day.join(ports)
        # self.demand_fare_stats_of_the_day.time_of_day_label.drop(['time_of_day_label'], axis=1, inplace=True)

        self.matching_stats_prior_dict = {
            tuple(row[['PULocationID', 'time_period_label']].values): ((1 if row['total_pickup'] <= 20 else 0), 15)
            for _, row in self.demand_fare_stats_prior.iterrows()}

        self.live_data = None
        self.optimal_si = None
        self.revenues = []
        # these should be probably enums, and accessed via functions

        self.SURGE_MULTIPLIER = surge_multiplier
        self.BONUS_POLICY = bonus_policy
        self.budget = budget
        self.revenue_report_dict = {
            'name': [self.name],
            'total_day_earning': None,
            'day': None
        }

    @staticmethod
    def random_bonus_function(ratio):
        if ratio >= 1.2:
            return np.around(np.random.uniform(0, MAX_BONUS), decimals=1)
        else:
            return 0

    def set_bonus(self, ratio):
        if self.BONUS_POLICY == "random":
            return self.random_bonus_function(ratio)
        if self.BONUS_POLICY == "const":
            return 5

    def collect_fare(self, fare):
        self.revenues.append(PHI * fare)

    @staticmethod
    def surge_step_function(ratio):
        """
        Calculates the surge charge based on an assumed step-wise function
        0.9-1 : 1.2
        1-1.2 : 1.5
        1.2-2: 2
        >2: 3
        @param ratio: (float)
        @return: the surge charge according to the function.
        """
        if ratio < 0.9:
            return 1
        if 1 >= ratio >= 0.9:
            return 1.1
        if 1.2 >= ratio > 1:
            return 1.5
        if 2 > ratio > 1.2:
            return 2
        else:
            return 2.5

    def update_zonal_info(self, t):
        """
        Updates the zonal information if it's a new demand update interval.
        @param t: current time
        """
        if t % DEMAND_UPDATE_INTERVAL == 0:
            self.get_true_zonal_info(t)

    def get_zonal_info_for_general(self):
        """
        Gets the zonal info, no manipulation

        @param true_demand: (bool)
        @return: (df)
        """
        return self.live_data

    def get_zonal_info_for_veh(self, zone_id, driver_information_count, t_seconds):
        """
        Gets the zonal info for driver # driver_information_count
        as asked by zone z_id

        @param t_seconds:
        @param zone_id:
        @param driver_information_count:
        @return: (df)
        """
        # filter based on time. if time is after optimization has started, then go ahead.
        # Otherwise, just return the raw data
        if t_seconds < (8 * 3600):
            return self.live_data
        if self.do_behavioral_opt is not True:
            # print("dont do beh opt")
            return self.live_data
        # optimal_si is
        try:
            assert self.optimal_si is not None
        except AssertionError:
            self.optimal_si
            raise AssertionError

        info_adjustments = self.optimal_si[zone_id]
        # print(f"zonal info requested for zone {zone_id}")
        # print(f"driver_information count is {driver_information_count}")
        # print(f"info adjustment is {info_adjustments}")
        # logger.info("####################")
        # logger.info(f"zonal info requested for zone {zone_id}")
        # logger.info(f"driver_information count is {driver_information_count}")
        # logger.info(f"info adjustment is {info_adjustments}")
        # logger.info("####################")

        # info_adjustments is {(dest, driver_id): adj}
        try:
            driver_adjustments = {k[0]: v for k, v in info_adjustments.items() if k[1] == driver_information_count}
        except:
            # print("get_zonal_info_for_veh")
            # print(driver_information_count)
            # print(info_adjustments)

            # if len(driver_adjustments) == 0:
            # all optimized information has already been given. return the raw data
            # logger.info(f"all optimized information has already been given. return the raw data at time {t_seconds} ")
            # num_expected_drivers = len(info_adjustments) / 66 # emm, what??
            # logger.info(f'num_expected_drivers {num_expected_drivers}')
            # logger.info(f'driver_information_count:  {driver_information_count}')
            # logger.info(f'difference is {num_expected_drivers - driver_information_count}')
            # print('all optimized information has already been given. return the raw data')
            return self.live_data

        # logger.info(f"driver adjustments is {driver_adjustments}")

        data_t = self.live_data.copy(deep=True)
        for k, v in driver_adjustments.items():
            data_t.at[k, 'total_pickup'] = data_t.at[k, 'total_pickup'] * v
            # should log the before/after pickups
        return data_t

    def get_true_zonal_info(self, t_sec):
        """

        @param t_sec:
        @return:
        """

        def _true_zonal_info_over_t(t_15):
            """
            Returns the correct zone demand.
            @param t: time (15 min index)
            @return: (df) zonal demand over t
            """
            # The below does not fill missing time periods per zone, i.e., the length is not fixed s
            df = self.demand_fare_stats_of_the_day[self.demand_fare_stats_of_the_day["time_of_day_index_15m"] == t_15]
            df = df.assign(surge=1)
            df = df.assign(bonus=0)
            df = df.assign(match_prob=1)
            df.index = df.PULocationID.values  # this  must be done for information manipulation to work
            df = df.sort_index()
            self.live_data = df
            return df

        fifteen_min = int(np.floor(t_sec / 900))
        _true_zonal_info_over_t(fifteen_min)
        assert self.live_data is not None
        # reset driver info for the next 15 mins
        return self.live_data

    def update_zone_policy(self, t, zones, WARMUP_PHASE):
        """
        This is meant to be called with the main simulation.
        It automatically sets pricing policies for each zone.
        e.g., surge pricing
        @param t:
        @param zones:
        @param WARMUP_PHASE:
        @return:
        """
        if t % POLICY_UPDATE_INTERVAL == 0:
            budget_left = False
            if self.budget > 0:
                budget_left = True
            for z in zones:
                ratio = len(z.demand) / (
                        len(z.idle_vehicles) + len(z.incoming_vehicles) + 1
                )
                if len(z.demand) > MIN_DEMAND:
                    m = self.surge_step_function(ratio)
                    if not WARMUP_PHASE:
                        z.surge = m
                        if m > 1:
                            z.num_surge += 1
                    if budget_left:
                        bonus = self.set_bonus(ratio)
                        if not WARMUP_PHASE:
                            z.bonus = bonus
                            print("bonus of ", bonus, " for zone ", z.id)

                else:
                    z.surge = 1  # resets surge
                    z.bonus = 0  # reset bonus
                if not budget_left:
                    z.bonus = 0

    def set_surge_multipliers_for_zones(self, t, zones, target_zone_ids, surge):
        """
        self.zones, coming from model. NOT USED!
        @param t:
        @param zones:
        @param target_zone_ids:
        @param surge:
        @return:
        """
        df = self.get_true_zonal_info(t)

        for zid in target_zone_ids:
            df.loc[df["Origin"] == zid, "surge"] = surge
            for zone in zones:
                if zone.id == zid:
                    zone.surge = surge

    def set_bonus_for_zones(self, t, zones, target_zone_ids, bonus):
        """
        Sets bonus for the zones
        self.zones, coming from model
        @param t: time
        @param zones: list of zones
        @param target_zone_ids: list of target zone ids (int)
        @param bonus: (float)
        """
        df = self.get_true_zonal_info(t)

        for zid in target_zone_ids:
            df.loc[df["Origin"] == zid, "bonus"] = bonus
            for zone in zones:
                if zone.id == zid:
                    zone.bonus = bonus

    def disseminated_zonal_demand_info(self, t):
        """
        Drivers will use this function to access the demand data.
        #this can be potentially updated to include supply as well. An Uber driver told me that he would switch to pax mode
        # and see how many cars were around, to get a sense of what would be the odds of getting a match
        """

        # 1) define a func that does the optimization
        # 2) store that info in a variable (driver specific, dict of dict?)
        # 3) this function serves two purposes
        #   3.1) filter the info based on the scenario
        #   3.2) persist that info for the whole 15 mins
        #   3.3) communicate it to drivers whenever called
        if self.scenario in (3, 4):
            # run optimization
            self.optimize_rebalancing()
            self.optimize_driver_information()
        if self.scenario in (1, 3):
            # only demand info
            pass
        pass

    @lru_cache(maxsize=None)
    def expected_fare_total_demand_per_zone_over_days(self, driver_type):
        """
        A professional driver will query this one time per (hour) to use as prior
        for naive drivers, it has a constant mean and std for every zone
        @param driver_type:
        @param t: time
        @return: (df) demand fare prior dataframe for the given time
        """

        # df = self.demand_fare_stats_prior.query("time_of_day_index_15m == {t_15}".format(t_15=t))
        # key: zone_id, time
        # value: avg_fare, std_fare
        if driver_type == DriverType.PROFESSIONAL:
            df = self.demand_fare_stats_prior_peak_off_peak_dict
        else:
            df = self.demand_fare_stats_prior_peak_off_peak_dict_naive
        return df

    @lru_cache(maxsize=None)
    def expected_matching_per_zone_over_days(self, driver_type):
        """

        @param driver_type:
        @return: {(zone_id, t_string):(m, n_obs)}
        """

        return self.matching_stats_prior_dict

    def get_optimal_si(self, optimal_si):
        """
        receives the optimal si as computed by the behavioral optimization unit
        it is used by  this to adjust demand info, wh
        @param optimal_si:
        @return:
        """
        self.optimal_si = optimal_si

    def bookkeep_one_days_revenue(self, day, month):
        if self.revenue_report_dict['total_day_earning'] is None:
            # first time
            self.revenue_report_dict['total_day_earning'] = [np.sum(self.revenues)]
            self.revenue_report_dict['day'] = [day]
            self.revenue_report_dict['month'] = [month]
        else:
            self.revenue_report_dict['total_day_earning'].extend([np.sum(self.revenues)])
            self.revenue_report_dict['day'].extend([day])
            self.revenue_report_dict['month'].extend([month])
            self.revenue_report_dict['name'].extend([self.name])

    def report_final_revenue(self):
        df = pd.DataFrame(data=self.revenue_report_dict)
        return df
