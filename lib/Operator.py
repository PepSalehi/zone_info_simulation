import numpy as np
import pandas as pd
from lib.Constants import (
    ZONE_IDS,
    DEMAND_UPDATE_INTERVAL,
    POLICY_UPDATE_INTERVAL,
    MIN_DEMAND,
    SURGE_MULTIPLIER,
    BONUS,
)


class Operator:
    def __init__(
        self,
        report,
        which_day_numerical=2,
        name="Uber",
        BONUS=BONUS,
        SURGE_MULTIPLIER=SURGE_MULTIPLIER,
    ):
        self.name = name
        self.demand_fare_stats_prior = pd.read_csv(
            "./Data/df_hourly_stats_over_days.csv"
        )
        self.demand_fare_stats_of_the_day = pd.read_csv(
            "./Data/df_hourly_stats.csv"
        ).query("Day=={d}".format(d=which_day_numerical))
        self.live_data = None
        self.revenues = []
        # these should be probably enums, and accessed via functions
        self.SHOULD_SURGE = True
        self.SHOULD_BONUS = False
        self.SHOULD_LIE_DEMAND = False
        self.SURGE_MULTIPLIER = SURGE_MULTIPLIER
        self.BONUS = BONUS

        self.report = report

    def surge_step_function(self, ratio):
        """
        Calculates the surge charge based on an assumed step-wise function 
        0.9-1 : 1.2
        1-1.2 : 1.5
        1.2-2: 2
        >2: 3
    
        """
        if ratio < 0.9:
            return 1
        if ratio <= 1 and ratio >= 0.9:
            return 1.2
        if ratio <= 1.2 and ratio > 1:
            return 1.5
        if ratio < 2 and ratio > 1.2:
            return 2
        else:
            return 3

    def true_zonal_info_over_t(self, t):
        """
        return the correct demand 
        """
        df = self.demand_fare_stats_of_the_day.query("Hour == {hour}".format(hour=t))
        df = df.assign(surge=1)
        df = df.assign(bonus=0)
        df = df.assign(match_prob=1)
        # df = df.assign(match_prob=df['total_pickup']/60)  # pax/min just the default
        # df = df.assign(match_prob=df['total_pickup']/df.total_pickup.sum())  # pax/min just the default
        if self.report is not None:
            # get the avg # of drivers per zone per price
            df = pd.merge(df, self.report, left_on="Origin", right_on="zone_id")
        self.live_data = df
        return df

    def false_zonal_info_over_t(self, t):
        """ 
        
        """
        False_mult = 3
        zone_ids = np.loadtxt("outputs/zones_los_less_50_f_2500.csv")
        df = self.demand_fare_stats_of_the_day.query("Hour == {hour}".format(hour=t))
        #
        df.loc[df["Origin"].isin(zone_ids), "total_pickup"] = (
            df[df["Origin"].isin(zone_ids)]["total_pickup"] * False_mult
        )
        df.loc[~df["Origin"].isin(zone_ids), "total_pickup"] = (
            df[~df["Origin"].isin(zone_ids)]["total_pickup"] / False_mult
        )
        df = df.assign(surge=1)
        df = df.assign(bonus=0)
        df = df.assign(match_prob=df["total_pickup"] / 60)  # pax/min just the default
        #        df = df.assign(match_prob=df['total_pickup']/df.total_pickup.sum())  # pax/min just the default

        self.live_data_false = df
        return df

    def update_zonal_info(self, t):

        if t % DEMAND_UPDATE_INTERVAL == 0:
            self.get_zonal_info(t)

    def zonal_info_for_veh(self, true_demand):
        if true_demand:
            return self.live_data
        else:
            return self.live_data_false

    def get_zonal_info(self, t):
        
        hour = int(np.floor(t / 3600))
        self.true_zonal_info_over_t(hour)
        self.false_zonal_info_over_t(hour)
        assert self.live_data is not None
        return self.live_data

    def update_zone_policy(self, t, zones, WARMUP_PHASE):
        """
        This is meant to be called with the main simulation. 
        It automatically sets pricing policies for each zone.
        e.g., surge pricing
        """
        if t % POLICY_UPDATE_INTERVAL == 0:
            for z in zones:
                ratio = len(z.demand) / (
                    len(z.idle_vehicles) + len(z.incoming_vehicles) + 1
                )
                if len(z.demand) > MIN_DEMAND:
                    m = self.surge_step_function(ratio)
                    z.surge = m
                    if not WARMUP_PHASE and m >= 1.2:
                        z.num_surge += 1
                    # print ("Zone {z} is currently surging at t = {t} with ratio {r} and surge of {s} !".format(z =z.id, t=t, r = ratio, s = m ))
                else:
                    z.surge = 1  # resets surge

    def set_surge_multipliers_for_zones(self, t, zones, target_zone_ids, surge):
        """
        self.zones, coming from model. NOT USED!
        """
        df = self.get_zonal_info(t)

        for zid in target_zone_ids:
            df.loc[df["Origin"] == zid, "surge"] = surge
            for zone in zones:
                if zone.id == zid:
                    zone.surge = surge

    def set_bonus_for_zones(self, t, zones, target_zone_ids, bonus):
        """
        self.zones, coming from model. 
        """
        df = self.get_zonal_info(t)

        for zid in target_zone_ids:
            df.loc[df["Origin"] == zid, "bonus"] = bonus
            for zone in zones:
                if zone.id == zid:
                    zone.bonus = bonus

    def dissiminate_zonal_demand_info(self, t, tell_truth=True):
        """
        Drivers will use this function to access the demand data. 
        #TODO this can be potentially updated to include supply as well. An Uber driver told me that he would switch to pax mode
        # and see how many cars were around, to get a sense of what would be the odds of getting a match  
        """

        if tell_truth:
            self.true_zonal_info(t)

    def expected_fare_totaldemand_per_zone_over_days(self, t):

        """ 
        A professional driver will query this one time per (hour) to use as prior
        
        """

        df = self.demand_fare_stats_prior.query("Hour == {hour}".format(hour=t))
        return df

