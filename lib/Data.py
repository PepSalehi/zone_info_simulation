import json

import numpy as np
import pandas as pd
from lib.Constants import DIST_MAT, ZONE_IDS


class Data:
    """
    Replaces Constants.py. Container for all constants, reference data that
    other modules will use.

    @todo: fix the paths to data. Not sure if the binning process is needed anymore. Most importantly, decouple this from Constants.py. Right now they are both handling demand and dist matrices.
    """

    def __init__(self,
                 path_zone_neighbors="./Data/zones_neighbors.json",
                 path_zones_w_neighbors="./Data/zones_w_neighbors.csv",
                 path_daily_demand="./Data/Daily_demand/",
                 day_of_run = 2,
                 bonus_policy="random",
                 budget=10000,
                 phi=0.25, fleet_size=None, pro_share=0,
                 percent_false_demand=0.0, av_share=0, penalty=0,
                 perce_know=0, const_fare=6, surge_multiplier=2, bonus=0, constant_speed=8,
                 ini_wait=400, ini_detour=1.25,
                 max_idle=300, int_assign=30, int_rebl=150,
                 fuel_cost=0.033 * 0.01, analysis_time_hour=8, warmup_time_hour=7,
                 demand_update_interval=3600, policy_update_interval=10 * 60,
                 min_demand=40, analysis_duration=4 * 3600,
                 t='15Min'
                 ):
        """
        Creates Data instance.
        @param path_zone_neighbors: path to zone neighbors file
        @param path_dist_mat: path to distance matrix file
        @param path_zones_w_neighbors: path to zone w neighbors file
        @param path_daily_demand: path to daily demand file
        @param phi: (float) operator commission
        @param fleet_size: (list of int) fleet size and breakdown #TODO: of what?
        @param pro_share: (float) proportion of pro drivers
        @param percent_false_demand: (float)
        @param av_share: (float) proportion of AVs
        @param penalty: (float)
        @param perce_know: (float) proportion of fare-aware vehicles
        @param const_fare: (float)
        @param surge_multiplier: (float)
        @param bonus: (float)
        @param constant_speed: (float) in meters per second
        @param ini_wait: (float) initial wait time when starting the interaction
        @param ini_detour: (float) initial detour factor when starting the interaction
        @param max_idle:
        @param int_assign: (int) interval for vehicle-request assignment
        @param int_rebl: (int) interval for rebalancing
        @param fuel_cost: (float) dollars/meter
            https://www.marketwatch.com/story/heres-how-much-uber-drivers-really-make-2017-12-01
        @param analysis_time_hour: (int) e.g. 8 (am)
        @param warmup_time_hour: (int)
        @param demand_update_interval: (int) in seconds
        @param policy_update_interval: (int) in seconds
        @param min_demand: (float) min demand to have surge
        @param analysis_duration: (int) in seconds
        @param t: (int) number of minutes for demand binning
        """

        # Zone neighbors dict, keys are strings
        self.day_of_run = day_of_run

        self.BUDGET = budget
        self.FLEET_SIZE = None
        if fleet_size is None:
            self.FLEET_SIZE = [1500]
        with open(path_zone_neighbors, 'r') as f:
            self.ZONES_NEIGHBORS = json.load(f)

        # Get zone ids
        # zone_ids_file = pd.read_csv(path_zones_w_neighbors)
        # self.ZONES_IDS = zone_ids_file.LocationID.values
        # self.ZONES_IDS = list(set(self.ZONES_IDS).intersection(DIST_MAT.PULocationID.unique()))
        # print("The number of zones is ", len(self.ZONES_IDS))

        # Get demand source

        # https://stackoverflow.com/questions/13651117/how-can-i-filter-lines-on-load-in-pandas-read-csv-function
        def __filter_data_to_day(day, fname):
            iter_csv = pd.read_csv(fname + "demand_for_day_{}.csv".format(day), iterator=True, chunksize=1000)
            df = pd.concat([chunk[chunk['Day'] == day] for chunk in iter_csv])
            return df

        # "daily_demand_day_2.csv"

        self.DEMAND_SOURCE = __filter_data_to_day(self.day_of_run, path_daily_demand)
        # self.DEMAND_SOURCE = pd.read_csv(path_daily_demand)
        print("The number of requests over all days is  ", self.DEMAND_SOURCE.shape)

        # Bins demand into 15 minute periods, populates variable
        self.BINNED_DEMAND = None
        self.BINNED_OD = None
        # self.bin_demand(t)

        # SET OTHER CONSTANTS
        # Operators commission
        self.PHI = phi

        # fleet size and its breakdown

        self.PRO_SHARE = pro_share
        self.PERCENT_FALSE_DEMAND = percent_false_demand  # given wrong demand info

        self.AV_SHARE = av_share

        self.PENALTY = penalty
        self.PERCE_KNOW = perce_know  # percentage of drivers that know the avg fare
        self.CONST_FARE = const_fare  # fare used when they don't know the true avg fare
        self.SURGE_MULTIPLIER = surge_multiplier
        self.BONUS = bonus
        self.BONUS_POLICY = bonus_policy
        self.CONSTANT_SPEED = constant_speed  # meters per second

        # initial wait time and detour factor when starting the interaction
        self.INI_WAIT = ini_wait
        self.INI_DETOUR = ini_detour
        self.MAX_IDLE = max_idle

        # intervals for vehicle-request assignment and rebalancing
        self.INT_ASSIGN = int_assign
        self.INT_REBL = int_rebl

        self.FUEL_COST = fuel_cost  # dollors per meter. roughly 54 cents/mile

        self.ANALYSIS_TIME_HOUR = analysis_time_hour
        self.ANALYSIS_TIME_SECONDS = self.ANALYSIS_TIME_HOUR * 3600
        self.WARMUP_TIME_HOUR = warmup_time_hour  # used for setting up the demand 8am
        self.WARMUP_TIME_SECONDS = self.WARMUP_TIME_HOUR * 3600

        self.DEMAND_UPDATE_INTERVAL = demand_update_interval  # seconds
        self.POLICY_UPDATE_INTERVAL = policy_update_interval  # 10 minutes
        self.MIN_DEMAND = min_demand  # min demand to have surge
        self.ANALYSIS_DURATION = analysis_duration  # hours

        # warm-up time, study time and cool-down time of the simulation (in seconds)
        # start_time_offset + 1 hour warm up + 1 hour analysis
        self.T_TOTAL_SECONDS = self.WARMUP_TIME_SECONDS + 3600 + self.ANALYSIS_DURATION

        # TODO: what to do with this?
        # T_WARM_UP = 60*30
        # T_STUDY = 60*60
        # T_COOL_DOWN = 60*30
        # T_TOTAL = (T_WARM_UP + T_STUDY + T_COOL_DOWN)

    def bin_demand(self, bin_width='15min'):
        """
        Counts the number of pickups/dropoffs per time interval, sectioned by zone.

        Args:
            bin_width (str): e.g. '5T'== 5 minutes, '1H' == 1 hour

        Returns:
            pandas df (indices: datetime, locationID. columns: frequency)
        """
        times = 'tpep_pickup_datetime'
        locationID = 'PULocationID'
        if '15' in bin_width:
            bin_size = 15
        else:
            bin_size = 60
        self.DEMAND_SOURCE[times] = pd.to_datetime(self.DEMAND_SOURCE[times])
        # Keep relevant columns
        df = self.DEMAND_SOURCE[[times, locationID, 'DOLocationID', 'passenger_count']]
        # df[times] = pd.to_datetime(df[times])
        df.set_index(times, inplace=True)
        # Resample and summarize
        pickups_df_binned = pd.DataFrame(
            df.groupby(locationID).resample(
                bin_width)['passenger_count'].sum())
        pickups_df_binned.reset_index(inplace=True)
        pickups_df_binned.columns = ['PULocationID', 'times', 'passenger_count']
        pickups_df_binned[
            "total_seconds"] = (pickups_df_binned.times.dt.hour * 3600 +
                                pickups_df_binned.times.dt.minute * 60 +
                                pickups_df_binned.times.dt.second)

        OD_df_binned = pd.DataFrame(
            df.groupby([locationID, "DOLocationID"]).resample(
                '15T')['passenger_count'].sum()
        )
        OD_df_binned.reset_index(inplace=True)
        OD_df_binned.columns = ['PULocationID', 'DOLocationID', 'times', 'passenger_count']
        OD_df_binned[
            "total_seconds"] = (OD_df_binned.times.dt.hour * 3600 +
                                OD_df_binned.times.dt.minute * 60 +
                                OD_df_binned.times.dt.second)
        # raw data
        self.DEMAND_SOURCE = self.DEMAND_SOURCE.rename(columns={times: 'times'})

        self.DEMAND_SOURCE[
            "total_seconds"] = (self.DEMAND_SOURCE.times.dt.hour * 3600 +
                                self.DEMAND_SOURCE.times.dt.minute * 60 +
                                self.DEMAND_SOURCE.times.dt.second)

        self.DEMAND_SOURCE["time_interval"] = np.floor(self.DEMAND_SOURCE["total_seconds"] / (bin_size * 60))
        self.BINNED_DEMAND = pickups_df_binned
        self.BINNED_OD = OD_df_binned
