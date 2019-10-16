import pandas as pd
import json


class Data:
    """
    Replaces Constants.py. Container for all constants, reference data that
    other modules will use.
    """
    def __init__(self,
                 path_zone_neighbors, path_dist_mat, path_zones_w_neighbors,
                 path_daily_demand, phi=0.25, fleet_size=[1500], pro_share=0,
                 percent_false_demand=0.0, av_share=0, penalty=0,
                 perce_know=0, const_fare=6, surge_multiplier=2, bonus=0, constant_speed=8,
                 ini_wait=400, ini_detour=1.25,
                 max_idle=300, int_assign=30, int_rebl=150,
                 fuel_cost=0.033*0.01, analysis_time_hour=8, warmup_time_hour=7,
                 demand_update_interval=3600, policy_update_interval=10*60,
                 min_demand=40, analysis_duration=4*3600,
                 t=15
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
        # SET ZONE INFORMATION
        # Zone neighbors dict, keys are strings
        with open(path_zone_neighbors, 'r') as f:
            self.ZONES_NEIGHBORS = json.load(f)

        # Distance matrix
        self.DIST_MAT = pd.read_csv(path_dist_mat)

        # Get zone ids
        zone_ids_file = pd.read_csv(path_zones_w_neighbors)
        self.ZONES_IDS = zone_ids_file.LocationID.values
        self.ZONES_IDS = list(set(self.ZONES_IDS).intersection(self.DIST_MAT.PULocationID.unique()))
        print("The number of zones is ", len(self.ZONES_IDS))

        # Get demand source
        self.DEMAND_SOURCE = pd.read_csv(path_daily_demand)
        print("The number of requests over all days is  ", self.DEMAND_SOURCE.shape)

        # Bins demand into 15 minute periods, populates variable
        self.BINNED_DEMAND = None
        self.bin_demand(t)

        # SET OTHER CONSTANTS
        # Operators commission
        self.PHI = phi

        # fleet size and its breakdown
        self.FLEET_SIZE = fleet_size
        self.PRO_SHARE = pro_share
        self.PERCENT_FALSE_DEMAND = percent_false_demand  # given wrong demand info

        self.AV_SHARE = av_share

        self.PENALTY = penalty
        self.PERCE_KNOW = perce_know # percentage of drivers that know the avg fare
        self.CONST_FARE = const_fare  # fare used when they don't know the true avg fare
        self.SURGE_MULTIPLIER = surge_multiplier
        self.BONUS = bonus
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

        self.DEMAND_UPDATE_INTERVAL = demand_update_interval # seconds
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

    def bin_demand(self, bin_width='15T'):
        """
        Counts the number of pickups/dropoffs per time interval, sectioned by zone.

        Args:
            bin_width (str): e.g. '5T'== 5 minutes, '1H' == 1 hour

        Returns:
            pandas df (indices: datetime, locationID. columns: frequency)
        """
        times = 'tpep_pickup_datetime'
        locationID = 'PULocationID'

        # Keep relevant columns
        pickups_df = self.DEMAND_SOURCE[[times, locationID, 'passenger_count']]
        pickups_df.set_index(times, inplace=True)

        # Resample and summarize
        result = pd.DataFrame(
            pickups_df.groupby(locationID).resample(
                bin_width)['passenger_count'].sum())

        result.index.names = ['locationID', 'times']

        self.BINNED_DEMAND = result
