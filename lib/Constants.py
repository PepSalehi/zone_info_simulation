from functools import lru_cache
import numpy as np
import pandas as pd
import json

# zone neighbors dict
# keys are strings
with open("./Data/zones_neighbors.json", "r") as f:
    zones_neighbors = json.load(f)

# distance matrix
DIST_MAT = pd.read_csv("./Data/dist_mat_2.csv")
# I know, from Taxi_data_new_iteration.ipynb, that zone 105 does not exist in the demand files.
# remove it
DIST_MAT.drop(DIST_MAT[DIST_MAT.DOLocationID == 105].index, inplace=True)

# temp = pd.read_csv( "./Data/zones.csv", header=None, names=["zone_id"])
zone_ids_file = pd.read_csv("./Data/zones_w_neighbors.csv")
ZONE_IDS = zone_ids_file.LocationID.values

ZONE_IDS = list(set(ZONE_IDS).intersection(DIST_MAT.PULocationID.unique()))
print("The number of zones is ", len(ZONE_IDS))
print(" is zone 202 in the list?:", (202 in ZONE_IDS))
# print( (202 in ZONE_IDS))

# FNAME = 'raw_transactions_with_time_index.csv'      #"daily_demand_day_2.csv"
# day_of_run = 2
# DEMAND_SOURCE = pd.read_csv("./Data/{demand}".format(demand=FNAME))
# DEMAND_SOURCE = filter_data_to_day(day_of_run, FNAME)
# print("The number of requests over all days is  ", DEMAND_SOURCE.shape)

DIST_MAT = DIST_MAT.set_index(['PULocationID', 'DOLocationID'])


class MyDist:
    def __init__(self, dist):
        self.dist = dist
        print("Initiated MyDist")

    @classmethod
    @lru_cache(maxsize=None)
    def return_distance(cls, origin, destination):
        return DIST_MAT.loc[origin, destination]["trip_distance_meter"]

    @classmethod
    @lru_cache(maxsize=None)
    def return_distance_from_origin_to_all(cls, origin):
        return DIST_MAT.loc[origin]
        # return DIST_MAT.query("PULocationID=={o}".format(o=ozone))


my_dist_class = MyDist(DIST_MAT)
# bin demand into 15 min periods,

# Operators commission
PHI = 0.25

# fleet size and its breakdown

FLEET_SIZE = [1500]
PRO_SHARE = 0
PERCENT_FALSE_DEMAND = 0.0  # given wrong demand info

AV_SHARE = 0

PENALTY = 0
PERCE_KNOW = 0  # percentage of drivers that know the avg fare
CONST_FARE = 6  # fare used when they don't know the true avg fare
SURGE_MULTIPLIER = 2
BONUS = 0
MAX_BONUS = 3
# constant speed
CONSTANT_SPEED = 8  # meters per second

# initial wait time and detour factor when starting the interaction
INI_WAIT = 400
INI_DETOUR = 1.25

MAX_IDLE = 5 * 60
# intervals for vehicle-request assignment and rebalancing
INT_ASSIGN = 30
INT_REBL = 150
# fuel cost
# https://www.marketwatch.com/story/heres-how-much-uber-drivers-really-make-2017-12-01
FUEL_COST = 0.033 * 0.01  # dollors per meter. roughly 54 cents/mile

# times is extremely ugly, but works
ANALYSIS_TIME_HOUR = 8  # 8-9 am
ANALYSIS_TIME_SECONDS = ANALYSIS_TIME_HOUR * 3600
WARMUP_TIME_HOUR = 7  # used for setting up the demand 8am
WARMUP_TIME_SECONDS = WARMUP_TIME_HOUR * 3600

DEMAND_UPDATE_INTERVAL = 3600  # seconds -> TODO: make it 900
POLICY_UPDATE_INTERVAL = 10 * 60  # 10 minutes
MIN_DEMAND = 40  # min demand to have surge
ANALYSIS_DURATION = 4 * 3600  # hours
# warm-up time, study time and cool-down time of the simulation (in seconds)
# start_time_offset + 1 hour warm up + 1 hour analysis
T_TOTAL_SECONDS = WARMUP_TIME_SECONDS + 3600 + ANALYSIS_DURATION
# T_WARM_UP = 60*30
# T_STUDY = 60*60
# T_COOL_DOWN = 60*30
# T_TOTAL = (T_WARM_UP + T_STUDY + T_COOL_DOWN)
if __name__ == "main":
    pass
