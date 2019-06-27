import numpy as np
import pandas as pd
import json

# zone neighbors dict
# keys are strings
with open("./Data/zones_neighbors.json", "r") as f:
    zones_neighbors = json.load(f)

# distance matrix
DIST_MAT = pd.read_csv("./Data/dist_mat.csv")


# temp = pd.read_csv( "./Data/zones.csv", header=None, names=["zone_id"])
zone_ids_file = pd.read_csv("./Data/zones_w_neighbors.csv")
ZONE_IDS = zone_ids_file.LocationID.values

ZONE_IDS = list(set(ZONE_IDS).intersection(DIST_MAT.PULocationID.unique()))
print("The number of zones is ", len(ZONE_IDS))

FNAME = "daily_demand_day_2.csv"
DEMAND_SOURCE = pd.read_csv("./Data/{demand}".format(demand=FNAME))
print("The number of requests over all days is  ", DEMAND_SOURCE.shape)

# bin demand into 15 min periods,

# Operators commission
PHI = 0.25

# fleet size and its breakdown

FLEET_SIZE = [1500]
PRO_SHARE = 0
PERCENT_FALSE_DEMAND = 0.0  # given wrong demand info

AV_SHARE = 0

PERCE_KNOW = 0  # percentage of drivers that know the avg fare
CONST_FARE = 6  # fare used when they don't know the true avg fare
SURGE_MULTIPLIER = 2
BONUS = 0
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

DEMAND_UPDATE_INTERVAL = 3600  # seconds
POLICY_UPDATE_INTERVAL = 10 * 60  # 10 minutes
MIN_DEMAND = 40  # min demand to have surge
ANALYSIS_DURATION = 4 * 3600 # hours
# warm-up time, study time and cool-down time of the simulation (in seconds)
# start_time_offset + 1 hour warm up + 1 hour analysis
T_TOTAL_SECONDS = WARMUP_TIME_SECONDS + 3600 + ANALYSIS_DURATION
# T_WARM_UP = 60*30
# T_STUDY = 60*60
# T_COOL_DOWN = 60*30
# T_TOTAL = (T_WARM_UP + T_STUDY + T_COOL_DOWN)
if __name__ == "main":
    pass
