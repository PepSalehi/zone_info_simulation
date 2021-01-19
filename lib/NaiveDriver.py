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
from lib.Vehicles import Veh, DriverType, VehState, _convect_time_to_peak_string, _choice, _convert_reporting_dict_to_df
from functools import lru_cache
from enum import Enum, unique, auto
import pickle

import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')



class NaiveDriver(Veh):
    """
    Class encapsulating a vehicle.
    """

    def __init__(self, rs, operator, day_of_run, output_path, beta=1, driver_type=DriverType.NAIVE,
                 ini_loc=None,
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
        super().__init__(rs,
                         operator, day_of_run, output_path, beta,  driver_type, ini_loc, dist_mat)

        # fh2 = logging.FileHandler(output_path + 'drivers_naive.log', mode='w')
        # fh2.setFormatter(formatter)
        # logger.addHandler(fh2)

    def _compute_attractiveness_of_zones(self, t, ozone):
        """
        TODO: inspecting the log, a consequence of considering every zone is that the individual probabilities are very
        low. Would make sense to limit the decision to just the neighboring zones.
        two possible issues:
        1. new bugs
        2. much slower performance
        @param t: time in seconds
        @param ozone: (int) current zone
        @return: (df) attractiveness to all zones and (df) probability to go to each zone
        """
        # 1)  get demand and distances
        # index is zone, trip_distance_meter is the only column
        dist = self._get_dist_to_all_zones(ozone)
        # 1.1) demand as told by the app
        df = (self.get_data_from_operator(t))  # .set_index('Origin')
        assert dist.shape[0] == df.shape[0]

        if df.empty:
            print(
                "corner case: take zone 127. there and no demand is left, df and therefore a will be empty. "
                "in this situation, it should just move to one of its neighbors"
            )
            print("ozone", self.ozone)
            # print("destination", neighbors_list[0])
            neighbors_list = self._get_neighboring_zone_ids(ozone)
            return neighbors_list[0]

        # 1.2) demand as expected from experience
        # naive: solely base decision on the app's info

        fare_to_use = CONST_FARE  # they should use the app's info, if it's given
        # 4) compute the expected revenue

        expected_revenue = (1 - PHI) * fare_to_use * df.surge.values + df.bonus.values
        # 5) compute the expected cost
        # is an array
        expected_cost = (
                dist.trip_distance_meter.values * self.unit_rebalancing_cost)  # doesn't take into account the distance travelled once the demand is picked up
        # 6) compute the expected profit
        # https://github.com/numpy/numpy/issues/14281
        # maybe this total pickup is an issue. it shows current and anticipated demand, instead of just current
        # prof = np.core.umath.clip(np.exp(df["total_pickup"].values), 0, 10000000)
        # prof = (expected_revenue - expected_cost) * df["total_pickup"].values
        THETA = 0.02 # 0.1 is too high (concentrated on 1 or 2 zones), 0.01 is too low (everyone is served!)
        # prof = np.core.umath.clip(np.exp(THETA * (expected_revenue - expected_cost) * df["total_pickup"].values), 0, 1e55)
        prof = np.exp(THETA * (expected_revenue - expected_cost) * df["total_pickup"].values)


        # prof = np.clip((expected_revenue - expected_cost) * df["total_pickup"].values, a_min=0, a_max=None)
        # 7) compute the probability of moving to each zone
        # http://cs231n.github.io/linear-classify/#softmax
        prob = prof / prof.sum()
        # return a.index.get_level_values(1).values, prob
        return df["PULocationID"].values, prob, None
        # return a, prob, a["Origin"]
