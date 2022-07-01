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
    my_dist_class,
THETA

)

from lib.Requests import Req
from lib.Vehicles import Veh, DriverType, VehState, _convect_time_to_peak_string, _choice, _convert_reporting_dict_to_df
from functools import lru_cache
from enum import Enum, unique, auto
import pickle

# import logging
#
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#

class NaiveDriver(Veh):
    """
    Class encapsulating a vehicle.
    """

    def __init__(self, rs, operator, day_of_run, output_path, beta=1, driver_type=DriverType.NAIVE,
                 ini_loc=None,
                 theta=THETA,
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
                         operator, day_of_run, output_path, beta, driver_type, ini_loc, dist_mat)
        self.theta = theta
        # fh = logging.FileHandler(output_path + 'naive_experiences.log', mode='a')
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # fh.setFormatter(formatter)
        # fh2 = logging.FileHandler(output_path + 'drivers_naive.log', mode='w')
        # fh2.setFormatter(formatter)
        # logger.addHandler(fh)

    # @profile
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
        # dist_dict = self._get_dist_to_all_zones_as_a_dict(ozone)
        # 1.1) demand as told by the app
        df = (self.get_data_from_operator(t))  # .set_index('Origin')
        # df = df.sort_index()
        assert dist.shape[0] == df.shape[0]
        # this merge operation is responsible for almost 8 times increase in running time

        # df = pd.merge(df, dist, left_index=True, right_index=True) # extremely expensive
        if df.empty:
            print(
                "corner case: take zone 127. there and no demand is left, df and therefore a will be empty. "
                "in this situation, it should just move to one of its neighbors"
            )
            print("ozone", self.ozone)
            # print("destination", neighbors_list[0])
            neighbors_list = self._get_neighboring_zone_ids(ozone)
            return neighbors_list[0]

        fare_to_use = CONST_FARE  # they should use the app's info, if it's given
        # total_demand = df["total_pickup"].sum()
        # expected_cost_dict = {}
        # expected_revenue_dict = {}
        # expected_profit_dict = {}
        # for dest in df["PULocationID"].values:
        #     expected_cost_dict[dest] = dist_dict[dest]['trip_distance_meter'] * self.unit_rebalancing_cost
        #     expected_revenue_dict[dest] = (1 - PHI) * fare_to_use * df[df.index.isin([dest])]['surge'].values[0] + df[df.index.isin([dest])]['bonus'].values[0]
        #     expected_profit_dict[dest] = THETA*((expected_revenue_dict[dest] - expected_cost_dict[dest] ) * df[df.index.isin([dest])]["total_pickup"].values[0] ) / total_demand
        # C = np.max(list(expected_profit_dict.values()))
        # z = np.array([np.exp(expected_profit_dict[dest] - C) for dest in df["PULocationID"].values])
        # prob = z / z.sum()
        # # return a.index.get_level_values(1).values, prob
        # return df["PULocationID"].values, prob, expected_profit_dict

        # 4) compute the expected revenue

        expected_revenue = (1 - PHI) * fare_to_use * df.surge.values + df.bonus.values
        # df.loc[:, "expected_cost"] = df.apply(lambda r: expected_cost_dict[r.PULocationID], axis=1)

        # 5) compute the expected cost
        # is an array
        # How do we know this "dist" corresponds the to the same zones/ has the same ordering?
        expected_cost = (
                dist.trip_distance_meter.values * self.unit_rebalancing_cost)  # doesn't take into account the distance travelled once the demand is picked up
        # 6) compute the expected profit
        # https://github.com/numpy/numpy/issues/14281
        # maybe this total pickup is an issue. it shows current and anticipated demand, instead of just current
        # THETA =  0.02 # 0.1 is too high (concentrated on 1 or 2 zones), 0.01 is too low (everyone is served!)
        # prof = np.core.umath.clip(np.exp(THETA * (expected_revenue - expected_cost) * df["total_pickup"].values), 0, 1e55)
        # normalized_demand = df["total_pickup"].values / (np.sum(df["total_pickup"].values) + 0.001)
        #####
        # Softmax is NOT scale invariant. This parameter \beta needs to be fitted using DCM.
        #####
        prof = self.theta*((expected_revenue - expected_cost) * df["total_pickup"].values) #/ df["total_pickup"].sum()
        # # See https://github.com/dlsys10714/notebooks/blob/main/8_nn_library_implementation.ipynb , numerical stability
        x = prof - np.max(prof) # this results in 0-1 prob for this problem
        z = np.exp(x)        #
        # # 7) compute the probability of moving to each zone
        # # http://cs231n.github.io/linear-classify/#softmax
        prob = z / z.sum()
        # # return a.index.get_level_values(1).values, prob
        return df["PULocationID"].values, prob, prof
        # return a, prob, a["Origin"]

    def record_daily_all_lr_rates(self, d_idx, month):
        pass
