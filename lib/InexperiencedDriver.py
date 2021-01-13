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
from lib.ProfessionalDriver import ProfessionalDriver

from lib.Requests import Req
from lib.Vehicles import Veh, DriverType, VehState, _convect_time_to_peak_string, _choice, _convert_reporting_dict_to_df
from functools import lru_cache
from enum import Enum, unique, auto
import pickle

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('drivers_inexperienced.log', mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)


class InexperiencedDriver(ProfessionalDriver):
    """
    This acts as a naive one in the beginning, but as it gains more experience starts behaving like the pro
    """

    def __init__(self, rs, operator, beta=1,  driver_type=DriverType.INEXPERIENCED,
                 ini_loc=None, know_fare=False,
                  dist_mat=DIST_MAT):
        """
        Creates a Vehicle object.

        @param rs: #
        @param operator (Operator): object describing operator-specific details (e.g. Uber)
        @param beta (float):
        @param true_demand (bool): #
        @param driver_type (enum): whether the driver is professional, naive, or inexperienced
        @param ini_loc (int): initial location (zone) of the driver
        @param know_fare (bool): whether the driver knows the fare

        @param dist_mat:
        """
        super().__init__(rs,
                         operator, beta, driver_type, ini_loc, know_fare,  dist_mat)

    def _compute_attractiveness_of_zones(self, t, ozone, true_demand):
        """
        @param t: time
        @param ozone: (int) current zone
        @param true_demand: (bool)
        @return: (df) attractiveness to all zones and (df) probability to go to each zone
        """
        # 1)  get demand and distances
        dist = self._get_dist_to_all_zones(ozone)
        # 1.1) demand as told by the app
        df = (self.get_data_from_operator(t))  # .set_index('Origin')
        assert dist.shape[0] == df.shape[0]
        # 1.2) demand as expected from experience
        # PRO: get estimates based on the prior
        # naive: solely base decision on the app's info
        # inexperienced: the first day, act naively. Then start to act like a PRO, with inferior estimates of course

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
        # 4) compute the expected revenue
        expected_revenue = (1 - PHI) * fare_to_use * df.surge.values  + df.bonus.values
        # 5) compute the expected cost
        expected_cost = (
                dist.trip_distance_meter.values * self.unit_rebalancing_cost)  # doesn't take into account the distance travelled once the demand is picked up
        # 6) compute the expected profit
        # https://github.com/numpy/numpy/issues/14281
        prof = np.core.umath.clip(np.exp(expected_revenue - expected_cost) * df["total_pickup"].values, 0, 10000)
        # prof = np.clip((expected_revenue - expected_cost) * df["total_pickup"].values, a_min=0, a_max=None)
        # 7) compute the probability of moving to each zone
        # http://cs231n.github.io/linear-classify/#softmax
        prob = prof / prof.sum()
        # return a.index.get_level_values(1).values, prob
        return df["PULocationID"], prob
        # return a, prob, a["Origin"]
