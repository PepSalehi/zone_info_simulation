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
from lib.configs import configs
from functools import lru_cache
from enum import Enum, unique, auto
import pickle

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('drivers_AV.log')
fh.setFormatter(formatter)
logger.addHandler(fh)




class AvDriver(Veh):
    """
    Class encapsulating a vehicle.
    """
    def __init__(self, rs, operator, beta=1, true_demand=True, driver_type=DriverType.AV,
                 ini_loc=None, know_fare=False,
                 is_AV=False, dist_mat=DIST_MAT):
        """
        Creates a Vehicle object.

        @param rs: # TODO: what is this
        @param operator (Operator): object describing operator-specific details (e.g. Uber)
        @param beta (float):
        @param true_demand (bool): # TODO: what is this
        @param driver_type (enum): whether the driver is professional, naive, or inexperienced
        @param ini_loc (int): initial location (zone) of the driver
        @param know_fare (bool): whether the driver knows the fare
        @param is_AV (bool)
        @param dist_mat:
        """
        super().__init__(rs,
                         operator, beta, true_demand, driver_type, ini_loc, know_fare, is_AV, dist_mat)


