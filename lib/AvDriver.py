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

    def __init__(self, rs, operator, day_of_run, output_path, beta=1, driver_type=DriverType.AV,
                 ini_loc=None,
                 dist_mat=DIST_MAT):
        """
        Creates a Vehicle object.

        @param rs:
        @param operator (Operator): object describing operator-specific details (e.g. Uber)
        @param beta (float):
        @param driver_type (enum): whether the driver is professional, naive, or inexperienced
        @param ini_loc (int): initial location (zone) of the driver
        @param know_fare (bool): whether the driver knows the fare
        @param is_AV (bool)
        @param dist_mat:
        """
        super().__init__(rs, operator, day_of_run, output_path, beta, driver_type, ini_loc, dist_mat)
        self.action = None
        if ini_loc is None:
            self.ozone = rs.choice(ZONE_IDS)
            self.locations.append(self.ozone)
            self._state = VehState.DECISION
            # self.ozone.incoming_vehicles.append(self)

    def rebalance(self, zones, target_zone):
        dist = None
        for z in zones:
            if z.id == target_zone:
                self._state = VehState.REBAL
                self.state_hist.append(self._state)
                # self.rebalancing = True
                # self.idle = False
                # self.TIME_TO_MAKE_A_DECISION  = False
                self.time_to_be_available = self._get_time_to_destination(
                    self.ozone, target_zone
                )
                self.tba.append(self.time_to_be_available)
                dist = self._get_distance_to_destination(self.ozone, target_zone)
                z.join_incoming_vehicles(self)
                self.zone = z
                self.ozone = z.id  # added Nov 27 2020, I think it's correct but maybe I missed sth
                break
        if dist is None:
            raise Exception('dist was None')

    # def set_action(self, action):
    #     """
    #     Use the RL agent to decide on the target.
    #     """
    #     assert action is not None
    #     assert action in ZONE_IDS, f"action {action} is not a zone"
    #     assert self._state == VehState.IDLE, f"AV's state is {self._state}, and not IDLE"
    #     f"AV's state is {self._state}, and not IDLE"
    #     self.action = int(action)
    #     # print("action is", action)

    def act(self, t, zones, WARMUP_PHASE):
        """

        catch all method called by the model. Should act accordingly
        @param t:
        @param Zones:
        @param WARMUP_PHASE:
        @return:
        """
        # self.available_for_assignment()
        #
        # if self.waited_too_long(t):
        #     self._state = VehState.DECISION
        # elif self.should_move():
        #     _ = self.move(t, zones)
        #     return
        if self.is_busy:
            self.keep_serving()
            return
        elif self.is_waiting_to_be_matched:
            # it's sitting somewhere
            self.keep_waiting()
            return
        elif self.is_rebalancing:  # and not self.busy:
            self.update_rebalancing(WARMUP_PHASE)
            return


