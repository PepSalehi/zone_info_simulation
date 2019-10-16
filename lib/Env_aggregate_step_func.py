import gym
from gym import spaces
import numpy as np
import copy

import argparse
import time
import pickle
from lib.utils import Model
from lib.Constants import (
    ZONE_IDS,
    DEMAND_SOURCE,
    INT_ASSIGN,
    FLEET_SIZE,
    PRO_SHARE,
    SURGE_MULTIPLIER,
    BONUS,
    PERCENT_FALSE_DEMAND,
)
from lib.Constants import (
    T_TOTAL_SECONDS,
    WARMUP_TIME_SECONDS,
    ANALYSIS_TIME_SECONDS,
    ANALYSIS_TIME_HOUR,
    WARMUP_TIME_HOUR,
)
from lib.Constants import PERCE_KNOW, INT_REBL

output_path = "./Outputs/"


class RebalancingEnv(gym.Env):
    """
    RebalancingEnv is the environment class for DQN
    Attributes:
        model: AMoD system to train
        dT: time interval for training
        penalty: penalty of rebalancing a vehicle
        action_space: action space
        state: the system state. It's (ui, vi, cik) for every zone, where cik is the cost of going to i. e.g., 67 zones -> 67  * 3.
        center: the centroid of cells
        input_dim: input dimension
    """

    def __init__(self, config):
        super(gym.Env, self).__init__()
        print("INSIDE INIT FUNCTION")
        print(config["av_share"])
        self.config = config
        self.model = Model(
            ZONE_IDS,
            DEMAND_SOURCE,
            WARMUP_TIME_HOUR,
            ANALYSIS_TIME_HOUR,
            FLEET_SIZE=config["fleet_size"],
            PRO_SHARE=config["pro_s"],
            SURGE_MULTIPLIER=config["surge"],
            bonus=config["bonus"],
            percent_false_demand=config["percent_false_demand"],
            percentage_know_fare=config["perc_k"],
            AV_share=config["av_share"],
        )

        self.dT = INT_REBL
        self.action_space = spaces.Discrete(len(ZONE_IDS))
        # why not define an observation space?
        self.state = np.zeros((len(ZONE_IDS), 3))
        self.observation_space = np.zeros((len(ZONE_IDS), 3))

        # self.center = np.zeros((Mlng, Mlat, 2))
        self.input_dim = 3 * len(ZONE_IDS)
        self.step_count = 0
        self.epi_count = 0
        self.total_reward = 0.0
        self.T = WARMUP_TIME_SECONDS
        self.old_income = 0

    def step(self, actions):
        """ 
        actions: a vector of length N_AV, which contains the target zone for idle veh, 
        and inaction for busy ones
        impelements action, returns new state, reward. 
        Currently the DQN is inside the model.dispatch_at_time function 
        """
        # print("Inside Step")
        # print("Step count: ", self.step_count)
        # print("T: ", self.T)
        flag = False
        self.step_count += 1
        for i, veh in enumerate([v for v in self.model.av_vehs]):
            # if the veh has to move, then move it
            if not np.isnan(actions[i]):
                veh.set_action(actions[i])

        # move the world forward
        self.model.dispatch_at_time(self.T)
        self.T = self.T + INT_ASSIGN
        # print("end T: ", self.T)

        state_n = []
        for i, veh in enumerate([v for v in self.model.av_vehs]):
            state_n.append(self.model.get_state(veh, self.T))

        # total_new_income = np.sum(veh.profits) - self.old_income
        # self.old_income = np.sum(veh.profits)
        # # normalize the reward.
        # # from previous runs, avg revenue is 35 with std of 5
        # # (base on Nuts and bolts of DRL)
        # normalized_income = (total_new_income ) #/10
        # reward = normalized_income
        # print("reward")
        # print(reward)
        # total_new_income = np.sum(model.operator.revenues) - self.old_income
        # self.old_income = np.sum(model.operator.revenues)
        # reward += total_new_income
        # report = self.model.get_service_rate_per_zone()
        # system_LOS = report.served.sum()/report.total.sum()
        # reward += system_LOS
        # self.T = self.T+INT_ASSIGN

        print("T_TOTAL_SECONDS", T_TOTAL_SECONDS)
        print("self.T", self.T)

        if self.T >= T_TOTAL_SECONDS:
            flag = True
            print("Episode is done!")

        return state_n, None, flag, {}

    def reset(self):
        print("Calling the reset method! ")
        self.model = Model(
            ZONE_IDS,
            DEMAND_SOURCE,
            WARMUP_TIME_HOUR,
            ANALYSIS_TIME_HOUR,
            FLEET_SIZE=self.config["fleet_size"],
            PRO_SHARE=self.config["pro_s"],
            SURGE_MULTIPLIER=self.config["surge"],
            bonus=self.config["bonus"],
            percent_false_demand=self.config["percent_false_demand"],
            percentage_know_fare=self.config["perc_k"],
            AV_share=self.config["av_share"],
        )

        self.total_reward = 0.0
        self.T = WARMUP_TIME_SECONDS
        self.old_income = 0

        state_n = []
        for _, veh in enumerate([v for v in self.model.av_vehs]):
            state_n.append(self.model.get_state(veh, self.T))

        return state_n

