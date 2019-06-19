import gym 
from gym import spaces
import numpy as np 
import copy

import argparse
import time 
import pickle 
from lib.utils import Model 
from lib.Constants import ZONE_IDS, DEMAND_SOURCE, INT_ASSIGN, FLEET_SIZE, PRO_SHARE, SURGE_MULTIPLIER, BONUS, PERCENT_FALSE_DEMAND
from lib.Constants import T_TOTAL_SECONDS, WARMUP_TIME_SECONDS, ANALYSIS_TIME_SECONDS, ANALYSIS_TIME_HOUR, WARMUP_TIME_HOUR
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
    def __init__(self, config, penalty=-10):
        print("INSIDE INIT FUNCTION")
        self.config = config
        self.model = Model(ZONE_IDS, DEMAND_SOURCE, WARMUP_TIME_HOUR, ANALYSIS_TIME_HOUR, 
            FLEET_SIZE=config["fleet_size"],
            PRO_SHARE=config["pro_s"],
            SURGE_MULTIPLIER=config["surge"], 
            BONUS=config["bonus"], 
            percent_false_demand=config["percent_false_demand"],
            percentage_know_fare = config["perc_k"])
        
        veh  = self.model.vehilcs[-1]
        veh.is_AV = True
  
        
        self.dT = INT_REBL
        self.penalty = penalty
        self.action_space = spaces.Discrete(len(ZONE_IDS))
        # why not define an observation space?
        self.state = np.zeros((len(ZONE_IDS),3 ))
        # self.center = np.zeros((Mlng, Mlat, 2))
        self.input_dim = 3 * len(ZONE_IDS)
        self.step_count = 0
        self.epi_count = 0
        self.total_reward = 0.0
        self.T = WARMUP_TIME_SECONDS
        self.old_income = 0
    

    def step(self, action):
        ''' 
        action: a vector of length N_AV, which contains the target zone for idle veh, and inaction for busy ones
        impelements action, returns new state, reward. 
        Currently the DQN is inside the model.dispatch_at_time function 
        '''
        flag = False 
        self.step_count += 1
        reward = 0 
        # AV
        veh = self.model.vehilcs[-1]
        # As long as a decision for AV is not needed, keep simulating
        while not veh.should_move():
            T = self.T
            T_ = self.T+INT_ASSIGN
            # dispatch the system for INT_ASSIGN seconds
            while T < T_:
                print("veh is currently")
                print(str(veh.ozone))
                print(str(veh.rebalancing))
                if veh.rebalancing:
                    print("time_to_be_available")
                    print(veh.time_to_be_available)
                self.model.dispatch_at_time(T, self.penalty)
                T += INT_ASSIGN
            self.T = self.T+INT_ASSIGN
        #check and see if the AV is ready to move. If not, keep simulating 
        print("AV should move ")
        T = self.T
        T_ = self.T+INT_ASSIGN
        # move it 
        while T < T_:
            self.model.dispatch_at_time(T, self.penalty, action)
            T += INT_ASSIGN
        self.T = self.T + INT_ASSIGN
        # calculate the reward of that action
        total_new_income = np.sum(veh.profits) - self.old_income 
        self.old_income = np.sum(veh.profits)
        # normalize the reward. 
        # from previous runs, avg revenue is 35 with std of 5
        # (base on Nuts and bolts of DRL)
        normalized_income = (total_new_income - 35) /10
        reward += normalized_income
        # total_new_income = np.sum(model.operator.revenues) - self.old_income 
        # self.old_income = np.sum(model.operator.revenues)
        # reward += total_new_income
        # report = self.model.get_service_rate_per_zone()
        # system_LOS = report.served.sum()/report.total.sum()
        # reward += system_LOS 
        # self.T = self.T+INT_ASSIGN
        self.update_state()

        print("T_TOTAL_SECONDS",T_TOTAL_SECONDS)
        print("self.T", self.T)

        if self.T >= T_TOTAL_SECONDS: 
            flag = True 
            print("Episode is done!")
        return self.state, reward, flag, {}

    def update_state(self, vid=-1):
        veh = self.model.vehilcs[vid]
        self.state = self.model.get_state(veh)

 
    # def act(self, action, vid=-1):
    #     veh = self.model.vehilcs[vid]
    #     assert veh.should_move()
    #     veh.move(model.zones, WARMUP_PHASE, action)

        

    
    def reset(self):
        print("Calling the reset method! ")
        self.model = Model(ZONE_IDS, DEMAND_SOURCE, WARMUP_TIME_HOUR, ANALYSIS_TIME_HOUR, 
            FLEET_SIZE=self.config["fleet_size"],
            PRO_SHARE=self.config["pro_s"],
            SURGE_MULTIPLIER=self.config["surge"], BONUS=self.config["bonus"], 
            percent_false_demand=self.config["percent_false_demand"], percentage_know_fare = self.config["perc_k"])

        veh  = self.model.vehilcs[-1]
        veh.is_AV = True
        self.total_reward = 0.0
        self.T = WARMUP_TIME_SECONDS
        self.old_income = 0

        self.update_state()
        # self.amods.append( copy.deepcopy(self.amod) )
        return self.state
    
