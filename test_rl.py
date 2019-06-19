#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:22:13 2019

@author: peyman
"""
# https://realpython.com/python-application-layouts/
import sys
import os
# current_path = os.path.dirname(os.getcwd()+"/Simulation/")
# sys.path.append(current_path)

import argparse
import time 
import pickle 
from lib.utils_with_rel_engine import Model 
from lib.Constants import ZONE_IDS, DEMAND_SOURCE, INT_ASSIGN, FLEET_SIZE, PRO_SHARE, SURGE_MULTIPLIER, BONUS, PERCENT_FALSE_DEMAND
from lib.Constants import T_TOTAL_SECONDS, WARMUP_TIME_SECONDS, ANALYSIS_TIME_SECONDS, ANALYSIS_TIME_HOUR, WARMUP_TIME_HOUR
from lib.Constants import PERCE_KNOW
from lib.Env import RebalancingEnv
output_path = "./Outputs/RL/"


import gym
from gym import spaces
import copy
import itertools
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


# from importlib import reload  # Python 3.4+ only.
# from lib.Env import RebalancingEnv
    
fleet_size = FLEET_SIZE[0]
surge = 2
perc_k = 1
bonus = 0
pro_s = 0
percent_false_demand = 0



config = {
"fleet_size" : FLEET_SIZE[0],
"surge" : 2,
"perc_k" : 1,
"bonus" : 0,
"pro_s" : 0,
"percent_false_demand" : 0
}
                
# m = Model(ZONE_IDS, DEMAND_SOURCE, WARMUP_TIME_HOUR, ANALYSIS_TIME_HOUR, FLEET_SIZE=fleet_size, PRO_SHARE=pro_s,
#         SURGE_MULTIPLIER=surge, BONUS=bonus, percent_false_demand=percent_false_demand, percentage_know_fare = perc_k)

# make one veh to be AV 
# veh  = m.vehilcs[-1]
# veh.is_AV = True
# 
# env = RebalancingEnv(m, penalty=-10, config=config )
env = RebalancingEnv(penalty=-10, config=config )
nb_actions = env.action_space.n
input_shape = (1,) + env.state.shape
input_dim = env.input_dim
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
memory = SequentialMemory(limit=2000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                target_model_update=1e-2, policy=policy, gamma=0.99)
dqn.compile(Adam(lr=0.001, epsilon=0.05, decay=0.0), metrics=['mae'])

# history = dqn.fit(env, nb_steps=100, action_repetition=1, visualize=False, verbose=2)

# dqn.save_weights('dqn_weights_%s.h5f' % (100), overwrite=True)

dqn.load_weights('dqn_weights_%s.h5f' % (3000))




# for perc_av in percent_av:


perc_av = 1 
print('Fleet size is {f}'.format(f=fleet_size))
print('Surge is {}'.format(surge))
print('Percentage knowing fares is {}'.format(perc_k))
print('Percentage of professional drivers {}'.format(pro_s))

m = Model(ZONE_IDS, DEMAND_SOURCE, WARMUP_TIME_HOUR, ANALYSIS_TIME_HOUR, FLEET_SIZE=fleet_size, PRO_SHARE=pro_s,
        SURGE_MULTIPLIER=surge, BONUS=bonus, percent_false_demand=percent_false_demand, percentage_know_fare = perc_k,
            RL_engine=dqn, AV_share=perc_av)

# start time
stime = time.time()
# # dispatch the system for T_TOTAL seconds, at the interval of INT_ASSIGN
for T in range(WARMUP_TIME_SECONDS, T_TOTAL_SECONDS, INT_ASSIGN):
    m.dispatch_at_time(T)

# end time
etime = time.time()
# run time of this simulation
runtime = etime - stime
print ("The run time was {runtime} minutes ".format(runtime = runtime/60))

report = m.get_service_rate_per_zone()

# So that it doesn't save a file with 1.5.py, rather 15.py
ss = str(surge).split('.')
ss = ''.join(ss)
    
    # report.to_csv(output_path + "report for fleet size " + str(fleet_size) + " surge "+
    #                 str(ss)+ "fdemand= "+ str(percent_false_demand)+
    #                     "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + ".csv")
    
    # pickle.dump( m, open( output_path + "model for fleet size " + str(fleet_size) + " surge "+ str(ss)
    # + "fdemand "+ str(percent_false_demand)+ "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + ".p", "wb" ) )





    
    
    
    
    
    
    
    