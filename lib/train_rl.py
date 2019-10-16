#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:22:13 2019

@author: peyman
"""
# https://realpython.com/python-application-layouts/
import argparse
import time
import pickle
from lib.utils import Model
from lib.Constants import ZONE_IDS, DEMAND_SOURCE, INT_ASSIGN, FLEET_SIZE, PRO_SHARE, SURGE_MULTIPLIER, BONUS, \
    PERCENT_FALSE_DEMAND
from lib.Constants import T_TOTAL_SECONDS, WARMUP_TIME_SECONDS, ANALYSIS_TIME_SECONDS, ANALYSIS_TIME_HOUR, \
    WARMUP_TIME_HOUR
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


def main():
    """
    Parses command line arguments, sets training environment parameters, creates deep Q-network and trains it
    on gym environment.
    """
    parser = argparse.ArgumentParser(description="Simulation of drivers' behavior")
    parser.add_argument('-f', '--fleet',
                        help='Fleet sizes to simulate, formatted as comma-separated list (i.e. "-f 250,275,300")')
    parser.add_argument('-m', '--multiplier',
                        help='Surge multiplier, formatted as comma-separated list (i.e. "-m 1,1.5,2")')
    parser.add_argument('-b', '--bonus', type=int,
                        help='Bonus')
    parser.add_argument('-d', '--demand',
                        help='Percent false demand ')
    parser.add_argument('-k', '--know',
                        help='Percent knowing fare, formatted as comma-separated list (i.e. "-m 1,1.5,2") ')
    parser.add_argument('-p', '--pro',
                        help='Percent pro drivers, formatted as comma-separated list (i.e. "-p 1,1.5,2") ')
    parser.add_argument('-av', '--av',
                        help='Percent AV drivers, formatted as comma-separated list (i.e. "-av 1,1.5,2") ')
    parser.add_argument('-nb', '--nb',
                        help='number of steps to train Rl ')

    args = parser.parse_args()
    if args.fleet:
        fleet_sizes = [int(x) for x in args.fleet.split(',')]
    #        fleet_sizes = args.fleet
    else:
        fleet_sizes = FLEET_SIZE

    if args.multiplier:
        # surge = args.multiplier
        surges = [float(x) for x in args.multiplier.split(',')]
    else:
        surges = [SURGE_MULTIPLIER]

    if args.know:
        # surge = args.multiplier
        perc_know = [float(x) for x in args.know.split(',')]
    else:
        perc_know = [PERCE_KNOW]

    if args.bonus:
        bonus = args.bonus
    else:
        bonus = BONUS

    if args.pro:

        pro_share = [float(x) for x in args.pro.split(',')]
    else:
        pro_share = [PRO_SHARE]

    if args.demand:
        percent_false_demand = float(args.demand)
    else:
        percent_false_demand = PERCENT_FALSE_DEMAND

    if args.av:
        av_share = [float(x) for x in args.av.split(',')]
    else:
        av_share = [1]
    if args.nb:
        nb_steps = args.nb
    else:
        nb_steps = 300

    for fleet_size in fleet_sizes:
        for surge in surges:
            for perc_k in perc_know:
                for pro_s in pro_share:
                    m = Model(ZONE_IDS, DEMAND_SOURCE, WARMUP_TIME_HOUR, ANALYSIS_TIME_HOUR, fleet_size=fleet_size,
                              pro_share=pro_s,
                              surge_multiplier=surge, bonus=bonus, percent_false_demand=percent_false_demand,
                              percentage_know_fare=perc_k)

                    # make one veh to be AV 
                    veh = m.vehilcs[-1]
                    veh.is_AV = True
                    # 
                    env = RebalancingEnv(m, penalty=-0)

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
                                   target_model_update=1e-2, policy=policy, gamma=.99)
                    dqn.compile(Adam(lr=0.001, epsilon=0.05, decay=0.0), metrics=['mae'])

                    dqn.fit(env, nb_steps=nb_steps, action_repetition=1, visualize=False, verbose=2)
                    dqn.save_weights('new_dqn_weights_%s.h5f' % (nb_steps), overwrite=True)


if __name__ == "__main__":
    main()
