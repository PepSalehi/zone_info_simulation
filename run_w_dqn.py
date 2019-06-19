#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:22:13 2019

@author: peyman
"""
# https://realpython.com/python-application-layouts/
import numpy as np 
import pandas as pd 
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

def main():
    
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
						help='Percent knowing fare, formatted as comma-separated list (i.e. "-k 1,1.5,2") ')
    parser.add_argument('-p', '--pro', 
						help='Percent pro drivers, formatted as comma-separated list (i.e. "-p 1,1.5,2") ')
    parser.add_argument('-av', '--AV', 
						help='Percent AV drivers, formatted as comma-separated list (i.e. "-av 1,1.5,2") ')  
    parser.add_argument('-r', '--replications',
                        help='number of times to run the simulation'   ) 
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
    
    if args.AV:
        # surge = args.multiplier
        perc_AV = [float(x) for x in args.AV.split(',')]
    else:
        perc_AV = [0] 

    if args.replications:
        n_rep = int(args.replications)
    else:
        n_rep = 1 
    if args.nb:
        nb_steps = int(args.nb)
    else:
        nb_steps = 3000
       
    for fleet_size in fleet_sizes:
        for surge in surges: 
            for perc_k in perc_know:
                for pro_s in pro_share:
                    for perc_av in perc_AV: 
                        for repl in range(n_rep):
                            print('Fleet size is {f}'.format(f=fleet_size))
                            print('Surge is {}'.format(surge))
                            print('Percentage knowing fares is {}'.format(perc_k))
                            print('Percentage of professional drivers {}'.format(pro_s))
                            print('Percentage of AV drivers {}'.format(perc_av))


                            config = {
                                "fleet_size" : fleet_size,
                                "surge" : surge,
                                "perc_k" : perc_k ,
                                "bonus" : 0,
                                "pro_s" : pro_s,
                                "percent_false_demand" : 0
                                }

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

                            dqn.load_weights('dqn_weights_%s.h5f' % (nb_steps))
                            
                        


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
                            

                            report.to_csv(output_path + "report for fleet size " + str(fleet_size) + " surge " +
                                    str(ss)+ "fdemand= "+ str(percent_false_demand)+
                                        "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + " perc_av " + str(perc_av)  + " repl"+ str(repl) +  ".csv")
                        
                            # pickle.dump( m, open( output_path + "model for fleet size " + str(fleet_size) + " surge "+ str(ss)
                            #         + "fdemand "+ str(percent_false_demand)+ "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + " repl"+ str(repl) + ".p", "wb" ) )
                            # report.to_csv(output_path + "report for fleet size " + str(fleet_size) + " surge "+
                            # str(ss)+ "fdemand= "+ str(percent_false_demand)+
                            #     "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + " perc_av " + str(perc_av) + ".csv")
                            # it seems like RL causes problems for pickling, remove it  
                            # m.RL_engine = None 

                            

                            # pickle.dump( np.sum(m.operator.revenues), open( output_path + "Operator_rev for fleet size " + str(fleet_size) + " surge "+ str(ss)
                            # + "fdemand "+ str(percent_false_demand)+ "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + " perc_av " + str(perc_av) + ".p", "wb" ) )

                            # # pickle.dump( m.operator, open( output_path + "Operator for fleet size " + str(fleet_size) + " surge "+ str(ss)
                            # # + "fdemand "+ str(percent_false_demand)+ "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + " perc_av " + str(perc_av) + ".p", "wb" ) )
                            
                            drivers_fares = [np.sum(v.collected_fares) for v in m.vehilcs]

                            pickle.dump( drivers_fares, open( output_path + "drivers_fares for fleet size " + str(fleet_size) + " surge "+ str(ss)
                            + "fdemand "+ str(percent_false_demand)+ "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + " perc_av " + str(perc_av) + " repl"+ str(repl)+ ".p", "wb" ) )



                            # pickle.dump( m, open( output_path + "model for fleet size " + str(fleet_size) + " surge "+ str(ss)
                            # + "fdemand "+ str(percent_false_demand)+ "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + " perc_av " + str(perc_av) + ".p", "wb" ) )
                        
        
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    