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
from lib.Constants import ZONE_IDS, DEMAND_SOURCE, INT_ASSIGN, FLEET_SIZE, PRO_SHARE, SURGE_MULTIPLIER, BONUS, PERCENT_FALSE_DEMAND
from lib.Constants import T_TOTAL_SECONDS, WARMUP_TIME_SECONDS, ANALYSIS_TIME_SECONDS, ANALYSIS_TIME_HOUR, WARMUP_TIME_HOUR
from lib.Constants import PERCE_KNOW
output_path = "./Outputs/"


def main():
    


    fleet_sizes = [2500]
    surges = [2]
    perc_know = [1]
    pro_share = [1]
    percent_false_demand = 0
    bonus = 0 
    

        
       
    for fleet_size in fleet_sizes:
        for surge in surges: 
            for perc_k in perc_know:
                for pro_s in pro_share:
                    print('Fleet size is {f}'.format(f=fleet_size))
                    print('Surge is {}'.format(surge))
                    print('Percentage knowing fares is {}'.format(perc_k))
                    print('Percentage of professional drivers {}'.format(pro_s))
        
                    m = Model(ZONE_IDS, DEMAND_SOURCE, WARMUP_TIME_HOUR, ANALYSIS_TIME_HOUR, FLEET_SIZE=fleet_size, PRO_SHARE=pro_s,
                            SURGE_MULTIPLIER=surge, BONUS=bonus, percent_false_demand=percent_false_demand, percentage_know_fare = perc_k)
                    
                    # start time
                    stime = time.time()
                    
                    # # dispatch the system for T_TOTAL seconds, at the interval of INT_ASSIGN
                    for T in range(WARMUP_TIME_SECONDS, T_TOTAL_SECONDS, INT_ASSIGN):
                        if T % (25200 + 3600) == 0 :
                            pass 
                    
                        
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
                    
                    report.to_csv(output_path + "report for fleet size " + str(fleet_size) + " surge "+
                                  str(ss)+ "fdemand= "+ str(percent_false_demand)+
                                     "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + ".csv")
                    
                    pickle.dump( m, open( output_path + "model for fleet size " + str(fleet_size) + " surge "+ str(ss)
                    + "fdemand "+ str(percent_false_demand)+ "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + ".p", "wb" ) )
                
        
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    