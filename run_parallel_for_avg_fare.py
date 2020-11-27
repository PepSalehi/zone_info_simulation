#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:22:13 2019

@author: peyman
"""
# https://realpython.com/python-application-layouts/
import argparse
import os
import pickle
import time
import datetime
from lib.Data import Data
# from lib.Constants import ZONE_IDS, DEMAND_SOURCE, INT_ASSIGN, FLEET_SIZE, PRO_SHARE, SURGE_MULTIPLIER, BONUS, \
#     PERCENT_FALSE_DEMAND
# from lib.Constants import T_TOTAL_SECONDS, WARMUP_TIME_SECONDS, ANALYSIS_TIME_SECONDS, ANALYSIS_TIME_HOUR, \
#     WARMUP_TIME_HOUR
# from lib.Constants import PERCE_KNOW
from lib.Vehicles import DriverType
from lib.configs import configs_dict
from lib.utils import Model


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
                        help='Percent knowing fare, formatted as comma-separated list (i.e. "-m 1,1.5,2") ')
    parser.add_argument('-p', '--pro',
                        help='Percent pro drivers, formatted as comma-separated list (i.e. "-m 1,1.5,2") ')
    parser.add_argument('-r', '--replications',
                        help='number of times to run the simulation')
    parser.add_argument('-bb', '--beta',
                        help='BETA')
    parser.add_argument('-b_policy', '--bonus_policy',
                        help='bonus per zone ')
    parser.add_argument('-budget', '--budget',
                        help='budget ')
    args = parser.parse_args()
    # TODO: argpars should get the bonus policy as input
    data_instance = Data()
    if args.fleet:
        fleet_sizes = [int(args.fleet)]
    else:
        fleet_sizes = data_instance.FLEET_SIZE

    if args.multiplier:
        # surge = args.multiplier
        surges = [float(x) for x in args.multiplier.split(',')]
    else:
        surges = [data_instance.SURGE_MULTIPLIER]

    if args.know:
        perc_know = [float(args.know)]
    else:
        perc_know = [data_instance.PERCE_KNOW]

    if args.bonus:
        bonus = args.bonus
    else:
        bonus = data_instance.BONUS
    if args.beta:
        beta = float(args.beta)
    else:
        beta = configs_dict["BETA"]

    if args.pro:
        pro_share = [float(x) for x in args.pro.split(',')]
    else:
        pro_share = [data_instance.PRO_SHARE]

    if args.demand:
        percent_false_demand = float(args.demand)
    else:
        percent_false_demand = data_instance.PERCENT_FALSE_DEMAND
    if args.replications:
        n_rep = int(args.replications)
    else:
        n_rep = 1
    if args.bonus_policy:
        bonus_policy = args.bonus_policy
    else:
        bonus_policy = data_instance.BONUS_POLICY
    if args.budget:
        budget = args.budget
    else:
        budget = data_instance.BUDGET
    # output_path = "./Outputs/avg_fare_info/" + str(beta) + "/"


    for fleet_size in fleet_sizes:
        for surge in surges:
            for perc_k in perc_know:
                for pro_s in pro_share:
                    for repl in range(n_rep):
                        output_path = "./Outputs/avg_fare_info/" + str(budget) + "_" + str(bonus_policy) + "_" + \
                                      str(datetime.datetime.now()).split('.')[0] + "/"
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)
                        print("iteration number ", repl)
                        print('Fleet size is {f}'.format(f=fleet_size))
                        print('Surge is {}'.format(surge))
                        print('Percentage knowing fares is {}'.format(perc_k))
                        print('Percentage of professional drivers {}'.format(pro_s))

                        data_instance.FLEET_SIZE = fleet_size
                        data_instance.PRO_SHARE = pro_s
                        data_instance.SURGE_MULTIPLIER = surge
                        data_instance.BONUS = bonus
                        data_instance.PERCENT_FALSE_DEMAND = percent_false_demand
                        data_instance.PERCE_KNOW = perc_k

                        m = Model(data_instance, configs_dict, beta)

                        # start time
                        stime = time.time()

                        # # dispatch the system for T_TOTAL seconds, at the interval of INT_ASSIGN
                        for T in range(data_instance.WARMUP_TIME_SECONDS,
                                       data_instance.T_TOTAL_SECONDS,
                                       data_instance.INT_ASSIGN):
                            m.dispatch_at_time(T)
                        print('Total drivers: ', len(m.vehicles))
                        print('# of Pro drivers: ', len([v for v in m.vehicles if v.driver_type== DriverType.PROFESSIONAL]))
                        print('# of naive drivers: ',
                              len([v for v in m.vehicles if v.driver_type == DriverType.NAIVE]))
                        print('# of inexperienced drivers: ',
                              len([v for v in m.vehicles if v.driver_type == DriverType.INEXPERIENCED]))
                        # end time
                        etime = time.time()
                        # run time of this simulation
                        runtime = etime - stime
                        print("The run time was {runtime} minutes ".format(runtime=runtime / 60))

                        report = m.get_service_rate_per_zone()

                        # So that it doesn't save a file with 1.5.py, rather 15.py
                        ss = str(surge).split('.')
                        ss = ''.join(ss)

                        report.to_csv(output_path + "report for fleet size " + str(fleet_size) + " surge " +
                                      str(ss) + "fdemand= " + str(percent_false_demand) +
                                      "perc_k " + str(perc_k) + "pro_s " + str(pro_s) + " repl" + str(repl) + ".csv")

                        # pickle.dump(m,
                        #             open(output_path + "model for fleet size " + str(fleet_size) + " surge " + str(ss)
                        #                  + "fdemand " + str(percent_false_demand) + "perc_k " + str(
                        #                 perc_k) + "pro_s " + str(pro_s) + " repl" + str(repl) + ".p", "wb"))
                        #

if __name__ == "__main__":
    main()
