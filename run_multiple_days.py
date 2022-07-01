#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:22:13 2019

@author: peyman
"""
# https://realpython.com/python-application-layouts/
import argparse
import logging

import numpy as np
import pandas as pd
import os
import pickle
import time
# import julia
# julia.install()
import datetime
from lib.Data import Data
# from lib.Constants import ZONE_IDS, DEMAND_SOURCE, INT_ASSIGN, FLEET_SIZE, PRO_SHARE, SURGE_MULTIPLIER, BONUS, \
#     PERCENT_FALSE_DEMAND
# from lib.Constants import T_TOTAL_SECONDS, WARMUP_TIME_SECONDS, ANALYSIS_TIME_SECONDS, ANALYSIS_TIME_HOUR, \
#     WARMUP_TIME_HOUR

from lib.Constants import THETA as theta
from lib.Constants import THETA_prof as theta_prof
from lib.Vehicles import DriverType
from lib.configs import configs_dict
from lib.utils import Model
import warnings
warnings.filterwarnings("ignore")



def main():
    parser = argparse.ArgumentParser(description="Simulation of drivers' behavior")
    # from lib.Constants import PERCE_KNOW
    parser.add_argument('-f', '--fleet',
                        help='Fleet sizes to simulate, formatted as comma-separated list (i.e. "-f 250,275,300")')
    parser.add_argument('-m', '--multiplier',
                        help='Surge multiplier, formatted as comma-separated list (i.e. "-m 1,1.5,2")')
    parser.add_argument('-b', '--bonus', type=int,
                        help='Bonus')
    parser.add_argument('-d', '--demand',
                        help='Percent false demand ')
    parser.add_argument('-AV', '--AV_fleet_size',
                        help="Number of Naive drivers ")
    parser.add_argument('-NAIVE', '--NAIVE_fleet_size',
                        help="Number of Naive drivers ")
    parser.add_argument('-PRO', '--PRO_fleet_size',
                        help="Number of Professional drivers ")
    parser.add_argument('-BH', '--behavioral_opt',
                        help="Perform behavioral optimization, pass 'yes' or 'no' ")
    parser.add_argument('-SURGE', '--surge_pricing',
                        help="should do surge pricing, pass 'yes' or 'no' ")

    parser.add_argument('-THETA', '--THETA',
                        help="choice param ")

    parser.add_argument('-THETA_prof', '--THETA_prof',
                        help="choice param ")

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
    parser.add_argument('-lb', '--LOWER_BOUND_SI',
                        help='LOWER_BOUND_SI ')
    parser.add_argument('-ub', '--UPPER_BOUND_SI',
                        help='UPPER_BOUND_SI ')
    parser.add_argument('-info', '--info_policy',
                        help='info_policy ')
    args = parser.parse_args()
    # TODO: argpars should get the bonus policy as input
    data_instance = Data()

    if args.LOWER_BOUND_SI:
        LOWER_BOUND_SI = args.LOWER_BOUND_SI
    else:
        LOWER_BOUND_SI = 0

    if args.UPPER_BOUND_SI:
        UPPER_BOUND_SI = args.UPPER_BOUND_SI
    else:
        UPPER_BOUND_SI = 2

    if args.info_policy:
        info_policy = args.info_policy
    else:
        info_policy = 'personalized'

    if args.fleet:
        fleet_sizes = [int(args.fleet)]
    else:
        fleet_sizes = data_instance.FLEET_SIZE

    if args.behavioral_opt:
        if args.behavioral_opt.lower() in ('yes', 'true'):
            do_behavioral_opt = True
        else:
            do_behavioral_opt = False
    else:
        do_behavioral_opt = data_instance.do_behavioral_opt

    if args.surge_pricing:
        if args.surge_pricing.lower() in ('yes', 'true'):
            do_surge_pricing = True
        else:
            do_surge_pricing = False
    else:
        do_surge_pricing = data_instance.do_surge_pricing

    if args.PRO_fleet_size:
        set_of_NUM_OF_PRO_DRIVERS = [int(args.PRO_fleet_size)]
    else:
        set_of_NUM_OF_PRO_DRIVERS = [data_instance.PRO_FLEET_SIZE]

    if args.NAIVE_fleet_size:
        set_of_NUM_OF_NAIVE_DRIVERS = [int(args.NAIVE_fleet_size)]
    else:
        set_of_NUM_OF_NAIVE_DRIVERS = [data_instance.NAIVE_FLEET_SIZE]

    if args.AV_fleet_size:
        set_of_NUM_OF_AV_DRIVERS = [int(args.AV_fleet_size)]
    else:
        set_of_NUM_OF_AV_DRIVERS = [data_instance.AV_FLEET_SIZE]

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
        beta =  (args.beta)
    else:
        beta = configs_dict["BETA"]

    if args.THETA:
        THETA = float(args.THETA)
        print("THETA: ", THETA)
    else:
        THETA = theta
        print("no change THETA: ", THETA)

    if args.THETA_prof:
        THETA_prof = float(args.THETA_prof)
        print("THETA_prof: ", THETA_prof)
    else:
        THETA_prof = theta_prof
        print("no change THETA_prof: ", THETA_prof)

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

    from lib.rebalancing_optimizer import RebalancingOpt

    for num_pros in set_of_NUM_OF_PRO_DRIVERS:
        for num_naives in set_of_NUM_OF_NAIVE_DRIVERS:
            for num_avs in set_of_NUM_OF_AV_DRIVERS:
                for surge in surges:
                    for repl in range(n_rep):
                        ## for some reason this jumps up by 1 day for each run. temp fix
                        data_instance = Data()
                        TOTAL_FLEET_SIZE = 4000
                        num_naives = TOTAL_FLEET_SIZE - num_pros
                        data_instance.AV_FLEET_SIZE = num_avs
                        data_instance.NAIVE_FLEET_SIZE = num_naives
                        data_instance.PRO_FLEET_SIZE = num_pros

                        # data_instance.do_behavioral_opt = do_behavioral_opt
                        # data_instance.do_surge_pricing = do_surge_pricing
                        do_surge_pricing = False
                        # do_behavioral_opt = True
                        print("######")
                        print ("do_behavioral_opt", do_behavioral_opt)
                        print("######")
                        if do_behavioral_opt:
                            st = '/with_behavioral_opt/'
                        elif do_surge_pricing:
                            st = '/with_surge_pricing/'
                        else:
                            st = '/no_intervention/'
                        output_path = "./Outputs/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + \
                                      st + str('Pro_') + str(num_pros) \
                                      + str('NAIVE_') + str(num_naives) \
                                      + str('AV_') + str(num_avs) \
                                      + str('THETA ') + str(THETA) \
                                      + str('THETA_prof ') + str(THETA_prof) \
                                      + str('budget_') + str(budget) + "_" \
                                      + str("bonus_policy") + "_" + str(bonus_policy) + "_" \
                                      + str('do_surge') + "_" + str(do_surge_pricing) + "_" \
                                      + str('do_opt') + "_" + str(do_behavioral_opt) + "_" \
                                      + str('UPPER_BOUND_SI') + "_" + str(UPPER_BOUND_SI) + "_" \
                                      + str('LOWER_BOUND_SI') + "_" + str(LOWER_BOUND_SI) + "_" \
                                      + str('info_policy') + "_" + str(info_policy) + "_" \
                                      + str(datetime.datetime.now()).split('.')[0] + "/"

                        if not os.path.exists(output_path):
                            os.makedirs(output_path)

                        print("iteration number ", repl)
                        print('Surge is {}'.format(surge))
                        print(f'lb is {LOWER_BOUND_SI}')
                        print(f'policy is {info_policy}')

                        data_instance.SURGE_MULTIPLIER = surge
                        data_instance.BONUS = bonus
                        data_instance.output_path = output_path
                        data_instance.LOWER_BOUND_SI = float(LOWER_BOUND_SI)
                        data_instance.UPPER_BOUND_SI = float(UPPER_BOUND_SI)
                        data_instance.info_policy = info_policy
                        data_instance.THETA = THETA
                        data_instance.THETA_prof = THETA_prof

                        # data_instance.do_behavioral_opt = False
                        m = Model(data_instance, configs_dict, beta, output_path)
                        # start time
                        stime = time.time()
                        # dispatch the system for T_TOTAL seconds, at the interval of INT_ASSIGN
                        # TODO: every run should in include the policy from the start
                        # TODO: process Feb's month as well.
                        months = [1, 2]
                        days = [30, 15]
                        # days = [30, 25]

                        # debug only
                        months = [1]
                        days = [10]
                        stop_month =  months[-1]
                        for ix, month in enumerate(months):
                            for d_idx in range(1, days[ix]):
                                # print(f"day index {d_idx}")
                                stop_day = days[ix]

                                if month == 1 and d_idx >= 2:
                                    # NOTE: THIS WILL NOT HAVE THE DESIRED EFFECT, BC OPERATOR has attribute set in the beginning
                                    if do_behavioral_opt:
                                        data_instance.do_behavioral_opt = True
                                        m.operator.do_behavioral_opt = True
                                    elif do_surge_pricing:

                                        #     pass
                                        data_instance.do_surge_pricing = True
                                        m.operator.do_surge_pricing = True
                                    else:
                                        data_instance.do_surge_pricing = False
                                        m.operator.do_surge_pricing = False
                                        data_instance.do_behavioral_opt = False
                                        m.operator.do_behavioral_opt = False

                                for T in range(data_instance.WARMUP_TIME_SECONDS,
                                               data_instance.T_TOTAL_SECONDS,
                                               data_instance.INT_ASSIGN):
                                    m.dispatch_at_time(T, day_idx=d_idx)
                                m.get_service_rate_per_zone(d_idx, month)
                                m.get_drivers_earnings_for_one_day(d_idx, month)
                                m.get_operators_earnings_for_one_day(d_idx, month)
                                print(f"Starting a new day, finished day number {d_idx + 1} of month {month}")
                                print(f"it took {(time.time() - stime) / 60}")
                                m.reset_after_one_day_of_operation(stop_month, stop_day)

                        if num_pros > 0:
                            all_dfs = pd.concat([v.report_learning_rates() for v in m.vehicles
                                                 if v.driver_type == DriverType.PROFESSIONAL
                                                 ], ignore_index=True)
                            all_dfs.to_csv(output_path + "fmean for all drivers.csv")
                            # TODO: experimental
                            all_dfs2 = pd.concat([v.report_m_learning_rates() for v in m.vehicles
                                                  if v.driver_type == DriverType.PROFESSIONAL
                                                  ], ignore_index=True)
                            all_dfs2.to_csv(output_path + "matching learning for all drivers.csv")

                            all_dfs2 = pd.concat([v.report_sd_learning_rates() for v in m.vehicles
                                                  if v.driver_type == DriverType.PROFESSIONAL
                                                  ], ignore_index=True)
                            all_dfs2.to_csv(output_path + "fsd for all drivers.csv")

                            drivers_folder_path = output_path + "drivers/"
                            if not os.path.exists(drivers_folder_path):
                                os.makedirs(drivers_folder_path)
                            # only save a handful of logs tho
                            max_save_info = 10
                            for v in m.vehicles:
                                if v.driver_type == DriverType.PROFESSIONAL:
                                    max_save_info -= 1
                                    if max_save_info > 0:
                                        print('save data')
                                        v.dump_lr_data(drivers_folder_path)

                            ##
                            all_fare_reliability_dfs = pd.concat(
                                [v.report_fare_reliability_evolution() for v in m.vehicles
                                 if v.driver_type == DriverType.PROFESSIONAL
                                 ], ignore_index=True)
                            all_fare_reliability_dfs.to_csv(output_path + "fare reliability for all drivers.csv")

                            all_m_reliability_dfs = pd.concat(
                                [v.report_matching_reliability_evolution() for v in m.vehicles
                                 if v.driver_type == DriverType.PROFESSIONAL
                                 ], ignore_index=True)
                            all_m_reliability_dfs.to_csv(output_path + "matching reliability for all drivers.csv")

                            all_fare_reliability_dfs = pd.concat(
                                [v.report_surge_bonus_behavior() for v in m.vehicles
                                 if v.driver_type == DriverType.PROFESSIONAL
                                 ], ignore_index=True)
                            all_fare_reliability_dfs.to_csv(output_path + "surge behavior for all drivers.csv")

                        all_earning_dfs = pd.concat([v.report_final_earnings() for v in m.vehicles
                                                     ], ignore_index=True)
                        all_earning_dfs.to_csv(output_path + "earnings for all drivers.csv")
                        operators_revenue = m.operator.report_final_revenue()
                        operators_revenue.to_csv(output_path + "operators_revenue.csv")

                        print('Total drivers: ', len(m.vehicles))
                        print('# of Pro drivers: ',
                              len([v for v in m.vehicles if v.driver_type == DriverType.PROFESSIONAL]))
                        print('# of naive drivers: ',
                              len([v for v in m.vehicles if v.driver_type == DriverType.NAIVE]))
                        print('# of inexperienced drivers: ',
                              len([v for v in m.vehicles if v.driver_type == DriverType.INEXPERIENCED]))
                        # end time
                        etime = time.time()
                        # run time of this simulation
                        runtime = etime - stime
                        print("The run time was {runtime} minutes ".format(runtime=runtime / 60))

                        report = m.report_final_performance()

                        # So that it doesn't save a file with 1.5.py, rather 15.py
                        ss = str(surge).split('.')
                        ss = ''.join(ss)

                        fleet_size = num_avs + num_pros + num_naives
                        report.to_csv(output_path + "report for fleet size " + str(fleet_size) + " surge " +
                                      str(ss) +
                                      "pro_ " + str(num_pros) + "naive_ " + str(num_naives) +
                                      "AV_" + str(num_avs)
                                      + " repl" + str(repl) + ".csv")

                        # pickle.dump(m,
                        #             open(output_path + "model for fleet size " + str(fleet_size) + " surge " + str(ss)
                        #                  + "fdemand " + str(percent_false_demand) + "perc_k " + str(
                        #                 perc_k) + "pro_s " + str(pro_s) + " repl" + str(repl) + ".p", "wb"))
                        #


if __name__ == "__main__":
    main()
