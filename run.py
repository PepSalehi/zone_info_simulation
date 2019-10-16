#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:22:13 2019

@author: peyman
"""
# https://realpython.com/python-application-layouts/
import argparse
import time
import os
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
from lib.Constants import PERCE_KNOW

output_path = "./Outputs/avg_fare_info/"
if not os.path.exists(output_path):
    os.makedirs(output_path)


def main():

    parser = argparse.ArgumentParser(description="Simulation of drivers' behavior")
    parser.add_argument(
        "-f",
        "--fleet",
        help='Fleet sizes to simulate, formatted as comma-separated list (i.e. "-f 250,275,300")',
    )
    parser.add_argument(
        "-m",
        "--multiplier",
        help='Surge multiplier, formatted as comma-separated list (i.e. "-m 1,1.5,2")',
    )
    parser.add_argument("-b", "--bonus", type=int, help="Bonus")
    parser.add_argument("-d", "--demand", help="Percent false demand ")
    parser.add_argument(
        "-k",
        "--know",
        help='Percent knowing fare, formatted as comma-separated list (i.e. "-m 1,1.5,2") ',
    )
    parser.add_argument(
        "-p",
        "--pro",
        help='Percent pro drivers, formatted as comma-separated list (i.e. "-m 1,1.5,2") ',
    )
    parser.add_argument(
        "-r", "--replications", help="number of times to run the simulation"
    )
    args = parser.parse_args()
    if args.fleet:
        fleet_sizes = [int(x) for x in args.fleet.split(",")]
    else:
        fleet_sizes = FLEET_SIZE

    if args.multiplier:
        # surge = args.multiplier
        surges = [float(x) for x in args.multiplier.split(",")]
    else:
        surges = [SURGE_MULTIPLIER]

    if args.know:
        # surge = args.multiplier
        perc_know = [float(x) for x in args.know.split(",")]
    else:
        perc_know = [PERCE_KNOW]

    if args.bonus:
        bonus = args.bonus
    else:
        bonus = BONUS

    if args.pro:
        pro_share = [float(x) for x in args.pro.split(",")]
    else:
        pro_share = [PRO_SHARE]

    if args.demand:
        percent_false_demand = float(args.demand)
    else:
        percent_false_demand = PERCENT_FALSE_DEMAND
    if args.replications:
        n_rep = int(args.replications)
    else:
        n_rep = 1

    for fleet_size in fleet_sizes:
        for surge in surges:
            for perc_k in perc_know:
                for pro_s in pro_share:
                    for repl in range(n_rep):
                        print("iteration number ", repl)
                        print("Fleet size is {f}".format(f=fleet_size))
                        print("Surge is {}".format(surge))
                        print("Percentage knowing fares is {}".format(perc_k))
                        print("Percentage of professional drivers {}".format(pro_s))

                        m = Model(
                            ZONE_IDS,
                            DEMAND_SOURCE,
                            WARMUP_TIME_HOUR,
                            ANALYSIS_TIME_HOUR,
                            fleet_size=fleet_size,
                            pro_share=pro_s,
                            surge_multiplier=surge,
                            bonus=bonus,
                            percent_false_demand=percent_false_demand,
                            percentage_know_fare=perc_k,
                        )

                        # start time
                        stime = time.time()

                        # # dispatch the system for T_TOTAL seconds, at the interval of INT_ASSIGN
                        for T in range(
                            WARMUP_TIME_SECONDS, T_TOTAL_SECONDS, INT_ASSIGN
                        ):

                            m.dispatch_at_time(T)

                        # end time
                        etime = time.time()
                        # run time of this simulation
                        runtime = etime - stime
                        print(
                            "The run time was {runtime} minutes ".format(
                                runtime=runtime / 60
                            )
                        )

                        m.runtime = runtime
                        report = m.get_service_rate_per_zone()

                        # So that it doesn't save a file with 1.5.py, rather 15.py
                        ss = str(surge).split(".")
                        ss = "".join(ss)

                        report.to_csv(
                            output_path
                            + "report for fleet size "
                            + str(fleet_size)
                            + " surge "
                            + str(ss)
                            + "fdemand= "
                            + str(percent_false_demand)
                            + "perc_k "
                            + str(perc_k)
                            + "pro_s "
                            + str(pro_s)
                            + " repl"
                            + str(repl)
                            + ".csv"
                        )

                        pickle.dump(
                            m,
                            open(
                                output_path
                                + "model for fleet size "
                                + str(fleet_size)
                                + " surge "
                                + str(ss)
                                + "fdemand "
                                + str(percent_false_demand)
                                + "perc_k "
                                + str(perc_k)
                                + "pro_s "
                                + str(pro_s)
                                + " repl"
                                + str(repl)
                                + ".p",
                                "wb",
                            ),
                        )


if __name__ == "__main__":
    main()
