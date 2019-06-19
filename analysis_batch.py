#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:31:41 2019
For rev analysis of pro vs naive 
@author: peyman
"""


import numpy as np
import pandas as pd 
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import os

import seaborn as sns
sns.set(style="whitegrid")

def get_operation_cost(pro_share, fleet):
    "hourly cost of Via driver: $30"
    return (fleet * 30)

#template = "./Outputs/RL/report for fleet size 2000 surge 2fdemand= 0.0perc_k 0pro_s 0 perc_av {} repl{}.csv"
template = "./Outputs/report for fleet size 1500 surge 2fdemand= 0.0perc_k {}pro_s 0 repl{}.csv"

template = "./Outputs/RL/report for fleet size 2000 surge 2fdemand= 0.0perc_k 0pro_s 0 perc_av {} repl{}.csv"
op_rev = []
op_cost = []
los_list = []
los_mean = []
los_median = []
denied_w = []
ff = []
fleet = 2000
for av_share in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for repl in range(5):
        report = pd.read_csv(template.format(av_share, repl))

        # op_rev.append(np.sum(m.operator.revenues))
        # op_cost.append(get_operation_cost(fleet,pro_share ))
        op_cost.append(fleet * 30 )
        system_LOS = report.served.sum()/report.total.sum()
        mean_los = report.LOS.mean()
        median_los = report.LOS.median()
        los_list.append(system_LOS)
        los_median.append(median_los)
        los_mean.append(mean_los)
        denied_w.append(report.w.sum())

print("los_list", los_list)

directory = "./Outputs/av"
if not os.path.exists(directory):
    os.makedirs(directory)
# plot revenue vs cost vs profit 

# visualized LOS 
los_list = np.array(los_list)
data = pd.DataFrame.from_records([los_list, los_mean])
df = data.transpose()
df.columns = columns=["LOS", "mean"]
df.index = np.repeat([0.0, 0.2,0.4, 0.6, 0.8, 1.0],5)
sns_plot = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
plt.savefig("{}/los.png".format(directory))
plt.clf()
# visualized denied/waiting 
denied_w = np.array(denied_w)
data = pd.DataFrame.from_records([denied_w])
df = data.transpose()
df.columns = columns=["Denied"]
df.index = np.repeat([0.0, 0.2,0.4, 0.6, 0.8, 1.0],5)
sns_plot = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
plt.savefig("{}/denied.png".format(directory))
plt.clf()

# df.loc[:,"Percent hired"] = [0.0, 0.2,0.4, 0.6, 0.8, 1.0]

