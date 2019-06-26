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
import json 
from collections import defaultdict
import seaborn as sns
sns.set(style="whitegrid")

def get_operation_cost(pro_share, fleet):
    "hourly cost of Via driver: $30"
    return (fleet * 30)


directory = "./Outputs/avg_fare_info/1/"
#template = "./Outputs/RL/report for fleet size 2000 surge 2fdemand= 0.0perc_k 0pro_s 0 perc_av {} repl{}.csv"
template = directory+"report for fleet size 1500 surge 2fdemand= 0.0perc_k {}pro_s 0 repl{}.csv"
pickle_template = directory+"model for fleet size 1500 surge 2fdemand 0.0perc_k {}pro_s 0 repl{}.p"
#template = "./Outputs/RL/report for fleet size 2000 surge 2fdemand= 0.0perc_k 0pro_s 0 perc_av {} repl{}.csv"
op_rev = defaultdict(list)
op_cost = []
los_list = []
los_mean = []
los_median = []
denied_w = []
ff = []
driver_revenue = defaultdict(list)
fleet = 1500
n_repl = 10
for av_share in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for repl in range(n_repl):
        report = pd.read_csv(template.format(av_share, repl))
        m = pickle.load(open(pickle_template.format(av_share, repl),'rb'))
        op_rev[av_share].append(np.sum(m.operator.revenues))
        driver_revenue[av_share].append([np.sum(v.collected_fares) for v in m.vehilcs])
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

directory = "./Outputs/avg_fare_info/1/"
if not os.path.exists(directory):
    os.makedirs(directory)

# save raw results 
with open(directory+'driver_revenue.p', 'wb') as f: 
    pickle.dump(driver_revenue, f)

with open(directory+'op_revenue.p', 'wb') as f: 
    pickle.dump(op_rev, f)



# plot revenue vs cost vs profit 




# visualized LOS 
los_list = np.array(los_list)
data = pd.DataFrame.from_records([los_list, los_mean])
df = data.transpose()
df.columns = columns=["LOS", "mean"]
df.index = np.repeat([0.0, 0.2,0.4, 0.6, 0.8, 1.0],n_repl)
df["Ratio"] = df.index
df.to_csv(directory + "los.csv")
sns_plot = sns.boxplot(x="Ratio",y="LOS",data=df, palette="tab10", linewidth=2.5)
plt.savefig("{}/los.png".format(directory))
plt.clf()
# visualized denied/waiting 
denied_w = np.array(denied_w)
data = pd.DataFrame.from_records([denied_w])
df = data.transpose()
df.columns = columns=["Denied"]
df.index = np.repeat([0.0, 0.2,0.4, 0.6, 0.8, 1.0],n_repl)
df.to_csv(directory + "denied.csv")
sns_plot = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
plt.savefig("{}/denied.png".format(directory))
plt.clf()

# df.loc[:,"Percent hired"] = [0.0, 0.2,0.4, 0.6, 0.8, 1.0]
# visualized LOS 
# op_rev = np.array(op_rev)
# data = pd.DataFrame.from_records([op_rev])
# df = data.transpose()
# df.columns = columns=["op_rev"]
# df.index = np.repeat([0.0, 0.2,0.4, 0.6, 0.8, 1.0],n_repl)
# df["Ratio"] = df.index
# df.to_csv(directory + "op_revenue.csv")
# sns_plot = sns.boxplot(x="Ratio",y="op_rev",data=df, palette="tab10", linewidth=2.5)
# plt.savefig("{}/op_rev.png".format(directory))
# plt.clf()

# driver_revenue = np.array(driver_revenue)
# data = pd.DataFrame.from_records([driver_revenue])
# df = data.transpose()
# df.columns = columns=["driver_revenue"]
# df.index = np.repeat([0.0, 0.2,0.4, 0.6, 0.8, 1.0],n_repl)
# df["Ratio"] = df.index
# df.to_csv(directory + "driver_revenue.csv")
# sns_plot = sns.boxplot(x="Ratio",y="driver_revenue",data=df, palette="tab10", linewidth=2.5)
# plt.savefig("{}/driver_revenue.png".format(directory))
# plt.clf()

