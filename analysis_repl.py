
import numpy as np
import pandas as pd 
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import seaborn as sns
sns.set(style="whitegrid")


def get_operation_cost(pro_share, fleet):
    "hourly cost of Via driver: $30"
    return (fleet * 30)


template = "./Outputs/model for fleet size 1800 surge 2fdemand 0.0perc_k 0pro_s {} repl{}.p"
op_rev = defaultdict(list)
op_cost =  defaultdict(list)
los_list =  defaultdict(list)
los_mean =  defaultdict(list)
los_median =  defaultdict(list)
denied_w =  defaultdict(list)
ff = []
for pro_share in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for repl in range(5):
        m= pickle.load(open(template.format(pro_share, repl), 'rb'))
        fleet = m.fleet_pro_size
        ff.append(fleet)
        op_rev[pro_share].append(np.sum(m.operator.revenues))
        # op_cost.append(get_operation_cost(fleet,pro_share ))
        op_cost[pro_share].append(fleet * 30 )
        report = m.get_service_rate_per_zone()
        system_LOS = report.served.sum()/report.total.sum()
        mean_los = report.LOS.mean()
        median_los = report.LOS.median()
        los_list[pro_share].append(system_LOS)
        los_median[pro_share].append(median_los)
        los_mean[pro_share].append(mean_los)
        denied_w[pro_share].append(report.w.sum())

op_rev_collected = []
op_cost_collected = []
op_profit_s = []
denied_w_collected = []
los_mean_coolected = []
# plot revenue vs cost vs profit 
for pro_share in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    op_rev_s = np.mean(op_rev[pro_share])
    op_rev_collected.append(op_rev_s)
    op_cost_s = np.mean(op_cost[pro_share])
    op_cost_collected.append(op_cost_s)
    op_profit_s.append(op_rev_s - op_cost_s)
    denied_w_s = np.mean(denied_w[pro_share]) 
    denied_w_collected.append(denied_w_s)
    los_mean_s = np.mean(los_mean[pro_share])
    los_mean_coolected.append(los_mean_s)

data = pd.DataFrame.from_records([op_rev_collected, op_cost_collected, op_profit_s])
df = data.transpose()
df.columns = columns=["Revenue", "Cost", "Profit"]
df.index = [0.0, 0.2,0.4, 0.6, 0.8, 1.0]
sns_plot = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
plt.show()
# visualized LOS 
los_mean_coolected = np.array(los_mean_coolected)
data = pd.DataFrame.from_records([los_mean_coolected])
df = data.transpose()
df.columns = columns=["LOS"]
df.index = [0.0, 0.2,0.4, 0.6, 0.8, 1.0]
sns_plot = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
plt.show()
# visualized denied/waiting 
denied_w_collected = np.array(denied_w_collected)
data = pd.DataFrame.from_records([denied_w_collected])
df = data.transpose()
df.columns = columns=["Denied"]
df.index = [0.0, 0.2,0.4, 0.6, 0.8, 1.0]
sns_plot = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
plt.show()
