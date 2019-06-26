import numpy as np
import pandas as pd
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns

sns.set(style="whitegrid")

m = pickle.load(
    open(
        "./Outputs/avg_fare_info/1/model for fleet size 1500 surge 2fdemand 0.0perc_k 0pro_s 0 repl0.p",
        "rb",
    )
)

report = m.get_service_rate_per_zone()
report
report.LOS.describe()
print("total_demand = {}".format(report.total.sum()))

total_demand = 20000
system_LOS = report.served.sum() / total_demand
system_LOS
np.sum(m.operator.revenues)
drivers_fares = [np.sum(v.collected_fares) for v in m.vehilcs]
stats.describe(drivers_fares)

np.median(drivers_fares)

# print("vehicle utilization = {}".format(report.idle.sum()/(report.idle.sum() + report.incoming.sum())))


z = m.zones[10]


l = [z.id for z in m.zones]
l.index(236)

directory = "./Outputs/zone_demand_viz/"
if not os.path.exists(directory):
    os.makedirs(directory)

for z in m.zones:
    # z = m.zones[l.index(88)]

    demand = z._demand_history
    supply = z._supply_history
    served = z._serverd_demand_history
    incoming = z._incoming_supply_history
    # https://datascience.stackexchange.com/questions/26333/convert-a-list-of-lists-into-a-pandas-dataframe
    data = pd.DataFrame.from_records([demand, served, supply, incoming])
    # data = pd.DataFrame.from_records([demand, supply, incoming])
    df = data.transpose()
    df.columns = columns = ["demand", "served", "supply", "incoming"]
    # f = open(directory +)
    sns.lineplot(data=df, palette="tab10", linewidth=2.5)
    f = open(directory + "zone time_demand {}.json".format(z.id), 'w')
    json.dump(z._time_demand, f)
    f.close()
    plt.savefig("{}/zone {}.png".format(directory, z.id))
    plt.clf()


