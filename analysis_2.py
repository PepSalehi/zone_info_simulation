#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:31:41 2019

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

m= pickle.load(open("./outputs/model for fleet size 2500 surge 2fdemand 0.0perc_k 1pro_s 1.0.p", 'rb'))

m= pickle.load(open("./outputs/model for fleet size 2500 surge 2fdemand 0.0perc_k 1pro_s 0.0.p", 'rb'))

m= pickle.load(open("./outputs/model for fleet size 2000 surge 2fdemand 0.0perc_k 1pro_s 0.0.p", 'rb'))



report = m.get_service_rate_per_zone()
report
report.LOS.describe()
print("total_demand = {}".format(report.total.sum()))

total_demand = 20000
system_LOS = report.served.sum()/total_demand
system_LOS
np.sum(m.operator.revenues)
drivers_fares = [np.sum(v.collected_fares) for v in m.vehilcs]
stats.describe(drivers_fares)

np.median(drivers_fares)

# print("vehicle utilization = {}".format(report.idle.sum()/(report.idle.sum() + report.incoming.sum())))


z = m.zones[10]



l = [z.id for z in m.zones ]
l.index(236)

directory = "./Outputs/pro 2k 0_0"
if not os.path.exists(directory):
    os.makedirs(directory)

for z in m.zones: 
	# z = m.zones[l.index(88)]

	demand = z._demand_history
	supply = z._supply_history 
	served = z._serverd_demand_history
	incoming = z._incoming_supply_history
	#https://datascience.stackexchange.com/questions/26333/convert-a-list-of-lists-into-a-pandas-dataframe
	data = pd.DataFrame.from_records([demand, served, supply, incoming])
	# data = pd.DataFrame.from_records([demand, supply, incoming])
	df = data.transpose()
	df.columns = columns=["demand", "served", "supply", "incoming"]
	# df.columns = columns=["demand", "supply", "incoming"]


	sns.lineplot(data=df, palette="tab10", linewidth=2.5)


	plt.savefig("{}/zone {}.png".format(directory,z.id))
	plt.clf()


plt.clf()

plt.show()







a = report.query('LOS<0.5')['zone_id']

np.savetxt(r'outputs/zones_los_less_50_f_2500.csv', a.values)



m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.6.p", 'rb'))

m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.6.p", 'rb'))



m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.6.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0.p", 'rb'))





report = m.get_service_rate_per_zone()
report
report.LOS.describe()
print("total_demand = {}".format(report.total.sum()))

total_demand = 18287
system_LOS = report.served.sum()/total_demand
system_LOS
np.sum(m.operator.revenues)
drivers_fares = [np.sum(v.collected_fares) for v in m.vehilcs]
stats.describe(drivers_fares)

np.median(drivers_fares)

print("vehicle utilization = {}".format(report.idle.sum()/(report.idle.sum() + report.incoming.sum())))


report.sort_values("total", ascending=False  )

x= {z.id:z.D for z in m.zones}
sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
sorted_x



z236 = [z for z in m.zones if z.id == 236][0]
idles = [v for v in z236.idle_vehicles]

np.mean([v.collected_fare_per_zone[236] for v in idles])

np.mean([v.collected_fare_per_zone[236] for v in m.vehilcs ])
np.median([v.collected_fare_per_zone[236] for v in m.vehilcs ])



rev_results = pd.DataFrame.from_dict({z.id: z.revenue_generated for z in m.zones}
, orient ='index')
rev_results.sort_values(0)






m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 0.0.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 0.6.p", 'rb'))

m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 0.0.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 0.6.p", 'rb'))


m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 0.0.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 0.6.p", 'rb'))


report = m.get_service_rate_per_zone()
report
report.LOS.describe()
print("total_demand = {}".format(report.total.sum()))

total_demand = 18287
system_LOS = report.served.sum()/total_demand
system_LOS
np.sum(m.operator.revenues)
drivers_fares = [np.sum(v.collected_fares) for v in m.vehilcs]
stats.describe(drivers_fares)

np.median(drivers_fares)

print("vehicle utilization = {}".format(report.idle.sum()/(report.idle.sum() + report.incoming.sum())))











m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 1.0perc_k 1pro_s 0.p", 'rb'))

m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 1.0perc_k 1pro_s 0.p", 'rb'))

m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 1.0perc_k 1pro_s 0.p", 'rb'))


report = m.get_service_rate_per_zone()
report
report.LOS.describe()
print("total_demand = {}".format(report.total.sum()))

total_demand = 18287
system_LOS = report.served.sum()/total_demand
system_LOS
np.sum(m.operator.revenues)
drivers_fares = [np.sum(v.collected_fares) for v in m.vehilcs]
stats.describe(drivers_fares)

np.median(drivers_fares)

print("vehicle utilization = {}".format(report.idle.sum()/(report.idle.sum() + report.incoming.sum())))




m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 1pro_s 0.0.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 1pro_s 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 1pro_s 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 1pro_s 0.6.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 1pro_s 1.0.p", 'rb'))



m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 1pro_s 0.0.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 1pro_s 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 1pro_s 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 1pro_s 0.6.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 15fdemand 0.0perc_k 1pro_s 1.0.p", 'rb'))



m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 1pro_s 0.0.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 1pro_s 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 1pro_s 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 1pro_s 0.6.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0perc_k 1pro_s 1.0.p", 'rb'))





report = m.get_service_rate_per_zone()
report
report.LOS.describe()
print("total_demand = {}".format(report.total.sum()))

total_demand = 18287
system_LOS = report.served.sum()/total_demand
system_LOS
np.sum(m.operator.revenues)
drivers_fares = [np.sum(v.collected_fares) for v in m.vehilcs]
stats.describe(drivers_fares)

np.median(drivers_fares)

print("vehicle utilization = {}".format(report.idle.sum()/(report.idle.sum() + report.incoming.sum())))







m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 0.2.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 0.4.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 2500 surge 10fdemand 0.0perc_k 0.6.p", 'rb'))



















m= pickle.load(open("outputs/model for fleet size 1500 surge 1.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 1500 surge 15.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 1500 surge 2.p", 'rb'))

fleet = '1500'
surge = '10'


def analyze_one_model(fleet, surge):

    fname = "outputs/model for fleet size {f} surge {s}.p".format(f=fleet, s=surge)
    m= pickle.load(open(fname, 'rb'))
    report = m.get_service_rate_per_zone()
    report.loc[:, "fleet"] = fleet
    report.loc[:, "surge"] = surge         
    return report 





fleets = ['1500', '2000','2500','3000']
surges = ['10','15','20','25','30']
f = fleets[0]
s = surges[0]
report = analyze_one_model(f, s)
for fleet in fleets:
    for surge in surges:
        r = analyze_one_model(fleet, surge)
        report= pd.merge(report, r, left_on='zone_id', right_on='zone_id' )
        

#    report
#    report.LOS.describe()
    print("total_demand = {}".format(report.total.sum()))
    system_LOS = report.served.sum()/report.total.sum()
    system_LOS
    np.sum(m.operator.revenues)
    drivers_fares = [np.sum(v.collected_fares) for v in m.vehilcs]
    stats.describe(drivers_fares)
    
    report.sort_values("total", ascending=False  )
    
    x= {z.id:z.D for z in m.zones}
    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x
    

