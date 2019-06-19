#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:22:46 2019

@author: peyman
"""

temp = pd.read_csv( "./Data/df_hourly_stats.csv")
This gives you the same thing as df_hourly_stats_over_days
 temp.groupby(['Hour','Origin'])['total_pickup'].mean()
 
 

 df = temp.query("Hour==6 & Day ==2") # true info 
 
 cl = DIST_MAT.query("PULocationID==4")
 a = pd.merge(df, cl, left_on='Origin', right_on='DOLocationID', how='left')
 
a['prof'] = a.avg_fare - a.trip_distance_meter/1600

a['prob'] = np.exp(a['prof'])/np.sum(np.exp(a['prof'])) 
 
a.sample(n=1, weights='prob', replace=True))


temp = pd.read_csv( "./Data/df_hourly_stats_over_days.csv")
df = temp.query("Hour==6") # true info 

cl = DIST_MAT.query("PULocationID==4")
a = pd.merge(df, cl, left_on='Origin', right_on='DOLocationID', how='left')
 
a['prof'] = a.avg_fare - a.trip_distance_meter/1600
 
  
a[["Origin", "avg_fare", "avg_pickup"]]

a['prob'] = np.exp(a['prof'])/np.sum(np.exp(a['prof'])) 
 
a.sample(n=1, weights='prob', replace=True))


a["weighted"] = a.avg_pickup * a.avg_fare
 
a[["Origin", "avg_fare", "avg_pickup", "weighted"]]
a['prob'] = np.exp(a['weighted'])/np.sum(np.exp(a['weighted'])) 
a['prob']
a['weighted']
a['weighted2'] = a['weighted'] - np.max(a['weighted'])
np.exp(a['weighted2'])/np.sum(np.exp(a['weighted2'])) 



a['weighted'] -= np.max(a['weighted'])
a['weighted']
np.exp(a['weighted'])/np.sum(np.exp(a['weighted'])) 
 

from collections import Counter 
from lib.Constants import DIST_MAT
Counter ([v.ozone for v in m.vehilcs])
Counter ([l  for v in m.vehilcs for l in v.locations])

Counter ([len(z.served_demand) for z in m.zones])

 s = {z.id: len(z.served_demand) for z in m.zones}
 
 a = [ len(z.served_demand) for z in m.zones]



 ([(z.N) for z in m.zones])

Counter ([len(z.demand) for z in m.zones])


len( [v for v in m.vehilcs if v.idle])

performance_results = {}
for z in m.zones:
    w = len(z.demand)
    served = len(z.served_demand)
    l = [w, served, served/(served+w+1)]
    r = {'waiting':w, 'served': served, 'total': w + served, 'LOS':served/(served+w+1) }
    performance_results[z.id] =  r

performance_results = pd.DataFrame.from_dict(performance_results, orient ='index')
performance_results.sort_values('LOS')



import numpy as np
import pandas as pd 
from scipy import stats
import pickle
m= pickle.load(open("outputs/model for fleet size 1500 surge 1.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 1500 surge 15.p", 'rb'))
m= pickle.load(open("outputs/model for fleet size 1500 surge 2.p", 'rb'))

fleet = '2500'
surge = '10'

fname = "outputs/model for fleet size {f} surge {s}.p".format(f=fleet, s=surge)
m= pickle.load(open(fname, 'rb'))
report = m.get_service_rate_per_zone()
report
report.LOS.describe()
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



z236 = [z for z in m.zones if z.id == 236][0]
idles = [v for v in z236.idle_vehicles]

np.mean([v.collected_fare_per_zone[236] for v in idles])

np.mean([v.collected_fare_per_zone[236] for v in m.vehilcs ])
np.median([v.collected_fare_per_zone[236] for v in m.vehilcs ])



rev_results = pd.DataFrame.from_dict({z.id: z.revenue_generated for z in m.zones}
, orient ='index')
rev_results.sort_values(0)



x = {z.id : v.collected_fare_per_zone[z.id] for v in m.vehilcs for z in m.zones }




# is the driver's income is negative, then sth is wrong.
import operator 
x = {z.id:z.num_surge for z in m.zones }
sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
sorted_x


x= {z.id: len(z.idle_vehicles) for z in m.zones}
sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
sorted_x

# total idle 
np.sum([len(z.idle_vehicles) for z in m.zones])
np.sum([len(z.incoming_vehicles) for z in m.zones])

np.sum([len(z.incoming_vehicles) for z in m.zones]) + np.sum([len(z.idle_vehicles) for z in m.zones])



x= {z.id: len(z.incoming_vehicles) for z in m.zones}
sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
sorted_x






m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0.p", 'rb'))
#m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.6.p", 'rb'))

report = m.get_service_rate_per_zone()

report


report.loc[:,'avg_num_drivers'] = report.idle + report.incoming


         
report.loc[:,'prob_of_s'] = report.total/report.avg_num_drivers
          
report.loc[np.isnan(report.prob_of_s),'prob_of_s'] = 0.0001
report.loc[np.isinf(report.prob_of_s),'prob_of_s'] = 1

m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.0.p", 'rb'))
#m= pickle.load(open("outputs/model for fleet size 2500 surge 20fdemand 0.6.p", 'rb'))

report = m.get_service_rate_per_zone()
           
def calc_s(df):
    df.loc[:,'avg_num_drivers'] = df.idle + report.incoming
    s = df.total/df.avg_num_drivers
    s[s>1] = 1
    s[np.isnan(s)] = 0.0001
    s[np.isinf(s)] = 1
      
    df.loc[:,'prob_of_s'] = s
    return df
          



































Counter( [int(v.distance_travelled) for v in m.vehilcs ])

Counter( [v.number_of_times_moved for v in m.vehilcs ])

Counter( [len(v.locations) for v in m.vehilcs ])

# number of distinct zones visited 
Counter( [len(set(v.locations)) for v in m.vehilcs ])


Counter( [v.number_of_times_overwaited for v in m.vehilcs ])

np.mean(( [int(v.distance_travelled) for v in m.vehilcs ]))
np.min(( [int(v.distance_travelled) for v in m.vehilcs ]))



Counter([z.num_surge for z in m.zones ])

Counter([len(z.idle_vehicles) for z in m.zones ])


{z.id: len(z.idle_vehicles) for z in m.zones }


v_busy = [v for v in m.vehilcs if v.id == 33][0]

v_busy = [v for v in m.vehilcs if v.number_of_times_moved >= 33 and v.idle] [20]

v_busy.locations
v_busy.number_of_times_moved
v_busy.distance_travelled
v_busy.idle
v_busy.rebalancing
v_busy.time_idled



v_busy = [v for v in m.vehilcs if  v.idle] 
v_busy = [v for v in m.vehilcs if v.number_of_times_moved >= 13 and v.idle and v.time_idled > 0] 

v_busy = ( [v for v in m.vehilcs if len(v.locations)>6])


v_busy = v_busy[0]

v= m.vehilcs[0]

v.tba 
v.total_waited 




