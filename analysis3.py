
import pickle 
import pandas as pd 
import numpy as  np 

fleet_size = 2500
ss = 10 
percent_false_demand = 0.0
perc_k = 1
pro_s = 0
output_path = "./Simulation/Outputs/"
s= output_path + "model for fleet size " + str(fleet_size) + " surge "+ str(ss) \
                    + "fdemand "+ str(percent_false_demand)+ "perc_k "+ str(perc_k) + "pro_s " + str(pro_s) + ".p"


m= pickle.load(open(s, 'rb'))



s = "model for fleet size 2500 surge 10fdemand 0.0perc_k 1pro_s 0.p"

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

