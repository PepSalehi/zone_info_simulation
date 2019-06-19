import numpy as np 
import pandas as pd 

config = {
    "fleet": 1500, 
    "perc_av": [0.2, 0.4, 0.6, 0.8, 1.0], #np.arange(0.2, 1.1, 0.2),
    "path" :"./Outputs/RL/"

}

def read_driver_fares(config):
    pass 

def read_reports(config):
    template = "report for fleet size {} surge 2fdemand= 0.0perc_k 1pro_s 0 perc_av {}.csv"
    fleet = config["fleet"]
    path = config["path"]
    l = []
    for av_share in config["perc_av"]:
        df = pd.read_csv(path + template.format(fleet,av_share)) 
        df.loc[:,"av_share"] = av_share
        l.append(df)
    # df_from_each_file = (pd.read_csv(path + template.format(fleet,av_share)) for av_share in config["perc_av"])
    concatenated_df   = pd.concat(l, ignore_index=True)
    return (concatenated_df)

def los_state(df):
    describe = df.LOS.describe()
    total_demand = df.total.sum()
    system_LOS = df.served.sum()/total_demand
    return({
        "system_LOS" : system_LOS,
        "describe" : describe
    })

def los(df):
    total_demand = df.total.sum()
    system_LOS = df.served.sum()/total_demand
    return system_LOS
   
all_reports = read_reports(config)

l = all_reports.groupby("av_share").apply(los)

# for av_share in config["perc_av"]:
#     mean = l[av_share]['describe']['mean'] 
#     los = l[av_share]['system_LOS'] 
#     df = pd.DataFrame([[mean,los]],columns = ["mean", "los"])

    
# l[0.2]