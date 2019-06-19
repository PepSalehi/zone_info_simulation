import numpy as np
import pandas as pd
import time 
import random 
import pickle 
from collections import Counter
from lib.Zones import Zone
from lib.Constants import ZONE_IDS, DEMAND_SOURCE, INT_ASSIGN, FLEET_SIZE, PRO_SHARE,SURGE_MULTIPLIER, BONUS,PERCENT_FALSE_DEMAND
from lib.Constants import T_TOTAL_SECONDS, WARMUP_TIME_SECONDS, ANALYSIS_TIME_SECONDS, ANALYSIS_TIME_HOUR, WARMUP_TIME_HOUR
from lib.Constants import PERCE_KNOW, CONST_FARE, AV_SHARE
from lib.Operator import Operator
from lib.Vehicles import Veh
from lib.AV_model import DQNAgent
# models creates all the zones. initializes the daily demand. this should be later be confined to hourly/15 mins demand 
# Zones must then create Requests from the matrix row by row 
#
#
class Model():
    ''' 
    encompassing object
    '''
    
    def __init__(self, zone_ids, daily_demand, WARMUP_TIME_HOUR, ANALYSIS_TIME_HOUR, 
                 FLEET_SIZE=FLEET_SIZE, PRO_SHARE=PRO_SHARE, SURGE_MULTIPLIER=SURGE_MULTIPLIER,
                 BONUS=BONUS, percent_false_demand =PERCENT_FALSE_DEMAND, percentage_know_fare=PERCE_KNOW,
                 AV_share = AV_SHARE, RL_engine = None):
        
        print("calling init function of Model")
        seed1 = 10 #np.random.randint(0,1000000)
        self.rs1 = np.random.RandomState(seed1)
        # 
        self.zone_ids = zone_ids
        self.zones = []
        self.daily_demand = daily_demand
        self._create_zones()
        print ("instantiated zones")
        self.set_analysis_time(WARMUP_TIME_HOUR)
        print ("generated demand")
        self.WARMUP_TIME_SECONDS = WARMUP_TIME_HOUR / 3600
        self.WARMUP_PHASE = True
        self.ANALYSIS_TIME_HOUR = ANALYSIS_TIME_HOUR
        self.ANALYSIS_TIME_SECONDS = ANALYSIS_TIME_HOUR * 3600
        
        self.PRO_SHARE = PRO_SHARE
        self.AV_SHARE = AV_share
        self.SURGE_MULTIPLIER = SURGE_MULTIPLIER
        self.FLEET_SIZE = FLEET_SIZE
        _s = str(self.SURGE_MULTIPLIER).split('.')
        _s = ''.join(_s)
                     
         # get expected number of drivers per zone 
        if self.PRO_SHARE >0:
            # m= pickle.load(open("outputs/model for fleet size {f} surge {s}fdemand 0.0perc_k 0pro_s 0.0.p".format(f=self.FLEET_SIZE, s= _s), 'rb'))
            m= pickle.load(open("Outputs/model for fleet size {f} surge {s}fdemand 0.0perc_k 0pro_s 0 repl0.p".format(f=self.FLEET_SIZE, s= _s), 'rb'))
            report = m.get_service_rate_per_zone()
            report = self.__calc_s(report)
        else:
            report = None
            
        #
        self.operator = Operator(report, BONUS=BONUS, SURGE_MULTIPLIER=SURGE_MULTIPLIER )
        #
        self.RL_engine = RL_engine
        # pro drivers know fares, so perc_know must take this into account
        self.fleet_pro_size = int(self.PRO_SHARE  * self.FLEET_SIZE ) 
        self.percent_false_demand = percent_false_demand
        self.fleet_deceived_size= int(self.percent_false_demand  * self.FLEET_SIZE )
        self.percentage_know_fare = np.minimum(self.PRO_SHARE + percentage_know_fare, 1) 
        self.fleet_DONT_know_fare = int((1-self.percentage_know_fare ) * self.FLEET_SIZE )
        self.fleet_AV = int(self.AV_SHARE * self.FLEET_SIZE)
        
        self._create_vehicles()
        
        print(Counter(v.ozone for v in self.vehilcs))
        # debug, delete this 
        self.targets = []
        self.performance_results = {} 
        ####
        
       
    def __calc_s(self,df):
        df.loc[:,'avg_num_drivers'] = df.idle + df.incoming
        s = df.total/df.avg_num_drivers
        s[s>1] = 1
        s[np.isnan(s)] = 0.0001
        s[np.isinf(s)] = 1
          
        df.loc[:,'prob_of_s'] = s
        df = df[['zone_id', 'prob_of_s']] 
        return df
    
        
    def _create_zones(self):
        ''' 
        make the zones, initiate their demand matrix
        '''
        for z_id in self.zone_ids: 
            Z = Zone(z_id, rs = self.rs1)
            Z.read_daily_demand(self.daily_demand)
            self.zones.append(Z)
            
    def _create_vehicles(self):
        '''
        '''
        self.vehilcs = [Veh(self.rs1, self.operator) for i in range(self.FLEET_SIZE)]
        # None of the below ones are mutually exclusive; a vehicle might be pro and don't know fare (which is wrong)
        if self.fleet_pro_size >0:
            print("fleet pro size", self.fleet_pro_size )
            # vs = random.choices(self.vehilcs, k=self.fleet_pro_size)
            vs = np.random.choice(self.vehilcs, self.fleet_pro_size, replace=False)
            for v in vs:
                v.professional = True
                v.know_fare = True 
            remaining_veh = list(set(self.vehilcs)-set(vs))
            
        if self.fleet_deceived_size >0:
            if not ('remaining_veh' in locals()):
                remaining_veh = self.vehilcs
            # vs = random.choices(remaining_veh, k=self.fleet_deceived_size)
            vs = np.random.choice(remaining_veh, self.fleet_deceived_size, replace=False)
            for v in vs:
                v.true_demand = False
                
            remaining_veh = list(set(remaining_veh)-set(vs))   

                
        if self.fleet_DONT_know_fare > 0:
            print("fleet Don't know fare", self.fleet_DONT_know_fare )
            if not ('remaining_veh' in locals()):
                remaining_veh = self.vehilcs
            

            vs = np.random.choice(remaining_veh, self.fleet_DONT_know_fare, replace=False)
            for v in vs:
                v.know_fare = False
                
            remaining_veh = list(set(remaining_veh)-set(vs))
            
        if self.fleet_AV > 0:
            if not ('remaining_veh' in locals()):
                remaining_veh = self.vehilcs
            # vs = random.choices(remaining_veh, k=self.AV_SHARE)
            vs = np.random.choice(remaining_veh, self.AV_SHARE, replace=False)
            for v in vs:
                v.is_AV = True
                v.RL_engine = self.RL_engine
                
            remaining_veh = list(set(remaining_veh)-set(vs))   
            
            
    def set_analysis_time(self, t):
        ''' 
        '''
       
        for z in self.zones:
            z.set_demand_rate_per_t(t)

    def generate_zonal_demand(self, t):
        '''
        first check and see if should change the demand rates
        then generate demand
        '''
        if self.WARMUP_PHASE and t >= self.ANALYSIS_TIME_SECONDS:
            print ("changing time")
            self.set_analysis_time(self.ANALYSIS_TIME_HOUR)
            self.WARMUP_PHASE = False
            # update drivers infor/expectations 
            
        for z in self.zones:
            z.generate_requests_to_time(t)
            
    # this is to be called by the RL algorithm 
    def act(self, t, action):
        for veh in self.vehilcs:
            if  veh.is_AV: 
                veh.move(t, self.zones, self.WARMUP_PHASE)

    def move_fleet(self,t, WARMUP_PHASE, action):
        for veh in self.vehilcs:
            if not veh.is_AV: # AV is already being moved by the engine 
                _=veh.move(t, self.zones, WARMUP_PHASE)
            if veh.is_AV: 
                # if veh.should_move(): this causes errors, since move is not just moving, but also rebalancing, waiting, etc.
                veh.move(t, self.zones, WARMUP_PHASE, action)
    
    def assign_zone_veh(self,t, WARMUP_PHASE, penalty):
        
        for z in self.zones:
            z.assign(self.zones, t, WARMUP_PHASE, penalty)
            
    def get_service_rate_per_zone(self):
        
        performance_results = {}
        for z in self.zones:
            w = len(z.demand)
            served = len(z.served_demand)
            los = served/(served+w) if (served+w) >0 else 0
           
            r = {'zone_id':z.id, 'w':w, 'served': served, 'total': w + served,
                 'LOS':los, "idle":len(z.idle_vehicles),
                'incoming':len(z.incoming_vehicles),'times_surged':z.num_surge }
            performance_results[z.id] =  r
        
        performance_results = pd.DataFrame.from_dict(performance_results, orient ='index')
        performance_results = performance_results.sort_values('LOS', ascending=False)
    
    
        return  performance_results
            
    # dispatch the AMoD system: move vehicles, generate requests, assign, reoptimize and rebalance
    def dispatch_at_time(self, t, penalty=-10, action=None):
#
        self.generate_zonal_demand(t)
        self.operator.update_zonal_info(t)
        self.operator.update_zone_policy(t, self.zones, self.WARMUP_PHASE)
        self.assign_zone_veh(t, self.WARMUP_PHASE, penalty)
        self.move_fleet(t, self.WARMUP_PHASE, action)
        
        if t % 500 == 0:
            print ("time is {time}".format(time= t))

    def _get_demand_per_zone(self):
        """ 
        
        Dataframe with zone_id as index and demand column
        """
        a = {z.id: len(z.demand) for z in self.zones}
        demand_df = pd.DataFrame.from_dict(a, orient='index', columns=['demand'])
        return demand_df

    def _get_supply_per_zone(self):
        """ 
        Dataframe with zone_id as index and supply column
        """
        b = {z.id: len(z.idle_vehicles) for z in self.zones}
        supply_df = pd.DataFrame.from_dict(b, orient='index', columns=['supply'])
        return supply_df

    def get_state(self, veh):
        """
        returns: matrix of size (#zones * 3), where each row is  (u_i, v_i, c_ij) 

        """
        
        dist = veh._get_dist_to_all_zones()[["DOLocationID","trip_distance_meter" ]]
        dist["costs"] = dist.trip_distance_meter * veh.rebl_cost
        dist["costs"] = dist["costs"].apply(lambda x: np.around(x,1))
        demand_df = self._get_demand_per_zone()
        supply_df = self._get_supply_per_zone()
        d_s = pd.merge(demand_df, supply_df, left_index=True, right_index=True)
        d_s_c = pd.merge(d_s, dist, left_index=True, right_on="DOLocationID")
        d_s_c = d_s_c [["demand", "supply", "costs"]]
        d_s_c = d_s_c.as_matrix()
        return d_s_c
        
    
if __name__ == "__main__":
    pass
    