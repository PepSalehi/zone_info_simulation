import numpy as np 
import pandas as pd 
from collections import defaultdict
from lib.Constants import ZONE_IDS, PHI, DIST_MAT, CONSTANT_SPEED, INT_ASSIGN, MAX_IDLE, FUEL_COST, CONST_FARE, zones_neighbors
from lib.Requests import Req
from lib.configs import configs
# from lib.rl_policy import DQNAgent
driver_id = 0

class Veh():
    def __init__(self, rs, operator, beta, true_demand=True, professional=False, ini_loc=None, 
    know_fare=False, is_AV=False, DIST_MAT = DIST_MAT):
        global driver_id  
        driver_id += 1
        self.id = driver_id
        self.just_started = True 
        self.is_AV = is_AV
        self.idle = True
        self.busy = False
        self.rebalancing = False 
        self.true_demand = true_demand
        self.professional = professional
        self.know_fare = know_fare
        self.rs = rs 
        self.DIST_MAT = DIST_MAT
        self.operator = operator
        self.locations = []
        self.req = None 
        self.beta = beta
        if ini_loc is None:
            self.ozone = rs.choice(ZONE_IDS)
#            self.ozone = 186
            self.locations.append(self.ozone)
        
        self.IDLE_COST = 0 # should be based on the waiting time 
        self.rebl_cost = FUEL_COST # should be based on dist to the destination zone 
        self.profits = []
        self.time_idled = 0
        self.MAX_IDLE = MAX_IDLE # 15 minutes
        
        self.t_since_idle = None 
        self.number_of_times_moved = 0
        self.number_of_times_overwaited = 0
        self.distance_travelled = 0
        
        self.tba = []
        self.total_waited =0
        self.zone = None 
        self.collected_fares = []
        self.collected_fare_per_zone = defaultdict(int)
#        self.prior = self.set_prior_info()
#        self.live_data = self.get_data_from_operator()

        # if self.is_AV:
        #     agent = DQNAgent(action_space = ZONE_IDS)
            
        
        
    def _sanity_check(self):
        assert (self.busy is not self.rebalancing) or (self.busy is not self.idle)


    def _calc_matching_prob(self):
        if not self.professional:
            return 1
        
    def get_data_from_operator(self, t, true_demand):
        df = self.operator.zonal_info_for_veh(true_demand)
        return df 
        
    def _get_dist_to_all_zones(self):
        dists = self.DIST_MAT.query("PULocationID=={o}".format(o=self.ozone))
        
        return dists 
    
    def _get_time_to_destination(self, dest):
        dist = self._get_distance_to_destination(dest)
        t = dist/CONSTANT_SPEED
        return t 
    
    def _get_distance_to_destination(self, dest):
        try: # because of the Nans, etc.  just a hack
            dist = np.ceil(self.DIST_MAT.query("PULocationID == {origin} & DOLocationID == {destination} ".format(origin=self.ozone,      
                                                                                              destination=dest))["trip_distance_meter"].values[0])
        
        except:
            dist = 1000
            print('Couldnt find the distance btw {o} and {d}'.format(
                    o=self.ozone, d=dest))
        
        return dist 

        
    def set_prior_info(self, t):
        '''
        prior demand/fare info
        '''
        if self.professional:
            prior = self.operator.expected_fare_totaldemand_per_zone_over_days(t)
        else:
            prior = None
        return prior 
            
    def cal_profit_per_zone_per_app(self, t):
#        df = self.get_data_from_operator(t)
        pass
        
        
    def waited_too_long(self):
        """
        Makes sure it's not idle nor rebalancing, then if it has been idle for too long returns True
        """
        return (self.idle and not self.rebalancing and self.time_idled > self.MAX_IDLE)
    
    
    def update_rebalancing(self, WARMUP_PHASE):
        self.time_to_be_available -= INT_ASSIGN #  delta t, this is a hack
        if self.time_to_be_available <= 0:
            self.rebalancing = False 
            self.idle = True 
            self.time_idled = 0
            if not WARMUP_PHASE:
                self.number_of_times_moved += 1
            # should it also get out of the waiting list?
    
    def keep_waiting(self):
        self.time_idled += INT_ASSIGN
        self.total_waited += INT_ASSIGN

    def get_neighboring_zone_ids(self):
        ''' 
        a list of ids of the neighboring zones 
        '''
        neighbors_list = zones_neighbors[str(self.ozone)]
        return neighbors_list
        
    def is_busy(self):
        return (not self.idle and not self.rebalancing)

    def set_target_zone(self, target):
        """
        use the RL agent to decide on the target  
        """
        assert self.is_AV

    def should_move(self):
        """
        just started or has been idle for too long 

        """
        return ( self.just_started or self.waited_too_long() )

    def move(self,t, Zones, WARMUP_PHASE, action=None):
        """ 
        1. just started or has been idle for too long -> choose zone 
        2. if it's rebalancing (i.e., on the way to the target zone) -> check whether or not it has gotten there 
        3. if it's idle, but the waiting time has not yet exceeded the threshold ->  keep waiting 
        4. if it's currently serving a demand -> update status 

        action is the INDEX of the zone, needs to be converted to the actual zone_id 
        """
        if self.should_move():
            # first, get out of the current zone's queue 
            if self.zone is not None :
                self.zone.remove_veh_from_waiting_list(self)
            # then choose the destination 
            if not self.is_AV :
                target_zone = self.choose_target_zone(t)

            if self.is_AV:
                assert action is not None 
                target_zone = Zones[action].id
                # print("action", action)
                # print("target zone id", Zones[action].id)
#            print(target_zone)            
            for z in Zones:
                if z.id == target_zone:
                    self.rebalancing = True
                    self.idle = False 
                    self.time_to_be_available = self._get_time_to_destination(target_zone)
                    self.tba.append(self.time_to_be_available)
                    dist = self._get_distance_to_destination(target_zone)
                    z.join_incoming_vehicles(self)
                    self.zone = z 
                    
                    break 
            
            self.ozone = target_zone # debugging for now (so what is this comment? should I delete this line? WTF is this doing?)
            
            if not WARMUP_PHASE:
                self.distance_travelled += dist
                self.number_of_times_moved += 1
                self.locations.append(self.ozone)
                
            self.time_idled = 0
            self.just_started = False 
            
            return target_zone
        
        if self.rebalancing: # and not self.busy:
            self.update_rebalancing(WARMUP_PHASE)
                
      
        elif self.idle and not self.rebalancing and self.time_idled <= self.MAX_IDLE:
            # it's sitting somewhere 
            self.keep_waiting()
#            print ("waiting")

        # this is the time it's busy serving demand
        elif self.is_busy():
#            print("BUSY SERVING DEMAND")
            self.time_to_be_available -= INT_ASSIGN 
            if self.time_to_be_available <= 0:
#                print("SERVED IT")
                self.rebalancing = False 
                self.idle = True 
                self.time_idled = 0
            
            
    
    
    def match_w_req (self, req, Zones, WARMUP_PHASE):
        

        self.idle = False
        self.rebalancing = False
        self.time_idled = 0 
        dest = req.dzone
        for z in Zones:
            if z.id == dest:
                self.time_to_be_available = self._get_time_to_destination(dest)
                dist = self._get_distance_to_destination(dest)
                
                self.ozone = dest 
                self.zone = z
                
                z.join_incoming_vehicles(self)
                #
                if not WARMUP_PHASE:
                    if not self.professional:
                        self.collected_fares.append((1-PHI) * req.fare )
                        self.operator.revenues.append(PHI * req.fare)
                        self.collected_fare_per_zone[req.ozone] += (1-PHI) * req.fare
                    elif self.professional:
                        self.collected_fares.append(req.fare )
                        self.operator.revenues.append(req.fare)
                        self.collected_fare_per_zone[req.ozone] +=req.fare


                    self.locations.append(dest)
                    self.distance_travelled += dist
                    self.profits.append(req.fare) # the absolute fare, useful for hired drivers 
                    
                #
                self.req = req 
                return True
        return False 
                
        
            
        
            
        
        
    def choose_target_zone(self,t):
        '''
        This has to be based on the information communicated by the app, as well as the prior experience
        '''
            
        dist = self._get_dist_to_all_zones()
        df = self.get_data_from_operator(t, self.true_demand) 
        b = pd.merge(df, dist, left_on='Origin', right_on='DOLocationID', how='left')
        
        neighbors_list = self.get_neighboring_zone_ids()
        assert len(neighbors_list) > 0
        # so that it can only choose from its neighboring zones
        a = b[b["Origin"].isin(neighbors_list)]


        
        
        if self.is_AV: 
            # if it's an AV
            return 

        if not self.know_fare: # they don't know the average fare for an area, they use one for all
            # print("They don't know the fare")
            a.avg_fare = CONST_FARE
        
        if not self.professional:
            match_prob = 1 # what?
            # the problem here is that p*q is way  
            
            # beta = 0.001 # coefficient to make calculation work out  
            # beta = 0.01 # remove this 
        # TODO : round up the fare 

        else:
            try:
                match_prob = a.prob_of_s
                # beta = 0.01 # change it to 0.01
            except:
                print(df)
                print("that was df")
                print(a)
                print("that was a")
                print(self.professional)

        a["relative_demand"] = a["total_pickup"]/ a["total_pickup"].sum()
        expected_revenue = (1-PHI) * a.avg_fare * a.surge * match_prob * self.beta  + a.bonus
        expected_cost = a.trip_distance_meter * self.rebl_cost # doesn't take into account the distance travelled once the demand is picked up
        
        a["expected_cost"] = expected_cost
        a['numerator'] = (expected_revenue - expected_cost) *  a["total_pickup"]
        a['expected_revenue'] = expected_revenue 
        a['expected_profit'] = expected_revenue - expected_cost
        a['prof'] = (expected_revenue - expected_cost) *  a["total_pickup"]
        a['prof'] = a['prof'].clip(lower=0)
        
        

        # http://cs231n.github.io/linear-classify/#softmax
        # for numerical stability
        # a['prof'] -= np.max(a['prof'])
        # a['prob'] = np.exp(a['prof'])/np.sum(np.exp(a['prof'])) 

        # so that probability doesn't end up being negative 
        
        a['prob'] = a['prof']/a['prof'].sum()
        

        # path_to_write = configs['output_path']
        # with open(path_to_write +'driver ' + str(self.id)+ '.csv', 'a') as f:
        #     a.to_csv(f, header = True, mode='a', index = False)

  



         
        try:
            selected = a.sample(n=1, weights='prob', replace=True)
            # if self.waited_too_long():
            #     cc = 0
            #     while selected['DOLocationID'].values[0] == self.zone:
            #         cc += 1 
            #         selected = a.sample(n=1, weights='prob', replace=True)
            #         if cc == 1000:
            #             print ('did 1000 iterations, still chose the same zone', self.z)
            #             break 
            
        except:
            print("ozone")
            print(self.ozone)
            raise Exception("selected was empty, here is a {}".format(a))
#                print(df)
#                print(dist)
            
        return selected['DOLocationID'].values[0]
    
    
    
    
    def _calculate_fare(self, request, surge):
        '''
        From Yellow Cab taxi webpage
        '''
        distance_meters = request.dist
        p1 = 2.5 
        p2 = 0.5 * 5 * 1609 * distance_meters
        f =  p1 + surge * p2 
        return f
    
    def _calc_rebl_cost(self, request): 
        '''
        This should be based on the value of time and time it took to get to the destination
        
        '''
        UNIT_COST = 1  # assume $1 per 1600 meter
        dest = request.ozone 
        current_loc = self.ozone
        dist = np.ceil(self.DIST_MAT.query("PULocationID == {origin} & DOLocationID == {destination} ".format(origin=current_loc,      
                                                                                                destination=dest))["trip_distance_meter"].values[0])
        cost = dist/1600 * UNIT_COST
        return cost 
    
    def realized_trip_profit(self, request, surge=1, driver_bonus = 0):
        assert isinstance(request, Req)
        fare = self._calculate_fare(request, surge)
        rebl_cost = self._calc_rebl_cost (request)
        profit = (1-PHI)* fare + driver_bonus - rebl_cost - self.IDLE_COST 
        return profit
        


