import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from lib.Constants import (
    ZONE_IDS,
    PHI,
    DIST_MAT,
    CONSTANT_SPEED,
    INT_ASSIGN,
    MAX_IDLE,
    FUEL_COST,
    CONST_FARE,
    zones_neighbors,
    PENALTY
)
from lib.Requests import Req
from lib.configs import configs
from functools import lru_cache
from enum import Enum, unique, auto    
import pickle
# from lib.rl_policy import DQNAgent
driver_id = 0

#https://stackoverflow.com/a/57516323/2005352
class VehState(Enum):
    IDLE  = auto()
    REBAL = auto()
    SERVING = auto()
    DECISION = auto()


class Veh:
    """
    A vehicle can have 4 states
    1. idle -> waiting to be matched 
    2. rebalacning -> travelling, but without a passenger. Upon arrival, should wait to be matched 
    3. serving -> currently saving demand. Should make a decision to move upon arrival at the req's destination
    4. should_make_a_decision
    """


    def __init__(
        self,
        rs,
        operator,
        beta=1,
        true_demand=True,
        professional=False,
        ini_loc=None,
        know_fare=False,
        is_AV=False,
        DIST_MAT=DIST_MAT
       
    ):
        global driver_id
        driver_id += 1
        self.id = driver_id
        # self.just_started = True
        self.is_AV = is_AV

        # self.idle = False 
        # self.serving = False
        #  #  if True, decide which zone to go to next
        # self.TIME_TO_MAKE_A_DECISION  = True 
        # self.rebalancing = False
        # self.should_make_a_decision = False

        self._state = VehState.IDLE
        

        self.true_demand = true_demand
        self.professional = professional
        self.know_fare = know_fare
        self.rs = rs
        # self.DIST_MAT = DIST_MAT
        self.operator = operator
        self.locations = []
        self.req = None
        self.beta = beta
        if ini_loc is None:
            self.ozone = rs.choice(ZONE_IDS)
            #            self.ozone = 186
            self.locations.append(self.ozone)
            self._state = VehState.DECISION

        self.IDLE_COST = 0  # should be based on the waiting time
        self.rebl_cost = FUEL_COST  # should be based on dist to the destination zone
        self.profits = []
        self.time_idled = 0
        self.MAX_IDLE = MAX_IDLE  # 15 minutes

        # self.t_since_idle = None
        self.number_of_times_moved = 0
        self.number_of_times_overwaited = 0
        self.distance_travelled = 0
        self.time_to_be_available  = 0

        self.tba = []
        self.total_waited = 0
        self.zone = None
        self.collected_fares = []
        self.collected_fare_per_zone = defaultdict(int)

        # debugging 
        self._times_chose_zone = []
        # to store (state, action, reward) for each vehicle 
        self._info_for_rl_agent = []
        self.reqs = []
        self.total_served = 0
        self.state_hist = [] 

  
    def _calc_matching_prob(self):
        if not self.professional:
            return 1

    @lru_cache(maxsize=None)
    def get_data_from_operator(self, t, true_demand):
        df = self.operator.zonal_info_for_veh(true_demand)
        return df

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_dist_to_all_zones( ozone):
        dists = DIST_MAT.query("PULocationID=={o}".format(o=ozone))

        return dists
    @lru_cache(maxsize=None)
    def _get_dist_to_only_neighboring_zones(self,ozone):
        # neighbors_list = self.get_neighboring_zone_ids()
        dists = DIST_MAT[(DIST_MAT["PULocationID"] == self.ozone) & (DIST_MAT["DOLocationID"].isin(self.get_neighboring_zone_ids(ozone)))]
        # dists = DIST_MAT.query(
        #     "PULocationID=={o} & DOLocationID.isin({destinations})".format(
        #         o=self.ozone, destinations=neighbors_list
        #     )
        # )
        return dists

    @lru_cache(maxsize=None)
    def _get_time_to_destination(self, ozone, dest):
        # dist = self._get_distance_to_destination(dest)
        t =  self._get_distance_to_destination(ozone, dest) / CONSTANT_SPEED
        return t

    @lru_cache(maxsize=None)
    def _get_distance_to_destination(self, ozone, dest):
        try:  # because of the Nans, etc.  just a hack
            dist = np.ceil(
                # DIST_MAT.query(
                #     "PULocationID == {origin} & DOLocationID == {destination} ".format(
                #         origin=self.ozone, destination=dest
                #     )
                # )["trip_distance_meter"].values[0]
                DIST_MAT[(DIST_MAT["PULocationID"] == ozone) & (DIST_MAT["DOLocationID"] == dest)]["trip_distance_meter"].values[0] 
            )
            

        except:
            dist = 1000
            print(
                "Couldnt find the distance btw {o} and {d}".format(o=self.ozone, d=dest)
            )

        return dist

    def set_prior_info(self, t):
        """
        prior demand/fare info
        """
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
        return self._state == VehState.IDLE and self.time_idled > self.MAX_IDLE
        # return self.idle and not self.rebalancing and self.time_idled > self.MAX_IDLE

    def update_rebalancing(self, WARMUP_PHASE):
        assert self._state == VehState.REBAL
        self.time_to_be_available -= INT_ASSIGN  #  delta t, this is a hack
        
        if self.time_to_be_available < 0:
            self._state = VehState.IDLE
            self.state_hist.append(self._state)
            self.time_idled = 0
            self.time_to_be_available = 0
            if not WARMUP_PHASE:
                self.number_of_times_moved += 1
           

    def keep_waiting(self):
        self.time_idled += INT_ASSIGN
        self.total_waited += INT_ASSIGN


    def keep_serving(self):
        self.time_to_be_available -= INT_ASSIGN
        if self.time_to_be_available < 0:
            assert self._state == VehState.SERVING
            self._state = VehState.DECISION
            if self.is_AV:
                try:
                    assert len(self._info_for_rl_agent) == 3
                except AssertionError:
                    print(self.waited_too_long())
                    print(self.time_idled)
                    print(self._info_for_rl_agent)
                    print(len(self.reqs))
                    print([r.fare for r in self.reqs])
                    print("time_to_be_available", self.time_to_be_available)
                    print("total_served", self.total_served)
                    print("self.is_busy", self.is_busy)
                    print("ozone", self.ozone)
                    print(self._state)
                    print("locations", self.locations)
                    print("self.state_hist", self.state_hist)
                    print("veh id ", self.id)
                    pickle.dump(self, open("veh.p", "wb"))
                    raise AssertionError

            

    @lru_cache(maxsize=None)
    def get_neighboring_zone_ids(self, ozone):
        """ 
        a list of ids of the neighboring zones 
        """
        neighbors_list = zones_neighbors[str(ozone)]
        neighbors_list.append(self.ozone)
        return neighbors_list 
    @property
    def is_busy(self):
        # try:
        #     assert self.serving == (not self.idle and not self.rebalancing )
       
        
        return self._state == VehState.SERVING

    def set_action(self, action):
        """
        use the RL agent to decide on the target  
        """
        assert self.is_AV
        assert action is not None
        self.action = int(action)
        # print("action is", action)

    @property
    def is_waiting_to_be_matched(self):
        if (self._state == VehState.IDLE and self.time_idled <= self.MAX_IDLE): 
            return True 
        else:
            return False 
        # if self.idle and not self.rebalancing and self.time_idled <= self.MAX_IDLE :
      
    @property
    def is_rebalancing(self):
        # True if self._state == VehState.REBAL else False 
        if self._state == VehState.REBAL :
            return True
        else:
            return False
            
    
    def should_move(self):
        """
        just started or has been idle for too long 
        """
        
        return self.waited_too_long() or self._state == VehState.DECISION

    def act(self, t, Zones, WARMUP_PHASE, action=None):
        """ 
        1. just started or has been idle for too long -> choose zone 
        2. if it's rebalancing (i.e., on the way to the target zone) -> check whether or not it has gotten there 
        3. if it's idle, but the waiting time has not yet exceeded the threshold ->  keep waiting 
        4. if it's currently serving a demand -> update status 
        5. idling is also an option 
        action is the INDEX of the zone, needs to be converted to the actual zone_id 
        """

        def _make_a_decision(t):
            
            # first, get out of the current zone's queue
            if self.zone is not None:
                self.zone.remove_veh_from_waiting_list(self)
            # then choose the destination
            if not self.is_AV:
                target_zone = self.choose_target_zone(t)

            if self.is_AV:
                # assert self.action is not None
                target_zone = Zones[self.action].id
                
                # print("action", self.action)
                # print("target zone id", target_zone)
            for z in Zones:
                if z.id == target_zone:
                    self._state = VehState.REBAL
                    self.state_hist.append(self._state)
                    # self.rebalancing = True
                    # self.idle = False
                    # self.TIME_TO_MAKE_A_DECISION  = False 
                    self.time_to_be_available = self._get_time_to_destination(
                        self.ozone, target_zone
                    )
                    self.tba.append(self.time_to_be_available)
                    dist = self._get_distance_to_destination(self.ozone, target_zone)
                    z.join_incoming_vehicles(self)
                    self.zone = z

                    break

            self.ozone = (
                target_zone
            )  # debugging for now (so what is this comment? should I delete this line? WTF is this doing?)

            if not WARMUP_PHASE:
                self.distance_travelled += dist
                self.number_of_times_moved += 1
                self.locations.append(self.ozone)

            # self.time_idled = 0
          

            return target_zone



        if self.should_move():
            _ = _make_a_decision(t)
            # self.update_rebalancing(WARMUP_PHASE)
            # return 
        if self.is_busy:
            self.keep_serving()
            return
        if self.is_waiting_to_be_matched:
            # it's sitting somewhere
            self.keep_waiting()
            return
        if self.is_rebalancing:  # and not self.busy:
          
            self.update_rebalancing(WARMUP_PHASE)
            return

              

        # for debugging
        # path_to_write = configs['output_path']
        # filepath = path_to_write +'driver ' + str(self.id)+ '.csv'
        # writer=csv.writer(open(filepath,'a'))
        # writer.writerow([self.ozone, self.time_to_be_available, self.rebalancing, self.idle, self.is_busy()])
            
    def match_w_req(self, req, Zones, WARMUP_PHASE):
        # try:
        #     assert self._state == VehState.IDLE
        # except:
        #     print(self.is_AV)
        #     print(self._state)
        #     print(self.time_to_be_available)
        #     raise AssertionError 
        assert self._state == VehState.IDLE
        self.time_idled = 0
        dest = req.dzone
        matched = False 
        for z in Zones:
            if z.id == dest:
                self._state = VehState.SERVING
                self.state_hist.append(self._state)
                self.time_to_be_available = self._get_time_to_destination(self.ozone, dest)
                dist = self._get_distance_to_destination(self.ozone, dest)

                self.ozone = dest
                self.zone = z
                
                matched = True
                # don't match incoming, rather join the undecided list. 
                # actually, don't join any list because in the next step, "act" will take care of it
                # z.join_incoming_vehicles(self)
                # z.join_undecided_vehicles(self)
                #
                # if not WARMUP_PHASE:
                if not self.professional and not self.is_AV:
                    self.collected_fares.append((1 - PHI) * req.fare)
                    self.operator.revenues.append(PHI * req.fare)
                    self.collected_fare_per_zone[req.ozone] += (1 - PHI) * req.fare

                elif self.professional:
                    self.collected_fares.append(req.fare)
                    self.operator.revenues.append(req.fare)
                    self.collected_fare_per_zone[req.ozone] += req.fare

                self.locations.append(dest)
                self.distance_travelled += dist
                self.profits.append(
                    req.fare
                )  
                if self.is_AV:
                    # print("thre fare was", req.fare)
                    # print("proftis are ", self.profits)
                    self.reqs.append(req)
                    self.locations.append(dest)
                    self.total_served += 1

                    try:
                        assert len(self._info_for_rl_agent)==2
                    except AssertionError:
                        print(self._state)
                        print(self.waited_too_long())
                        print(self.time_idled)
                        print(self._info_for_rl_agent)
                        print(len(self.reqs))
                        print([r.fare for r in self.reqs])
                        print(self.time_to_be_available)
                        print(self.total_served)
                        raise AssertionError


                    self._info_for_rl_agent.append(np.round(req.fare, 4)) # doesn't account for rebl cost yet
                    try:
                        assert len(self._info_for_rl_agent)==3
                    except AssertionError:
                        print(self._state)
                        print(self.waited_too_long())
                        print(self.time_idled)
                        print(self._info_for_rl_agent)
                        print(len(self.reqs))
                        print([r.fare for r in self.reqs])
                        print(self.time_to_be_available)
                        print(self.total_served)
                        raise AssertionError


                self.req = req
                return True


        if not matched:
            print("zone {} does not exist ".format(dest))
        # why and when would it return False?
        return False
    
    @lru_cache(maxsize=None)
    def _compute_attractiveness_of_zones(self, t, ozone, true_demand):

        dist = self._get_dist_to_all_zones(ozone)
        df = self.get_data_from_operator(t, true_demand)
        a = pd.merge(df, dist, left_on="Origin", right_on="DOLocationID", how="left")
        # a = pd.merge(df.set_index('Origin'), dist.set_index('DOLocationID'),  how="left", 
        # left_index=True, right_index=True)
        
        neighbors_list = self.get_neighboring_zone_ids(ozone)
        a = a[a["Origin"].isin(neighbors_list)] 
        # a = a[a.index.isin(neighbors_list)] 


        if a.empty:
            print(
                "corner case: take zone 127. there and no demand is left, df and therefore a will be empty. in this situation, it should just move to one of its neighbors"
            )
            print("ozone", self.ozone)
            print("destination", neighbors_list[0])
            return neighbors_list[0]

        if (
            not self.know_fare
        ):  # they don't know the average fare for an area, they use one for all
            fare_to_to_use = CONST_FARE
        else:
            fare_to_to_use = a.avg_fare
            # a.avg_fare = CONST_FARE
            # TODO : round up the fare

        if not self.professional:
            # if not professional, you don't consider supply in your decision making
            match_prob = 1
        else:
            try:
                match_prob = a.prob_of_s
            except:
                print(df)
                print("that was df")
                print(a)
                print("that was a")
                print(self.professional)


        # a["relative_demand"] = a["total_pickup"] / a["total_pickup"].sum()

        expected_revenue = (
            1 - PHI
        ) * fare_to_to_use * a.surge * match_prob * self.beta + a.bonus
        expected_cost = (
            a.trip_distance_meter * self.rebl_cost
        )  # doesn't take into account the distance travelled once the demand is picked up

        # a["expected_cost"] = expected_cost
        # a["numerator"] = (expected_revenue - expected_cost) * a["total_pickup"]
        # a["expected_revenue"] = expected_revenue
        # a["expected_profit"] = expected_revenue - expected_cost
        prof = (((expected_revenue - expected_cost) * a["total_pickup"]).clip(lower=0))

        # http://cs231n.github.io/linear-classify/#softmax
        # for numerical stability
        # a['prof'] -= np.max(a['prof'])
        # a['prob'] = np.exp(a['prof'])/np.sum(np.exp(a['prof']))

        # so that probability doesn't end up being negative
        # a["prob"] = a["prof"] / a["prof"].sum()
        prob = prof/ prof.sum()
        return (a, prob)

    def choose_target_zone(self, t):
        """
        This has to be based on the information communicated by the app, as well as the prior experience
        It should have the option of not moving. Maybe include that in the neighbors list 
        """



        # debugging.
        # self._times_chose_zone.append([t, self.idle, self.rebalancing, self.is_busy(), self.should_move(), self.time_to_be_available, self.waited_too_long() ])

        # dist = self._get_dist_to_all_zones(self.ozone)
        # df = self.get_data_from_operator(t, self.true_demand)
        # a = pd.merge(df, dist, left_on="Origin", right_on="DOLocationID", how="left")

        # neighbors_list = self.get_neighboring_zone_ids(self.ozone)
        # # assert len(neighbors_list) > 0
        # # so that it can only choose from its neighboring zones
        # # TODO : this line bumped the running time of the code from 5 to 63 minutes!
        # # a = b[b["Origin"].isin(neighbors_list)]
        # a = a[a["Origin"].isin(neighbors_list)]

        # # corner case: take zone 127. When a taxi is there and no demand is left, df and therefore a will be empty.
        # # in this situation, it should just move to one of its neighbors
        # if a.empty:
        #     print(
        #         "corner case: take zone 127. there and no demand is left, df and therefore a will be empty. in this situation, it should just move to one of its neighbors"
        #     )
        #     print("ozone", self.ozone)
        #     print("destination", neighbors_list[0])
        #     return neighbors_list[0]

        # # TODO: this should be a very interesting pandas question. In the above, if I use
        # # b = pd.merge(df, dist, left_on='Origin', right_on='DOLocationID', how='left')
        # # a = b[b["Origin"].isin(neighbors_list)]
        # # it will take 63 minutes!!!
        # # if it's just a all along, it's 14 minutes.
        # # none of this (i.e. isin(list)) and it's 5 minutes. OMG!
        # # After investigating this more, it seems that the most problematic one was indeed reassignment, i.e. a = b[b["Origin"].isin(neighbors_list)]
        # # but I have no idea why



        # if (
        #     not self.know_fare
        # ):  # they don't know the average fare for an area, they use one for all
        #     # print("They don't know the fare")
        #     a.avg_fare = CONST_FARE
        #     # TODO : round up the fare

        # if not self.professional:
        #     # if not professional, you don't consider supply in your decision making
        #     match_prob = 1
        # else:
        #     try:
        #         match_prob = a.prob_of_s
        #     except:
        #         print(df)
        #         print("that was df")
        #         print(a)
        #         print("that was a")
        #         print(self.professional)

        # a["relative_demand"] = a["total_pickup"] / a["total_pickup"].sum()
        # expected_revenue = (
        #     1 - PHI
        # ) * a.avg_fare * a.surge * match_prob * self.beta + a.bonus
        # expected_cost = (
        #     a.trip_distance_meter * self.rebl_cost
        # )  # doesn't take into account the distance travelled once the demand is picked up

        # a["expected_cost"] = expected_cost
        # a["numerator"] = (expected_revenue - expected_cost) * a["total_pickup"]
        # a["expected_revenue"] = expected_revenue
        # a["expected_profit"] = expected_revenue - expected_cost
        # a["prof"] = (expected_revenue - expected_cost) * a["total_pickup"]
        # a["prof"] = a["prof"].clip(lower=0)

        # # http://cs231n.github.io/linear-classify/#softmax
        # # for numerical stability
        # # a['prof'] -= np.max(a['prof'])
        # # a['prob'] = np.exp(a['prof'])/np.sum(np.exp(a['prof']))

        # # so that probability doesn't end up being negative

        # a["prob"] = a["prof"] / a["prof"].sum()

        # path_to_write = configs['output_path']
        # with open(path_to_write +'driver ' + str(self.id)+ '.csv', 'a') as f:
        #     a.to_csv(f, header = True, mode='a', index = False)

        a, prob = self._compute_attractiveness_of_zones(t, self.ozone, self.true_demand)
        try:
            selected = a.sample(n=1, weights=prob, replace=True)
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


        return selected["DOLocationID"].values[0]
        # return selected.index.values[0]


    def _calculate_fare(self, request, surge):
        """
        From Yellow Cab taxi webpage
        """
        distance_meters = request.dist
        p1 = 2.5
        p2 = 0.5 * 5 * 1609 * distance_meters
        f = p1 + surge * p2
        return f
    
    # @lru_cache(maxsize=None)
    # def _calc_rebl_cost(self, dist):
    #     """
    #     This should be based on the value of time and time it took to get to the destination
        
    #     """
    #     # dist = veh._get_dist_to_all_zones(veh.ozone)[["DOLocationID", "trip_distance_meter"]]
    #     # this is the costliest operation! 
    #     dist["costs"] = dist.trip_distance_meter * self.rebl_cost
    #     dist["costs"] = dist["costs"].apply(lambda x: np.around(x, 1))

    #     return dist

    # def realized_trip_profit(self, request, surge=1, driver_bonus=0):
    #     assert isinstance(request, Req)
    #     fare = self._calculate_fare(request, surge)
    #     rebl_cost = self._calc_rebl_cost(request)
    #     profit = (1 - PHI) * fare + driver_bonus - rebl_cost - self.IDLE_COST
    #     return profit

