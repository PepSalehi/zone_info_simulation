from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
from lib.Constants import PENALTY
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger
from lib.Env_aggregate_step_func import RebalancingEnv
import json
import argparse
import time 
import pickle 
from lib.utils import Model 
from lib.Constants import ZONE_IDS, DEMAND_SOURCE, INT_ASSIGN, FLEET_SIZE, PRO_SHARE, SURGE_MULTIPLIER, BONUS, PERCENT_FALSE_DEMAND
from lib.Constants import T_TOTAL_SECONDS, WARMUP_TIME_SECONDS, ANALYSIS_TIME_SECONDS, ANALYSIS_TIME_HOUR, WARMUP_TIME_HOUR
from lib.Constants import PERCE_KNOW
from lib.dqn_agent import DQNAgent
if __name__ == '__main__':
    
    ####
    parser = argparse.ArgumentParser(description="Simulation of drivers' behavior")
    parser.add_argument('-f', '--fleet', 
						help='Fleet sizes to simulate, formatted as comma-separated list (i.e. "-f 250,275,300")')
    parser.add_argument('-m', '--multiplier', 
						help='Surge multiplier, formatted as comma-separated list (i.e. "-m 1,1.5,2")')
    parser.add_argument('-b', '--bonus', type=int,
						help='Bonus')
    parser.add_argument('-d', '--demand', 
						help='Percent false demand ')
    parser.add_argument('-k', '--know', 
						help='Percent knowing fare, formatted as comma-separated list (i.e. "-m 1,1.5,2") ')
    parser.add_argument('-p', '--pro', 
						help='Percent pro drivers, formatted as comma-separated list (i.e. "-p 1,1.5,2") ')
    parser.add_argument('-av', '--av', 
						help='Percent AV drivers, formatted as comma-separated list (i.e. "-av 1,1.5,2") ')   
    parser.add_argument('-nb', '--nb', 
						help='number of steps to train Rl ') 

    args = parser.parse_args()
    if args.fleet:
        fleet_sizes = [int(x) for x in args.fleet.split(',')]
#        fleet_sizes = args.fleet
    else:
        fleet_sizes = FLEET_SIZE

    if args.multiplier:
        # surge = args.multiplier
        surges = [float(x) for x in args.multiplier.split(',')]
    else:
        surges = [SURGE_MULTIPLIER]
        
    if args.know:
        # surge = args.multiplier
        perc_know = [float(x) for x in args.know.split(',')]
    else:
        perc_know = [PERCE_KNOW]    
    
    if args.bonus:
        bonus = args.bonus
    else:
        bonus = BONUS
        
    if args.pro:
        
        pro_share = [float(x) for x in args.pro.split(',')]
    else:
        pro_share = [PRO_SHARE]  
        
    if args.demand:
        percent_false_demand = float(args.demand)
    else:
        percent_false_demand = PERCENT_FALSE_DEMAND
    
    if args.av:
        av_share = [float(x) for x in args.av.split(',')]
    else:
        av_share = [0.2]
   

    ####
    # 
    #   
    # stores the reward per episode
    scores = deque(maxlen=1000)
    p_trials = 10
    logger.setLevel(logger.ERROR)
    # env = gym.make(args.env_id)
    config = {
    "fleet_size" : fleet_sizes[0],
    "surge" : surges[0],
    "perc_k" : perc_know[0],
    "bonus" : bonus,
    "pro_s" : 0,
    "percent_false_demand" : 0,
    "av_share" : av_share[0]
    }
    env = RebalancingEnv(config=config )

    outdir = "./Outputs/dqn/"
    # if args.ddqn:
    #     outdir = "/tmp/ddqn-%s" % args.env_id
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    # env.seed(0)

    # instantiate the DQN/DDQN agent
    agent = DQNAgent(env.observation_space, env.action_space)

    # should be solved in this number of episodes
    episode_count = 100
    state_size = env.observation_space.shape[0]
    batch_size = 64

    # by default, CartPole-v0 has max episode steps = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000

    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state_n = env.reset()
        print("state_n dimension", state_n[0].shape)
        print("state")
        print(state_n[0])
        _sar = {}
        # state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        print("len(av)", len(env.model.av_vehs))
        loop_counter = 0
        while not done:
           
            loop_counter += 1
            _sar = {}
            actions = np.zeros(env.model.fleet_AV)
            for i, veh in enumerate([v for v in env.model.av_vehs]):
                # if it's rebalancing or serving demand, leave it alone
                if veh.should_move() :
                    # if it has to move because didn't get any matches, record that as a huge penalty 
                    if veh.waited_too_long():
                        print("waited too long after ", veh.time_idled)
                        _sar[i] =  [veh._info_for_rl_agent[0], veh._info_for_rl_agent[1], PENALTY]
                        veh._info_for_rl_agent = [] # reset state_action_reward 

                    elif not veh.just_started: 
                        # so it must have finished serving a match 
                        try:
                            assert len(veh._info_for_rl_agent) == 3
                        except AssertionError:
                            print(veh.idle)
                            print(veh.rebalancing)
                            print(veh.waited_too_long())
                            print(veh.TIME_TO_MAKE_A_DECISION)
                            print(veh.time_idled)
                            print(len(veh.reqs))
                            print([r.fare for r in veh.reqs])
                            print(veh._info_for_rl_agent)
                            raise AssertionError

                        _sar[i] = veh._info_for_rl_agent[:]
                        veh._info_for_rl_agent = [] # reset state_action_reward 

                    # state 
                    veh._info_for_rl_agent.append(state_n[i])
                    # action 
                    actions[i] = agent.act(veh._info_for_rl_agent[0])
                    veh._info_for_rl_agent.append( actions[i] )
                    
                else:
                    actions[i] = np.nan
          
            # get all the rewards  
            try:
                reward_n = [v[2] for _ , v in _sar.items()]
            except:
                print('sar', _sar)
                raise IndexError
            
            # Simulate 
            state_n, _ , done, _ = env.step(actions)
            

            for key, mem in _sar.items():
                agent.remember(_sar[key][0], _sar[key][1], _sar[key][2], state_n[key], done)

            print("len(agent.memory) ", len(agent.memory) )


            if loop_counter == 1000:
                print("state")
                print(state_n[0])
            # For profiling
            # if loop_counter == 10:
            #     done = True  
            # state = [pos, vel, theta, angular speed]
            # next_state = np.reshape(next_state, [1, state_size])
            # # store every experience unit in replay buffer
            # agent.remember(state, actions, reward, next_state, done)
            # state = next_state
            total_reward += np.sum(reward_n)
            print("total_reward", total_reward)

            # call experience relay
            if len(agent.memory) >= batch_size: # should this be indented and moved into the while loop? 
                # since this is now only happening once per episode 
                # I did, and it took forever to do anything
                print("started learning")
                # pickle.dump(agent, open( "my_agent.p", "wb" ) )
                agent.replay(batch_size)

    
        scores.append(total_reward)
        mean_score = np.mean(scores)

        # if mean_score >= win_reward[args.env_id] and episode >= win_trials:
        #     print("Solved in episode %d: Mean survival = %0.2lf in %d episodes"
        #           % (episode, mean_score, win_trials))
        #     print("Epsilon: ", agent.epsilon)
        #     agent.save_weights()
        #     break
        if episode % p_trials == 0:
            print("Episode %d: Mean reward = %0.2lf " %
                  (episode, mean_score))

    # close the env 