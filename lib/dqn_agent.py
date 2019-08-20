"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""

from keras.layers import Dense, Input, Conv2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from collections import deque
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger
from time import time

class DQNAgent():
    def __init__(self, state_space, action_space, args=None, episodes=500):

        self.action_space = action_space
        self.ddqn = True
        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.95
        # self.tensorboard = TensorBoard(log_dir="Outputs/{}".format(time()))

        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0
        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(episodes))

        # Q Network weights filename
        self.weights_file = 'dqn_cartpole.h5'
        # Q Network for training
        print("state_space", state_space.shape)
        self.state_space = state_space
        n_inputs = state_space.shape
        n_outputs = action_space.n
        self.q_model = self.build_model(n_inputs, n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam())
        # target Q Network
        self.target_q_model = self.build_model(n_inputs, n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0
        # self.ddqn = True if args.ddqn else False
        if self.ddqn:
            print("----------Double DQN--------")
        else:
            print("-------------DQN------------")

    
    # Q Network 
    def build_model(self, n_inputs, n_outputs):
        inputs = Input(shape=(n_inputs[0], n_inputs[1], 1), name='state')
        x = Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_outputs, activation='softmax', name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model


    def load_weights(self, f):
        assert isinstance(f, str)
        self.q_model.load_weights(f)

    # save Q Network params to a file
    def save_weights(self, f):
        self.q_model.save_weights(f)


    # copy trained Q Network params to target Q Network
    def update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())


    # eps-greedy policy
    def act(self, state):
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit
        state = state.reshape(-1, state.shape[0], state.shape[1], 1)
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        return np.argmax(q_values[0])


    # store experiences in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory.append(item)


    # compute Q_max
    # use of target Q Network solves the non-stationarity problem
    def get_target_q_value(self, next_state, reward):
        # max Q value among next state's actions
        if self.ddqn:
            # DDQN
            # current Q Network selects the action
            # a'_max = argmax_a' Q(s', a')
            action = np.argmax(self.q_model.predict(next_state)[0])
            # target Q Network evaluates the action
            # Q_max = Q_target(s', a'_max)
            q_value = self.target_q_model.predict(next_state)[0][action]
        else:
            # DQN chooses the max Q value among next actions
            # selection and evaluation of action is on the target Q Network
            # Q_max = max_a' Q_target(s', a')
            q_value = np.amax(self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value


    # experience replay addresses the correlation issue between samples
    def replay(self, batch_size):
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            # state is (64 , 3), need to change it to (-1, 64, 3 ,1)
            state = state.reshape(-1, state.shape[0], state.shape[1], 1)
            next_state = next_state.reshape(-1, next_state.shape[0], next_state.shape[1], 1)

            q_values = self.q_model.predict(state) # mind the 's': q_values and q_value 

            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            # print("q_value", q_value)
            # print("q_values", q_values)
            # print("action", action)
            # print("len(q_values)",len(q_values))
            # print(q_values[0])
            q_values[0][int(action)] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch), # so, q_model will predict q(s,a), q_values are q_max, so the difference will be the loss
                         batch_size=batch_size)
                         # epochs=1,
                         # verbose=0)
                         # callbacks=[self.tensorboard])

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_weights()

        self.replay_counter += 1

    
    # decrease the exploration, increase exploitation
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay