from keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from collections import deque
import numpy as np
import random
from collections import deque
import argparse
import gym
from gym import wrappers, logger
from time import time


class DQNAgent():
    """
    Trains a DQN/DDQN to solve CartPole-v0 problem, applicable to the rideshare case.
    """

    def __init__(self, state_space, action_space, args=None, episodes=500):
        """
        Creates an agent that interacts with the gym environment.

        @param state_space
        @param action_space
        @param args
        @param episodes (int)
        """
        self.action_space = action_space
        self.ddqn = True
        # experience buffer
        self.memory = deque(maxlen=10000)

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
        self.q_model.compile(loss='mse', optimizer=Adam(lr=0.01))
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

    def build_model(self, n_inputs, n_outputs):
        """
        Builds Q-network architecture.
        @param n_inputs: (int) input dimensions
        @param n_outputs: (int) output dimensions
        @return: (Model) q-network object
        """
        inputs = Input(shape=(n_inputs[0], n_inputs[1], 1), name='state')
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
        # x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(n_outputs, activation='softmax', name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model

    def load_weights(self, f):
        """
        Assigns weights to q-network given filename f.
        @param f: (str) filename
        """
        assert isinstance(f, str)
        self.q_model.load_weights(f)
        print("used prior weights")

    def save_weights(self, f):
        """
        Save Q Network params to a file
        @param f: (str) filename
        """
        self.q_model.save_weights(f)

    def update_weights(self):
        """
        Copy trained Q Network params to target Q Network
        """
        self.target_q_model.set_weights(self.q_model.get_weights())

    def act(self, state):
        """
        Eps-greedy policy. With probability epsilon, performs a random action. With
        probability 1- epsilon, exploits by choosing the action with the maximum
        Q-value.
        @param state
        @return: the chosen action
        """
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit
        state = state.reshape(-1, state.shape[0], state.shape[1], 1)
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Store experiences in the replay buffer
        @param state
        @param action
        @param reward: (float)
        @param next_state
        @param done: (bool) indicator whether simulation is done
        """
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

    def get_target_q_value(self, next_state, reward):
        """
        Computes Q_max
        Use of target Q Network solves the non-stationarity problem.
        @param next_state
        @param reward: (float)
        @return: the maximized q-value
        """
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

    def replay(self, batch_size):
        """
        Fits DQN using experience replay, which addresses the correlation issue between samples
        @param batch_size: (int)
        @return: None
        """
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

            q_values = self.q_model.predict(state)  # mind the 's': q_values and q_value

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
                         np.array(q_values_batch),
                         # so, q_model will predict q(s,a), q_values are q_max, so the difference will be the loss
                         batch_size=batch_size,
                         epochs=10,
                         verbose=0)
        # callbacks=[self.tensorboard])

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_weights()

        self.replay_counter += 1

    def update_epsilon(self):
        """
        Decreases the exploration, increase exploitation by updating epsilon.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
