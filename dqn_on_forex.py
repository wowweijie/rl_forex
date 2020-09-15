# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 02:48:48 2020

@author: User
"""

from collections import deque
from pyrfc3339 import parse
from influxdb import InfluxDBClient
import math
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta, timezone
from tensorflow.keras import layers, models

# initializing data from InfluxDB

client = InfluxDBClient(host='localhost', port=8086)
results = client.query('select "bid" from "dukascopy"."autogen"."EURUSD" order by time desc limit 3')

upper_limit_timestamp = 1577717700
lower_limit_timestamp = 1577716500
bid_values = []
ask_values = []
bid_volume = []
ask_volume = []

query_statement = 'select "bid", "ask", "bid_vol", "ask_vol" from "dukascopy"."autogen"."EURUSD" where time >= ' +\
str(int(lower_limit_timestamp)) + '000000000' +\
' and time <= ' + str(int(upper_limit_timestamp)) + '000000000 order by time asc'
results = client.query(query_statement)

df = pd.DataFrame(columns = ["bid", "ask", "bid_vol", "ask_vol"])
for point in results.get_points(): 
    row = {"time" : point["time"],
           "bid" : point["bid"], 
           "ask" : point["ask"],
           "bid_vol" : point["bid_vol"],
           "ask_vol" : point["ask_vol"],
           }
    df = df.append(row, ignore_index = True)

#df.to_csv('chicago_pmi_time_frame.csv')

class Timer: 
        def start(self):
            self.starting_time = pd.Timestamp.utcnow()

        def stop(self):
            snapshot_time = pd.Timestamp.utcnow()-self.starting_time
            print(snapshot_time)
            return snapshot_time
    
class Agent():
    """Sets up a reinforcement learning agent to play in a game environment."""
    def __init__(self, network, memory, epsilon_decay, action_size):
        """Initializes the agent with DQN and memory sub-classes.

        Args:
            network: A neural network created from deep_q_network().
            memory: A Memory class object.
            epsilon_decay (float): The rate at which to decay random actions.
            action_size (int): The number of possible actions to take.
        """
        self.network = network
        self.action_size = action_size
        self.memory = memory
        self.epsilon = 1  # The chance to take a random action.
        self.epsilon_decay = epsilon_decay

    def act(self, state, training=False):
        """Selects an action for the agent to take given a game state.

        Args:
            state (list of numbers): The state of the environment to act on.
            traning (bool): True if the agent is training.

        Returns:
            (int) The index of the action to take.
        """
        if training:
            # Random actions until enough simulations to train the model.
            if len(self.memory.buffer) >= self.memory.batch_size:
                self.epsilon *= self.epsilon_decay

            if self.epsilon > np.random.rand():
                print("Exploration!")
                return random.randint(0, self.action_size-1)

        # If not acting randomly, take action with highest predicted value.
        print("Exploitation!")
        state_batch = np.expand_dims(state, axis=0)
        print("state_batch :", state_batch)
        predict_mask = np.ones((1, self.action_size,))
        action_qs = self.network.predict([state_batch, predict_mask])
        return np.argmax(action_qs[0])
    
    def learn(self):
        """Trains the Deep Q Network based on stored experiences."""
        batch_size = self.memory.batch_size
        if len(self.memory.buffer) < batch_size:
            return None
    
        # Obtain random mini-batch from memory.
        state_mb, action_mb, reward_mb, next_state_mb, done_mb = (
            self.memory.sample())
    
        # Get Q values for next_state.
        predict_mask = np.ones(action_mb.shape + (self.action_size,))
        next_q_mb = self.network.predict([next_state_mb, predict_mask])
        next_q_mb = tf.math.reduce_max(next_q_mb, axis=1)
    
        # Apply the Bellman Equation
        target_qs = (next_q_mb * self.memory.gamma) + reward_mb
        target_qs = tf.where(done_mb, reward_mb, target_qs)
    
        # Match training batch to network output:
        # target_q where action taken, 0 otherwise.
        action_mb = tf.convert_to_tensor(action_mb, dtype=tf.int32)
        action_hot = tf.one_hot(action_mb, self.action_size)
        target_mask = tf.multiply(tf.expand_dims(target_qs, -1), action_hot)
    
        return self.network.train_on_batch(
            [state_mb, action_hot], target_mask, reset_metrics=False
        )

        
class Memory():
    """Sets up a memory replay buffer for a Deep Q Network.

    A simple memory buffer for a DQN. Randomly selects state
    transitions with uniform probability.
    
    Args:
        memory_size (int): How many elements to hold in the memory buffer.
        batch_size (int): The number of elements to include in a replay batch.
        gamma (float): The "discount rate" used to assess Q values.
    """
    def __init__(self, memory_size, batch_size, gamma):
        self.buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma

    def add(self, experience):
        """Adds an experience into the memory buffer.

        Args:
            experience: a (state, action, reward, state_prime, done) tuple.
        """
        self.buffer.append(experience)

    def sample(self):
        """Uniformally selects from the replay memory buffer.

        Uniformally and randomly selects sequential experiences to train the nueral
        network on. Transposes the experiences to allow batch math on
        the experience components.

        Returns:
            (list): A list of lists with structure [
                [states], [actions], [rewards], [state_primes], [dones]
            ]
        """
        buffer_size = len(self.buffer)        
        index = np.random.choice(
            np.arange(buffer_size), size=self.batch_size, replace=False)

        # Columns have different data types, so numpy array would be awkward.
        batch = np.array([self.buffer[i] for i in index], dtype=object).T.tolist()
        states_mb = tf.convert_to_tensor(np.array(batch[0], dtype=np.float32))
        actions_mb = np.array(batch[1], dtype=np.int8)
        rewards_mb = np.array(batch[2], dtype=np.float32)
        states_prime_mb = np.array(batch[3], dtype=np.float32)
        dones_mb = batch[4]
        return states_mb, actions_mb, rewards_mb, states_prime_mb, dones_mb
    
class Environment():
    """Sets up a continuous timeseries environment from unevenly spaced tick data of window containing news event
        
        Args: 
            timeframe (pd.Series) : A pandas series of timestamp in RFC3339 format
            features (pd.DataFrame) : A pandas dataframe of features recorded at each timestamp
            event_time (datetime) : datetime of news event (tzinfo='UTC')
        
        Attributes:
            time_series (pd.DataFrame) : A pandas dataframe with timedelta index (timedelta==0 at event_time) and features
            feature_list (list) : list of feature names in time_series column order
            event_time (datetime) :  datetime of news event 
            
    """
    def __init__(self, timeframe, features, event_time):
        
        self.slippage = 500000
        self.feature_list = list(features)
        self.event_time = event_time
        self.action_space = [0, 1, -1]
        

        timedelta_index = pd.TimedeltaIndex(timeframe.apply(lambda x : parse(x) - event_time))
        
        self.time_series = pd.DataFrame(features)       
        self.time_series = self.time_series.set_index(timedelta_index)
        self.time_series["signal"] = np.zeros(len(self.time_series.index))
        
        self.start_timedelta_index = np.timedelta64(timedelta_index[0])
        
        
        
    def run(self):
        
                
        # np timedelta is converted to float
        time = np.array([np.timedelta64(self.start_timedelta_index)]).astype('float64')
        data = np.array(self.time_series.iloc[0])
        return np.concatenate((time, data))
    
    # updates features including timedelta of state with latest features
    def update_features(self, state):
        
        pd_timedelta_now = np.timedelta64(1, 's') + np.timedelta64(int(state[0]), 'us')
        state[0] = pd_timedelta_now.astype('int')
        
        print(self.time_series.iloc[self.time_series.index.get_loc(pd_timedelta_now, method='ffill'), :-1])
        state[1:-1] = np.array(self.time_series.iloc[self.time_series.index.get_loc(pd_timedelta_now, method='ffill'), : -1])
        print(state)
        return state 
        
        
    
    def take_action(self, state, action):
        """Sets up a continuous timeseries environment from unevenly spaced tick data of window containing news event
        
            Args: 
                action (int) : 0 : HOLD, 1 : BUY, 2 : SELL        
                state (pd.Series) : current state where action is executed on
            
            Attributes:
                time_series (pd.DataFrame) : A pandas dataframe with timedelta index (timedelta==0 at event_time) and features
                feature_list (list) : list of feature names in time_series column order
                event_time (datetime) : datetime of news event
                    
        """
    
        # whether state is at end of time_series
        terminal_state = False
        
        # transform action(argmax) into directional action, 0 - HOLD, 1 - BUY, -1 - SELL
        action = self.action_space[action]
        
        # if state signal is different from action:
        if state[-1] != action:
                            
                # adjusting for slippage
                state[0] += self.slippage
                pd_timedelta_now = pd.Timedelta(state[0], unit='microsecond')
                
                state[-1] = action
                
                state[1:-1] = np.array(self.time_series.iloc[self.time_series.index.get_loc(pd_timedelta_now, method='ffill'), :-1])
        
        else :
                
                # return state_prime as one second from state
                state = self.update_features(state)
            
        if pd.Timedelta(state[0], unit='microsecond') > self.time_series.iloc[-1].name:
            terminal_state = True
        
            
        return state, terminal_state
    
        
    # get reward(state, state_prime)
    def get_reward(self, state, state_prime, terminal_state):
        
        reward = 0
        
        prev_signal = state[-1]
        new_signal = state_prime[-1]
        
        # if previous action was in hold then reward remains at zero
        
        # if previous action was long
        if new_signal == 1:
            
            
            exit_price = state_prime[self.feature_list.index("ask")+1]
            exit_price = self.truncate(exit_price)
            print("exit_price :", exit_price)
            entry_price = state[self.feature_list.index("ask")+1]
            entry_price = self.truncate(entry_price)
            print("entry_price :", entry_price)
            
            reward = exit_price - entry_price
            
            
        # if previous action was in short 
        elif new_signal == -1:
            
            exit_price = state_prime[self.feature_list.index("bid")+1]
            exit_price = self.truncate(exit_price)
            print("exit_price :", exit_price)
            entry_price = state[self.feature_list.index("bid")+1]
            entry_price = self.truncate(entry_price)
            print("entry_price :", entry_price)
            
            reward = entry_price - exit_price
            
        # if entering into a new position
        if new_signal != 0 and prev_signal != new_signal :
            
            bid_price = state_prime[self.feature_list.index("bid")+1]
            bid_price = self.truncate(bid_price)
            print("bid_price :", bid_price)
            ask_price = state_prime[self.feature_list.index("ask")+1]
            ask_price = self.truncate(ask_price)
            print("ask_price :", ask_price)
            
            # bid-ask spread is subtracted from reward
            reward += bid_price - ask_price
            
                
        print("get_reward :", reward)
        
        return reward
    
    def truncate(self, number):
        """truncates to pips
        
        """
        
        factor = 10.0 ** 5
        return math.trunc(number * factor) / 10
        
        
        
    

def deep_q_network(state_shape, action_size, learning_rate, hidden_neurons):
    """Creates a Deep Q Network to emulate Q-learning.

    Creates a two hidden-layer Deep Q Network. Similar to a typical nueral
    network, the loss function is altered to reduce the difference between
    predicted Q-values and Target Q-values.

    Args:
        state_shape: a tuple of ints representing the observation space.
        action_size (int): the number of possible actions.
        learning_rate (float): the neural network's learning rate.
        hidden_neurons (int): the number of neurons to use per hidden
            layer.
    """
    state_input = layers.Input(state_shape, name='frames')
    actions_input = layers.Input((action_size,), name='mask')

    hidden_1 = layers.Dense(hidden_neurons, activation='relu')(state_input)
    hidden_2 = layers.Dense(hidden_neurons, activation='relu')(hidden_1)
    q_values = layers.Dense(action_size)(hidden_2)
    masked_q_values = layers.Multiply()([q_values, actions_input])

    model = models.Model(
        inputs=[state_input, actions_input], outputs=masked_q_values)
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

def print_state(state, step, reward=None):
    format_string = 'Step {0} - time: {1:.3f}, bid: {2:.3f}, ask: {3:.3f}, bid_vol:{4:.3f}, ask_vol:{5:.3f}, position:{6:.1f}'
    print(format_string.format(step, *tuple(state), reward))


#network
state_shape = (6,)
action_size = 3
test_learning_rate = 0.3
test_hidden_neurons = 3
test_network = deep_q_network(state_shape, action_size, test_learning_rate, test_hidden_neurons)

#memory
test_memory_size = 1000
test_batch_size = 200
test_sequence_size = 60
test_gamma = .9  # Unused here. For learning.
test_memory = Memory(test_memory_size, test_batch_size, test_gamma)

#agent
test_epsilon_decay = .95
test_agent = Agent(
    test_network, test_memory, test_epsilon_decay, action_size)


#training - one episode
test_env = Environment(df['time'], df[["bid", "ask", "bid_vol", "ask_vol"]], 
                       datetime(2019, 12, 30, 14, 45, tzinfo=timezone.utc))



def EpisodicTrain(episode_reward,  limit): 
    step = 0
    episode_reward = 0
    done = False
    qwert = 0
    
    #initialize state
    state = test_env.run()
    
    
    while not done:
        #state = test_env.update_features(state, step)
        print("state :" , state)
        action = test_agent.act(state, training=True)
        print("action taken :" , action)
        state_prime, done = test_env.take_action(state, action)
        print("state_prime :" , state_prime)
        reward = test_env.get_reward(state, state_prime, done)
        if action != 1 :
            qwert += 1
            if qwert > limit :
                print("limit reached")
                break
        episode_reward += reward
        test_agent.memory.add((state, action, reward, state_prime, done)) # New line here
        step += 1
        state = state_prime
        print_state(state, step, reward)
    
    loss = test_agent.learn()
    print("loss :", loss)
    print("Game over! Score =", episode_reward)
    
    return episode_reward
    
    