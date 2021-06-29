"""Agents

This file contains agents that are associated with
different types of learning algorithms

Contains the following classes:
    
    * Agent - parent class
    * QAgent - Q-learning agent
    * DeepAgent - deep Q-learning agent
"""

import numpy as np
from config import X, Y, actions
import deepQ

from tensorflow.keras.models import load_model

############################################
################## AGENTS ##################
############################################
class Agent:
    """Parent agent class"""
    
    def __init__(self):
        self.alpha = 0.8
        self.discount = 0.8
        self.actions_map = {'l': 0, 'r': 1, 'u': 2, 'd': 3}
        self.actions_opp = {0: 'l', 1: 'r', 2: 'u', 3: 'd'}
        self.epsilon = 1

    def get_pos(self, state):
        """Gets position of car, trophy as int type."""
        car = np.where(state==1)
        trophy = np.where(state==2)
        car = car[0][0]*Y + car[1][0]
        trophy = trophy[0][0]*Y + trophy[1][0]
        return (car, trophy)
    
    def time_until_epsilon_min(self, epsilon_decrement, epsilon_min):
        """Just a helper function to calculate time until epsilon reaches min."""
        
        return np.log(epsilon_min)/np.log(epsilon_decrement)
    
    
##############################################################################
    
class QAgent(Agent):
    """Q-learning agent"""
    
    def __init__(self):
        super().__init__()
        self.Q = np.zeros((X*Y, X*Y, 4))
        # first dim is car position, second is trophy,third is actions
        
        self.epsilon_decrement = 0.996
        self.epsilon_min = 0.01
    
    def action(self, state, weird=False):
        """Returns an action for a given state."""
        positions = self.get_pos(state)
        value_func = self.Q[positions[0], positions[1]]
        ideal = np.argmax(value_func)
        uni = np.random.uniform()
        if (uni <= self.epsilon) or weird: # weird means choose a random action
            return np.random.choice(actions)
        else:
            return self.actions_opp[ideal]
    
    def train(self, state, action, reward, new_state, success):
        """Trains the bot."""
        action = self.actions_map[action]
        state_pos = self.get_pos(state)
        car = state_pos[0]
        trophy = state_pos[1]
        if not success:
            newstate_pos = self.get_pos(new_state)
            carnew = newstate_pos[0]
            trophynew = newstate_pos[1]
            self.Q[car, trophy, action] += self.alpha * (reward + self.discount*max(self.Q[carnew, trophynew]) - self.Q[car, trophy, action])
        elif success:
            self.Q[car, trophy, action] = reward
            self.epsilon = self.epsilon*self.epsilon_decrement if self.epsilon > \
                self.epsilon_min else self.epsilon_min
        # update Q value at state_pos[0], state_pos[1]
        # print("Q updated")
        
        

##############################################################################
    
class DeepAgent(Agent):
    """Deep Q-learning agent"""
    
    def __init__(self):
        super().__init__()
        
        self.targ_net = deepQ.create_network()
        self.store_counter = 0
        self.STORE_LIMIT = 50
        self.train_counter = 0
        self.TRAIN_COUNT = 25 # train after every 20 actions
        
        # storage for training
        self.state_store = np.zeros((self.STORE_LIMIT , X*Y))
        self.new_state_store = np.zeros((self.STORE_LIMIT , X*Y))
        self.action_store = np.zeros(self.STORE_LIMIT, dtype=np.int32)
        self.reward_store = np.zeros(self.STORE_LIMIT)
        self.success_store = np.zeros(self.STORE_LIMIT)
        
        
        # epsilon stuff
        self.epsilon_decrement = 0.985
        self.epsilon_min = 0.01
        
        # to save trained model
        self.network_fname = r"trained_model//model.h5"
        
    def action(self, state, weird=False):
        """
        Takes an action for a given state. 
        Can be random/ideal depending on epsilon.
        """
        value_func = self.targ_net.predict(state.reshape((1, X*Y)), batch_size=1)[0]
        ideal = np.argmax(value_func)
        uni = np.random.uniform()
        if (uni <= self.epsilon) or weird: # weird means choose a random action
            return np.random.choice(actions)
        else:
            return self.actions_opp[ideal]
        
    def train(self, state, action, reward, new_state, success):
        """
        Trains the bot.
        
        First, it calculates the updated Qvalue for the given data tuple using
        the Bellman equation. Then it stores this updated Qvalue and the state.
        
        Secondly, the train counter is incremented. If the train counter has
        reached its limit, the model is trained by calling train_do, and the 
        counter is reset.        
        """
        
        # store state & updated Qvals
        self.save_dat(state, action, reward, new_state, success)
        self.store_counter += 1
        
        self.train_counter += 1
        if self.train_counter == self.TRAIN_COUNT:
            self.train_do()
            self.train_counter = 0
    
    def train_do(self):
        """
        Actually implements training of the neural network.
        
        It draws a random sample of stored data to train on.
        Then, the neural network is fit on the sampled training data.
        """
        
        # takes a random set of stored data
        max_store = min(self.STORE_LIMIT, self.store_counter)
        train_indices = np.random.choice(max_store, 
                                         size=self.TRAIN_COUNT)
        train_states = self.state_store[train_indices]
        train_new_states = self.new_state_store[train_indices]
        train_action = self.action_store[train_indices]
        train_reward = self.reward_store[train_indices]
        train_success = self.success_store[train_indices]
        
        
        Qvals = self.targ_net.predict(train_states)
        new_Qvals = self.targ_net.predict(train_new_states) # of next states
        batch_index = np.arange(self.TRAIN_COUNT, dtype=np.int32)
        Qvals[batch_index, train_action] = train_reward +  \
            self.alpha * np.max(new_Qvals, axis=1) * train_success
        
        # train the NN with states & updated Qvals
        self.targ_net.fit(train_states, Qvals)
        
        self.epsilon = self.epsilon*self.epsilon_decrement if self.epsilon > \
            self.epsilon_min else self.epsilon_min
        
    def save_dat(self, state, action, reward, new_state, success):
        save_index = self.store_counter % self.STORE_LIMIT
        self.state_store[save_index] = state.flatten()
        self.new_state_store[save_index] = state.flatten()
        self.action_store[save_index] = self.actions_map[action]
        self.reward_store[save_index] = reward
        self.success_store[save_index] = success
        
    def save_network(self):
        """Helper function to save trained model."""
        self.targ_net.save(self.network_fname)
        
    def load_network(self):
        """Helper function to load trained model."""
        self.targ_net = load_model(self.network_fname)        