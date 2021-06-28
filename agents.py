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
    
class QAgent(Agent):
    """Q-learning agent"""
    
    def __init__(self):
        super().__init__()
        self.Q = np.zeros((X*Y, X*Y, 4)) # first dim is car position, second is trophy,third is actions
    
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
            self.epsilon -= 0.005
        # update Q value at state_pos[0], state_pos[1]
        # print("Q updated")
        
        

############################################################################



    
class DeepAgent(Agent):
    """Deep Q-learning agent"""
    
    def __init__(self):
        super().__init__()
        self.hist_net = deepQ.create_network()
        self.target_net = deepQ.create_network()
        self.target_net.set_weights(self.hist_net.get_weights())
        self.BUFFER = 2 # copy target_net to hist_net
        self.buffer_count = 0
        
        self.TRAIN_BUFFER = 1 # train network after these many moves
        self.train_buffer_count = 0
        self.states = np.zeros((self.TRAIN_BUFFER, X, Y))
        self.actions = [i for i in range(self.TRAIN_BUFFER)]
        self.rewards = np.zeros(self.TRAIN_BUFFER)
        self.new_states = np.zeros((self.TRAIN_BUFFER, X, Y))
        # self.successes = [i for i in range(self.TRAIN_BUFFER)]
        self.successes = np.zeros(self.TRAIN_BUFFER)
        pass
    
    def prep_states(self, states):
        """Flattens states."""
        if len(states.shape) == 3:
            num_states = states.shape[0]
        else:
            num_states = 1
        return states.reshape(num_states, X*Y)
    
    def buffer_stuff(self):
        """Handles updation of neural network."""
        self.buffer_count += 1
        if self.buffer_count == self.BUFFER:
            self.hist_net.set_weights(self.target_net.get_weights()) 
            self.buffer_count = 0
            print("buffer cleared")
        
    
    def action(self, state, weird=False):
        """Returns an action for a given state."""
        value_func = self.target_net.predict(self.prep_states(state))
        ideal = np.argmax(value_func)
        uni = np.random.uniform()
        if (uni <= self.epsilon) or weird: # weird means choose a random action
            return np.random.choice(actions)
        else:
            return self.actions_opp[ideal]
        
    def train(self, state, action, reward, new_state, success):
        """Stores history and starts training if 
        training buffer has been reached."""
        
        # update training buffer
        if self.train_buffer_count < self.TRAIN_BUFFER - 1:
            self.train_buffer_count += 1
        else:
            self.train_do()
            self.train_buffer_count = 0
        
        action = self.actions_map[action]
        index = self.train_buffer_count
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.successes[index] = success
    
    def train_do(self):
        """Trains the bot once TRAIN_BUFFER is reached."""
        
        # get current Q-value for given state and for each action from hist_net
        Qvals = self.target_net.predict(self.prep_states(np.stack(self.states)))
        future_Qvals = self.hist_net.predict(self.prep_states(np.stack(self.new_states)))
        
        ### for debugging
        print(Qvals)
        
        # update the Q-value for chosen action
        for count in range(self.TRAIN_BUFFER):
            action = self.actions[count]
            print(action)
            if not self.successes[count]:
                Qvals[count, action] += (self.alpha 
                                       * (self.rewards[count] 
                                          + self.discount*max(future_Qvals[count]) 
                                          - Qvals[count, action])
                                       )
                # train target_net with input as state and output as updated Q_values                    
            elif self.successes[count]:
                Qvals[count, action] = self.rewards[count]
                
                
        self.target_net.fit(self.prep_states(np.stack(self.states)), Qvals)
        
        
        ### debugging
        print(Qvals)
        print(self.target_net.predict(self.prep_states(np.stack(self.states))))
        
        self.epsilon -= 0.001
        print("epsilon", self.epsilon)
        # update Q value at state_pos[0], state_pos[1]
        # print("Q updated")
        self.buffer_stuff()