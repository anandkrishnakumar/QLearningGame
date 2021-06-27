import numpy as np
from config import X, Y, actions

############################################
################## AGENTS ##################
############################################
class Agent:
    def __init__(self):
        self.alpha = 0.8
        self.discount = 0.8
        self.actions = {'l': 0, 'r': 1, 'u': 2, 'd': 3}
        self.actions_opp = {0: 'l', 1: 'r', 2: 'u', 3: 'd'}
        self.epsilon = 1

    def get_pos(self, state):
        # given state, get position as an int of car and trophy
        car = np.where(state==1)
        trophy = np.where(state==2)
        car = car[0][0]*Y + car[1][0]
        trophy = trophy[0][0]*Y + trophy[1][0]
        return (car, trophy)
    
class QAgent(Agent):
    def __init__(self):
        super().__init__()
        self.Q = np.zeros((X*Y, X*Y, 4)) # first dim is car position, second is trophy,third is actions
    
    def action(self, state, weird=False):
        positions = self.get_pos(state)
        value_func = self.Q[positions[0], positions[1]]
        ideal = np.argmax(value_func)
        uni = np.random.uniform()
        if (uni <= self.epsilon) or weird: # weird means choose a random action
            return np.random.choice(actions)
        else:
            return self.actions_opp[ideal]
    
    def train(self, state, action, reward, new_state, success):
        action = self.actions[action]
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
    
class DeepAgent(Agent):
    def __init__(self):
        super().__init__()
        pass