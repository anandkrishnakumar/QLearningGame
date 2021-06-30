"""Games

This file contains the game defintions.

Work in progress. At the moment, this
only contains one game.
"""

import numpy as np
from config import X, Y, actions

###########################################
################## GAMES ##################
###########################################
class Game:
    """Game class"""
    
    def __init__(self):
        # print("Game created")
        self.state = np.zeros((X, Y))
        startx = np.random.randint(X)
        starty = np.random.randint(Y)

        self.state[startx, starty] = 1

        trophyx = np.random.randint(X)
        trophyy = np.random.randint(Y)

        while (startx==trophyx) and (starty==trophyy):
            trophyx = np.random.randint(X)
            trophyy = np.random.randint(Y)

        self.state[trophyx, trophyy] = 2

        self.actions = ['l', 'r', 'u', 'd']
        self.success = False
        self.moves = 0 # number of moves
        return None
    
    def play(self, action):
        """Updates state, given an action."""
        self.moves += 1
        pos = np.where(self.state==1)
        self.state[pos] = 0
        pos = [pos[0][0], pos[1][0]]
        if action=='l':
            pos[1] -= 1
            # print("left")
        elif action=='r':
            pos[1] += 1
            # print("right")
        elif action=='u':
            pos[0] -= 1
            # print("up")
        elif action=='d':
            pos[0] += 1
            # print("down")
        else:
            print("Invalid action")
        unadjusted_pos = [i for i in pos]
        pos[0] = max(0, pos[0])
        pos[0] = min(X-1, pos[0])
        pos[1] = max(0, pos[1])
        pos[1] = min(Y-1, pos[1])
        # negative reward for invalid moves
        reward = -50*np.any(np.array(pos) != np.array(unadjusted_pos))
        # penalise number of moves
        reward -= self.moves
        pos = (np.array([pos[0]]), np.array([pos[1]]))
        if self.state[pos] == 2:
            # print("Success!")
            self.success = True
            reward = 100
            return reward
        else:
            self.state[pos] = 1
            return reward
        
        
    def get_state(self):
        """Returns current state."""
        return self.state
    
    def set_state(self, state):
        """Set game state explicitly."""
        self.state = np.copy(state)