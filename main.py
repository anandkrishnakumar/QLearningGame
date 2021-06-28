import numpy as np

import config; config.init()
from config import X, Y, actions

from agents import QAgent, DeepAgent
from games import Game

# training libraries
from train import train
# from deepQ import deepQtrain

from tests import test
import plots

if __name__ == "__main__":
    episodes = 2
    
    # a = QAgent()
    # times_taken = train(Game, a)
    
    a = DeepAgent()
    times_taken = train(Game, a, episodes)
    
    # testing with extreme initial state
    istate = np.zeros((X, Y)); istate[0, 0] = 1; istate[-1, -1] = 2
    # test(Game, a, init=istate)