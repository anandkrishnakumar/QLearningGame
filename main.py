import numpy as np

import config; config.init()
from config import X, Y, actions

from classes import Agent, Game
from qlearning import Qtrain
from tests import test

import plots

a = Agent()
episodes = 1000
times_taken = Qtrain(Game, a)

# plotting training time

# testing with extreme initial state
istate = np.zeros((X, Y)); istate[0, 0] = 1; istate[-1, -1] = 2
test(Game, a)