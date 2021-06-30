"""Plotter

Contains functions that can be used to implement
various plots.

This file contains the following functions:
    
    * train_time - to plot the time taken to train an agent
"""

import numpy as np
import matplotlib.pyplot as plt
# import seaborn; seaborn.set()

def train_time(times_taken):
    """2D plot: iterations vs. cumulative training time."""
    plt.plot(np.cumsum(times_taken))
    plt.xlabel("Training iteration")
    plt.ylabel("Cumulative time taken (s)")
    plt.title("Training time")
    plt.show()
    
def plot_conv_filter(agent):
    filters, bias = agent.targ_net.layers[1].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    f = filters.reshape(2, 2)
    plt.imshow(f, cmap='gray')