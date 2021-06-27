import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

def train_time(times_taken):
    plt.plot(np.cumsum(times_taken))
    plt.xlabel("Training iteration")
    plt.ylabel("Cumulative time taken (s)")
    plt.title("Training time")
    plt.show()