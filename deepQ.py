import numpy as np
import time
import logging
logging.captureWarnings(True)

import config

from config import X, Y, actions

# loading neural network libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_network():
    """Creates a new Keras network for Deep Q-learning."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(X*Y,)))
    model.add(layers.Dense(X*Y, activation='relu'))
    model.add(layers.Dense(len(actions), activation='linear'))
    
    model.compile(loss="mean_squared_error", optimizer='rmsprop')
    
    return model