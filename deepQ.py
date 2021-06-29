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
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(len(actions)))
    
    model.compile(loss="mse", optimizer='adam')
    
    return model