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
    model = keras.Sequential()
    model.add(keras.Input(shape=(X, Y,)))
    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(1))
    
    return model

def deepQtrain(Game, agent, episodes=1000):
    """Deep Q-learning training"""
    
    model = create_network()
    
    return model.output_shape