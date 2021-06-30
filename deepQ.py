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
    learning_rate = 0.01
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.Input(shape=(X*Y,)))
    model.add(layers.Dense(100, activation='relu', 
                            kernel_initializer=init))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(len(actions)))
    
    model.compile(loss="huber", 
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate))

    return model