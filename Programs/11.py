import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    LeakyReLU,
    Reshape
)
from tensorflow.keras.optimizers import Adam

(x_train,_), (_,_) = mnist.load_data()

x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

def build_generator():
    return Sequential([
        Dense(128 * 7 * 7, input_dim = 100),
        LeakyReLU(0.2),
        Reshape((7,7,128)),

        Conv2DTranspose(64, kernal_size = 4, strides = 2, padding = "same"),
        LeakyReLU(0.2),
        Conv2DTranspose(1, kernal_size = 1, strides = 2, padding = "same", activation = "tanh")
        
    ])

def build_discriminator():
    return Sequential([
        Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape = (28,28,1)),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1, activation="sigmoid")
    ])
