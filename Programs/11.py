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

generator = build_generator()
discriminator = build_discriminator()

discriminator.complie(
    optimizer = Adam(0.0002, 0.5),
    loss = "binary_crossentropy",
    metrics = ['accuracy']
)

discriminator.trainable = False

gan_input = tf.keras.Input(shape = (100,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.models(gan_input,gan_output)

gan.compile(
    optimizer = Adam(0.0002, 0.5),
    loss = "binary_crossentropy",
    metrics = ['accuracy']
)

def show_imgs(epochs):
    noise = np.random.normal(0,1, (16, 100))
    gen_imgs = generator.predict(noise, verbose = 0)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(4,4, figsize = (4,4))
    cnt = 0

    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            cnt += 1
    plt.show()

def train_gan(epochs = 1000, batch_size = 128):
    half_batch = batch_size // 2

    for epoch in epochs:
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_imgs = x_train[idx]

        noise = np.random.normal(0,1, (half_batch,100))
        fake_imgs = generator.predict(noise, verbose = 0)

        d_loss_real = discriminator.train_on_batch(
            real_imgs, np.ones((half_batch, 1))
        )

        d_loss_fake = discriminator.train_on_batch(
            fake_imgs, np.zeros((half_batch,1))
        )

        noise = np.random.normal(0,1, (batch_size,100))        
        valid_y = np.ones((batch_size,1))
        g_loss = gan.train_on_batch(noise, valid_y)

        print()
        show_imgs(epoch)

train_gan(epochs=5, batch_size=128)        
                