import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1. Load and Normalize MNIST Dataset
# ==========================================
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Scale to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)


# ==========================================
# 2. Build Models (Generator & Discriminator)
# ==========================================
def build_generator():
  return Sequential([
      Dense(128 * 7 * 7, input_dim=100),
      LeakyReLU(0.2),
      Reshape((7, 7, 128)),
      Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
      LeakyReLU(0.2),
      Conv2DTranspose(
          1, kernel_size=4, strides=2, padding="same", activation="tanh"
      ),
  ])


def build_discriminator():
  return Sequential([
      Conv2D(
          64, kernel_size=4, strides=2, padding="same", input_shape=(28, 28, 1)
      ),
      LeakyReLU(0.2),
      Flatten(),
      Dense(1, activation="sigmoid"),
  ])


# Instantiate models
generator = build_generator()
discriminator = build_discriminator()

# Compile Discriminator
discriminator.compile(
    optimizer=Adam(0.0002, 0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ==========================================
# 3. Combined GAN Model Setup
# ==========================================
discriminator.trainable = False  # Freeze discriminator during generator training

gan_input = tf.keras.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)

gan.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy")


# ==========================================
# 4. Visualization Function
# ==========================================
def show_images(epoch):
  noise = np.random.normal(0, 1, (16, 100))
  gen_imgs = generator.predict(noise, verbose=0)
  gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale back to [0, 1]

  fig, axs = plt.subplots(4, 4, figsize=(4, 4))
  cnt = 0
  for i in range(4):
    for j in range(4):
      axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
      axs[i, j].axis("off")
      cnt += 1
  plt.suptitle(f"Epoch: {epoch}")
  plt.tight_layout()
  plt.show()


# ==========================================
# 5. Training Loop
# ==========================================
def train_gan(epochs=1000, batch_size=128):
  half_batch = batch_size // 2

  for epoch in range(epochs):
    # --- Train Discriminator ---
    idx = np.random.randint(0, x_train.shape[0], half_batch)
    real_imgs = x_train[idx]

    noise = np.random.normal(0, 1, (half_batch, 100))
    fake_imgs = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(
        real_imgs, np.ones((half_batch, 1))
    )
    d_loss_fake = discriminator.train_on_batch(
        fake_imgs, np.zeros((half_batch, 1))
    )

    # --- Train Generator ---
    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_y)

    # Print progress and plot generated samples periodically
    
    print(
        f"Epoch {epoch:4d} | D Loss Real: {d_loss_real[0]:.4f} | D Loss"
        f" Fake: {d_loss_fake[0]:.4f} | G Loss: {g_loss:.4f}"
    )
    show_images(epoch)


# Run Training
train_gan(epochs=5, batch_size=128)