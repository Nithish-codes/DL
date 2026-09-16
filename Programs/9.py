import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_length = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_length)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential([
    Embedding(
        input_dim=vocab_length, output_dim=32, input_length=max_len
    ),
    SimpleRNN(64),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=3, verbose=1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)

print(acc)