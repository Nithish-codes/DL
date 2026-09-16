import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Attention, Concatenate, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Dataset & Tokenization
input_texts = ["hello", "how are you", "what is your name", "bye"]
target_texts = ["hi", "i am fine", "i am a bot", "goodbye"]

# Add spaces around target text to act as start/end boundaries
target_texts = [" " + txt + " " for txt in target_texts]

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(input_texts + target_texts)

input_seqs = pad_sequences(tokenizer.texts_to_sequences(input_texts), padding="post")
target_seqs = pad_sequences(
    tokenizer.texts_to_sequences(target_texts), padding="post"
)

vocab_size = len(tokenizer.word_index) + 1
max_len = input_seqs.shape[1]
embedding_dim, lstm_units = 64, 128

# 2. Training Model (Encoder-Decoder with Attention)
# Encoder
enc_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(enc_inputs)
enc_out, state_h, state_c = LSTM(
    lstm_units, return_sequences=True, return_state=True
)(enc_emb)

# Decoder
dec_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size, embedding_dim)(dec_inputs)
dec_lstm, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(
    dec_emb, initial_state=[state_h, state_c]
)

# Attention & Output Layer
context = Attention()([dec_lstm, enc_out])
decoder_concat = Concatenate(axis=-1)([context, dec_lstm])
outputs = Dense(vocab_size, activation="softmax")(decoder_concat)

# Compile & Train
model = Model([enc_inputs, dec_inputs], outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(
    [input_seqs, target_seqs],
    np.expand_dims(target_seqs, -1),
    epochs=150,
    verbose=0,
)

# 3. Inference Models Setup
encoder_model = Model(enc_inputs, [enc_out, state_h, state_c])

dec_in_h, dec_in_c, enc_out_in = (
    Input(shape=(lstm_units,)),
    Input(shape=(lstm_units,)),
    Input(shape=(None, lstm_units)),
)
dec_emb_inf = Embedding(vocab_size, embedding_dim)(dec_inputs)
dec_lstm_inf, h_inf, c_inf = LSTM(
    lstm_units, return_sequences=True, return_state=True
)(dec_emb_inf, initial_state=[dec_in_h, dec_in_c])

context_inf = Attention()([dec_lstm_inf, enc_out_in])
concat_inf = Concatenate(axis=-1)([context_inf, dec_lstm_inf])
dec_out_inf = Dense(vocab_size, activation="softmax")(concat_inf)

decoder_model = Model(
    [dec_inputs, enc_out_in, dec_in_h, dec_in_c],
    [dec_out_inf, h_inf, c_inf],
)


# 4. Response Generator Function
def generate_response(input_text):
  seq = pad_sequences(
      tokenizer.texts_to_sequences([input_text]), maxlen=max_len, padding="post"
  )
  e_out, h, c = encoder_model.predict(seq, verbose=0)

  # Initialize target sequence with start space token
  target_seq = np.array([[tokenizer.word_index.get(" ", 1)]])
  response = ""

  for _ in range(max_len):
    output_tokens, h, c = decoder_model.predict(
        [target_seq, e_out, h, c], verbose=0
    )
    sampled_idx = np.argmax(output_tokens[0, -1, :])
    word = tokenizer.index_word.get(sampled_idx, "")

    if not word or word == " ":
      break
    response += word + " "
    target_seq = np.array([[sampled_idx]])

  return response.strip()


# 5. Test the Bot
test_input = "how are you"
print("Input :", test_input)
print("Bot   :", generate_response(test_input))