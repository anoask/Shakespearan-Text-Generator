# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time
## code adopted from tf, pytorch and karpathy blog

# %%
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# %%
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# %%
# Take a look at the first 400 characters in text
print(text[:400])
# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')
example_texts = ['NLPUSF', 'Assignment3']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
chars = chars_from_ids(ids)
tf.strings.reduce_join(chars, axis=-1).numpy()

# %%
def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))
seq_length = 140
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
  print(chars_from_ids(seq))

for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
split_input_target(list("Tensorflow"))

dataset = sequences.map(split_input_target)
for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 256

class NLPUSFModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

# %%
model = NLPUSFModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

# %%
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

# %%
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", example_batch_mean_loss)
tf.exp(example_batch_mean_loss).numpy()

# %%
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 20
# Start training your model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# %%
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

# %%
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# %%
start = time.time()
states = None
next_char = tf.constant(['Queen:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)

# %%
def BeamSearch(RNN, start_sequence:str, beam_width:int, temperature = 1.0, gen_length=1000): #Without States
  skip_ids = ids_from_chars(['[UNK]'])[:, None]
  sparse_mask = tf.SparseTensor(
      # Put a -inf at each bad index.
      values=[-float('inf')]*len(skip_ids),
      indices=skip_ids,
      # Match the shape to the vocabulary
      dense_shape=[len(ids_from_chars.get_vocabulary())])
  prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def one_step(RNN,inputs):
    #print("inputs",inputs)
    input_chars = tf.strings.unicode_split(inputs,'UTF-8')
    #print("inputs_chars",input_chars)
    input_ids = ids_from_chars(input_chars).to_tensor()
    #print("inputs_chars",input_ids)
    predicted_logits = RNN(inputs=input_ids)
    predicted_logits = predicted_logits[:,-1,:]
    predicted_logits = predicted_logits/temperature
    predicted_logits += prediction_mask
    return predicted_logits #-1 is the last character in the sequence


  next_char = tf.constant([start_sequence])
  #print(tf.get_static_value(next_char)[0])
  #print(len(tf.get_static_value(next_char)[0]))
  candidates = [(0,next_char)]#,states)] # We can set our starting prob to zero because all sequences share the starting probability value
  final_candidates = []
  incomplete = True
  while(incomplete):
    all_expansions = []
    for sequence in candidates:
      current_seq = sequence[1]
      if(len(tf.get_static_value(current_seq)[0])>=gen_length):
        final_candidates.append(sequence)
        continue
      predicted_logits = one_step(RNN,current_seq) # Predicted Logits for the last letter in the sequence
      softmax = tf.nn.softmax(predicted_logits,1)
      beam_values, beam_indices = tf.nn.top_k(softmax, k=beam_width)
      beam_ids = tf.get_static_value(beam_indices)
      beam_scores = tf.get_static_value(beam_values)
      for i in range(0,len(beam_ids[0])):
        # print(beam_ids[0][i])
        # print(chars_from_ids(beam_ids[0][i]))
        new_sequence = tf.strings.join([current_seq]+chars_from_ids(beam_ids[0][i])) #Appends new char to sequence
        new_score = sequence[0] + np.log(beam_scores[0][i])
        #print(new_score,new_sequence)
        all_expansions.append((new_score,new_sequence))#,states))
    candidates = sorted(all_expansions, key=lambda seq: seq[0], reverse=True)[:beam_width] #Replace candidates with top N sequences where N is the beamwidth of all_expansions
    #print(candidates[0][1])
    incomplete = False #Check end condition
    for seq in candidates:
      if(len(tf.get_static_value(seq[1])[0]) < gen_length):
        incomplete = True
        break
    #print(all_expansions)
    #incomplete = False
  final_candidates += candidates
  return sorted(final_candidates, key=lambda seq: seq[0], reverse=True)


# %%
gru0 = NLPUSFModel(
    vocab_size=vocab_size,
    embedding_dim=256,
    rnn_units=256)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = gru0(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
gru0.summary()
gru0.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './gru0_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# Stops training if there is no improvement for threee consec epochs
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 20
# Start training your model
gru0_history = gru0.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback,early_stop_callback])

# %%
gru1 = NLPUSFModel(
    vocab_size=vocab_size,
    embedding_dim=256,
    rnn_units=256)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = gru1(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
gru1.summary()
gru1.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './gru1_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# Stops training if there is no improvement for threee consec epochs
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 100
# Start training your model
gru1_history = gru1.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback,early_stop_callback])

# %%
gru = NLPUSFModel(
    vocab_size=vocab_size,
    embedding_dim=512,
    rnn_units=512)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = gru(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
gru.summary()
gru.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './gru_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# Stops training if there is no improvement for threee consec epochs
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 100
# Start training your model
gru_history = gru.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback,early_stop_callback])

# %%
final = BeamSearch(gru,"Queen: ",5,gen_length=500)

# %%
final0 = BeamSearch(gru0,"Queen: ",5,gen_length=500)
final1 = BeamSearch(gru1,"Queen: ",5,gen_length=500)


# %%
#print(final[0])
print(final0[0][1][0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print(final1[1][1][0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print(final[2][1][0].numpy().decode('utf-8'), '\n\n' + '_'*80)

# %%
class CustomLSTMCell(keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super(CustomLSTMCell, self).__init__(**kwargs)
    self.units = units
    self.state_size = [units, units]  # Hidden state size and cell state size

  def build(self, input_shape):
    input_dim = input_shape[-1]
    # One can play with init to stabalize learning, remember what we discussed for MLP
    # As described in class LSTM is simply 4 different RNNs (h_t = sigma(Wx_t + Uh_{t-1} + b)) working in parallel, but connected jointly.
    # Weights for the input gate
    self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', name='W_i')
    self.U_i = self.add_weight(shape=(self.units, self.units), initializer='random_normal', name='U_i')
    self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')

    # Weights for the forget gate
    self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', name='W_f')
    self.U_f = self.add_weight(shape=(self.units, self.units), initializer='random_normal', name='U_f')
    self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', name='b_f')

    # Weights for the cell state
    self.W_c = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', name='W_c')
    self.U_c = self.add_weight(shape=(self.units, self.units), initializer='random_normal', name='U_c')
    self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', name='b_c')

    # Weights for the output gate
    self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', name='W_o')
    self.U_o = self.add_weight(shape=(self.units, self.units), initializer='random_normal', name='U_o')
    self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')

    super(CustomLSTMCell, self).build(input_shape)

  def call(self, inputs, states, return_state=None,training=None):
    #print("called")
    h_tm1, c_tm1 = states  # Previous state
    # Input gate
    i = tf.sigmoid(tf.matmul(inputs, self.W_i) + tf.matmul(h_tm1, self.U_i) + self.b_i)
    # Forget gate
    f = tf.sigmoid(tf.matmul(inputs, self.W_f) + tf.matmul(h_tm1, self.U_f) + self.b_f)

    # Cell state
    c_ = tf.tanh(tf.matmul(inputs, self.W_c) + tf.matmul(h_tm1, self.U_c) + self.b_c)
    c = f * c_tm1 + i * c_

    # Output gate
    o = tf.sigmoid(tf.matmul(inputs, self.W_o) + tf.matmul(h_tm1, self.U_o) + self.b_o)
    # New hidden state
    h = o * tf.tanh(c)
    return h, [h, c]

# %%
vocab_size = 66
embedding_dim = 512
rnn_units = 512 # Number of LSTM units
input_shape = (None, embedding_dim)  # Example input shape (timesteps, features)
# Create the LSTM layer using the custom cell
lstm_layer = keras.layers.RNN(CustomLSTMCell(rnn_units), input_shape=input_shape,return_sequences=True)
lstm = keras.Sequential([
  keras.layers.Embedding(vocab_size, embedding_dim),
  lstm_layer,
  keras.layers.Dense(vocab_size)  # Example output layer
])
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

lstm.compile(optimizer='adam', loss=loss)
lstm.summary()


# %%
checkpoint_dir = './lstm_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# Stops training if there is no improvement for threee consec epochs
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 100
# Start training your model
lstm_history = lstm.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback,early_stop_callback])

# %%
vocab_size = 66
embedding_dim = 256
rnn_units = 512 # Number of units
input_shape = (None, embedding_dim)  # Example input shape (timesteps, features)
# Create the Elman layer using SimpleRNN
elman_layerR512 = keras.layers.SimpleRNN(rnn_units,input_shape=input_shape,return_sequences=True)
elmanR512 = keras.Sequential([
  keras.layers.Embedding(vocab_size, embedding_dim),
  elman_layerR512,
  keras.layers.Dense(vocab_size)  # Example output layer
])
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

elmanR512.compile(optimizer='adam', loss=loss)
elmanR512.summary()

checkpoint_dir = './elmanR512_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# Stops training if there is no improvement for threee consec epochs
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 100
# Start training your model
elmanR512_history = elmanR512.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback,early_stop_callback])

# %%
elman_final = BeamSearch(elmanR512,"Queen: ",5,gen_length=500)

# %%
print(elman_final[0][1][0].numpy().decode('utf-8'), '\n\n' + '_'*80)

# %%
vocab_size = 66
embedding_dim = 256
rnn_units = 128 # Number of units
input_shape = (None, embedding_dim)  # Example input shape (timesteps, features)
# Create the Elman layer using SimpleRNN
elman_layerR128 = keras.layers.SimpleRNN(rnn_units,input_shape=input_shape,return_sequences=True)
elmanR128 = keras.Sequential([
  keras.layers.Embedding(vocab_size, embedding_dim),
  elman_layerR128,
  keras.layers.Dense(vocab_size)  # Example output layer
])
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

elmanR128.compile(optimizer='adam', loss=loss)
elmanR128.summary()

checkpoint_dir = './elmanR128_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# Stops training if there is no improvement for threee consec epochs
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 100
# Start training your model
elmanR128_history = elmanR128.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback,early_stop_callback])


