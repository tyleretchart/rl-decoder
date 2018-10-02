import tensorflow as tf

# def model(features, labels, mode, params):
#     return None

# tf.layers.

# tf.estimator.Estimator(
#     model_fn=model,
#     params={
#         'feature_columns': ""
#     }
# )

words_in_dataset = [
    ["The", "The"],
    ["brown", "red"],
    ["fox", "fox"],
    ["is", "jumped"],
    ["quick", "high"],
]
batch_size = 2
time_steps = 5

lstm_size = 512
rnn = tf.contrib.rnn.BasicLSTMCell(lstm_size)

print(rnn)

# Initial state of the LSTM memory.
hidden_state = tf.zeros([batch_size, *rnn.state_size])
current_state = tf.zeros([batch_size, *rnn.state_size])
state = hidden_state, current_state
probabilities = []
loss = 0.0


for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = rnn(current_batch_of_words[0], state)

    print(output)
    print(state)

    # The LSTM output can be used to make next word predictions
    # logits = tf.matmul(output, softmax_w) + softmax_b
    # probabilities.append(tf.nn.softmax(logits))
    # loss += loss_function(probabilities, target_words)