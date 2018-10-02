import timeit
import os
import collections

os.environ["TFHUB_CACHE_DIR"] = "/mnt/pccfs/not_backed_up/hub/"
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow.contrib.legacy_seq2seq as seq2seq # I don't want to use legacy...
from tensorflow.python.ops import rnn_cell

import scipy.spatial as spatial



with tf.Graph().as_default():

    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

    dataset = [
        "The quick brown fox jumps over the lazy dog.",
        "I am a sentence for which I would like to get its embedding",
        "I am a sentence",
        "I am another sentence",
        "I am a camera",
        "I love lamp",
        "Don't kill the whale"
    ]

    embeddings = embed(dataset)

    counter = collections.Counter(" ".join(dataset))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    chars.append("$")
    vocab_size = len(chars)
    vocab = dict(zip(chars, range(len(chars))))

    embedded_input = tf.placeholder(tf.float32, [1, 512], name="embedded_input")
    input_char = tf.placeholder(tf.int32, [1, vocab_size], name="input_char")

    state_dim = 512
    num_layers = 1
    batch_size = 1





    cells = [rnn_cell.GRUCell(state_dim) for i in range(num_layers)]
    stacked_cells = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    # initial_state = stacked_cells.zero_state(batch_size, tf.float32)

    # temp_char = input_char
    # state = embedded_input

    outputs, state = seq2seq.rnn_decoder(input_char, embedded_input, stacked_cells)

    W = tf.get_variable("W", [state_dim, vocab_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [vocab_size], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(o, W) + b for o in outputs]


    # LOSS FUNCTION #

    char_list = [chars[tf.argmax(l)] for l in logits]
    output_sentence = "".join(char_list)

    embedded_output = embed([output_sentence])[0]

    loss = spatial.distance.cosine(embedded_output, embedded_input)
    optim = tf.train.AdamOptimizer(0.001).minimize(loss)


    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        # start = timeit.default_timer()

        initial_input = np.array([0] * vocab_size)
        initial_input[-1] = 1


        for epoch in range(1000):
            for i in len(range(dataset)):
                sentence = dataset[i]
                e_input = embeddings[i]

                result, _ = sess.run([output_sentence, optim], feed_dict={embedded_input:e_input, input_char:initial_input})

                print("\n**************************")
                print("TRUE:", sentence)
                print("OUTPUT:", result)
        # print(result, timeit.default_timer() - start, sep='\n')
