import numpy as np

import torch
import torch.nn as nn
from torch import optim

from encoding_environment import Environment
from pytorch_rnn import Decoder
from utils import char_to_input_tensor, char_to_target_tensor

# set up environment
use_cuda = torch.cuda.is_available()
target = "The quick brown fox jumps over the lazy dog."
env = Environment(target)
target_embedded = env.embed([target])
# target_embedded = np.random.random((1, 1, 512))
target_embedded = torch.Tensor(target_embedded)
target_embedded = target_embedded.view((1, 1, 512))
target_embedded = torch.cat((target_embedded, target_embedded), 0)

# create model
# rnn = Decoder(input_size=env.vocab_size, hidden_size=512)
rnn = torch.load('rl_weights_embed.pt')
if use_cuda:
    rnn = rnn.cuda()

# loss structure
LEARNING_RATE = .001
DISCOUNT_FACTOR = .95
EPSILON = .05
EPOCHS = 100000
TEACHER_FORCING = .7
criterion = nn.MSELoss()    
optimizer = optim.SGD(rnn.parameters(), lr=LEARNING_RATE)

# for e in range(EPOCHS):
e = 0
while True:
    e += 1

    # set up inputs
    onehot = char_to_input_tensor(">", env.vocab)
    hidden = target_embedded
    _, index = torch.max(onehot, 2)
    sentence = env.chars[index]
    optimizer.zero_grad()
    loss = 0

    if use_cuda:
        onehot = onehot.cuda()
        hidden = hidden.cuda()

    if e == 500:
        print("TEST")
        EPSILON = 0

    # main training loop
    times_in_loop = 0
    for char in target:
        times_in_loop += 1

        # run network
        q_table, hidden = rnn(onehot, hidden)

        # choose action
        if np.random.random() < EPSILON:
            index = np.random.choice(range(env.vocab_size))
            current_action_reward = q_table[0, 0, index]
        else:
            current_action_reward, index = torch.max(q_table, 2)

        # pick char
        if np.random.random() < TEACHER_FORCING:
            chosen_char = char
        else:
            chosen_char = env.chars[index]

        # get reward
        if chosen_char == "<":
            reward = env.get_reward(state=sentence[1:-1], target=target)
        else:
            reward = -1

        # prep next sentence onehot
        onehot = char_to_input_tensor(chosen_char, env.vocab)
        if use_cuda:
            onehot = onehot.cuda()

        future_q_table, future_hidden = rnn(onehot, hidden)
        next_possible_reward, next_index = torch.max(future_q_table, 2)
        discounted_next_reward = DISCOUNT_FACTOR * next_possible_reward

        target_q_table = q_table.clone()
        target_q_table[0][0][index] = reward + discounted_next_reward
        target_q_table = target_q_table.detach()

        loss += criterion(input=q_table, target=target_q_table)

        # get for print purposes
        sentence += chosen_char
        if chosen_char == env.terminal_char:
            break

        # if e % 1000 == 0:
        #     EPSILON *= .9

    # loss /= times_in_loop
    loss.backward()
    optimizer.step()

    # print output
    print("LOSS: {}".format(loss))
    print(sentence + "+++")

    if e == 500:
        print("TEST OVER")
        EPSILON = .1

    if e > 1000:
        e = 0
        torch.save(rnn, "rl_weights_embed.pt")