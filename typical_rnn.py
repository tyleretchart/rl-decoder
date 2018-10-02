import torch
import torch.nn as nn
from torch import optim

from encoding_environment import Environment
from pytorch_rnn import Decoder
from utils import get_char, char_to_input_tensor, char_to_target_tensor

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
rnn = Decoder(input_size=env.vocab_size, hidden_size=512)
log_softmax = nn.LogSoftmax(dim=2)
if use_cuda:
    rnn = rnn.cuda()
    log_softmax = log_softmax.cuda()

# loss structure
LEARNING_RATE = .001
EPOCHS = 100000
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=LEARNING_RATE)

for e in range(EPOCHS):

    # set up inputs
    onehot = char_to_input_tensor(">", env.vocab)
    hidden = target_embedded
    sentence = get_char(onehot=onehot, chars=env.chars)
    optimizer.zero_grad()
    loss = 0

    if use_cuda:
        onehot = onehot.cuda()
        hidden = hidden.cuda()

    # main training loop
    times_in_loop = 0
    for char in target:
        times_in_loop += 1

        # run network
        onehot, hidden = rnn(onehot, hidden)
        onehot = log_softmax(onehot)

        # create loss
        loss += criterion(onehot[0], char_to_target_tensor(char, env.vocab, use_cuda=use_cuda))

        # get for print purposes
        chosen_char = get_char(onehot=onehot, chars=env.chars)
        sentence += chosen_char
        if chosen_char == env.terminal_char:
            break

    # loss /= times_in_loop
    loss.backward()
    optimizer.step()

    # print output
    print("LOSS: {}".format(loss))
    print(sentence + "+++")

    if e % 1000 == 0:
        torch.save(rnn, "weights.pt")