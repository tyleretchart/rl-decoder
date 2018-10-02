import collections
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from embeddings.encoding_environment import Environment

#
# ---------------------------------------------------
# build network


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(1, vocab_size, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(vocab_size, vocab_size, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(vocab_size, vocab_size, kernel_size=3, stride=2)
        self.conv4 = nn.Conv1d(vocab_size, vocab_size, kernel_size=3, stride=1)
        # self.conv5 = nn.Conv1d(vocab_size, vocab_size, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        return x


#
# ---------------------------------------------------
# build dataset and action space

# denote target
target = "The quick brown fox jumps over the lazy dog."
env = Environment(target)
target_embedded = torch.Tensor(env.embed([target]))
target_embedded = target_embedded.view((1, 1, 512))

# collect action space
counter = collections.Counter(target)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
chars, _ = zip(*count_pairs)
chars = list(chars)
vocab_size = len(chars)
vocab = dict(zip(chars, range(len(chars))))

# global vars
SPACE_INDEX = vocab[" "]
OUTPUT_SIZE = 61

# massage target
if len(target) > OUTPUT_SIZE:
    target = target[:OUTPUT_SIZE]
if len(target) < OUTPUT_SIZE:
    target += " " * (OUTPUT_SIZE - len(target))

# generate target matrix
target_matrix = np.zeros((1, 61))
for i, char in enumerate(target):
    target_matrix[0, i] = vocab[char]
target_matrix = torch.LongTensor(target_matrix)
print(type(target_matrix))

#
# ---------------------------------------------------
# train

USE_CUDA = torch.cuda.is_available()

# create convnet
convnet = Decoder(vocab_size)
if USE_CUDA:
    convnet.cuda()

if USE_CUDA:
    target_embedded = target_embedded.cuda()
    target_matrix = target_matrix.cuda()

print(target_matrix)

# loss structure
LEARNING_RATE = .001
# DISCOUNT_FACTOR = .95
# EPSILON = .05
EPOCHS = 100000
# TEACHER_FORCING = .7
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(convnet.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    # produce output
    output = convnet(target_embedded)

    loss = criterion(input=output, target=target_matrix)

    loss.backward()
    optimizer.step()

    # print output
    if epoch % 5 == 0:
        out = output[0]
        value, index_mat = torch.max(out, dim=0)
        sentence = ""
        for index in index_mat:
            sentence += chars[index]
        print("EPOCH: {}, LOSS: {}, SENTENCE: {}".format(epoch, loss, sentence))