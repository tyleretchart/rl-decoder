import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2)
        self.linear = nn.Linear(hidden_size, input_size, bias=True)

    def forward(self, input_, hidden):
        output, hidden = self.rnn(input_, hidden)
        reshaped = self.linear(output)
        return reshaped, hidden