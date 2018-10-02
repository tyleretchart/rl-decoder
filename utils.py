import torch


def get_char(onehot, chars):
    _, index = torch.max(onehot, 2)
    return chars[index]


def char_to_input_tensor(char, vocab, use_cuda=False):
    index = vocab[char]
    tensor = torch.zeros((1, 1, len(vocab)))
    tensor[0, 0, index] = 1.
    if use_cuda:
        return tensor.cuda()
    else:
        return tensor


def char_to_target_tensor(char, vocab, use_cuda=False):
    index = vocab[char]
    tensor = torch.LongTensor([index])
    if use_cuda:
        return tensor.cuda()
    else:
        return tensor