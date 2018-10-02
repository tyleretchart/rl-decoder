import numpy as np


class Environment:
    def __init__(self):
        self.reset()
        self.vocab = ["a", "b", "c", "d", "T"]

    def reset(self):
        self.state = ""

    def get_numerical_state(self):
        numerical_state = np.zeros((1, 10))
        index = len(self.state) - 1
        if index > 9:
            index = 9
        numerical_state[0][index] = 1
        return numerical_state

    def step(self, action):
        new_letter = self.vocab[action]
        if self.state:
            reward = -1 if new_letter == self.state[-1] else 1
        else:
            reward = 0
        terminate = action == len(self.vocab) - 1
        self.state += new_letter
        return reward, terminate