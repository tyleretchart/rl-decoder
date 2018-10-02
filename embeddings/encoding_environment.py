from embeddings.encoders import UniversalSentenceEncoderLite
from scipy.spatial import distance as sp_distance
import tensorflow as tf

import collections

class Environment:
    def __init__(self, target):
        self.encoder = UniversalSentenceEncoderLite()
        
        # collect action space
        self.terminal_char = "<"
        start_char = ">"
        counter = collections.Counter(target)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.chars = list(self.chars)
        self.chars.append(start_char)
        self.chars.append(self.terminal_char)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def embed(self, sentences):
        return self.encoder.embed(sentences)

    def distance(self, sentence1, sentence2):
        embeddings = self.encoder.embed([sentence1, sentence2])
        return sp_distance.euclidean(embeddings[0], embeddings[1])

    def get_reward(self, state, target):
        distance = self.distance(state, target)
        if distance < .2:
            return 10
        else:
            return -10


if __name__ == "__main__":
    env = Environment()
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I am a sentence for which I would like to get its embedding",
        "I am a sentence.",
        "I am another sentence.",
        "I am a camera.",
        "I love lamp.",
        "Don't kill the whale.",
    ]

    embeddings = env.embed(sentences)
    print(embeddings)

    dist = env.distance("Dkljs s skldfj asldkj ",
                        "The quick brown fox jumps over the lazy dog.")
    print(dist)