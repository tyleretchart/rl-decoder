import collections
import numpy as np
from encoding_environment import Environment

# target and state
target = "The quick brown fox jumps over the lazy dog."
state = "  "
env = Environment(target)

# collect action space
terminal_char = "<END"
counter = collections.Counter(target)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
chars, _ = zip(*count_pairs)
chars = list(chars)
chars.append(terminal_char)
vocab_size = len(chars)
vocab = dict(zip(chars, range(len(chars))))

# print(vocab)
# print(chars)

# create table
q_table = np.zeros([vocab_size, vocab_size, vocab_size])
# dim 0 = 2 steps ago
# dim 1 = 1 step ago
# dim 2 = next step

# # Set learning parameters
learning_rate = .8
discount_factor = .95
num_episodes = 200000
epsilon = .1

# rewards over time
reward_list = []

# start learning
for i in range(num_episodes):

    # reset state
    state = "  "
    total_reward = 0
    terminate = False
    steps = 0
    current_state_index = 2

    # run through episode
    while steps < 99:
        steps += 1

        # understand current state
        prev_token = state[current_state_index - 1]
        before_prev_token = state[current_state_index - 2]

        prev_token_index = vocab[prev_token]
        before_prev_token_index = vocab[before_prev_token]

        # choose an action by greedily (with noise) picking from Q table
        if np.random.random() < epsilon:
            action = np.random.choice(range(vocab_size))
        else:
            action = np.argmax(q_table[before_prev_token_index, prev_token_index, :])

        # get new state and reward from environment
        new_state = state + chars[action]
        if chars[action] == terminal_char:
            terminate = True
            reward = env.get_reward(new_state[2:], target) * 10
        else:
            if current_state_index - 2 >= len(target):
                reward = -1
            elif target[current_state_index - 2] == new_state[current_state_index]:
                reward = 1
            else:
                reward = -1

        # update Q-Table with new knowledge
        current_action_reward = q_table[before_prev_token_index, prev_token_index, action]
        next_possible_reward = np.max(q_table[prev_token_index, action, :])
        discounted_next_reward = discount_factor * next_possible_reward
        percieved_reward = reward + discounted_next_reward - current_action_reward

        q_table[before_prev_token_index, prev_token_index,
                action] = current_action_reward + (learning_rate * percieved_reward)
        total_reward += reward
        state = new_state
        current_state_index += 1

        # stop current episode
        if terminate == True:
            break

    reward_list.append(total_reward)
    if i % 1000 == 0:
        print("\nEpisode: {}, Output: >{}<".format(i, state[2:]))