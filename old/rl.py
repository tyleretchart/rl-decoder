import numpy as np

print("RL")


class ENV:
    def __init__(self):
        self.observation_space = 3
        self.action_space = 3

    def step(self, action):
        if action == 0:
            return 0, 1, False
        elif action == 1:
            return 1, -1, False
        elif action == 2:
            return 2, 0, True
        else:
            raise ValueError("Not true action")

    def reset(self):
        return 0


env = ENV()


#Initialize table with all zeros
Q = np.zeros([env.observation_space, env.action_space])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j += 1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s, :] +
                      np.random.randn(1, env.action_space) * (1. / (i + 1)))
        #Get new state and reward from environment
        s1, r, d = env.step(a)
        #Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)

print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)