import tensorflow as tf
import numpy as np

from environment import Environment

# set up environment
env = Environment()

# Set learning parameters
y = .99
e = 0.1
learning_rate = 0.1

# reset default graph
tf.reset_default_graph()

# define graph
inputs = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="inputs")
next_q = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="next_q")

q_out = tf.contrib.layers.fully_connected(
    inputs=inputs, num_outputs=5, activation_fn=None)

predicted_action = tf.argmax(q_out, 1)

# define loss function
loss = tf.reduce_sum(tf.square(next_q - q_out))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update_model = optimizer.minimize(loss)

# init
init = tf.initialize_all_variables()
state = env.get_numerical_state()

# start network
with tf.Session() as sess:
    # init session
    sess.run(init)

    for e in range(10000):
        env.reset()
        total_reward = 0

        for i in range(100):

            # get action and predicted_rewards
            action, predicted_rewards = sess.run(
                [predicted_action, q_out], feed_dict={inputs: state})

            # with epsilon chance, choose random action
            if np.random.rand(1) < e:
                action[0] = np.random.choice(range(5))

            # Get new state and reward from environment
            reward, terminate = env.step(action[0])
            new_state = env.get_numerical_state()

            # Obtain the Q' values by feeding the new state through our network
            new_predicted_rewards = sess.run(q_out, feed_dict={inputs: new_state})

            # Obtain maxQ' and set our target value for chosen action.
            max_predicted_rewards = np.max(new_predicted_rewards)
            target_rewards = predicted_rewards
            target_rewards[0, action[0]] = reward + (y * max_predicted_rewards)

            # Train our network using target and predicted Q values
            sess.run(
                update_model, feed_dict={
                    inputs: new_state,
                    next_q: target_rewards
                })
            total_reward += reward
            state = new_state

            # Print
            # print("PREDICT: {}, OUTPUTS: {}, TOTAL REWARD: {}".format(
                # action, predicted_rewards, total_reward))

            # Terminate
            if terminate:
                #Reduce chance of random action as we train the model.
                e = 1. / ((e / 50) + 10)
                break

        print("STATE: {}, REWARD: {}\n\n".format(env.state, total_reward))