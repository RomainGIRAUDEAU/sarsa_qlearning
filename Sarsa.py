import gym
import numpy as np
import time
import matplotlib.pyplot as plt



# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005


def learn(Q,state,state2,reward,action,action2):
    predict = Q[state,action]
    target = reward+gamma * Q[state2,action2]
    Q[state,action] += alpha * (target - predict)

def init_q(s, a, type="ones"):
    """
    @param s the number of states
    @param a the number of actions
    @param type random, ones or zeros for the initialization
    """
    if type == "ones":
        return np.ones((s, a))
    elif type == "random":
        return np.random.random((s, a))
    elif type == "zeros":
        return np.zeros((s, a))


def epsilon_greedy(Q, epsilon, n_actions, s, train=False):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param s number of states
    @param train if true then no random actions selected
    """
    if train or np.random.rand() < epsilon:
        action = np.argmax(Q[s, :])
    else:
        action = np.random.randint(0, n_actions)
    return action



def sarsa(env,alpha, gamma, epsilon, episodes, max_steps, n_tests, render=False, test=False):

    plt.axis([0,500,0,episodes])
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_q(n_states, n_actions, type="zeros")
    timestep_reward = []

    for episode in range(episodes):
        print(f"Episode: {episode}")
        total_reward = 0
        s = env.reset()
        a = epsilon_greedy(Q, epsilon, n_actions, s)
        t = 0
        done = False
        while t < max_steps:
            if render:
                env.render()
            t += 1
            s_, reward, done, info = env.step(a)
            total_reward += reward
            a_ = epsilon_greedy(Q, epsilon, n_actions, s_)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            Q[s, a] += alpha * (reward + (gamma * Q[s_, a_]) - Q[s, a])
            s, a = s_, a_

            if done:
                if render:
                    print(f"This episode took {t} timesteps and reward {total_reward}")
                timestep_reward.append(total_reward)
                break


    print(f"Here are the Q values:\n{Q}\nTesting now:")
    return Q

def testing_game(env,qtable):
    winning_game =0
    #### test our matrix #############################
    env.reset()

    for episode in range(10000):
        state = env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(max_steps):

            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(qtable[state, :])

            new_state, reward, done, info = env.step(action)

            if done:
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                env.render()
                if (reward == 1):
                    winning_game += 1

                # We print the number of step it took.
                print("Number of steps", step)
                break
            state = new_state
    env.close()
    return winning_game



if __name__ == "__main__":
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.9
    episodes = 30000
    max_steps = 250
    n_tests = 20
    env = gym.make('FrozenLake8x8-v0')
    Q = sarsa(env,alpha, gamma, epsilon, episodes, max_steps, n_tests)

    print(testing_game(env,Q))
