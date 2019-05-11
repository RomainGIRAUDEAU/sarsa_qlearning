import gym
import numpy as np
import time, pickle, os
import matplotlib.pyplot as plt

env = gym.make('FrozenLake8x8-v0')

epsilon = 0.9
total_episodes = 100000
max_steps = 2500

lr_rate = 0.81
gamma = 0.96

step_recorder_win = []
lost_game = []
time_out = []
Q = np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    return action


def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] +=  lr_rate * (target - predict)


# Start
def change_reward(done, reward,state1,state2):
    if reward == 1 and done == True:
        return +3
    elif reward == 0 and done == True:
        return -3
    elif reward == 0 and done == False and state1 == state2:
        return -1
    elif reward == 0 and done == False:
        return -2

for episode in range(total_episodes):
    print("episose numero :"+str(episode))
    state = env.reset()
    t = 0

    while t < max_steps:
        if(t == max_steps - 1):
            time_out.append(episode)

        #env.render()

        action = choose_action(state)

        state2, reward, done, info = env.step(action)
      #  reward = change_reward(done, reward,state,state2)
      #  learn(state, state2, reward, action)
        #epsilon = np.sqrt(epsilon)

        state = state2

        t += 1

        if done:
            if reward == 1:
                step_recorder_win.append(episode)
            else :
                lost_game.append(episode)
            break



plt.plot(step_recorder_win)
plt.plot(time_out)
plt.plot(lost_game)
plt.show()
print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)


