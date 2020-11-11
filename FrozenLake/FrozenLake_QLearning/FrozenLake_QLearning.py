import gym
import numpy as np
import matplotlib.pyplot as plt
from FrozenLake_QLearning_Agent import Agent

env = gym.make('FrozenLake-v0')

n_games = 1000
win_pct = []
scores = []

agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01, eps_dec=0.01, n_states=16, n_actions=4)

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)
plt.plot(win_pct)
plt.show()