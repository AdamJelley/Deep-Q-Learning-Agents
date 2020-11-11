import gym
import numpy as np
import matplotlib.pyplot as plt
from FrozenLake_QLearning_Agent import Agent

if __name__ == "__main__":

    n_games = 500000
    win_pct_list = []
    scores = []

    env = gym.make('FrozenLake-v0')
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_states=16, n_actions=4)

    for i in range(n_games):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            state = obs
            action = agent.choose_action(state)
            obs, reward, done, info = env.step(action)
            agent.learn(state, action, reward, obs)
            score += reward
        scores.append(score)

        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i % 1000 == 0:
                print(f"Episode {i}, win_pct {win_pct:.2f}, epsilon {agent.epsilon:.2f}")

    plt.plot(win_pct_list)
    plt.show()