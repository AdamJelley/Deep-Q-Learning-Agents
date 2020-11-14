import numpy as np
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from utils import plot_learning_curve

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions

class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99, eps_start=1.0, eps_dec=1e-5, eps_end=0.01):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor(obs, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon-self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, new_state):
        self.Q.optimizer.zero_grad()

        stateT = T.tensor(obs, dtype=T.float).to(self.Q.device)
        new_stateT = T.tensor(new_obs, dtype=T.float).to(self.Q.device)
        rewardT = T.tensor(reward, dtype=T.float).to(self.Q.device)

        q_value = T.max(self.Q.forward(stateT))
        new_q_value = T.max(self.Q.forward(new_stateT))

        q_target = rewardT + self.gamma*(new_q_value)

        cost = self.Q.loss(q_target, q_value).to(self.Q.device)

        cost.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

if __name__ == "__main__":

    n_games = 10
    win_pct_list = []
    scores = []
    eps_history = []

    env = gym.make("CartPole-v1")
    agent = Agent(input_dims=env.observation_space.shape,
                    n_actions=env.action_space.n,
                    lr=0.0001, gamma=0.99, eps_start=1.0, eps_dec=1e-5, eps_end=0.01)

    for i in range(n_games):
        done=False
        obs=env.reset()
        score=0
        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, new_obs)
            obs = new_obs
            score += reward
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            print(f"Episode {i}, win_pct {win_pct:.2f}, epsilon {agent.epsilon:.2f}")

    filename_path="./NaiveDQN/cartpole_naive_dqn_learning.png"
    x = [x+1 for x in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename_path)
