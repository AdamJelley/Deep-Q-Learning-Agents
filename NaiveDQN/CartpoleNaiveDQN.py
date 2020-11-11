import munpy as np
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork).__init__()

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
    def __init__(self, input_dims, n_actions, lr, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in  range(self.n_actions)]

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor(obs, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random_choice(self.action_space)
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

        q_target = rewardT + self.gamma*(new_q_value))

        cost = self.Q.loss(q_target, q_value).to(self.Q.device)

        cost.backwards()
        self.Q.optimizer.step()
        self.decrement_epsilon()
