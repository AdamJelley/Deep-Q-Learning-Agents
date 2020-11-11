import numpy as np

class Agent():
    def __init__(self, lr, gamma, eps_start, eps_end, eps_dec, n_states, n_actions):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.n_states = n_states
        self.n_actions = n_actions

        self.Q = {}
        self._init_Q()

    def _init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            actions = np.array([self.Q[(state, action)] for action in range(self.n_actions)])
            action = np.argmax(actions)
        else:
            action = np.random.choice(list(range(self.n_actions)))
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, new_state):
        new_action = np.argmax(np.array([self.Q[(new_state, action)] for action in range(self.n_actions)]))
        
        self.Q[(state, action)] += self.lr*(reward + self.gamma*\
                                (self.Q[(new_state, new_action)]-self.Q[(state, action)]))
        
        self.decrement_epsilon()