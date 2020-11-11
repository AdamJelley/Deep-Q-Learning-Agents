
class Agent():
    def __init__(self, lr, gamma, eps_start, eps_end, eps_dec, n_states, n_actions):
        self.lr = lr
        self.gamma = gamma
        self.eps_max = eps_start
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

    