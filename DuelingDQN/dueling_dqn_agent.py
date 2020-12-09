import numpy as np
import torch as T
from deep_q_network import DuelingDeepQNetwork
from replay_memory import ReplayBuffer

class DuelingDQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01,
                eps_dec=5e-7, replace=1000, algo=None, env_name=None, checkpoint_dir='tmp/dqn'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_cntr = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                    name = self.env_name+'_'+self.algo+'_q_eval', 
                                    checkpoint_dir=self.checkpoint_dir)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                            name = self.env_name+'_'+self.algo+'_q_next', 
                            checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        new_states = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        return states, actions, rewards, new_states, dones

    def replace_target_network(self):
        if self.learn_step_cntr % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_model(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, new_states, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_next, A_s_next = self.q_next.forward(new_states)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_next, (A_s_next - A_s_next.mean(dim=1, keepdim=True))).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()

        self.learn_step_cntr += 1

        self.decrement_epsilon()