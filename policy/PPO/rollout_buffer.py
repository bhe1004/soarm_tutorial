import numpy as np

class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.raw_actions = np.zeros((size, act_dim), dtype=np.float32)
        self.logps = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.ptr = 0
        self.max_size = size

    def store(self, obs, action, raw_action, reward, value, logp, done):
        if self.ptr < self.max_size:
            self.obs[self.ptr] = obs
            self.actions[self.ptr] = action
            self.raw_actions[self.ptr] = raw_action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.logps[self.ptr] = logp
            self.dones[self.ptr] = done
            self.ptr += 1

    def reset(self):
        self.ptr = 0
