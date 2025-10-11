import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class Pi(nn.Module):
    """Policy network for REINFORCE (Categorical policy for discrete action spaces)."""
    def __init__(self, in_dim, out_dim, hidden_sizes):
        super(Pi, self).__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())  # 활성화 함수 고정
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.model = nn.Sequential(*layers)

        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        """에피소드 종료 후 log_probs, rewards 초기화"""
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        """Forward pass (state → logits)"""
        return self.model(x)

    def act(self, state):
        """Sample action from the current policy"""
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()
