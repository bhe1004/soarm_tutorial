import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64],
                 init_action_std=0.6, action_low=None, action_high=None):
        super().__init__()
        # Actor
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        self.actor = nn.Sequential(*layers, nn.Linear(last_dim, act_dim))

        # Critic
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        self.critic = nn.Sequential(*layers, nn.Linear(last_dim, 1))

        self.action_std = init_action_std
        self.action_low = torch.tensor(action_low, dtype=torch.float32).cuda() if action_low is not None else None
        self.action_high = torch.tensor(action_high, dtype=torch.float32).cuda() if action_high is not None else None

    def squash_and_scale(self, raw_action):
        squashed = torch.tanh(raw_action)  # [-1, 1]
        if self.action_low is not None and self.action_high is not None:
            scaled = self.action_low + 0.5 * (squashed + 1.0) * (self.action_high - self.action_low)
            return scaled
        return squashed

    def get_action_and_value(self, obs):
        mean = self.actor(obs)
        std = torch.ones_like(mean) * self.action_std
        dist = torch.distributions.Normal(mean, std)

        raw_action = dist.sample()
        logp = dist.log_prob(raw_action).sum(axis=-1)

        scaled_action = self.squash_and_scale(raw_action)  # 환경에 전달할 action
        value = self.critic(obs)

        return scaled_action, raw_action, logp, value

    def evaluate_raw_actions(self, obs, raw_actions):
        """raw_action을 기반으로 logp, entropy 계산"""
        mean = self.actor(obs)
        std = torch.ones_like(mean) * self.action_std
        dist = torch.distributions.Normal(mean, std)

        logp = dist.log_prob(raw_actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs)
        return logp, entropy, value

    def get_deterministic_action(self, obs):
        mean = self.actor(obs)
        return self.squash_and_scale(mean)
    
    def get_value(self, obs):
        return self.critic(obs)
    
    def set_action_std(self, new_action_std: float):
        """외부에서 action std 값을 갱신"""
        self.action_std = new_action_std
