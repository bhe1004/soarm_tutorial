import torch
import torch.optim as optim
import numpy as np
import os
from .actor_critic import ActorCritic


class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, clip_eps=0.2, epochs=10,
                 batch_size=64, gamma=0.99, lam=0.95, hidden_sizes=[64, 64],
                 init_action_std=0.6, decay_rate=0.01, min_action_std=0.1,
                 action_low=None, action_high=None):
        # ActorCritic 네트워크에 초기 std 전달
        self.net = ActorCritic(
            obs_dim, act_dim, hidden_sizes, 
            init_action_std, action_low, action_high
            ).cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam

        # best reward 추적
        self.best_reward = -float("inf")

        # action std 관리
        self.action_std = init_action_std
        self.decay_rate = decay_rate
        self.min_action_std = min_action_std

    def compute_advantages(self, rewards, values, dones, last_value):
        adv = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * \
                    (values[t+1] if t+1 < len(values) else last_value) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    # def update(self, buffer, last_value):
    #     adv, returns = self.compute_advantages(buffer.rewards, buffer.values, buffer.dones, last_value)
    #     adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    #     obs = torch.tensor(buffer.obs[:buffer.ptr]).float().cuda()
    #     raw_actions = torch.tensor(buffer.raw_actions[:buffer.ptr]).float().cuda()
    #     actions = torch.tensor(buffer.actions[:buffer.ptr]).float().cuda()
    #     old_logps = torch.tensor(buffer.logps[:buffer.ptr]).float().cuda()
    #     adv = torch.tensor(adv).float().cuda()
    #     returns = torch.tensor(returns).float().cuda()

    #     dataset_size = buffer.ptr
    #     inds = np.arange(dataset_size)

    #     for _ in range(self.epochs):
    #         np.random.shuffle(inds)
    #         for start in range(0, dataset_size, self.batch_size):
    #             end = start + self.batch_size
    #             batch_inds = inds[start:end]
    #             batch_obs = obs[batch_inds]
    #             # batch_actions = actions[batch_inds]
    #             batch_raw_actions = raw_actions[batch_inds]
    #             batch_old_logps = old_logps[batch_inds]
    #             batch_adv = adv[batch_inds]
    #             batch_returns = returns[batch_inds]

    #             # logp, entropy, values = self.net.evaluate_actions(batch_obs, batch_actions)
    #             logp, entropy, values = self.net.evaluate_raw_actions(batch_obs, batch_raw_actions)
    #             ratio = (logp - batch_old_logps).exp()
    #             surr1 = ratio * batch_adv
    #             surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
    #             policy_loss = -torch.min(surr1, surr2).mean()
    #             value_loss = (batch_returns - values.squeeze())**2
    #             value_loss = value_loss.mean()
    #             loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    def update(self, buffer, last_value):
        adv, returns = self.compute_advantages(buffer.rewards, buffer.values, buffer.dones, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.tensor(buffer.obs[:buffer.ptr]).float().cuda()
        raw_actions = torch.tensor(buffer.raw_actions[:buffer.ptr]).float().cuda()
        old_logps = torch.tensor(buffer.logps[:buffer.ptr]).float().cuda()
        adv = torch.tensor(adv).float().cuda()
        returns = torch.tensor(returns).float().cuda()

        dataset_size = buffer.ptr
        inds = np.arange(dataset_size)

        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_inds = inds[start:end]
                batch_obs = obs[batch_inds]
                batch_raw_actions = raw_actions[batch_inds]
                batch_old_logps = old_logps[batch_inds]
                batch_adv = adv[batch_inds]
                batch_returns = returns[batch_inds]

                # 새 logp, entropy, value 예측
                logp, entropy, values = self.net.evaluate_raw_actions(batch_obs, batch_raw_actions)
                values = values.squeeze()

                # ----- Policy Loss (PPO Clipping) -----
                ratio = (logp - batch_old_logps).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ----- Value Loss (with clipping) -----
                with torch.no_grad():
                    old_values = self.net.get_value(batch_obs).squeeze()

                value_loss_unclipped = (batch_returns - values) ** 2
                value_pred_clipped = old_values + torch.clamp(
                    values - old_values,
                    -self.clip_eps,
                    self.clip_eps
                )
                value_loss_clipped = (batch_returns - value_pred_clipped) ** 2

                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # ----- Total Loss -----
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    # -------------------- Exploration 제어 --------------------
    def decay_action_std(self):
        """action std를 줄여 탐험을 점점 줄임"""
        self.action_std = max(self.action_std - self.decay_rate, self.min_action_std)
        self.net.set_action_std(self.action_std)  # ActorCritic 내부에 적용
        # print(f"[PPO] Decayed action_std to {self.action_std:.3f}")

    def set_action_std(self, new_action_std):
        """action std를 직접 설정"""
        self.action_std = new_action_std
        self.net.set_action_std(new_action_std)
        # print(f"[PPO] Set action_std to {new_action_std:.3f}")

    # -------------------- 모델 저장 --------------------
    def save(self, save_dir, episode=None, reward=None, is_best=False):
        os.makedirs(save_dir, exist_ok=True)

        if not is_best and episode is not None:
            path = os.path.join(save_dir, f"ppo_soarm_{episode}.pth")
            checkpoint = {
                "model_state": self.net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "action_std": self.action_std,
            }
            torch.save(checkpoint, path)
            print(f"Saved PPO model to {path}")

        if is_best and reward is not None:
            best_path = os.path.join(save_dir, f"best_{reward:.2f}.pth")
            for f in os.listdir(save_dir):
                if f.startswith("best_"):
                    os.remove(os.path.join(save_dir, f))
            checkpoint = {
                "model_state": self.net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "action_std": self.action_std,
            }
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")

    # -------------------- 모델 불러오기 --------------------
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "action_std" in checkpoint:
            self.set_action_std(checkpoint["action_std"])
        print(f"Loaded PPO model from {path}")
