import argparse
import yaml
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import wandb
import datetime
from pi import Pi


# -------------------- Argument Parser --------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="렌더링 완전 비활성화 (기본값: 렌더링 활성화)")
    parser.add_argument("--wandb", action="store_true",
                        help="wandb 로깅 활성화 여부 (기본: 비활성화)")
    return parser.parse_args()


# -------------------- 설정 불러오기 --------------------
with open("/home/home/soarm_tutorial/policy/REINFORCE/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

gamma = cfg["train"]["gamma"]
lr = cfg["train"]["learning_rate"]
num_episodes = cfg["train"]["num_episodes"]
max_steps = cfg["train"]["max_steps"]
hidden_sizes = cfg["model"]["hidden_sizes"]
solved_threshold = cfg["env"]["solved_threshold"]


# -------------------- 학습 함수 --------------------
def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret

    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets
    loss = torch.sum(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# -------------------- 메인 루프 --------------------
def main():
    args = get_args()

    # 기본값: human, --headless면 None
    render_mode = None if args.headless else "human"

    # wandb 설정
    use_wandb = args.wandb
    if use_wandb:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project="REINFORCE-CartPole",
            config=cfg,
            name=f"reinforce_train_run_{timestamp}"
        )
        wandb.watch_called = False
        wandb.config.update({"headless": args.headless})

    # 환경 초기화
    env = gym.make("CartPole-v1", render_mode=render_mode)
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim, hidden_sizes)
    optimizer = optim.Adam(pi.parameters(), lr=lr)

    for epi in range(num_episodes):
        state, _ = env.reset()
        for t in range(max_steps):
            action = pi.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            pi.rewards.append(reward)
            if render_mode == "human":
                env.render()
            if done:
                break

        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > solved_threshold
        pi.onpolicy_reset()

        print(f"Episode {epi}, loss: {loss:.3f}, total_reward: {total_reward}, solved: {solved}")

        if use_wandb:
            wandb.log({
                "episode": epi,
                "loss": loss,
                "total_reward": total_reward,
            })

    env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
