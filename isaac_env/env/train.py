import numpy as np
import sys
import torch
import argparse
import os
import wandb
import datetime
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from isaacsim import SimulationApp

SOARM100_STAGE_PATH = "/World/soarm100_robot"
SOARM100_USD_PATH = "/home/home/soarm_tutorial/isaac_env/asset/soarm100.usd"

CONFIG = {"renderer": "RaytracedLighting", "headless": True}
simulation_app = SimulationApp(CONFIG)

import carb
import omni.graph.core as og
from isaacsim.core.utils import viewports
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction

from task import Soarm_Task


from policy.PPO.agent import PPO
from policy.PPO.rollout_buffer import RolloutBuffer
from policy.PPO.utils import load_config


# -------------------- 환경 래퍼 --------------------
class IsaacSoarmEnv:
    def __init__(self, max_steps=1000):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()

        # 카메라 세팅
        viewports.set_camera_view(
            eye=np.array([1.2, 1.2, 0.8]),
            target=np.array([0, 0, 0.5])
        )

        # 월드 생성
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 태스크 추가
        self.task = Soarm_Task(name="Soarm_task")
        self.world.add_task(self.task)
        self.world.reset()
        simulation_app.update()

        # 로봇 핸들
        self.robot = self.world.scene.get_object("soarm100_robot")
        self.controller = self.robot.get_articulation_controller()
        simulation_app.update()

        # 물리 초기화
        self.world.initialize_physics()
        self.world.play()

        self.step_count = 0
        self.max_steps = max_steps  # 에피소드 길이 제한

    def reset(self):
        self.world.reset()
        simulation_app.update()
        self.step_count = 0

        if hasattr(self.task, "reset_cube_position"):
            cube_pos = self.task.reset_cube_position()
            # print(f"[Env] Cube reset to {cube_pos}")
            
        obs = self._get_obs()
        return obs

    def step(self, action):
        # 액션 적용 (예: joint target positions)
        self.controller.apply_action(
            ArticulationAction(joint_positions=action)
        )

        # 한 스텝 진행
        self.world.step(render=True)
        simulation_app.update()
        self.step_count += 1

        # 관측
        obs = self._get_obs()

        # 보상 계산 (예시: EE와 큐브 거리의 음수)
        reward, info = self._compute_reward()

        # 종료 조건
        done = self.step_count >= self.max_steps

        return obs, reward, done, info

    def _get_obs(self):
        obs_dict = self.world.get_observations()

        joint_pos = obs_dict["soarm100"]["joint_position"]
        eef_pos   = obs_dict["soarm100"]["eef_pos"]
        cube_pos  = obs_dict["object"]["cube"]["pos"]

        # 하나의 벡터로 합치기
        obs_vec = np.concatenate([joint_pos, eef_pos, cube_pos]).astype(np.float32)

        return obs_vec
    
    def _compute_reward(self, std: float = 0.1):
        """보상 함수: EE와 큐브 사이 거리 기반 (혼합형: penalty + tanh reward)."""
        obs_dict = self.world.get_observations()
        ee_pos = obs_dict["soarm100"]["eef_pos"]
        cube_pos = obs_dict["object"]["cube"]["pos"]

        # L2 distance
        dist = np.linalg.norm(ee_pos - cube_pos)

        # Penalty: 거리가 멀면 음수 보상 (weight = -0.1)
        penalty = -0.1 * dist

        # Reward: tanh 커널 (weight = +0.2)
        reward_tanh = 0.2 * (1.0 - np.tanh(dist / std))

        # 최종 reward
        reward = penalty + reward_tanh

        info = {
            "distance": dist,
            "penalty": penalty,
            "reward_tanh": reward_tanh,
            "final_reward": reward,
        }
        return reward, info


# -------------------- Train 실행 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    args = parser.parse_args()

    cfg = load_config("/home/home/soarm_tutorial/policy/PPO/config.yaml")

    joint_lower = np.array(cfg["robot"]["joint_limits"]["lower"], dtype=np.float32)
    joint_upper = np.array(cfg["robot"]["joint_limits"]["upper"], dtype=np.float32)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = os.path.join(cfg["model"]["save_path"], timestamp)
    os.makedirs(save_path, exist_ok=True)

    config_src = "/home/home/soarm_tutorial/policy/PPO/config.yaml"
    config_dst = os.path.join(save_path, "config.yaml")
    shutil.copy(config_src, config_dst)

    # wandb 초기화
    if args.wandb:
        wandb.init(
            project="soarm-ppo",
            config=cfg,
            name=f"ppo_train_run_{timestamp}"
        )

    env = IsaacSoarmEnv(max_steps=cfg["env"]["max_steps"])
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = len(env.robot.get_joints_state().positions)

    agent = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr=float(cfg["train"]["learning_rate"]),
        clip_eps=float(cfg["train"]["clip_eps"]),
        epochs=int(cfg["train"]["epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        gamma=float(cfg["train"]["gamma"]),
        lam=float(cfg["train"]["lam"]),
        hidden_sizes=cfg["model"]["hidden_sizes"],
        decay_rate=cfg["train"]["action_std_decay"],
        min_action_std=cfg["train"]["min_action_std"],
        action_low=joint_lower,
        action_high=joint_upper,
        entropy_bonus=cfg["train"]["entropy_bonus"]
    )

    buffer = RolloutBuffer(size=cfg["train"]["rollout_size"], obs_dim=obs_dim, act_dim=act_dim)

    # -------------------- 학습 루프 --------------------
    for episode in range(cfg["train"]["total_episodes"]):
        obs = env.reset()
        done = False
        buffer.reset()
        total_reward = 0.0

        while not done and simulation_app.is_running():
            obs_t = torch.tensor(obs).float().unsqueeze(0).cuda()
            with torch.no_grad():
                action, raw_action, logp, value = agent.net.get_action_and_value(obs_t)

            action_np = action.cpu().numpy().squeeze()
            raw_action_np = raw_action.cpu().numpy().squeeze()
            logp_np = logp.cpu().numpy().squeeze()
            value_np = value.cpu().numpy().squeeze()

            next_obs, reward, done, info = env.step(action_np)

            # rollout buffer에 저장
            buffer.store(obs, action_np, raw_action_np, reward, value_np, logp_np, done)

            total_reward += reward
            obs = next_obs

        # critic value를 사용해서 마지막 상태의 value 추정
        with torch.no_grad():
            last_value = agent.net.get_value(
                torch.tensor(obs).float().unsqueeze(0).cuda()
            ).item()

        # PPO 업데이트 (last_value 반영)
        agent.update(buffer, last_value=last_value)

        # --- 탐험 제어 (action std decay) ---
        if hasattr(agent, "decay_action_std"):
            agent.decay_action_std()

        # wandb 로그
        if args.wandb:
            wandb.log({
                "episode": episode + 1,
                "reward": total_reward,
                "distance": info["distance"],
            })

        # best 저장
        if total_reward > agent.best_reward:
            agent.best_reward = total_reward
            agent.save(save_path, episode=episode+1, reward=total_reward, is_best=True)
            print(f"New best model saved at episode {episode+1} | Reward: {total_reward:.3f}")

        # interval 저장
        elif (episode + 1) % cfg["train"]["save_interval"] == 0:
            agent.save(save_path, episode=episode+1, reward=total_reward, is_best=False)
            print(f"Model checkpoint saved at episode {episode+1}")

        print(f"Episode {episode+1} finished | Reward: {total_reward:.3f}")


    if args.wandb:
        wandb.finish()