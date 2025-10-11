import numpy as np
import sys
import torch
import argparse
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from isaacsim import SimulationApp

SOARM100_STAGE_PATH = "/World/soarm100_robot"
SOARM100_USD_PATH = "/home/home/soarm_tutorial/isaac_env/asset/soarm100.usd"

CONFIG = {"renderer": "RaytracedLighting", "headless": False}  # 시각화 ON
simulation_app = SimulationApp(CONFIG)

import carb
import omni.graph.core as og
from isaacsim.core.utils import viewports
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction

from task import Soarm_Task
from policy.PPO.agent import PPO
from policy.PPO.utils import load_config


# -------------------- 환경 --------------------
class IsaacSoarmEnv:
    def __init__(self, max_steps=1000):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()

        viewports.set_camera_view(
            eye=np.array([1.2, 1.2, 0.8]),
            target=np.array([0, 0, 0.5])
        )

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        self.task = Soarm_Task(name="Soarm_task")
        self.world.add_task(self.task)
        self.world.reset()
        simulation_app.update()

        self.robot = self.world.scene.get_object("soarm100_robot")
        self.controller = self.robot.get_articulation_controller()
        simulation_app.update()

        self.world.initialize_physics()
        self.world.play()

        self.step_count = 0
        self.max_steps = max_steps

    def reset(self):
        self.world.reset()
        simulation_app.update()
        self.step_count = 0
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.controller.apply_action(
            ArticulationAction(joint_positions=action)
        )
        self.world.step(render=True)
        simulation_app.update()
        self.step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward()
        done = self.step_count >= self.max_steps
        return obs, reward, done, info

    def _get_obs(self):
        obs_dict = self.world.get_observations()
        joint_pos = obs_dict["soarm100"]["joint_position"]
        eef_pos   = obs_dict["soarm100"]["eef_pos"]
        cube_pos  = obs_dict["object"]["cube"]["pos"]
        obs_vec = np.concatenate([joint_pos, eef_pos, cube_pos]).astype(np.float32)
        return obs_vec
    
    def _compute_reward(self, std: float = 0.1):
        obs_dict = self.world.get_observations()
        ee_pos = obs_dict["soarm100"]["eef_pos"]
        cube_pos = obs_dict["object"]["cube"]["pos"]
        dist = np.linalg.norm(ee_pos - cube_pos)
        penalty = -0.1 * dist
        reward_tanh = 0.2 * (1.0 - np.tanh(dist / std))
        reward = penalty + reward_tanh
        info = {"distance": dist, "final_reward": reward}
        return reward, info


# -------------------- 플레이 실행 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained PPO checkpoint (.pth)")
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode (no rendering)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of play episodes to run")
    args = parser.parse_args()

    # 렌더링 비활성화 옵션
    if args.headless:
        simulation_app.set_setting("/app/window/drawGpu", False)

    # config 로드
    cfg_path = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
    cfg = load_config(cfg_path)

    env = IsaacSoarmEnv(max_steps=cfg["env"]["max_steps"])
    obs = env.reset()

    obs_dim = obs.shape[0]
    act_dim = len(env.robot.get_joints_state().positions)

    joint_lower = np.array(cfg["robot"]["joint_limits"]["lower"], dtype=np.float32)
    joint_upper = np.array(cfg["robot"]["joint_limits"]["upper"], dtype=np.float32)

    # PPO 모델 로드
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

    agent.load(args.checkpoint)
    print(f"✅ Loaded checkpoint from: {args.checkpoint}")

    # -------------------- 에피소드 루프 --------------------
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        print(f"\n[Episode {ep+1}] ------------------------------")

        while not done and simulation_app.is_running():
            obs_t = torch.tensor(obs).float().unsqueeze(0).cuda()
            with torch.no_grad():
                action = agent.net.get_deterministic_action(obs_t)

            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, done, info = env.step(action_np)
            total_reward += reward
            obs = next_obs

        print(f"Episode {ep+1} finished | Total reward: {total_reward:.3f}")

    print("✅ Play completed successfully.")
    simulation_app.close()
