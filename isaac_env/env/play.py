import numpy as np
import sys
import torch
import argparse
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from isaacsim import SimulationApp

SOARM100_STAGE_PATH = "/World/soarm100_robot"
SOARM100_USD_PATH = "/home/home/soarm_tutorial/isaac_env/asset/soarm100.usd"

CONFIG = {"renderer": "RaytracedLighting", "headless": False}
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
        # done = self.step_count >= self.max_steps

        done = False

        return obs, reward, done, info

    def _get_obs(self):
        obs_dict = self.world.get_observations()

        joint_pos = obs_dict["soarm100"]["joint_position"]
        eef_pos   = obs_dict["soarm100"]["eef_pos"]
        cube_pos  = obs_dict["object"]["cube"]["pos"]

        # 하나의 벡터로 합치기
        obs_vec = np.concatenate([joint_pos, eef_pos, cube_pos]).astype(np.float32)

        return obs_vec
    
    def _compute_reward(self):
        """보상 함수: EE와 큐브 사이 거리 기반"""
        obs_dict = self.world.get_observations()
        ee_pos = obs_dict["soarm100"]["eef_pos"]
        cube_pos = obs_dict["object"]["cube"]["pos"]

        dist = np.linalg.norm(ee_pos - cube_pos)
        reward = -dist

        info = {"distance": dist}
        return reward, info


# -------------------- Play 실행 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint path for play mode")
    args = parser.parse_args()

    cfg = load_config("/home/home/soarm_tutorial/policy/PPO/config.yaml")

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
    )

    agent.load(args.checkpoint)
    print(f"▶ Play mode 시작 (checkpoint: {args.checkpoint})")

    done = False
    total_reward = 0.0
    while not done and simulation_app.is_running():
        obs_t = torch.tensor(obs).float().unsqueeze(0).cuda()
        with torch.no_grad():
            action = agent.net.get_deterministic_action(obs_t)
        action_np = action.cpu().numpy().squeeze()

        obs, reward, done, info = env.step(action_np)
        total_reward += reward
        print(f"Reward: {reward:.3f}, Info: {info}")

    print(f"총 Reward: {total_reward:.3f}")