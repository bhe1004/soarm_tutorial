from abc import ABC
import numpy as np

import omni
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.manipulators import SingleManipulator


class Soarm_Task(ABC, BaseTask):
    def __init__(self, name: str) -> None:
        BaseTask.__init__(self, name=name)
        self._robot = None
        self._asset_root_path = get_assets_root_path()
        if self._asset_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")
        self._robot_stage_path = "/World/robot"
        self._asset_path = "/home/home/soarm_tutorial/isaac_env/asset"
        self._robot_asset_path = self._asset_path + "/soarm100.usd"
        self._ee_xform_path = self._robot_stage_path + "/wrist"

        # 🔑 cube 위치 범위 설정 (원하는 대로 수정 가능)
        self._cube_x_range = (-0.2, 0.2)
        self._cube_y_range = (-0.2, -0.3)
        self._cube_z_height = 0.2

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)

        scene.add_default_ground_plane(z_position=0.0)

        # Visual cube 초기화 (위치는 reset에서 다시 조정됨)
        self.visual_cube = VisualCuboid(
            prim_path="/World/objects/cube/visual",
            name="visual_cube",
            size=1.0,
            scale=(0.02, 0.02, 0.02),
            position=np.array([0.0, -0.3, self._cube_z_height]),
            color=np.array([1.0, 0.0, 0.0])
        )

        self._robot = self.set_robot()

        scene.add(self.visual_cube)
        scene.add(self._robot)

    def set_robot(self) -> SingleManipulator:
        robot_prim_path = find_unique_string_name(
            initial_name=self._robot_stage_path,
            is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        add_reference_to_stage(usd_path=self._robot_asset_path, prim_path=robot_prim_path)

        manipulator = SingleManipulator(
            prim_path=robot_prim_path,
            name="soarm100_robot",
            end_effector_prim_path=robot_prim_path + "/gripper",
        )

        manipulator.set_joints_default_state(
            positions=np.deg2rad([0.0, 90.0, -70.0, 40.0, 0.0, 0.0])
        )

        return manipulator

    def reset_cube_position(self):
        """큐브를 특정 범위 내에서 무작위 위치로 리셋"""
        new_pos = np.array([
            np.random.uniform(*self._cube_x_range),
            np.random.uniform(*self._cube_y_range),
            self._cube_z_height,
        ], dtype=np.float32)

        _, ori = self.visual_cube.get_local_pose()
        self.visual_cube.set_local_pose(new_pos, ori)

        return new_pos

    def get_observations(self) -> dict:
        joints_state = self._robot.get_joints_state()
        end_effector_pos, end_effector_ori = self._robot.end_effector.get_local_pose()
        cube_pos, cube_ori = self.visual_cube.get_local_pose()

        return {
            "soarm100": {
                "joint_position": joints_state.positions,
                "eef_pos": end_effector_pos,
                "eef_ori": end_effector_ori,
            },
            "object": {
                "cube": {
                    "pos": cube_pos,
                    "ori": cube_ori,
                },
            },
        }
