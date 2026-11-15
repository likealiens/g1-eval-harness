"""BeyondMimic policy exported to ONNX by mjlab: network, reference motion and metadata in one file."""
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from adapters.base import PolicyAdapter
from harness.config import DEFAULT_MODEL_XML
from harness.types import JointCommand, RobotState

MOTION_FPS = 50.0
EXPECTED_OBS = ["command", "motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel",
                "base_ang_vel", "joint_pos", "joint_vel", "actions"]


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split(",")])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


class BeyondMimicAdapter(PolicyAdapter):
    name = "beyondmimic"
    uses_privileged_state = True  # anchor position and base velocity; a state estimator on the robot

    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        import onnx
        import onnxruntime
        from onnx import numpy_helper

        model = onnx.load(checkpoint_path)
        meta = {p.key: p.value for p in model.metadata_props}
        if meta["observation_names"].split(",") != EXPECTED_OBS:
            raise ValueError(f"Unexpected observation layout: {meta['observation_names']}")
        self.POLICY_JOINT_ORDER = meta["joint_names"].split(",")
        super().__init__(checkpoint_path, config)

        self.default_q = _floats(meta["default_joint_pos"])
        self.action_scale = _floats(meta["action_scale"])
        self.kp = self.mapper.to_source(_floats(meta["joint_stiffness"]))
        self.kd = self.mapper.to_source(_floats(meta["joint_damping"]))

        motion = {i.name.split(".")[0]: numpy_helper.to_array(i) for i in model.graph.initializer}  # "joint_pos.1"
        self.ref_joint_pos = motion["joint_pos"].astype(np.float64)
        self.ref_joint_vel = motion["joint_vel"].astype(np.float64)
        self.ref_body_pos = motion["body_pos_w"].astype(np.float64)
        self.ref_body_quat = motion["body_quat_w"].astype(np.float64)
        body_names = meta["body_names"].split(",")
        self.anchor = body_names.index(meta["anchor_body_name"])
        self.root = body_names.index("pelvis")
        self.frames = len(self.ref_joint_pos)

        self.fk_model = mujoco.MjModel.from_xml_path(DEFAULT_MODEL_XML)
        self.fk_data = mujoco.MjData(self.fk_model)
        self.fk_anchor = self.fk_model.body(meta["anchor_body_name"]).id
        self.fk_qadr = np.array([self.fk_model.joint(n).qposadr[0] for n in self.POLICY_JOINT_ORDER])
        self.imu_offset = self.fk_model.site("imu_in_pelvis").pos.copy()  # pelvis frame

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1  # one core per run, like SONIC
        self.session = onnxruntime.InferenceSession(checkpoint_path, options, providers=["CPUExecutionProvider"])
        self.last_action = np.zeros(29)

    def _frame(self, t: float) -> int:
        return min(int(round(t * MOTION_FPS)), self.frames - 1)

    def motion_duration(self) -> float:
        return (self.frames - 1) / MOTION_FPS

    def default_base_pose(self):
        return self.ref_body_pos[0, self.root], self.ref_body_quat[0, self.root]

    def default_pose(self) -> np.ndarray:
        return self.mapper.to_source(self.ref_joint_pos[0])

    def reset(self, seed: int) -> None:
        self.last_action = np.zeros(29)

    def reference(self, t: float) -> Dict[str, np.ndarray]:
        k = self._frame(t)
        return {"q": self.mapper.to_source(self.ref_joint_pos[k]),
                "root_pos": self.ref_body_pos[k, self.root], "root_quat": self.ref_body_quat[k, self.root]}

    def act(self, state: RobotState) -> JointCommand:
        k = self._frame(state.t)
        q = self.mapper.to_target(state.q)
        dq = self.mapper.to_target(state.dq)

        d = self.fk_data
        d.qpos[:3], d.qpos[3:7], d.qpos[self.fk_qadr] = state.base_pos, state.base_quat, q
        mujoco.mj_kinematics(self.fk_model, d)
        robot_pos, robot_rot = d.xpos[self.fk_anchor], d.xmat[self.fk_anchor].reshape(3, 3)

        # Reference anchor in the robot's anchor frame, as mjlab's subtract_frame_transforms.
        ref_pos, ref_rot = self.ref_body_pos[k, self.anchor], _quat_to_mat(self.ref_body_quat[k, self.anchor])
        anchor_pos_b = robot_rot.T @ (ref_pos - robot_pos)
        anchor_ori_b = (robot_rot.T @ ref_rot)[:, :2].reshape(-1)

        lin_vel = state.base_lin_vel + np.cross(state.base_ang_vel, self.imu_offset)  # velocimeter at the IMU site

        obs = np.concatenate([
            self.ref_joint_pos[k], self.ref_joint_vel[k], anchor_pos_b, anchor_ori_b,
            lin_vel, state.base_ang_vel, q - self.default_q, dq, self.last_action,
        ]).astype(np.float32)[None]
        action = self.session.run(["actions"], {"obs": obs, "time_step": np.array([[k]], dtype=np.float32)})[0][0]
        self.last_action = action.astype(np.float64)

        q_target = self.default_q + self.action_scale * self.last_action
        return JointCommand(q=self.mapper.to_source(q_target), kp=self.kp, kd=self.kd)

    def describe(self) -> Dict[str, Any]:
        return {"source": "mjlab BeyondMimic ONNX export", "motion_frames": self.frames,
                "motion_fps": MOTION_FPS, "observation": EXPECTED_OBS}
