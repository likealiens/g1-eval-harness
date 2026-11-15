"""SONIC (ViT-H): ViT-H's fine-tune of SONIC, run as its g1_tracker runs it. Inputs: --set tracker_root=... --set motion=motion.npz."""
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from adapters.base import PolicyAdapter
from harness.types import JointCommand, RobotState

CONTROL_HZ = 50.0


class SonicVitHAdapter(PolicyAdapter):
    name = "sonic"
    uses_privileged_state = True  # base position

    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(checkpoint_path, config)
        root = Path(self.config["tracker_root"]).expanduser().resolve()
        sys.path.insert(0, str(root))
        from vith_tracker import runtime, tracker
        from vith_tracker.observation import GlobalObservationMixin

        self.rt, self.tr = runtime, tracker
        with np.load(self.config["motion"]) as payload:
            values = {k: np.asarray(payload[k]) for k in payload.files}
        fps = float(self.config.get("motion_fps", values.get("fps", 0)))
        if not fps:
            raise ValueError("motion fps unknown: store `fps` in the motion file or --set motion_fps=...")
        values = {k: v[None] if k in tracker.REQUIRED_INPUT_FIELDS else v for k, v in values.items()}

        # Same preparation as g1_tracker.track_file.
        motion = tracker.resample_motion(values, fps, CONTROL_HZ)
        grounder = tracker.InitialSupportGrounder(root / "assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml")
        self.motion, self.grounding_offset = grounder.ground(motion)
        initial_isaaclab, _ = grounder.clip_initial_joint_pos(self.motion.joint_pos[0])
        self.initial_q = initial_isaaclab[runtime.ISAACLAB_TO_MUJOCO]

        self.policy = tracker.DeployPolicy(Path(checkpoint_path), int(self.config.get("ort_threads", 1)))
        self.obs = type("SonicObservation", (GlobalObservationMixin,), {})()
        self.obs.motion = self.motion

    def _frame(self, t: float) -> int:
        return min(int(round(t * CONTROL_HZ)), self.motion.timesteps - 1)

    def motion_duration(self) -> float:
        return (self.motion.timesteps - 1) / CONTROL_HZ

    def default_base_pose(self):
        return self.motion.root_pos[0].astype(np.float64), self.motion.root_quat[0].astype(np.float64)

    def default_pose(self) -> np.ndarray:
        return self.initial_q.copy()

    def reset(self, seed: int) -> None:
        self.obs.history = deque(maxlen=10)
        self.obs.last_action = np.zeros(self.rt.HW_DOF)
        self.started = False

    def reference(self, t: float) -> Dict[str, np.ndarray]:
        k = self._frame(t)
        return {"q": self.motion.joint_pos[k][self.rt.ISAACLAB_TO_MUJOCO].astype(np.float64),
                "root_pos": self.motion.root_pos[k].astype(np.float64),
                "root_quat": self.motion.root_quat[k].astype(np.float64)}

    def act(self, state: RobotState) -> JointCommand:
        o, rt = self.obs, self.rt
        o.current_frame = self._frame(state.t)
        o.base_pos, o.base_quat = state.base_pos.copy(), rt.normalize_quat(state.base_quat.copy())
        o.base_ang_vel = state.base_ang_vel.copy()
        o.joint_pos_mj, o.joint_vel_mj = state.q.copy(), state.dq.copy()
        if not self.started:
            o.reset_reference_alignment()
            for _ in range(10):
                o._append_history()
            self.started = True
        o._append_history()

        obs = np.concatenate([o._command_multi_future(), o._motion_anchor_orientation_10frame_step5(),
                              o._normal_actor_observation()]).astype(np.float32).reshape(1, -1)
        action = self.policy.act(obs).reshape(-1).astype(np.float64)
        o.last_action = action
        target = rt.DEFAULT_ANGLES + action[rt.ISAACLAB_TO_MUJOCO] * rt.ACTION_SCALE
        return JointCommand(q=target, kp=rt.KP, kd=rt.KD)

    def describe(self) -> Dict[str, Any]:
        return {"source": "ViT-H g1_tracker (SONIC deploy policy)", "motion": self.config["motion"],
                "motion_frames": int(self.motion.timesteps), "grounding_offset_m": float(self.grounding_offset)}
