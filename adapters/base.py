"""
adapters/base.py

Every policy owner writes ONE adapter that subclasses PolicyAdapter.

The harness gives you a RobotState (canonical joint order) and expects a
JointCommand (canonical joint order) back - the same information as Unitree's
LowState / LowCmd. Everything in between (observation layout, scaling, history,
joint order, action scaling, recurrent state) is yours.

Use Marvarid's mapper (self.mapper) to go between your policy's joint order and
the canonical order. Never index joints by hand.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from harness.types import RobotState, JointCommand, NUM_JOINTS
from utils.joint_mapping import JointOrderMapper, CANONICAL_G1_JOINTS


class PolicyAdapter(ABC):
    # ---- Fill these in your subclass ----
    name: str = "unnamed_policy"
    # Your policy's joint order (all 29 names, in the order your network uses).
    POLICY_JOINT_ORDER: Sequence[str] = CANONICAL_G1_JOINTS
    # Control period your policy was trained for. Harness refuses to run a mismatch.
    policy_dt: float = 0.02
    # "control": act() is called once per control step (normal case).
    # "physics": act() is called every physics step (external real-time stacks, see adapters/external/).
    command_rate: str = "control"
    # Set True if you read RobotState.base_lin_vel or base_pos (not measurable on hardware).
    uses_privileged_state: bool = False

    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.checkpoint_path = checkpoint_path
        self.config: Dict[str, Any] = dict(config or {})
        self.mapper = JointOrderMapper(target_joint_names=list(self.POLICY_JOINT_ORDER),
                                       source_joint_names=list(CANONICAL_G1_JOINTS))

    # ---------------- required ----------------
    @abstractmethod
    def default_pose(self) -> np.ndarray:
        """[29] CANONICAL joint angles the robot is spawned in (usually your nominal stance)."""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Start of every episode: clear action history, RNN memory, phase clocks, RNGs."""

    @abstractmethod
    def act(self, state: RobotState) -> JointCommand:
        """Return a JointCommand (CANONICAL order) for this control step."""

    # ---------------- optional ----------------
    def reference(self, t: float) -> Optional[Dict[str, np.ndarray]]:
        """Motion-tracking policies (BeyondMimic, SONIC, ...): return the reference the policy is
        supposed to track at time t, e.g. {"q": canonical[29]} and optionally
        {"q_mask": bool[29]}. Enables the tracking-error metric. Velocity policies return None.
        Adding {"root_pos": xyz, "root_quat": wxyz} (pelvis, world) scores it as a motion tracker (README §3d)."""
        return None

    def motion_duration(self) -> Optional[float]:
        """Episode length in seconds for motion trackers. None keeps the scenario's 10 s."""
        return None

    def default_base_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(pelvis xyz, pelvis quaternion wxyz) to spawn at; the harness sets the height. None: origin."""
        return None

    def describe(self) -> Dict[str, Any]:
        """Free-form metadata saved with the results (training repo, commit, obs layout...)."""
        return {}

    def close(self) -> None:
        """Release sockets, DDS participants, threads, GPU memory."""

    # ---------------- helpers for subclasses ----------------
    def indices_of(self, names: Sequence[str]) -> np.ndarray:
        """Indices of `names` inside POLICY_JOINT_ORDER (for policies that control a subset)."""
        order = list(self.POLICY_JOINT_ORDER)
        missing = [n for n in names if n not in order]
        if missing:
            raise ValueError(f"Unknown joints: {missing}")
        return np.array([order.index(n) for n in names], dtype=np.int64)

    @staticmethod
    def full(value: float) -> np.ndarray:
        return np.full(NUM_JOINTS, value, dtype=np.float64)
