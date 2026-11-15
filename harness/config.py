"""
harness/config.py

Everything that must be IDENTICAL for every policy lives here.
Adapters cannot change these values. If you think one of them is wrong,
change it here (in a reviewed commit) so it changes for everyone.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import os

HARNESS_VERSION = "2.0.0"

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
# robot models (assets/g1/build_*_model.py); ros2 = the ROS 2 deployment stack's G1
ROBOT_MODELS = {
    "ros2": os.path.join(REPO_ROOT, "assets", "g1", "g1_29dof_ros2.xml"),
    "unitree_rl_gym": os.path.join(REPO_ROOT, "assets", "g1", "g1_29dof_harness.xml"),
}
DEFAULT_ROBOT = "ros2"
DEFAULT_MODEL_XML = ROBOT_MODELS[DEFAULT_ROBOT]


# Reference PD gains for the G1 (by joint name). Only used when the harness is
# run with --gains harness. By default each policy sends its own gains, exactly
# as it would on the real robot (see DESIGN.md, "PD gains").
# Legs: unitree_rl_gym deploy config (g1.yaml). Waist/arms: stiff "hold" gains.
G1_REFERENCE_GAINS: Dict[str, Tuple[float, float]] = {}
for side in ("left", "right"):
    G1_REFERENCE_GAINS.update({
        f"{side}_hip_pitch_joint": (100.0, 2.0),
        f"{side}_hip_roll_joint": (100.0, 2.0),
        f"{side}_hip_yaw_joint": (100.0, 2.0),
        f"{side}_knee_joint": (150.0, 4.0),
        f"{side}_ankle_pitch_joint": (40.0, 2.0),
        f"{side}_ankle_roll_joint": (40.0, 2.0),
        f"{side}_shoulder_pitch_joint": (60.0, 2.0),
        f"{side}_shoulder_roll_joint": (60.0, 2.0),
        f"{side}_shoulder_yaw_joint": (60.0, 2.0),
        f"{side}_elbow_joint": (60.0, 2.0),
        f"{side}_wrist_roll_joint": (20.0, 1.0),
        f"{side}_wrist_pitch_joint": (20.0, 1.0),
        f"{side}_wrist_yaw_joint": (20.0, 1.0),
    })
G1_REFERENCE_GAINS.update({
    "waist_yaw_joint": (300.0, 5.0),
    "waist_roll_joint": (300.0, 5.0),
    "waist_pitch_joint": (300.0, 5.0),
})


# real fall of a motion tracker (unless --stop-at-tracking-failure): head below this for this long
REAL_FALL_HEAD_HEIGHT_M = 0.5
REAL_FALL_TIME_S = 2.0
HEAD_LIDAR_IN_TORSO = (0.0002835, 0.00003, 0.41618)  # ros2 model's mid360 site in torso_link


@dataclass(frozen=True)
class HarnessConfig:
    # ---- Simulation ----
    model_xml: str = DEFAULT_MODEL_XML
    physics_dt: float = 0.002            # 500 Hz physics
    decimation: int = 10                 # policy at 500/10 = 50 Hz
    gains_mode: str = "policy"           # "policy" (adapter sends kp/kd) or "harness" (G1_REFERENCE_GAINS)

    # ---- Episode start ----
    spawn_clearance_m: float = 0.002     # robot is placed this far above the floor, never inside it
    settle_time_s: float = 0.0           # optional pre-roll under the policy (not scored) - kept 0 for strictness

    # ---- Seeded randomization (identical for every policy at a given seed) ----
    init_joint_noise_rad: float = 0.02   # uniform +/- on every joint at spawn
    push_time_jitter_s: float = 0.30     # uniform +/- on every push start time
    push_dir_jitter_deg: float = 10.0    # uniform +/- rotation of every push direction (about z)

    # ---- Fall detection ----
    fall_min_base_height_m: float = 0.40     # pelvis height
    fall_max_tilt_rad: float = 1.047         # 60 deg between pelvis z-axis and world z
    # Bodies allowed to touch the floor. Any other body touching it = fall.
    allowed_ground_contact_bodies: Tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")

    # ---- Motion tracking (reference with a root pose): mjlab's failure rules and bodies ----
    tracking_anchor_body: str = "torso_link"
    tracking_anchor_height_threshold_m: float = 0.25    # |reference - robot| anchor height
    tracking_anchor_gravity_threshold: float = 0.8      # |reference - robot| projected-gravity z of the anchor
    tracking_end_effector_height_threshold_m: float = 0.25
    tracking_end_effector_bodies: Tuple[str, ...] = (
        "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    tracking_key_bodies: Tuple[str, ...] = (
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
    tracking_strict_mpkpe_m: float = 0.15                # strict success (PHUMA)

    # ---- Metrics ----
    tracking_warmup_s: float = 1.0       # velocity tracking ignores the first second (starting from rest)
    tracking_window_s: float = 1.0       # moving-average window (>= one gait cycle) for primary velocity metrics
    soft_joint_limit_fraction: float = 0.95  # "near limit" band = outer 5% of each joint's range

    # ---- Outputs ----
    video_width: int = 640
    video_height: int = 480
    video_stride: int = 2                # render every 2nd control step -> 25 fps video

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation

    def to_dict(self) -> dict:
        d = asdict(self)
        d["control_dt"] = self.control_dt
        d["harness_version"] = HARNESS_VERSION
        return d
