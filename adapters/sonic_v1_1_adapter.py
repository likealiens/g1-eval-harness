"""NVIDIA's SONIC v1.1 (GEAR-SONIC), encoder mode g1, run as their C++ deploy (g1_deploy_onnx_ref) runs it."""

import hashlib
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from adapters.base import PolicyAdapter
from harness.types import JointCommand, RobotState
from utils.joint_orders import ISAACLAB_G1_29DOF_ORDER

CONTROL_HZ = 50.0
HISTORY = 10
FUTURE_FRAMES, FUTURE_STEP = 10, 5
ENCODER_DIM, DECODER_DIM, TOKEN_DIM = 1751, 994, 64
ENCODER_MODE_G1 = 0

# policy_parameters.hpp, MuJoCo order
ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
OMEGA = 10 * 2.0 * 3.1415926535
LEG = ("7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020")
ARM = ("5020",) * 5 + ("4010",) * 2
MOTORS = LEG * 2 + ("7520_14", "5020", "5020") + ARM * 2
DOUBLE_GAIN = (4, 5, 10, 11, 13, 14)  # ankles, waist roll and pitch
ARMATURES = np.array([ARMATURE[m] for m in MOTORS])
GAIN = np.where(np.isin(np.arange(29), DOUBLE_GAIN), 2.0, 1.0)
KP = GAIN * ARMATURES * OMEGA ** 2
KD = GAIN * 2.0 * 2.0 * ARMATURES * OMEGA
ACTION_SCALE = 0.25 * np.array([EFFORT[m] for m in MOTORS]) / (ARMATURES * OMEGA ** 2)
DEFAULT_ANGLES = np.array([
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
  0.0, 0.0, 0.0,
  0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
  0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product of wxyz quaternions, broadcast over leading axes."""
  w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
  w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
  return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def heading_quat(q: np.ndarray) -> np.ndarray:
  """calc_heading_quat_d: (w, 0, 0, z) normalised."""
  h = np.array([q[0], 0.0, 0.0, q[3]])
  return h / max(np.linalg.norm(h), 1e-9)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """quat_rotate_d, including its result for the all-zero quaternion of padded history."""
  w, u = q[0], q[1:]
  return v * (2.0 * w * w - 1.0) + np.cross(u, v) * w * 2.0 + u * np.dot(u, v) * 2.0


def first_two_columns(q: np.ndarray) -> np.ndarray:
  """Rotation matrices of [N, 4] quaternions, first two columns flattened row by row."""
  w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
  return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                   2 * (x * z - w * y), 2 * (y * z + w * x)], axis=-1).reshape(-1)


def lerp_rows(values: np.ndarray, src_t: np.ndarray, t: np.ndarray) -> np.ndarray:
  return np.stack([np.interp(t, src_t, col) for col in values.T], axis=-1)


def slerp_rows(quats: np.ndarray, src_t: np.ndarray, t: np.ndarray) -> np.ndarray:
  quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
  i = np.clip(np.searchsorted(src_t, t, side="right") - 1, 0, len(src_t) - 2)
  s = ((t - src_t[i]) / (src_t[i + 1] - src_t[i]))[:, None]
  a, b = quats[i], quats[i + 1].copy()
  dot = np.sum(a * b, axis=-1)
  b[dot < 0] *= -1.0
  angle = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))[:, None]
  sin = np.sin(angle)
  near = sin < 1e-6
  wa = np.where(near, 1.0 - s, np.sin((1.0 - s) * angle) / np.where(near, 1.0, sin))
  wb = np.where(near, s, np.sin(s * angle) / np.where(near, 1.0, sin))
  out = wa * a + wb * b
  return out / np.linalg.norm(out, axis=-1, keepdims=True)


def load_motion(path: str, fps: float = 0.0) -> Dict[str, np.ndarray]:
  """Harness motion .npz (joint_pos IsaacLab order, root_pos_w, root_quat_w, fps) at 50 Hz with joint velocities."""
  with np.load(path) as payload:
    values = {k: np.asarray(payload[k]) for k in payload.files}
  fps = float(fps or values.get("fps", 0))
  if not fps:
    raise ValueError("motion fps unknown: store `fps` in the motion file or --set motion_fps=...")
  joint_pos = values["joint_pos"].astype(np.float64).reshape(-1, 29)
  src_t = np.arange(len(joint_pos)) / fps
  # IsaacLab MotionLib timeline (endpoint excluded), as for the other SONIC adapter
  t = np.arange(np.float32(0.0), np.float32((len(joint_pos) - 1) / fps), np.float32(1.0 / CONTROL_HZ),
                dtype=np.float32).astype(np.float64)
  # gear_sonic torch_humanoid_batch: forward difference, last frame repeats the one before
  joint_vel = np.diff(joint_pos, axis=0) * fps
  joint_vel = np.concatenate([joint_vel, joint_vel[-2:-1]])
  return {"joint_pos": lerp_rows(joint_pos, src_t, t), "joint_vel": lerp_rows(joint_vel, src_t, t),
          "root_pos": lerp_rows(values["root_pos_w"].astype(np.float64).reshape(-1, 3), src_t, t),
          "root_quat": slerp_rows(values["root_quat_w"].astype(np.float64).reshape(-1, 4), src_t, t)}


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class SonicV11Adapter(PolicyAdapter):
  name = "sonic_v1_1"
  POLICY_JOINT_ORDER = ISAACLAB_G1_29DOF_ORDER

  def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    import onnxruntime

    super().__init__(checkpoint_path, config)
    self.decoder_path = Path(checkpoint_path)
    self.encoder_path = Path(self.config.get("encoder", self.decoder_path.with_name("model_encoder.onnx")))
    self.motion = load_motion(self.config["motion"], float(self.config.get("motion_fps", 0)))
    self.frames = len(self.motion["joint_pos"])
    self.hold = int(round(float(self.config.get("hold_s", 0.0)) * CONTROL_HZ))
    self.default_il = self.mapper.to_target(DEFAULT_ANGLES)

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = int(self.config.get("ort_threads", 1))
    self.encoder = onnxruntime.InferenceSession(str(self.encoder_path), options, providers=["CPUExecutionProvider"])
    self.decoder = onnxruntime.InferenceSession(str(self.decoder_path), options, providers=["CPUExecutionProvider"])
    for session, dim in ((self.encoder, ENCODER_DIM), (self.decoder, DECODER_DIM)):
      if session.get_inputs()[0].shape[-1] != dim:
        raise ValueError(f"{session.get_inputs()[0].name}: expected {dim} inputs, got {session.get_inputs()[0].shape}")

  def _frame(self, t: float) -> int:
    """Motion frame at time t; negative while holding the first frame before playback."""
    return min(int(round(t * CONTROL_HZ)) - self.hold, self.frames - 1)

  def motion_duration(self) -> float:
    return (self.hold + self.frames - 1) / CONTROL_HZ

  def default_base_pose(self):
    return self.motion["root_pos"][0], self.motion["root_quat"][0]

  def default_pose(self) -> np.ndarray:
    return self.mapper.to_source(self.motion["joint_pos"][0])

  def reset(self, seed: int) -> None:
    zero = (np.zeros(3), np.zeros(29), np.zeros(29), np.zeros(29), quat_rotate(np.zeros(4), np.array([0.0, 0.0, -1.0])))
    self.history = deque([zero] * HISTORY, maxlen=HISTORY)
    self.last_action = np.zeros(29)
    self.apply_delta_heading = None

  def reference(self, t: float) -> Dict[str, np.ndarray]:
    k = max(self._frame(t), 0)
    return {"q": self.mapper.to_source(self.motion["joint_pos"][k]),
            "root_pos": self.motion["root_pos"][k], "root_quat": self.motion["root_quat"][k]}

  def _encoder_obs(self, frame: int, base_quat: np.ndarray) -> np.ndarray:
    playing = frame >= 0
    idx = np.minimum(frame + FUTURE_STEP * np.arange(FUTURE_FRAMES), self.frames - 1) if playing \
      else np.zeros(FUTURE_FRAMES, dtype=np.int64)
    joint_vel = self.motion["joint_vel"][idx] if playing else np.zeros((FUTURE_FRAMES, 29))
    ref_quat = quat_mul(self.apply_delta_heading, self.motion["root_quat"][idx])
    anchor = first_two_columns(quat_mul(quat_conj(heading_quat(base_quat)), ref_quat))
    obs = np.zeros(ENCODER_DIM)
    obs[0] = ENCODER_MODE_G1
    body = np.concatenate([self.motion["joint_pos"][idx].reshape(-1), joint_vel.reshape(-1), anchor])
    obs[4:4 + len(body)] = body
    return obs

  def act(self, state: RobotState) -> JointCommand:
    quat = state.base_quat.astype(np.float64)
    if self.apply_delta_heading is None:
      ref_heading = heading_quat(self.motion["root_quat"][0])
      self.apply_delta_heading = quat_mul(heading_quat(quat), quat_conj(ref_heading))
    self.history.append((state.base_ang_vel.copy(), self.mapper.to_target(state.q) - self.default_il,
                         self.mapper.to_target(state.dq), self.last_action.copy(),
                         quat_rotate(quat_conj(quat), np.array([0.0, 0.0, -1.0]))))

    encoder_obs = self._encoder_obs(self._frame(state.t), quat).astype(np.float32)[None]
    token = self.encoder.run(None, {"obs_dict": encoder_obs})[0].reshape(-1)
    columns = [np.stack(c).reshape(-1) for c in zip(*self.history)]
    obs = np.concatenate([token, *columns]).astype(np.float32)[None]
    self.last_action = self.decoder.run(None, {"obs_dict": obs})[0].reshape(-1).astype(np.float64)

    target = DEFAULT_ANGLES + self.mapper.to_source(self.last_action) * ACTION_SCALE
    return JointCommand(q=target, kp=KP.copy(), kd=KD.copy())

  def describe(self) -> Dict[str, Any]:
    return {"source": "NVIDIA GEAR-SONIC sonic_v1_1 (g1_deploy_onnx_ref, encoder mode g1)",
            "encoder": str(self.encoder_path), "encoder_sha256": sha256(self.encoder_path),
            "motion": self.config["motion"], "motion_frames": self.frames, "hold_s": self.hold / CONTROL_HZ}
