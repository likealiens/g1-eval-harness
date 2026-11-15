"""Both robot models: same robot, different contacts and motor inertia."""

import mujoco
import numpy as np

from harness.config import DEFAULT_ROBOT, ROBOT_MODELS, HarnessConfig
from harness.robot import RobotModel


def models():
  return {name: RobotModel(path, 0.002) for name, path in ROBOT_MODELS.items()}


def test_default_is_ros2():
  assert DEFAULT_ROBOT == "ros2"
  assert HarnessConfig().model_xml == ROBOT_MODELS["ros2"]


def test_same_mass_limits_and_kinematics():
  a, b = models().values()
  assert np.isclose(a.model.body_subtreemass[a.base_body], b.model.body_subtreemass[b.base_body])
  assert np.allclose(a.q_lower, b.q_lower) and np.allclose(a.q_upper, b.q_upper)
  assert np.allclose(a.torque_limit, b.torque_limit)
  q = np.random.default_rng(0).uniform(a.q_lower, a.q_upper)
  pos = []
  for r in (a, b):
    d = mujoco.MjData(r.model)
    d.qpos[r.qpos_adr] = q
    mujoco.mj_kinematics(r.model, d)
    pos.append({r.body_name(i): d.xpos[i].copy() for i in range(1, r.model.nbody)})
  for name in set(pos[0]) & set(pos[1]):
    assert np.allclose(pos[0][name], pos[1][name], atol=1e-9), name


def test_ros2_has_motor_armature_per_joint():
  r = models()["ros2"]
  armature = r.model.dof_armature[r.model.jnt_dofadr[r.joint_ids]]
  assert len(set(np.round(armature, 6))) > 1
  assert r.pelvis_acc_sensor is not None
