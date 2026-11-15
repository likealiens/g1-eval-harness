# Design

## 1. How existing projects run sim2sim

| project | how the policy is run in MuJoCo | consequence for us |
|---|---|---|
| **unitree_rl_gym** (`deploy/deploy_mujoco`) | One Python script loads a TorchScript policy, builds the observation itself, runs the policy at 50 Hz and a PD loop at 500 Hz on Unitree's G1 MJCF. Gains, default angles, scales and the 0.8 s gait period live in a per-robot YAML (`configs/g1.yaml`). Verified: our adapter reproduces it. | Direct-call policies. Observation and gains are policy-specific: they must live in the adapter. |
| **unitree_mujoco** | Runs MuJoCo as a stand-in for the robot: publishes `rt/lowstate` and consumes `rt/lowcmd` over DDS, so the *same* binary that drives the real robot can drive the sim. | Pattern for evaluating stacks we cannot (or should not) call directly. |
| **BeyondMimic** (whole_body_tracking) | Trained in IsaacLab; policy exported (ONNX) together with its deployment parameters. Deployed by a ROS 2 controller that also runs the reference motion. Joint order is IsaacLab's, not Unitree's. mjlab's export (v2.0.0 adapter) stores the network, reference motion, gains, default pose and joint names in one ONNX file. | Needs joint remapping. Best evaluated by loading the exported policy in an adapter (deterministic) and providing `reference(t)` for tracking error (`adapters/beyondmimic_adapter.py`); the ROS 2 path works through the DDS bridge if the controller is pointed at the Unitree SDK interface. |
| **SONIC** | C++ deployment stack communicating with the robot over Unitree DDS; sim2sim runs it against a MuJoCo process that behaves like the robot. Its topics and start-up handshake are not yet checked. | Exactly the unitree_mujoco pattern → DDS bridge. Since v2.0.0 its ONNX networks also run in-process (`adapters/sonic_v1_1_adapter.py`, `adapters/sonic_vith_adapter.py`). |

Common lesson: **sim2sim failures are almost always mismatches between training and
deployment** (joint order, default pose, observation scaling/sign, gains, control rate,
robot model). The first version of this harness had all of these (see FIXES.md). The
design below tries to make each of them either impossible or loudly visible.

## 2. Split of responsibilities

| harness owns (identical for everyone) | adapter owns (per policy) |
|---|---|
| robot model, physics 500 Hz, control 50 Hz | observation layout, scaling, history |
| PD law and torque limits, clipping | network loading and inference, RNN state |
| spawning, scenarios, seeded disturbances | joint order (via the joint-order mapper) |
| fall detection, metrics, logs, videos | default pose, action scaling |
| result format, config snapshot | gains the policy was trained with |
| | reference motion (tracking policies) |

The interface between them mirrors Unitree's LowState / LowCmd:
`RobotState` (q, dq, tau, IMU quaternion/gyro/accelerometer, command, t) in,
`JointCommand` (q, dq, kp, kd, tau_ff per joint) out. Because it is the same information
the real robot exchanges, an in-process adapter, a socket client and a DDS deploy stack are
three transports for the same thing, and the metrics code never knows which one was used.

## 3. Three ways to run a policy

1. **In-process adapter** (`adapters/*.py`) - deterministic, fastest, easiest to debug. Default.
2. **Socket bridge** (`adapters/external/socket_bridge.py`) - policy in another process
   (another venv, ROS 2 Python node, C++). Lock-step: sim time stops until the command
   arrives, so results are bit-identical to (1). Tested.
3. **DDS bridge** (`adapters/external/unitree_dds_bridge.py`) - harness acts as the robot for an
   unmodified deploy stack. Wall-clock paced, so not bit-reproducible; latency/staleness stats
   are saved. Untested against real stacks at the time of writing.

Recommendation: use (1) or (2) for comparisons; use (3) to check that the real deploy stack
behaves the same as the exported policy.

## 4. PD gains - an open decision

The strict reading of a fair comparison is "same PD gains". On the real G1, however, **gains are sent by the policy in every
LowCmd**; unitree_rl_gym, BeyondMimic and SONIC were each trained with different gains. Forcing
one gain set on all of them runs every policy outside the conditions it was trained for, so the
comparison would measure gain mismatch, not policy quality.

Implemented: `--gains policy` (default) - each adapter sends its training gains, the harness
applies them with the same PD law and torque limits and records them in the log.
`--gains harness` forces the shared table in `harness/config.py` for anyone who wants the strict
reading. Both are one flag apart; the choice is recorded in `summary.json`.

## 5. Fairness and reproducibility details

- **Robot model** (v2.0.0). Two models, each built by a script from an unmodified upstream copy, so
  the changes are auditable:
  - `ros2` (default): the G1 of the ROS 2 deployment stack that runs on the robot
    (`unitree_description` 1.3.0, ROS 2 Jazzy, BSD): motor armature per joint from the motor specs,
    capsule soles and hands. BeyondMimic's sim2sim (mjlab and the ROS 2 stack) uses it.
  - `unitree_rl_gym` (v1): Unitree's `g1_29dof_rev_1_0.xml` plus one armature (0.01) for every
    joint, taken from unitree_rl_gym's 12-DoF sim2sim; 4 spheres per foot, mesh hands.

  Masses, kinematics, joint ranges and torque limits are the same, yet results change with the model:

  | | `ros2` | `unitree_rl_gym` |
  |---|---|---|
  | BeyondMimic, LAFAN1 demo dance | finishes (as in mjlab and the ROS 2 stack) | fails at the cartwheel, 106.9 s |
  | SONIC (ViT-H), LAFAN1 demo dance | fails at 106.7 s | fails at 106.8 s |
  | SONIC (ViT-H), tennis serve (GVHMR + GMR motion) | fails at 6.7 s | fails at 7.8 s |
  | Example walking policy, flat-ground speed error | 0.03 m/s | 0.11 m/s |
  | Example walking policy, falls on ice | 0 % | 67 % |
  | Example walking policy, falls under the sideways push | 67 % | 33 % |

  Walking rows: seeds 0 1 2, both models on the same Linux machine with the PyTorch backend. The
  reference run in `examples/sample_results/` (Windows) falls on ice in all 3 seeds; that is the
  machine difference described in README, Reproducibility, not the model.

  `ros2` is the default because it is the model the policy meets on the robot's own software stack,
  its armature comes from the motor specs rather than one value for every joint, and BeyondMimic's
  sim2sim already uses it. `--robot unitree_rl_gym` keeps v1.0.0 results exactly. The choice is
  recorded in `summary.json → robot.model`; the default is still open (STATUS.md, item 5).
- **Name-based indexing everywhere** (`harness/robot.py`): a wrong model or joint order fails at
  start-up instead of silently scrambling joints.
- **Seeds**: per-trial `numpy.random.default_rng(seed)` draws initial joint noise, push time and
  push direction. Same seed → same disturbances for every policy; same seed + same policy on the same
  machine/backend → bit-identical trial (tested). Across machines or backends, trials on the edge
  of falling can flip (see README, Reproducibility), so comparisons are run on one machine.
- **Tracking failure vs real fall** (motion trackers): the episode runs on after the first tracking
  failure, so a policy that loses the reference but stays on its feet is told apart from one that
  falls (METRICS.md §1). `--stop-at-tracking-failure` ends it at the failure, as BeyondMimic's
  termination does; success and the tracking error up to the failure are the same either way.
- **Friction scenarios** set friction on the floor *and* the robot's geoms, because MuJoCo takes
  the maximum of the two.
- **Torque saturation** is counted on the commanded torque before clipping.
- **Policy rate** must match the harness rate; a policy trained at 100 Hz is rejected rather than
  silently run at 50 Hz.
- **Results** store the harness version, full config, scenario definitions, actual seed
  perturbations, checkpoint SHA-256 and library versions.

## 6. Known limitations / next steps

- No terrain scenarios yet (slopes, steps, rough ground). Adding one = a new MJCF scene + a
  `Scenario` entry.
- No sensor noise or latency injection (deliberately: the harness measures the policy, not the
  adapter's noise model). A `latency_steps` scenario option would be a useful addition.
- Joint velocity limits are not checked (not in Unitree's MJCF).
- The DDS bridge needs a first real run with the SONIC and BeyondMimic deploy stacks. Since v2.0.0
  both can be evaluated in-process instead (README §3d).
- `ISAACLAB_G1_29DOF_ORDER` is not checked against an IsaacLab BeyondMimic export. mjlab exports
  don't need it: they store `joint_names` in the ONNX metadata (STATUS.md, item 2).
