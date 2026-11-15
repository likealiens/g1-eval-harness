# G1 Policy Evaluation Harness

Fork of [likealiens/g1-eval-harness](https://github.com/likealiens/g1-eval-harness) (v1.0.0, commit `9916b20`).

One shared MuJoCo evaluation tool for every Unitree G1 policy: walking policies that follow a
velocity command, and motion-tracking policies that follow a reference motion.
Every policy runs on the **same robot model, physics, control rate, torque limits,
scenarios and seeds**, and gets the **same metrics, logs and videos**, so results
from different policies can be compared directly.

The harness owns everything that must stay identical. Each policy owner writes
one small **adapter** that translates between the harness and their policy.

The harness tells the two kinds apart from the adapter: if `reference(t)` returns the pelvis
position and orientation, the policy is scored as a motion tracker (§3d); otherwise as a walking
policy. Each run prints `policy type: ...` and records it in `summary.json`.

```
g1-eval-harness/
├── evaluate.py                 # the one command
├── harness/                    # OWNED BY THE HARNESS - do not change per policy
│   ├── config.py               #   physics 500 Hz, control 50 Hz, fall rules, seeds, robot models
│   ├── robot.py                #   real G1 29-DOF model, name-based indexing, limits
│   ├── scenarios.py            #   flat, stand, low friction, ice, 3 pushes
│   ├── engine.py               #   sim loop, PD law, torque limits, pushes, logging
│   ├── metrics.py              #   all metrics (see METRICS.md)
│   ├── reporting.py            #   summary.json / summary.md / logs / videos
│   └── types.py                #   RobotState in, JointCommand out (LowState / LowCmd)
├── adapters/                   # ONE FILE PER POLICY
│   ├── base.py                 #   PolicyAdapter interface
│   ├── unitree_rl_gym_adapter.py   # walking policy: unitree_rl_gym pretrained G1 (example)
│   ├── beyondmimic_adapter.py  #   motion tracker: BeyondMimic (mjlab ONNX export)
│   ├── sonic_v1_1_adapter.py   #   motion tracker: SONIC v1.1 (NVIDIA's release)
│   ├── sonic_vith_adapter.py   #   motion tracker: SONIC (ViT-H), run by ViT-H's g1_tracker
│   ├── template_adapter.py     #   copy this to start your own
│   └── external/               #   policies that run in another process
│       ├── protocol.py         #     standalone TCP protocol (copy into your env)
│       ├── socket_bridge.py    #     harness side, lock-step, deterministic
│       └── unitree_dds_bridge.py #   harness acts as the robot over Unitree DDS
├── utils/
│   ├── joint_mapping.py        # Marvarid's joint-order mapper
│   ├── joint_orders.py         # known orders (IsaacLab, ...)
│   └── policy_runtime.py       # TorchScript loader (torch, or numpy fallback)
├── assets/g1/                  # G1 models + meshes (§4)
│   ├── g1_ros2_upstream.xml    #   ROS 2 stack's G1 (unitree_description 1.3.0, BSD)
│   ├── NOTICE_unitree_description.txt  #   its source and license
│   ├── build_ros2_model.py     #   -> g1_29dof_ros2.xml, the default model
│   ├── g1_29dof_rev_1_0.xml    #   Unitree's G1 (unitree_rl_gym, BSD-3)
│   ├── build_harness_model.py  #   -> g1_29dof_harness.xml, the v1 model
│   └── meshes/
├── checkpoints/unitree_rl_gym_g1_motion.pt
├── examples/external_unitree_rl_gym_client.py
├── tests/
├── STATUS.md                   # deliverables + open items  <- start here
├── METRICS.md  DESIGN.md  FIXES.md
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.9+
pip install -r requirements.txt
pip install torch            # recommended; without it the numpy fallback runs the example policy
pip install onnx onnxruntime scipy   # only for the motion trackers (§2, §3d)
python -m pytest -q tests    # 30 tests, ~5 s
```

Videos are rendered offscreen. On Linux `evaluate.py` uses `MUJOCO_GL=egl`
(headless GPU); on a Linux machine without a GPU use `export MUJOCO_GL=osmesa` or `--no-video`.
On Windows and macOS nothing needs to be set. If you set `MUJOCO_GL` yourself on Windows,
remove it (`set MUJOCO_GL=`), because `egl`/`osmesa` are Linux-only.

**Windows `cmd`:** the multi-line commands below use `\` (Linux/macOS). In `cmd`, put the
command on one line or end lines with `^` instead.
Rendering is the slow part on CPU-only machines; the simulation itself runs
~20x faster than real time.

## 2. Evaluate a policy - one command

```bash
python evaluate.py \
  --adapter adapters.unitree_rl_gym_adapter:UnitreeRLGymAdapter \
  --checkpoint checkpoints/unitree_rl_gym_g1_motion.pt
```

Defaults: all 7 scenarios, seeds `0 1 2` (quick check; use 10 seeds for reports), command `vx=0.5 m/s`, gains from the policy.
Motion trackers ignore the speed command, so they skip `stand_still` and run the other 6.

**Motion trackers.** Checkpoints and motion files are not in the repo; §3d says where they come from.

```bash
# BeyondMimic: the motion is inside the ONNX file
python evaluate.py --adapter adapters.beyondmimic_adapter:BeyondMimicAdapter --checkpoint policy.onnx

# SONIC v1.1: model_encoder.onnx is read from the decoder's folder
python evaluate.py --adapter adapters.sonic_v1_1_adapter:SonicV11Adapter \
    --checkpoint GR00T-WholeBodyControl/gear_sonic_deploy/policy/sonic_v1_1/model_decoder.onnx \
    --set motion=motion.npz

# SONIC (ViT-H)
python evaluate.py --adapter adapters.sonic_vith_adapter:SonicVitHAdapter \
    --checkpoint ViT-H-release/inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx \
    --set tracker_root=ViT-H-release/g1_tracker --set motion=motion.npz
```

Common options:

| option | meaning |
|---|---|
| `--scenarios flat_nominal ice` | subset (`--list-scenarios` shows all) |
| `--seeds 0 1 2 3 4` | more seeds = tighter confidence |
| `--cmd-vel 0.8 0 0` | velocity command `vx vy yaw_rate` (scenarios like `stand_still` override it) |
| `--robot unitree_rl_gym` | the v1 robot model instead of the default `ros2` (§4) |
| `--gains harness` | force the shared G1 reference gains instead of each policy's own (see DESIGN.md) |
| `--set key=value` / `--adapter-config cfg.yaml` | pass options to your adapter |
| `--no-video`, `--video-all-seeds` | video control (default: first seed of each scenario) |
| `--output-dir results/x` | default `results/<policy>_<timestamp>` |
| `--stop-at-tracking-failure` | motion trackers: end the episode at the first tracking failure, with no real falls (§3d) |

### What you get

```
results/<policy>_<timestamp>/
├── summary.md        # table: mean ± std over seeds per scenario  <- read this first
├── summary.json      # everything: config snapshot, robot model, per-trial metrics, aggregates,
│                     #   seed perturbations, checkpoint sha256, versions
├── logs/<scenario>_seed<k>.npz   # per-control-step time series (q, dq, targets, gains,
│                                  #   velocities, base pose, torques, latency)
└── videos/<scenario>_seed<k>.mp4
```

Motion trackers also get MPKPE, root-relative MPKPE, drift and strict success instead of the
velocity metrics (METRICS.md §3).

Expected result for the example policy (reference run in `examples/sample_results/`, on the v1
robot model, so run with `--robot unitree_rl_gym`; PyTorch backend, Windows 10, Python 3.10,
MuJoCo 3.15, seeds 0 1 2):

| scenario | falls | v_xy RMSE m/s | power W |
|---|---|---|---|
| flat_nominal | 0% | 0.114 | 161 |
| stand_still | 0% | 0.095 | 119 |
| low_friction | 0% | 0.065 | 153 |
| ice | 100% | 0.313 | 347 |
| push_lateral_medium | 33% | 0.250 | 160 |
| push_forward_hard | 33% | 0.583 | 203 |
| push_backward_medium | 0% | 0.174 | 146 |

Videos for four scenarios are in `examples/sample_results/videos/`.

**Reproducibility.** The same policy + seed on the same machine and setup gives identical
results. Across machines, operating systems or backends (PyTorch vs the numpy fallback, CPU vs
GPU) tiny floating-point differences can change the outcome of trials where the robot is on the
edge of falling. Example: with the numpy backend, ice seed 0 survives the full 10 s; with
PyTorch it falls at 8.6 s, so the ice fall rate reads 67 % vs 100 %. All other scenarios match.
Therefore:

- **Compare policies on the same machine with the same environment.**
- **Use at least 10 seeds for any reported comparison** (`--seeds 0 1 2 3 4 5 6 7 8 9`);
  with 3 seeds one flipped trial moves a fall rate by 33 %.
- **Compare on the same robot model** (`summary.json → robot.model`); results depend on it (§4).

If you get something very different from the table with the same seeds, `--robot unitree_rl_gym`
and the PyTorch backend, your setup differs - check `summary.json → environment` and the
checkpoint hash (`cf668f75b90d1abf`).

## 3. Add your own policy

You never edit `harness/`. You write one adapter.

### 3a. Policy you can call from Python (most cases)

1. `cp adapters/template_adapter.py adapters/<yourname>_adapter.py`
2. Set `POLICY_JOINT_ORDER` to the joint order your network uses (29 names).
   The mapper (`self.mapper`, `utils/joint_mapping.py`) converts to/from the harness's canonical order:
   `self.mapper.to_target(canonical) -> yours`, `self.mapper.to_source(yours) -> canonical`.
   Known orders are in `utils/joint_orders.py`; add new ones there.
3. Implement four methods:

```python
def default_pose(self):          # 29 canonical angles to spawn in
def reset(self, seed):           # clear RNN memory, last action, phase clock
def act(self, state):            # RobotState -> JointCommand(q, kp, kd[, dq, tau_ff])
def reference(self, t):          # optional: {"q": canonical[29]} for motion-tracking policies (§3d)
```

`state` gives you `q, dq, tau, base_quat (wxyz), base_ang_vel (body), base_lin_acc (body),
projected_gravity ([0,0,-1] upright), command, t` - what the real robot measures.
`base_lin_vel` / `base_pos` are privileged; if you use them set
`uses_privileged_state = True` (it is recorded in the results).

4. Run it:

```bash
python evaluate.py --adapter adapters.<yourname>_adapter:YourAdapter --checkpoint path/to/policy.pt
```

5. Sanity-check before trusting numbers: watch `videos/flat_nominal_seed0.mp4`.
   If the robot collapses in the first second, it is almost always the adapter
   (joint order, default pose, gravity sign, scaling, gains) - see the checklist at
   the top of `template_adapter.py`, and compare with your training repo's own sim2sim.

### 3b. Policy in another process / environment (socket, deterministic)

If your policy lives in another venv, a ROS 2 node, or C++, copy
`adapters/external/protocol.py` there and subclass `PolicyClient`
(`examples/external_unitree_rl_gym_client.py` is a complete example).
The harness waits for your answer every control step, so results are
**bit-identical** to running in-process (verified by `tests/test_reference_policy.py`).

```bash
python evaluate.py --adapter adapters.external.socket_bridge:SocketBridgeAdapter --set port=7788
python examples/external_unitree_rl_gym_client.py --checkpoint checkpoints/unitree_rl_gym_g1_motion.pt --port 7788
```

(or let the harness launch the client: `--set "launch=python my_client.py --port 7788"`).

### 3c. Policy inside its own deploy stack over Unitree DDS (SONIC, BeyondMimic)

The harness pretends to be the robot: it publishes `rt/lowstate` from MuJoCo every
physics step and applies whatever arrives on `rt/lowcmd`, with the same PD law,
limits, scenarios and metrics. Simulation is paced to wall clock.

```bash
pip install unitree_sdk2py
python evaluate.py --adapter adapters.external.unitree_dds_bridge:UnitreeDDSBridgeAdapter \
    --set interface=lo --set domain_id=1 --scenarios flat_nominal --seeds 0
# then start the deploy stack on the same interface / domain id
```

> **Status:** written against the unitree_sdk2_python API but **not yet tested with a real
> SONIC / BeyondMimic deployment**. Expect to adjust `interface`, `domain_id`, `mode_machine`
> and the stack's start-up handshake. Real-time runs are not bit-reproducible; use more seeds.
> Where possible, prefer exporting the policy (ONNX/TorchScript) and using 3a - it is
> deterministic and much easier to debug. BeyondMimic and SONIC already run this way (§3d).

### 3d. Motion trackers

A motion tracker is a 3a adapter whose `reference(t)` also returns the root pose:
`{"q": canonical[29], "root_pos": xyz, "root_quat": wxyz}` (pelvis, world frame). Implement:

```python
def reference(self, t):          # {"q", "root_pos", "root_quat"} of the motion at time t
def motion_duration(self):       # episode length in seconds (the motion's length)
def default_base_pose(self):     # (pelvis xyz, pelvis wxyz) of the first frame; the harness sets the height
```

How a motion tracker is scored:

- each episode lasts `motion_duration()` and starts at `default_base_pose()`, at rest, in the
  first frame's joint angles plus the seeded joint noise;
- it fails when the torso height, torso tilt or a hand/foot height drifts from the reference
  (thresholds in `harness/config.py`, as in BeyondMimic); the absolute fall rules don't apply,
  since a motion may kneel or jump on purpose;
- the episode runs on after a tracking failure, so the tracking failure (`tracking_failure`, its time
  and rule) and real falls (`fell`: head below 0.5 m for over 2 s, or still down when the motion ends) are
  reported separately (METRICS.md §1); `fall` is the first failure of either kind. With
  `--stop-at-tracking-failure` the episode ends at the first tracking failure, as in BeyondMimic;
  success and tracking error up to the failure are the same either way;
- MPKPE, root-relative MPKPE, drift and strict success are computed on mjlab's 14 key bodies;
  velocity metrics are n/a (METRICS.md §3).

Physics, PD law, scenarios, seeds and the other metrics are the same as for walking policies.

Inputs of the shipped trackers:

- **BeyondMimic:** an ONNX file exported by mjlab's tracking task (`Mjlab-Tracking-Flat-Unitree-G1`);
  the adapter reads the network, the reference motion and the gains from it. Files from older
  exporters lack `anchor_body_name`; re-export them.
- **SONIC v1.1:** NVIDIA's [GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) v1.1 encoder and
  decoder, run as their C++ deploy (`g1_deploy_onnx_ref`) runs them in encoder mode `g1`: 10 future
  frames at step 5 of joint positions and velocities, plus the anchor orientation relative to the
  robot's heading. Joint order, gains, action scale, history and heading follow that code. Files:
  `python download_from_hf.py --sonic-v1-1 --no-planner` in
  [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) (revision `6733128a`:
  encoder `fb97de22…8365bae`, decoder `34bae857…f39c24a`). `--set encoder=…` points to another
  encoder file. `--set hold_s=…` holds the first frame before playback, as their deploy does
  before `T`.
- **SONIC (ViT-H):** NVIDIA's SONIC fine-tuned by ViT-H's authors; checkpoint and `g1_tracker` from
  [ViT-H](https://github.com/mileret/ViT-H-release)
  (`hf download eerrr1/ViT-H-checkpoints model_step_020000_g1.onnx --repo-type dataset`).
- **Motion file (both SONIC adapters):** a `g1_tracker` input `.npz` with `joint_pos` (T, 29,
  IsaacLab order, rad), `root_pos_w` (T, 3, z-up), `root_quat_w` (T, 4, wxyz),
  `body_pos_w` (T, 30, 3; zeros are fine) and `fps`.

## 4. Rules that keep it fair (owned by `harness/config.py`)

- Robot: `ros2` by default, `--robot unitree_rl_gym` for the v1 model (table below)
- Physics 500 Hz, policy 50 Hz; policies trained at another rate are rejected
- Torque limits from Unitree's model (88/139/50/25/5 N·m), commanded torque is clipped
- PD law `tau = kp(q* − q) + kd(dq* − dq) + tau_ff` computed by the harness every physics step
- Spawn on the ground with seeded ±0.02 rad joint noise; seeded push time/direction jitter
- Fall = pelvis < 0.40 m, tilt > 60°, or any body other than the feet touches the floor
- Motion trackers instead fail when the torso height (0.25 m), torso tilt (projected-gravity z, 0.8)
  or a hand/foot height (0.25 m) differs from the reference by more than the threshold (§3d)
- Any change to these goes through a reviewed commit to `harness/`, and bumps `HARNESS_VERSION`

| | `ros2` (default) | `unitree_rl_gym` (v1) |
|---|---|---|
| Source | ROS 2 stack's G1: `unitree_description` 1.3.0, ROS 2 Jazzy (BSD); `build_ros2_model.py` | Unitree's `g1_29dof_rev_1_0` (unitree_rl_gym); `build_harness_model.py` |
| Motor inertia (armature) | per joint, from the motor specs (0.0036–0.025) | 0.01 for every joint, from Unitree's 12-DoF model |
| Contact shapes | capsule soles and hands | 4 spheres per foot, mesh hands |

Masses, kinematics, joint ranges and torque limits are the same. `ros2` is the default because it is
the model of the ROS 2 stack that runs on the robot, with motor inertia from the motor specs.
Results depend on the model (DESIGN.md §5), so report it with every result (`summary.json → robot.model`).

## 5. Adapter checks

Each shipped adapter was compared with the policy's own sim-to-sim on the default `ros2` model.

| adapter | checked against | result |
|---|---|---|
| `unitree_rl_gym_adapter.py` (walking) | unitree_rl_gym's `deploy/deploy_mujoco` | reproduces it (DESIGN.md §1) |
| `beyondmimic_adapter.py` | mjlab's evaluation, LAFAN1 demo dance | finishes, as 93 % of mjlab's runs do; pose error (r-MPKPE) 3.5 vs 3.8 cm; joint error 0.082 vs 0.084 rad |
| | the ROS 2 deploy stack, same dance | finishes, as the stack does (10/10 runs) |
| `sonic_v1_1_adapter.py` | NVIDIA's sim-to-sim: tennis serve (ViT-H motion) | finishes, as theirs does; joint error 0.139 vs 0.148 rad |
| | same, jump kick (ViT-H motion) | fails at 1.0 s; theirs at 0.96 s |
| | same, NVIDIA's squat example | finishes, as theirs does; joint error 0.102 vs 0.107 rad |
| `sonic_vith_adapter.py` | ViT-H's `g1_tracker`: LAFAN1 demo dance | fails at the cartwheel, 106.7 s, as `g1_tracker` does |
| | same, tennis serve (GVHMR + GMR motion) | fails at 6.7 s; `g1_tracker` at 7.8 s |

For SONIC v1.1, NVIDIA's sim-to-sim ran their deploy with TensorRT swapped for ONNX Runtime on CPU
(same networks). Given the robot states it logged, the adapter builds the same encoder and decoder
inputs on every tick (within 5e-6), and the pelvis height stays within 2 cm of theirs.

**Privileged state.** BeyondMimic (anchor position, base linear velocity) and SONIC (ViT-H) (base
position) read the true base state, so they set `uses_privileged_state = True`; on the robot it comes
from a state estimator. SONIC v1.1 and the walking example read only the IMU and joints, as on the robot.

Metric definitions: **[METRICS.md](METRICS.md)**. Why it is designed this way: **[DESIGN.md](DESIGN.md)**.
What was wrong with the first version: **[FIXES.md](FIXES.md)**. Versions, status and open items: **[STATUS.md](STATUS.md)**.
