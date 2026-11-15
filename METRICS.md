# Metrics

Computed by `harness/metrics.py` for every (scenario, seed) trial, then aggregated as
mean / std / min / max over seeds. All metrics come from the MuJoCo state, so they are
identical whether the policy runs in-process, over the socket, or over DDS.

Notation: control step k (50 Hz, Δt = 0.02 s), physics step i (500 Hz). Joint arrays are
29-DOF in canonical order. "Heading frame" = world frame rotated by the pelvis yaw.

## 1. Falls and survival

| key | unit | definition |
|---|---|---|
| `fall` | bool | trial ended early for any reason below |
| `fall_reason` | text | `base_too_low` (pelvis z < 0.40 m), `base_tilt` (angle between pelvis z-axis and world z > 60°), `ground_contact:<body>` (any body except the two `ankle_roll_link`s touches the floor), `sim_unstable` (MuJoCo NaN/Inf acceleration), `invalid_action` (policy returned NaN/Inf) |
| `time_to_fall_s` | s | sim time at the end of the trial if it fell, else empty |
| `survival_fraction` | 0–1 | time survived / episode length (1.0 if no fall); the episode is the scenario's 10 s, or the motion for motion trackers (§3) |

In aggregates, the mean of `fall` is the **fall rate** over seeds.

### Tracking failure vs real fall (motion trackers)

A tracking failure (§3) is not a fall: a policy can lose the reference and stay on its feet. So a motion
tracker's first tracking failure is recorded and the episode runs on until the motion ends or the robot
**really falls**: `head_down`, the head below 0.5 m for longer than 2 s, as in our ROS 2 analysis
(`analyze_ros2.py`). The head is the `mid360` lidar site of the `ros2` model (the same point, `torso_link`
+ (0.0003, 0, 0.416) m, on `unitree_rl_gym`); it stands at 1.25 m. The trial ends once the head has been
down for 2 s; a head still below 0.5 m when the motion ends is a real fall too. `fell_time_s` is when the
head went down. Pelvis height, tilt and ground contact are not checked, since motions may crouch, kneel or
cartwheel on purpose (the LAFAN1 cartwheel tilts the pelvis to 149°, with the head below 0.5 m for 0.56 s).
`sim_unstable` and `invalid_action` still end the trial at once. With `--stop-at-tracking-failure` the
episode ends at the first tracking failure instead (BeyondMimic's termination), counted as `fall`.

`fall`, `fall_reason`, `time_to_fall_s`, `survival_fraction` and `success_strict` keep their meaning: the
first failure of either kind (same values as with `--stop-at-tracking-failure`, unless the robot really
falls before any tracking rule fires). Added per trial:

| key | unit | definition |
|---|---|---|
| `tracking_failure` | bool | a tracking rule (§3) fired |
| `tracking_failure_reason`, `tracking_failure_time_s` | text, s | the first one, and its sim time |
| `fell` | bool | real fall: head below 0.5 m for over 2 s, or still below when the motion ends |
| `fell_reason`, `fell_time_s` | text, s | `head_down`, `sim_unstable` or `invalid_action`, and its sim time (for `head_down`, when the head went down) |
| `pre_failure_mpkpe_m`, `pre_failure_r_mpkpe_m` | m | §3 metrics up to the first tracking failure (whole run if none); same as with `--stop-at-tracking-failure` |
| `pre_failure_joint_pos_tracking_mae` | rad | same for `joint_pos_tracking_mae` |

All other metrics, including `mpkpe_m`, `r_mpkpe_m` and the joint errors, cover the whole run. Means over
seeds: `tracking_failure` = **tracking-failure rate**, `fell` = **real-fall rate**, 1 − `fall` = success
rate. With `--stop-at-tracking-failure` these fields are absent. Walking policies are unaffected.

## 2. Velocity tracking

Measured only for t ≥ 1.0 s (the robot starts from rest) and only while upright.
v = pelvis linear velocity in the heading frame, ω_z = world yaw rate, c = command.

| key | unit | definition |
|---|---|---|
| `lin_vel_xy_rmse` | m/s | **primary.** RMSE between the 1 s moving average of (v_x, v_y) and the command: sqrt(mean_k ‖c̄_xy − v̄_xy‖²) |
| `yaw_rate_rmse` | rad/s | **primary.** same with the 1 s moving average of ω_z |
| `lin_vel_xy_rmse_instant` | m/s | same without averaging |
| `yaw_rate_rmse_instant` | rad/s | same without averaging |
| `mean_forward_vel`, `mean_lateral_vel` | m/s | mean heading-frame v_x, v_y |

Why the 1 s window: a walking G1's pelvis sways and twists every step (instantaneous yaw rate
oscillates by ~±1 rad/s while walking straight). Without averaging, the metric would mostly
measure gait style rather than whether the robot goes where it is told. 1 s covers at least
one full gait cycle. The instantaneous versions are kept for completeness.

Returns empty (n/a) if the robot fell before 1 s; use the fall metrics in that case.

## 3. Motion tracking error (motion-tracking policies only)

Only computed if the adapter implements `reference(t)` (BeyondMimic, SONIC, any imitation policy).

| key | unit | definition |
|---|---|---|
| `joint_pos_tracking_rmse` | rad | sqrt(mean over steps and masked joints of (q − q_ref)²) |
| `joint_pos_tracking_mae` | rad | mean over steps and masked joints of \|q − q_ref\| (the joint error we report for mjlab runs) |

If the reference also has a root pose, the policy is a **motion tracker**. Failure, which ends the
episode only with `--stop-at-tracking-failure` (§1), is then judged against the reference
(`fall_reason` = `tracking:anchor_height`, `tracking:anchor_orientation` or
`tracking:end_effector_height`, thresholds in `harness/config.py`, same as BeyondMimic / mjlab), and the
14 key bodies (`tracking_key_bodies`, mjlab's set) of robot and reference are compared every control
step (reference at the start of the step, robot after it, as in mjlab's reward). The reference is posed
on the harness robot model and shifted vertically so that it starts at the robot's spawn height.

| key | unit | definition |
|---|---|---|
| `mpkpe_m` | m | mean over steps of the mean key-body distance ‖p_ref − p‖, world frame |
| `max_mpkpe_m` | m | worst step of the above |
| `r_mpkpe_m` | m | same after re-anchoring the reference on the robot as mjlab's `body_pos_relative_w` does: turned about z by the yaw of q_anchor · q_ref_anchor⁻¹ and moved onto the robot's anchor in xy (keeps its own height), using the robot's anchor at the start of the step. Pose error without drift. |
| `max_r_mpkpe_m` | m | worst step of the above |
| `drift_mean_m`, `drift_max_m` | m | anchor (torso) xy distance from the reference path |
| `success_strict` | 0/1 | no failure and `max_mpkpe_m` ≤ 0.15 m throughout (PHUMA's strict success) |

Velocity-tracking metrics are n/a for motion trackers.

The adapter provides the reference (it owns the motion file and its timing); the harness
measures the error against the simulated robot. Velocity policies show n/a.

## 4. Smoothness

| key | unit | definition |
|---|---|---|
| `action_rate_rms` | rad/s | RMS over steps and joints of (q*_k − q*_(k−1)) / Δt, where q* is the commanded joint target. Measures how jumpy the policy's output is. **Primary smoothness metric.** |
| `joint_acc_rms` | rad/s² | RMS of (dq_k − dq_(k−1)) / Δt |
| `joint_jerk_rms` | rad/s³ | RMS of the second difference of dq divided by Δt² |

Accelerations and jerk are finite differences at 50 Hz. Foot impacts produce real spikes,
so compare jerk between policies on the same scenario, not as an absolute number.

## 5. Effort

| key | unit | definition |
|---|---|---|
| `mean_mech_power_w` | W | mean over physics steps of Σ_j \|τ_j · dq_j\| (applied torque) |
| `torque_rms` | N·m | RMS over physics steps and joints of applied torque |

## 6. Limit violations

| key | unit | definition |
|---|---|---|
| `joint_limit_violation_rate` | 0–1 | fraction of control steps where at least one joint is outside its range in Unitree's model (+1 mrad tolerance; MuJoCo limits are soft so a joint can be pushed past them) |
| `joint_limit_violation_count` | count | number of (step, joint) pairs outside range |
| `near_joint_limit_rate` | 0–1 | fraction of (step, joint) pairs in the outer 5 % of the range (2.5 % at each end) |
| `torque_saturation_rate` | 0–1 | fraction of physics steps where the **commanded** torque of at least one motor exceeded its limit (counted *before* clipping) |
| `torque_saturation_count` | count | number of (physics step, joint) pairs that saturated |

Torque limits per joint come from Unitree's model: hip pitch/yaw and waist yaw 88 N·m,
hip roll and knee 139, ankles and waist roll/pitch 50, shoulders/elbows/wrist roll 25,
wrist pitch/yaw 5.

## 7. Bookkeeping (not for ranking)

`episode_length_steps`, `sim_time_s`, `mean_policy_latency_ms` (wall-clock time of `act()`;
depends on the machine, informational only).

## How to compare two policies

1. Same harness version (`summary.json → harness.harness_version`), same seeds, same command,
   **same machine and environment** (see README, Reproducibility).
2. Look at fall rate per scenario first. A policy that falls is not "smoother".
3. Then tracking, then smoothness/effort/limits.
4. With 3 seeds, differences smaller than ~2× the std are not meaningful. Use 5–10 seeds for final reports.
