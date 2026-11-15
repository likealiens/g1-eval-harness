"""
harness/metrics.py

Turns one trial's log into the standard metrics. Definitions: METRICS.md.
Pure numpy, no MuJoCo, so it is easy to unit-test and re-run on saved logs.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np


@dataclass
class TrialMetrics:
    # Falls / survival
    fall: bool
    fall_reason: str
    time_to_fall_s: float            # NaN if no fall
    survival_fraction: float
    # Velocity tracking (heading frame, after warm-up, while upright)
    lin_vel_xy_rmse: float           # m/s,   1 s moving-average velocity vs command (primary)
    yaw_rate_rmse: float             # rad/s, 1 s moving-average yaw rate vs command (primary)
    lin_vel_xy_rmse_instant: float   # m/s,   instantaneous (includes in-gait sway)
    yaw_rate_rmse_instant: float     # rad/s, instantaneous (includes pelvis twist each step)
    mean_forward_vel: float          # m/s (heading frame x)
    mean_lateral_vel: float          # m/s (heading frame y)
    # Motion tracking (only if the adapter provides a reference)
    joint_pos_tracking_rmse: float   # rad, NaN if not applicable
    # Smoothness
    action_rate_rms: float           # rad/s   (change of joint targets between control steps)
    joint_acc_rms: float             # rad/s^2 (50 Hz finite difference of joint velocity)
    joint_jerk_rms: float            # rad/s^3 (50 Hz finite difference)
    # Effort
    mean_mech_power_w: float         # W, sum over joints of |tau * dq|
    torque_rms: float                # N*m
    # Limits
    joint_limit_violation_rate: float   # fraction of control steps with any joint outside its range
    joint_limit_violation_count: int    # joint-steps outside range
    near_joint_limit_rate: float        # fraction of joint-steps in the outer band of the range
    torque_saturation_rate: float       # fraction of physics steps where any motor hit its torque limit
    torque_saturation_count: int        # joint-physics-steps where commanded torque exceeded the limit
    # Bookkeeping
    episode_length_steps: int
    sim_time_s: float
    mean_policy_latency_ms: float       # wall-clock, informational only (hardware dependent)
    # Motion trackers only (NaN otherwise), mjlab's definitions.
    joint_pos_tracking_mae: float = float("nan")  # rad, mean |q - q_ref|
    success_strict: float = float("nan")   # no failure and MPKPE <= 0.15 m throughout
    mpkpe_m: float = float("nan")          # world frame
    max_mpkpe_m: float = float("nan")
    r_mpkpe_m: float = float("nan")        # reference re-anchored on the robot
    max_r_mpkpe_m: float = float("nan")
    drift_mean_m: float = float("nan")     # anchor xy distance from the reference
    drift_max_m: float = float("nan")
    # Only for motion trackers without --stop-at-tracking-failure (None and left out of to_dict otherwise).
    tracking_failure: Optional[bool] = None
    tracking_failure_reason: Optional[str] = None
    tracking_failure_time_s: Optional[float] = None
    fell: Optional[bool] = None
    fell_reason: Optional[str] = None
    fell_time_s: Optional[float] = None
    pre_failure_mpkpe_m: Optional[float] = None
    pre_failure_r_mpkpe_m: Optional[float] = None
    pre_failure_joint_pos_tracking_mae: Optional[float] = None

    def to_dict(self) -> Dict:
        out = {}
        for k, v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v, (bool, np.bool_)):
                out[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                out[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                out[k] = None if not np.isfinite(v) else float(v)
            else:
                out[k] = v
        return out


# Metrics where lower is better / higher is better (used by summary tables).
LOWER_IS_BETTER = ["joint_pos_tracking_mae", "mpkpe_m", "pre_failure_mpkpe_m", "pre_failure_r_mpkpe_m",
                   "pre_failure_joint_pos_tracking_mae", "max_mpkpe_m", "r_mpkpe_m", "max_r_mpkpe_m", "drift_mean_m", "drift_max_m",
                   "lin_vel_xy_rmse", "yaw_rate_rmse", "lin_vel_xy_rmse_instant", "yaw_rate_rmse_instant", "joint_pos_tracking_rmse", "action_rate_rms",
                   "joint_acc_rms", "joint_jerk_rms", "mean_mech_power_w", "torque_rms",
                   "joint_limit_violation_rate", "near_joint_limit_rate", "torque_saturation_rate"]
HIGHER_IS_BETTER = ["survival_fraction", "success_strict"]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x if len(x) >= k else x[:0]
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], axis=0), axis=0)
    return (c[k:] - c[:-k]) / k


def _yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul_conj(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """p * conj(q) for (..., 4) wxyz quaternions."""
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]
    return np.stack([pw * qw - px * qx - py * qy - pz * qz, pw * qx + px * qw + py * qz - pz * qy,
                     pw * qy - px * qz + py * qw + pz * qx, pw * qz + px * qy - py * qx + pz * qw], axis=-1)


def body_tracking_metrics(log: Dict[str, np.ndarray], anchor_index: int, strict_mpkpe_m: float,
                          fell: bool) -> Dict[str, float]:
    """MPKPE, root-relative MPKPE and drift (mjlab's definitions). Empty if not logged."""
    if "body_pos" not in log or not len(log["body_pos"]):
        return {}
    body, ref = log["body_pos"], log["ref_body_pos"]          # (T, bodies, 3)
    a = anchor_index
    mpkpe = np.linalg.norm(ref - body, axis=-1).mean(axis=1)
    drift = np.linalg.norm(ref[:, a, :2] - body[:, a, :2], axis=-1)
    # As mjlab's body_pos_relative_w: turn by yaw(q_robot * q_ref^-1), move onto the robot anchor in xy.
    turn = _yaw(_quat_mul_conj(log["anchor_quat_start"], log["ref_anchor_quat"]))
    c, s_ = np.cos(turn)[:, None], np.sin(turn)[:, None]
    rel = ref - ref[:, a:a + 1]
    rel_turned = np.stack([c * rel[..., 0] - s_ * rel[..., 1], s_ * rel[..., 0] + c * rel[..., 1], rel[..., 2]], axis=-1)
    anchor = np.concatenate([log["anchor_pos_start"][:, :2], ref[:, a, 2:]], axis=-1)[:, None]
    r_mpkpe = np.linalg.norm(rel_turned + anchor - body, axis=-1).mean(axis=1)
    return {
        "success_strict": float((not fell) and mpkpe.max() <= strict_mpkpe_m),
        "mpkpe_m": float(mpkpe.mean()), "max_mpkpe_m": float(mpkpe.max()),
        "r_mpkpe_m": float(r_mpkpe.mean()), "max_r_mpkpe_m": float(r_mpkpe.max()),
        "drift_mean_m": float(drift.mean()), "drift_max_m": float(drift.max()),
    }


def _joint_tracking(log: Dict[str, np.ndarray]):
    """(RMSE, MAE) of q against q_ref over masked joints; NaN without a reference."""
    if "q_ref" in log and log["q_ref"].size:
        m = log["q_ref_mask"].astype(bool)
        err = (log["q"] - log["q_ref"])[m]
        return _rms(err), float(np.mean(np.abs(err)))
    return float("nan"), float("nan")


def compute_metrics(log: Dict[str, np.ndarray], *, control_dt: float, duration_s: float,
                    warmup_s: float, window_s: float, q_lower: np.ndarray, q_upper: np.ndarray,
                    soft_fraction: float, fall_reason: str, fall_time: Optional[float],
                    phys: Dict[str, float], anchor_index: int = 7,
                    strict_mpkpe_m: float = 0.15, split: Optional[Dict] = None) -> TrialMetrics:
    """With `split` (run kept going after a tracking failure), fall_reason/fall_time are the first failure of either kind."""
    m = _compute_metrics(log, control_dt=control_dt, duration_s=duration_s, warmup_s=warmup_s, window_s=window_s,
                         q_lower=q_lower, q_upper=q_upper, soft_fraction=soft_fraction, fall_reason=fall_reason,
                         fall_time=fall_time, phys=phys, anchor_index=anchor_index, strict_mpkpe_m=strict_mpkpe_m)
    if split is None:
        return m
    nan = float("nan")
    m.tracking_failure = split["tracking_failure_reason"] != ""
    m.tracking_failure_reason = split["tracking_failure_reason"]
    m.tracking_failure_time_s = split["tracking_failure_time_s"] if m.tracking_failure else nan
    m.fell = split["fell_reason"] != ""
    m.fell_reason = split["fell_reason"]
    m.fell_time_s = split["fell_time_s"] if m.fell else nan
    n = len(log.get("t", []))
    if n:
        if m.fall and fall_time is not None:
            m.survival_fraction = float(min(1.0, fall_time / duration_s))
        pre = {k: v[:split["tracking_failure_steps"] or n] for k, v in log.items()}
        body = body_tracking_metrics(pre, anchor_index, strict_mpkpe_m, m.fall)
        m.pre_failure_mpkpe_m = body.get("mpkpe_m", nan)
        m.pre_failure_r_mpkpe_m = body.get("r_mpkpe_m", nan)
        m.pre_failure_joint_pos_tracking_mae = _joint_tracking(pre)[1]
    else:
        m.pre_failure_mpkpe_m = m.pre_failure_r_mpkpe_m = m.pre_failure_joint_pos_tracking_mae = nan
    return m


def _compute_metrics(log: Dict[str, np.ndarray], *, control_dt: float, duration_s: float,
                     warmup_s: float, window_s: float, q_lower: np.ndarray, q_upper: np.ndarray,
                     soft_fraction: float, fall_reason: str, fall_time: Optional[float],
                     phys: Dict[str, float], anchor_index: int, strict_mpkpe_m: float) -> TrialMetrics:
    nan = float("nan")
    if len(log.get("t", [])) == 0:
        # Failed before the first control step completed (e.g. NaN action at t=0).
        return TrialMetrics(fall=True, fall_reason=fall_reason or "no_data", time_to_fall_s=0.0,
                            survival_fraction=0.0, lin_vel_xy_rmse=nan, yaw_rate_rmse=nan,
                            lin_vel_xy_rmse_instant=nan, yaw_rate_rmse_instant=nan, mean_forward_vel=nan,
                            mean_lateral_vel=nan, joint_pos_tracking_rmse=nan, action_rate_rms=nan,
                            joint_acc_rms=nan, joint_jerk_rms=nan, mean_mech_power_w=nan, torque_rms=nan,
                            joint_limit_violation_rate=nan, joint_limit_violation_count=0,
                            near_joint_limit_rate=nan, torque_saturation_rate=nan, torque_saturation_count=0,
                            episode_length_steps=0, sim_time_s=0.0, mean_policy_latency_ms=nan)
    t = log["t"]
    n = len(t)

    fell = fall_reason != ""
    sim_time = float(phys["sim_time"])
    survival = min(1.0, sim_time / duration_s) if fell else 1.0

    # --- velocity tracking (heading frame), only after warm-up; not for motion trackers ---
    mask = (t >= warmup_s) & ("body_pos" not in log)
    lin_rmse = yaw_rmse = lin_i = yaw_i = mean_fwd = mean_lat = nan
    if mask.any():
        cmd, v, w = log["cmd"][mask], log["vel_heading"][mask, :2], log["yaw_rate"][mask]
        lin_i = float(np.sqrt(np.mean(np.sum((cmd[:, :2] - v) ** 2, axis=1))))
        yaw_i = _rms(cmd[:, 2] - w)
        mean_fwd, mean_lat = float(np.mean(v[:, 0])), float(np.mean(v[:, 1]))
        k = max(1, int(round(window_s / control_dt)))
        v_avg, w_avg, c_avg = _moving_average(v, k), _moving_average(w, k), _moving_average(cmd, k)
        if len(v_avg):
            lin_rmse = float(np.sqrt(np.mean(np.sum((c_avg[:, :2] - v_avg) ** 2, axis=1))))
            yaw_rmse = _rms(c_avg[:, 2] - w_avg)

    # --- motion tracking (optional) ---
    track, track_mae = _joint_tracking(log)

    # --- smoothness (control rate) ---
    q_tgt, dq = log["q_target"], log["dq"]
    action_rate = _rms(np.diff(q_tgt, axis=0) / control_dt) if n > 1 else float("nan")
    acc = np.diff(dq, axis=0) / control_dt if n > 1 else np.zeros((0, dq.shape[1]))
    jerk = np.diff(acc, axis=0) / control_dt if n > 2 else np.zeros((0, dq.shape[1]))
    acc_rms, jerk_rms = _rms(acc), _rms(jerk)

    # --- joint limits ---
    q = log["q"]
    tol = 1e-3  # MuJoCo limits are soft constraints; ignore sub-millirad numerical overshoot
    outside = (q < q_lower - tol) | (q > q_upper + tol)
    span = q_upper - q_lower
    band = 0.5 * (1.0 - soft_fraction) * span
    near = (q < q_lower + band) | (q > q_upper - band)

    phys_steps = max(1, int(phys["steps"]))
    return TrialMetrics(
        fall=fell,
        fall_reason=fall_reason,
        time_to_fall_s=float(fall_time) if fell and fall_time is not None else float("nan"),
        survival_fraction=float(survival),
        lin_vel_xy_rmse=lin_rmse,
        yaw_rate_rmse=yaw_rmse,
        lin_vel_xy_rmse_instant=lin_i,
        yaw_rate_rmse_instant=yaw_i,
        mean_forward_vel=mean_fwd,
        mean_lateral_vel=mean_lat,
        joint_pos_tracking_rmse=track,
        joint_pos_tracking_mae=track_mae,
        action_rate_rms=action_rate,
        joint_acc_rms=acc_rms,
        joint_jerk_rms=jerk_rms,
        mean_mech_power_w=float(phys["power_sum"] / phys_steps),
        torque_rms=float(np.sqrt(phys["tau_sq_sum"] / (phys_steps * q.shape[1]))),
        joint_limit_violation_rate=float(np.mean(outside.any(axis=1))),
        joint_limit_violation_count=int(outside.sum()),
        near_joint_limit_rate=float(np.mean(near)),
        torque_saturation_rate=float(phys["sat_steps"] / phys_steps),
        torque_saturation_count=int(phys["sat_joint_steps"]),
        episode_length_steps=int(n),
        sim_time_s=sim_time,
        mean_policy_latency_ms=float(np.mean(log["latency_ms"])) if n else float("nan"),
        **body_tracking_metrics(log, anchor_index, strict_mpkpe_m, fell),
    )


def aggregate(trials: Dict[str, Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    """Mean / std / min / max across seeds for every numeric metric (NaN-aware)."""
    keys = [k for k, v in next(iter(trials.values())).items() if isinstance(v, (int, float, bool)) or v is None]
    out = {}
    for k in keys:
        vals = np.array([np.nan if tr[k] is None else float(tr[k]) for tr in trials.values()], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std()) if finite.size else None,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "n": int(finite.size),
        }
    return out
