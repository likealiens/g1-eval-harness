"""
harness/reporting.py

Writes everything a run produces:
  summary.json          config snapshot + every trial + per-scenario aggregates
  summary.md            human-readable table (paste into reports)
  logs/<scen>_seed<k>.npz   per-control-step time series of each trial
  videos/<scen>_seed<k>.mp4 one video per scenario (first seed)
"""
import json
import os
from typing import Dict, List

import numpy as np

TABLE_COLUMNS = [
    ("fall", "falls", "{:.0%}"),
    ("survival_fraction", "survival", "{:.2f}"),
    ("lin_vel_xy_rmse", "v_xy RMSE m/s", "{:.3f}"),
    ("yaw_rate_rmse", "yaw RMSE rad/s", "{:.3f}"),
    ("joint_pos_tracking_rmse", "track RMSE rad", "{:.3f}"),
    ("success_strict", "strict success", "{:.0%}"),
    ("mpkpe_m", "MPKPE m", "{:.3f}"),
    ("r_mpkpe_m", "r-MPKPE m", "{:.3f}"),
    ("action_rate_rms", "action rate rad/s", "{:.2f}"),
    ("joint_jerk_rms", "jerk rad/s^3", "{:.0f}"),
    ("mean_mech_power_w", "power W", "{:.0f}"),
    ("joint_limit_violation_rate", "q-limit viol", "{:.1%}"),
    ("torque_saturation_rate", "tau sat", "{:.1%}"),
]
# Motion trackers that keep running after a tracking failure (the default): `fall` is the first failure of either kind.
SPLIT_COLUMNS = [
    ("fall", "failed", "{:.0%}"),
    ("tracking_failure", "track fail", "{:.0%}"),
    ("fell", "real falls", "{:.0%}"),
] + TABLE_COLUMNS[1:7] + [("pre_failure_mpkpe_m", "MPKPE pre-fail m", "{:.3f}")] + TABLE_COLUMNS[7:]


def save_video(frames: List[np.ndarray], path: str, fps: int) -> bool:
    if not frames:
        return False
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8) as w:
            for f in frames:
                w.append_data(f)
        return True
    except Exception:
        pass
    try:
        import cv2
        h, w_ = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return True
    except Exception as e:
        print(f"    [VIDEO] Could not write video: {e}")
        return False


def save_trial_log(log: Dict[str, np.ndarray], path: str):
    np.savez_compressed(path, **log)


def _fmt(v, fmt):
    if v is None:
        return "n/a"
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def markdown_table(aggregates: Dict[str, Dict]) -> str:
    columns = SPLIT_COLUMNS if any("fell" in agg for agg in aggregates.values()) else TABLE_COLUMNS
    head = "| scenario | " + " | ".join(c[1] for c in columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    lines = [head, sep]
    for scen, agg in aggregates.items():
        cells = []
        for key, _, fmt in columns:
            a = agg.get(key)
            if a is None or a["mean"] is None:
                cells.append("n/a")
            elif key in ("fall", "tracking_failure", "fell"):
                cells.append(_fmt(a["mean"], fmt))
            else:
                cells.append(_fmt(a["mean"], fmt) + (f" ± {_fmt(a['std'], fmt)}" if a["n"] > 1 else ""))
        lines.append(f"| {scen} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(out_dir: str, summary: Dict):
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = [f"# Evaluation: {summary['adapter']['name']}", "",
          f"- checkpoint: `{summary['adapter'].get('checkpoint')}` (sha256 `{summary['adapter'].get('checkpoint_sha256', 'n/a')}`)",
          f"- harness {summary['harness']['harness_version']}, physics {1 / summary['harness']['physics_dt']:.0f} Hz, "
          f"control {1 / summary['harness']['control_dt']:.0f} Hz, gains from **{summary['harness']['gains_mode']}**",
          f"- seeds: {summary['seeds']}, command (vx, vy, yaw_rate): "
          f"{'n/a (motion tracker)' if summary['adapter'].get('policy_type') == 'motion_tracker' else summary['command']}", "",
          "Mean ± std over seeds. Velocity metrics exclude the first "
          f"{summary['harness']['tracking_warmup_s']} s and are n/a if the robot fell before that.", ""]
    if summary["harness"].get("keep_running_after_tracking_failure"):
        md += ["Motion trackers kept running after a tracking failure. `failed` = tracking failure or fall, "
               "`real falls` = head below 0.5 m for over 2 s or at the end; tracking metrics cover the whole run, "
               "`pre-fail` only up to the first tracking failure.", ""]
    md += [markdown_table(summary["aggregates"]), ""]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))
