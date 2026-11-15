"""
evaluate.py - evaluate one G1 policy with one command.

  python evaluate.py --adapter adapters.unitree_rl_gym_adapter:UnitreeRLGymAdapter \
                     --checkpoint checkpoints/unitree_rl_gym_g1_motion.pt

See README.md for all options.
"""
import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
import time

# Headless rendering on Linux servers / WSL. Windows and macOS use their native OpenGL,
# where MUJOCO_GL=egl is invalid. Override by setting MUJOCO_GL yourself.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from harness.config import (DEFAULT_ROBOT, HARNESS_VERSION, REAL_FALL_HEAD_HEIGHT_M, REAL_FALL_TIME_S,  # noqa: E402
                            ROBOT_MODELS, HarnessConfig)
from harness.engine import SimulationEngine, is_motion_tracker  # noqa: E402
from harness.metrics import aggregate  # noqa: E402
from harness.reporting import save_trial_log, save_video, write_summary, markdown_table  # noqa: E402
from harness.scenarios import TEST_SCENARIOS, DEFAULT_SCENARIOS  # noqa: E402


def load_adapter_class(spec: str):
    """'package.module:Class', 'package.module.Class' or 'path/to/file.py:Class'."""
    if ":" in spec:
        mod_part, cls = spec.rsplit(":", 1)
    else:
        mod_part, cls = spec.rsplit(".", 1)
    if mod_part.endswith(".py"):
        path = os.path.abspath(mod_part)
        s = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        module = importlib.util.module_from_spec(s)
        s.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)
    return getattr(module, cls)


def parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.startswith("[") or v.startswith("{"):
        return json.loads(v)
    return v


def load_adapter_config(path, overrides):
    cfg = {}
    if path:
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                cfg = yaml.safe_load(f) or {}
            else:
                cfg = json.load(f)
    for item in overrides or []:
        k, v = item.split("=", 1)
        cfg[k] = parse_value(v)
    return cfg


def sha256(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description="Standardized MuJoCo evaluation harness for Unitree G1 policies")
    ap.add_argument("--adapter", required=True, help="module:Class or file.py:Class")
    ap.add_argument("--checkpoint", default=None, help="policy weights (.pt/.onnx/...), if the adapter needs one")
    ap.add_argument("--adapter-config", default=None, help="YAML/JSON file passed to the adapter as `config`")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="adapter config override (repeatable)")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help=f"default: all ({', '.join(DEFAULT_SCENARIOS)}); motion trackers skip stand_still")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cmd-vel", nargs=3, type=float, default=[0.5, 0.0, 0.0], metavar=("VX", "VY", "YAW_RATE"))
    ap.add_argument("--robot", choices=sorted(ROBOT_MODELS), default=DEFAULT_ROBOT,
                    help="robot model: ros2 (default) or unitree_rl_gym (v1 model)")
    ap.add_argument("--gains", choices=["policy", "harness"], default="policy",
                    help="policy: each policy sends its own kp/kd (like on hardware). harness: force G1 reference gains.")
    ap.add_argument("--output-dir", default=None, help="default: results/<adapter>_<timestamp>")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-all-seeds", action="store_true")
    ap.add_argument("--list-scenarios", action="store_true")
    ap.add_argument("--stop-at-tracking-failure", action="store_true",
                    help="motion trackers: end the episode at the first tracking failure (BeyondMimic's termination). "
                         "Default: keep running until the motion ends or the robot really falls (head below 0.5 m "
                         "for over 2 s, or still down when the motion ends); see METRICS.md")
    args = ap.parse_args()

    if args.list_scenarios:
        for s in TEST_SCENARIOS.values():
            print(f"{s.name:24s} {s.description}")
        return 0

    unknown = [s for s in args.scenarios or [] if s not in TEST_SCENARIOS]
    if unknown:
        ap.error(f"Unknown scenarios {unknown}. Use --list-scenarios.")

    cfg = HarnessConfig(gains_mode=args.gains, model_xml=ROBOT_MODELS[args.robot])
    adapter_cls = load_adapter_class(args.adapter)
    adapter = adapter_cls(checkpoint_path=args.checkpoint,
                          config=load_adapter_config(args.adapter_config, args.set))
    name = getattr(adapter, "name", adapter_cls.__name__)
    tracker = is_motion_tracker(adapter.reference(0.0))
    skipped = args.scenarios is None and tracker
    if args.scenarios is None:
        args.scenarios = [s for s in DEFAULT_SCENARIOS if not (tracker and s == "stand_still")]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir or os.path.join("results", f"{name}_{stamp}")
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    if not args.no_video:
        os.makedirs(os.path.join(out, "videos"), exist_ok=True)

    engine = SimulationEngine(cfg)
    command = np.array(args.cmd_vel, dtype=np.float64)

    print(f"\n[HARNESS {HARNESS_VERSION}] policy={name}  scenarios={len(args.scenarios)}  seeds={args.seeds}")
    cmd = "n/a (motion tracker)" if tracker else command.tolist()
    print(f"  physics {1/cfg.physics_dt:.0f} Hz | control {1/cfg.control_dt:.0f} Hz | gains: {cfg.gains_mode} | cmd {cmd}")
    print(f"  robot model: {args.robot} ({os.path.basename(cfg.model_xml)})")
    print("  policy type: " + ("motion tracker (reference(t) has a root pose)" if tracker
                               else "walking (no root pose in reference(t))"))
    if skipped:
        print("  stand_still skipped: a motion tracker ignores the speed command (ask for it with --scenarios)")
    print(f"  output -> {out}\n")

    trials, aggregates, perturbations = {}, {}, {}
    t_start = time.time()
    try:
        for scen_name in args.scenarios:
            scen = TEST_SCENARIOS[scen_name]
            trials[scen_name], perturbations[scen_name] = {}, {}
            for i, seed in enumerate(args.seeds):
                record = (not args.no_video) and (i == 0 or args.video_all_seeds)
                res = engine.run_trial(adapter, scen, seed, command, record_video=record,
                                       stop_at_tracking_failure=args.stop_at_tracking_failure)
                m = res.metrics
                trials[scen_name][f"seed_{seed}"] = m.to_dict()
                perturbations[scen_name][f"seed_{seed}"] = res.perturbations
                save_trial_log(res.log, os.path.join(out, "logs", f"{scen_name}_seed{seed}.npz"))
                status = f"FELL ({m.fall_reason} @ {m.time_to_fall_s:.2f}s)" if m.fall else "ok"
                if m.fell is not None:
                    status = (f"track-fail ({m.tracking_failure_reason} @ {m.tracking_failure_time_s:.2f}s)"
                              if m.tracking_failure else "tracking ok") + ", " + \
                             (f"FELL ({m.fell_reason} @ {m.fell_time_s:.2f}s)" if m.fell else "no fall")
                if np.isfinite(m.mpkpe_m):  # motion tracker
                    score = f"MPKPE {m.mpkpe_m:6.3f}"
                else:
                    score = "v_xy rmse " + ("   n/a" if not np.isfinite(m.lin_vel_xy_rmse) else f"{m.lin_vel_xy_rmse:6.3f}")
                print(f"  {scen_name:22s} seed {seed:<4d} {status:36s} surv {m.survival_fraction:4.2f} | "
                      f"{score} | act-rate {m.action_rate_rms:6.2f} | power {m.mean_mech_power_w:6.0f} W")
                if record and res.frames:
                    vp = os.path.join(out, "videos", f"{scen_name}_seed{seed}.mp4")
                    if save_video(res.frames, vp, fps=int(round(1 / (cfg.control_dt * cfg.video_stride)))):
                        print(f"      video -> {vp}")
            aggregates[scen_name] = aggregate(trials[scen_name])
    finally:
        adapter.close()

    summary = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "harness": cfg.to_dict(),
        "robot": {"model": args.robot, **engine.robot.describe()},
        "adapter": {"name": name, "policy_type": "motion_tracker" if tracker else "walking", "class": f"{adapter_cls.__module__}.{adapter_cls.__name__}",
                    "checkpoint": args.checkpoint, "checkpoint_sha256": sha256(args.checkpoint),
                    "config": adapter.config, "policy_dt": adapter.policy_dt,
                    "command_rate": adapter.command_rate,
                    "uses_privileged_state": adapter.uses_privileged_state,
                    "describe": adapter.describe()},
        "scenarios": {s: TEST_SCENARIOS[s].to_dict() for s in args.scenarios},
        "seeds": args.seeds,
        "command": command.tolist(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "mujoco": __import__("mujoco").__version__, "numpy": np.__version__},
        "aggregates": aggregates,
        "trials": trials,
        "perturbations": perturbations,
        "wall_time_s": round(time.time() - t_start, 1),
    }
    if tracker and not args.stop_at_tracking_failure:
        summary["harness"].update(keep_running_after_tracking_failure=True,
                                  real_fall_head_height_m=REAL_FALL_HEAD_HEIGHT_M, real_fall_time_s=REAL_FALL_TIME_S)
    write_summary(out, summary)
    print("\n" + markdown_table(aggregates))
    print(f"\n[HARNESS] done in {summary['wall_time_s']} s -> {out}/summary.json, summary.md\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
