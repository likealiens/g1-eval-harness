# Status and open items

## Versions

- **v1.0.0** (`9916b20`): walking policies; the deliverables marked 1.0.0 below.
- **v2.0.0**: motion-tracking policies and a second robot model.
  - Motion trackers: episodes as long as the motion, failure judged against the reference
    (BeyondMimic's rules), and MPKPE, root-relative MPKPE, drift and strict success. The episode runs
    on after a tracking failure, so real falls (head down) are reported apart;
    `--stop-at-tracking-failure` ends it at the failure instead.
  - Adapters for BeyondMimic (mjlab ONNX export), SONIC v1.1 (NVIDIA's release) and SONIC (ViT-H)
    (ViT-H's fine-tune, run by its `g1_tracker`), each checked against the policy's own
    sim-to-sim (README §5).
  - Two robot models, default `ros2`. `--robot unitree_rl_gym` gives v1.0.0 results exactly.

## Deliverables

| deliverable | where | version | status |
|---|---|---|---|
| Evaluate a policy with one command | `evaluate.py` | 1.0.0 | done, verified on Windows 10 (PyTorch) and Linux (numpy backend) |
| Metric definitions (falls, tracking error, velocity tracking, smoothness, limit violations) | `METRICS.md` | 1.0.0 | done |
| Working example adapter | `adapters/unitree_rl_gym_adapter.py` (unitree_rl_gym pretrained G1) | 1.0.0 | done, walks in all non-ice scenarios |
| README: how to add your own policy | `README.md` §3 | 1.0.0, 2.0.0 | done (in-process, socket, DDS; motion trackers in §3d) |
| Direct call **and** externally running policies, same metrics | `adapters/external/` | 1.0.0 | socket path done and tested; DDS path written, untested (below) |
| Design based on existing sim2sim setups | `DESIGN.md` | 1.0.0 | done |
| Tests | `tests/` (`python -m pytest -q tests`) | 1.0.0, 2.0.0 | 32/32 pass (15 from v1.0.0) |
| Motion-tracking policies | `harness/`, `METRICS.md` §3, `README.md` §3d | 2.0.0 | done; walking-policy results identical to v1.0.0 |
| BeyondMimic adapter | `adapters/beyondmimic_adapter.py` | 2.0.0 | done, checked against mjlab and the ROS 2 stack on the LAFAN1 demo dance |
| SONIC v1.1 adapter (encoder + decoder, mode `g1`) | `adapters/sonic_v1_1_adapter.py` | 2.0.0 | done, checked against NVIDIA's sim-to-sim on a tennis serve, a jump kick and their squat |
| SONIC (ViT-H) adapter (1570-D deploy policy) | `adapters/sonic_vith_adapter.py` | 2.0.0 | done, checked against `g1_tracker` on the LAFAN1 demo dance and a tennis serve |
| Robot-model choice, default `ros2` | `assets/g1/build_ros2_model.py`, `--robot` | 2.0.0 | done; `--robot unitree_rl_gym` reproduces v1.0.0 |

Reference results: `examples/sample_results/` (summary, per-trial logs, videos; v1 robot model).
What was wrong with the first version and what changed: `FIXES.md`.

## Open items

1. **DDS bridge not yet tested with real stacks.** `adapters/external/unitree_dds_bridge.py`
   (for SONIC's C++ deploy and BeyondMimic's ROS 2 controller) follows the unitree_sdk2py API and
   the unitree_mujoco pattern, but needs one session with each policy owner to confirm network
   interface, domain id, `mode_machine` and the stack's start-up handshake. Since v2.0.0 both
   policies can be evaluated in-process instead.
2. **IsaacLab joint order.** `utils/joint_orders.py` lists the order IsaacLab-trained policies are
   expected to use. Since v2.0.0: an mjlab BeyondMimic export stores `joint_names` in its ONNX
   metadata, in the canonical MuJoCo order, so BeyondMimic needs no assumed order. The SONIC v1.1
   adapter uses `ISAACLAB_G1_29DOF_ORDER` and builds the same network inputs as NVIDIA's deploy
   (README §5). SONIC (ViT-H) reuses `g1_tracker`'s own mapping. The joint-mapping script
   (`utils/joint_mapping.py`) is used unchanged.
3. **PD gains - decision needed.** Default: each policy sends the gains it was trained with
   (as on the real robot). `--gains harness` forces one shared gain set instead. Rationale in
   `DESIGN.md` §4. Both are supported; only the default needs agreeing on. The v2.0.0 adapters
   use the default (their own gains).
4. **Comparison protocol.** For reported comparisons: same machine and environment for all
   policies, same robot model, 10 seeds (`--seeds 0 1 2 3 4 5 6 7 8 9`). Trials on the edge of
   falling can flip across machines/backends (see README, Reproducibility).
5. **Default robot model - open decision.** v2.0.0 defaults to `ros2`; `--robot unitree_rl_gym`
   keeps the v1.0.0 model and results. Results change with the model (`DESIGN.md` §5), so the
   default needs agreeing on. `examples/sample_results/` are v1-model numbers.
6. **Motion trackers and pushes.** Pushes start at about 4 s, so motions shorter than about 4.3 s
   are never pushed. `stand_still` is the same as `flat_nominal` for motion trackers (they have no
   velocity command).
7. **Privileged state in the tracking adapters.** BeyondMimic (anchor position, base linear
   velocity) and SONIC (ViT-H) (base position) read the true base state; on the robot it comes from
   a state estimator. Both adapters set `uses_privileged_state = True`; SONIC v1.1 doesn't need it.
   An estimator-noise scenario would make this closer to the robot.
8. **Starting state of motion trackers.** The robot starts at rest on the reference's first frame
   (plus the seeded joint noise). mjlab and `g1_tracker` also start it with the reference
   velocities, which matters for motions that begin while moving.

## Possible extensions

Terrain scenarios (slopes, steps, rough ground), observation latency/noise scenarios,
joint-velocity limit checks, a leaderboard script that merges several `summary.json` files.
