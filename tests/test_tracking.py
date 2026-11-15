"""Motion-tracking support."""
import mujoco
import numpy as np
import pytest

from adapters.base import PolicyAdapter
from harness.config import HarnessConfig
from harness.engine import SimulationEngine
from harness.metrics import aggregate, body_tracking_metrics
from harness.reporting import markdown_table
from harness.scenarios import TEST_SCENARIOS
from harness.types import JointCommand
from tests.test_harness import STIFF_KD, STIFF_KP, HoldPose, short


class HoldPoseTracker(PolicyAdapter):
    """Holds the zero pose; the reference is the same pose, `lift` metres higher."""
    name = "hold_pose_tracker"

    def __init__(self, lift=0.0, duration=2.0):
        super().__init__()
        self.lift, self.duration, self.root_z = lift, duration, None

    def default_pose(self):
        return np.zeros(29)

    def reset(self, seed):
        pass

    def act(self, state):
        if self.root_z is None:
            self.root_z = state.base_pos[2]
        return JointCommand(q=np.zeros(29), kp=STIFF_KP, kd=STIFF_KD)

    def motion_duration(self):
        return self.duration

    def reference(self, t):
        # Any constant height works: the engine moves the reference to the spawn height.
        return {"q": np.zeros(29), "root_pos": np.array([0.0, 0.0, 0.8 + self.lift]),
                "root_quat": np.array([1.0, 0.0, 0.0, 0.0])}


@pytest.fixture(scope="module")
def engine():
    return SimulationEngine(HarnessConfig())


def test_reference_matching_robot_is_not_a_failure(engine):
    res = engine.run_trial(HoldPoseTracker(), TEST_SCENARIOS["flat_nominal"], 0, np.zeros(3))
    m = res.metrics
    assert not m.fall, m.fall_reason
    assert m.sim_time_s == pytest.approx(2.0, abs=0.03)       # motion length, not the scenario's 10 s
    assert m.mpkpe_m < 0.05 and m.r_mpkpe_m < 0.05
    assert m.success_strict == 1.0
    assert np.isnan(m.lin_vel_xy_rmse)                        # no velocity metrics for motion trackers


def test_reference_far_above_robot_fails_on_anchor_height(engine):
    adapter = HoldPoseTracker()
    adapter.reference = lambda t: {"q": np.zeros(29), "root_pos": np.array([0.0, 0.0, 0.8 + (0.5 if t > 0 else 0)]),
                                   "root_quat": np.array([1.0, 0.0, 0.0, 0.0])}
    res = engine.run_trial(adapter, TEST_SCENARIOS["flat_nominal"], 0, np.zeros(3))
    assert res.metrics.fall and res.metrics.fall_reason == "tracking:anchor_height"
    assert res.metrics.time_to_fall_s < 0.1


SPLIT_KEYS = {"tracking_failure", "tracking_failure_reason", "tracking_failure_time_s", "fell", "fell_reason",
              "fell_time_s", "pre_failure_mpkpe_m", "pre_failure_r_mpkpe_m", "pre_failure_joint_pos_tracking_mae"}


def rising_reference(duration=2.0):
    """Robot stands still while the reference is 0.5 m higher: tracking fails at once, the robot never falls."""
    adapter = HoldPoseTracker(duration=duration)
    adapter.reference = lambda t: {"q": np.zeros(29), "root_pos": np.array([0.0, 0.0, 0.8 + (0.5 if t > 0 else 0)]),
                                   "root_quat": np.array([1.0, 0.0, 0.0, 0.0])}
    return adapter


def without_latency(m):
    d = m.to_dict()
    d.pop("mean_policy_latency_ms")
    return d


def test_stop_at_tracking_failure_ends_the_trial(engine):
    a = engine.run_trial(rising_reference(), TEST_SCENARIOS["flat_nominal"], 0, np.zeros(3),
                         stop_at_tracking_failure=True).metrics
    da = without_latency(a)
    assert not SPLIT_KEYS & set(da)
    assert a.fall and a.fall_reason == "tracking:anchor_height" and a.sim_time_s == a.time_to_fall_s
    assert markdown_table({"flat_nominal": aggregate({"seed_0": da})}).startswith("| scenario | falls | survival |")


def test_default_keeps_running_after_tracking_failure(engine):
    flat = TEST_SCENARIOS["flat_nominal"]
    a = engine.run_trial(rising_reference(), flat, 0, np.zeros(3)).metrics
    b = engine.run_trial(rising_reference(), flat, 0, np.zeros(3), stop_at_tracking_failure=False).metrics
    assert without_latency(a) == without_latency(b) and SPLIT_KEYS <= set(a.to_dict())


def test_walking_policy_ignores_the_option(engine):
    scen = short("push_lateral_medium", 5.0)
    a = engine.run_trial(HoldPose(), scen, 7, np.zeros(3)).metrics
    b = engine.run_trial(HoldPose(), scen, 7, np.zeros(3), stop_at_tracking_failure=True).metrics
    assert without_latency(a) == without_latency(b) and not SPLIT_KEYS & set(a.to_dict())


def test_keep_running_separates_tracking_failure_from_fall(engine):
    flat = TEST_SCENARIOS["flat_nominal"]
    stopped = engine.run_trial(rising_reference(), flat, 0, np.zeros(3), stop_at_tracking_failure=True).metrics
    m = engine.run_trial(rising_reference(), flat, 0, np.zeros(3)).metrics
    assert m.tracking_failure and m.tracking_failure_reason == "tracking:anchor_height"
    assert m.tracking_failure_time_s == stopped.time_to_fall_s
    assert not m.fell and m.fell_reason == "" and np.isnan(m.fell_time_s)
    assert m.sim_time_s == pytest.approx(2.0, abs=0.03)       # ran the whole motion
    for k in ("fall", "fall_reason", "time_to_fall_s", "survival_fraction", "success_strict"):
        assert getattr(m, k) == getattr(stopped, k), k
    assert m.pre_failure_mpkpe_m == pytest.approx(stopped.mpkpe_m)
    assert m.pre_failure_r_mpkpe_m == pytest.approx(stopped.r_mpkpe_m)
    assert m.pre_failure_joint_pos_tracking_mae == pytest.approx(stopped.joint_pos_tracking_mae)
    assert SPLIT_KEYS <= set(m.to_dict())
    assert "| failed | track fail | real falls |" in markdown_table({"flat_nominal": aggregate({"seed_0": m.to_dict()})})


def limp(duration):
    adapter = HoldPoseTracker(duration=duration)
    adapter.act = lambda state: JointCommand(q=np.zeros(29), kp=np.zeros(29), kd=np.zeros(29))
    return adapter


def test_real_fall_is_head_down_for_over_two_seconds(engine):
    flat = TEST_SCENARIOS["flat_nominal"]
    res = engine.run_trial(limp(4.0), flat, 0, np.zeros(3))
    m = res.metrics
    assert m.fell and m.fell_reason == "head_down"
    assert m.sim_time_s - m.fell_time_s == pytest.approx(2.0, abs=1e-6)  # ends 2 s after the head went down
    assert res.log["head_height"][-1] < 0.5
    assert m.tracking_failure and m.tracking_failure_time_s <= m.fell_time_s
    assert m.fall_reason == m.tracking_failure_reason

    res = engine.run_trial(limp(2.0), flat, 0, np.zeros(3))
    m = res.metrics
    assert res.log["head_height"][-1] < 0.5 and m.sim_time_s == pytest.approx(2.0, abs=0.03)
    assert m.fell and m.fell_reason == "head_down"            # down for less than 2 s, but still down at the end
    assert m.sim_time_s - m.fell_time_s < 2.0


@pytest.mark.parametrize("down_from, down_to, fell", [(0.5, 1.0, False), (1.51, 9.0, True)])
def test_head_down_briefly_or_at_the_end(engine, monkeypatch, down_from, down_to, fell):
    """Head down for 0.5 s mid-motion is no fall; head down when the motion ends is one."""
    monkeypatch.setattr(engine, "_head_height", lambda d: 0.3 if down_from < d.time < down_to else 1.25)
    m = engine.run_trial(HoldPoseTracker(), TEST_SCENARIOS["flat_nominal"], 0, np.zeros(3)).metrics
    assert not m.tracking_failure and m.sim_time_s == pytest.approx(2.0, abs=0.03)
    assert m.fell == fell and m.fall == fell
    if fell:
        assert m.fell_reason == "head_down" and m.fell_time_s == pytest.approx(1.52, abs=0.011)


def test_spawn_quaternion_is_normalised(engine):
    seen = []

    class Yawed(HoldPoseTracker):
        def default_base_pose(self):  # 90 degrees about z, not unit length
            return np.array([0.3, -0.2, 0.8]), np.array([1.0, 0.0, 0.0, 1.0])

        def reference(self, t):
            return {**super().reference(t), "root_pos": np.array([0.3, -0.2, 0.8]),
                    "root_quat": np.array([1.0, 0.0, 0.0, 1.0])}

        def act(self, state):
            seen.append(state.base_quat.copy())
            return super().act(state)

    res = engine.run_trial(Yawed(duration=0.5), TEST_SCENARIOS["flat_nominal"], 0, np.zeros(3))
    assert not res.metrics.fall, res.metrics.fall_reason
    assert np.linalg.norm(seen[0]) == pytest.approx(1.0)
    assert seen[0] == pytest.approx(np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2), abs=1e-6)


def test_reference_that_stops_partway_is_a_clear_error(engine):
    adapter = HoldPoseTracker()
    full = adapter.reference
    adapter.reference = lambda t: full(t) if t < 0.5 else None
    with pytest.raises(ValueError, match="motion tracker must give them"):
        engine.run_trial(adapter, TEST_SCENARIOS["ice"], 0, np.zeros(3))
    assert np.array_equal(engine.robot.model.geom_friction, engine.robot._nominal_friction)  # ice undone
    adapter.reference = lambda t: {"root_pos": np.array([0.0, 0.0, 0.8]), "root_quat": np.array([1.0, 0.0, 0.0, 0.0])}
    with pytest.raises(ValueError, match="no joint angles"):
        engine.run_trial(adapter, TEST_SCENARIOS["flat_nominal"], 0, np.zeros(3))
    with pytest.raises(ValueError, match="positive"):
        engine.run_trial(HoldPoseTracker(duration=0.0), TEST_SCENARIOS["flat_nominal"], 0, np.zeros(3))


def test_body_metrics_zero_for_identical_robot_and_reference():
    rng = np.random.default_rng(0)
    body = rng.normal(size=(20, 14, 3))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (20, 1))
    log = {"body_pos": body, "ref_body_pos": body.copy(), "anchor_quat": quat, "ref_anchor_quat": quat,
           "anchor_pos_start": body[:, 7], "anchor_quat_start": quat}
    m = body_tracking_metrics(log, anchor_index=7, strict_mpkpe_m=0.15, fell=False)
    assert m["mpkpe_m"] == pytest.approx(0) and m["r_mpkpe_m"] == pytest.approx(0)
    assert m["drift_max_m"] == pytest.approx(0) and m["success_strict"] == 1.0


def test_body_metrics_shift_counts_in_world_error_but_not_root_relative():
    body = np.zeros((10, 14, 3))
    body[:, :, 2] = np.linspace(0, 1, 14)
    ref = body.copy()
    ref[:, :, 0] += 0.3                                        # whole reference 0.3 m ahead
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
    log = {"body_pos": body, "ref_body_pos": ref, "anchor_quat": quat, "ref_anchor_quat": quat,
           "anchor_pos_start": body[:, 7], "anchor_quat_start": quat}
    m = body_tracking_metrics(log, anchor_index=7, strict_mpkpe_m=0.15, fell=False)
    assert m["mpkpe_m"] == pytest.approx(0.3) and m["drift_mean_m"] == pytest.approx(0.3)
    assert m["r_mpkpe_m"] == pytest.approx(0, abs=1e-9)
    assert m["success_strict"] == 0.0


def _mjlab_r_mpkpe(body, ref, anchor_pos_start, anchor_quat_start, ref_anchor_quat, a=7):
    """mjlab's r-MPKPE, step by step."""
    errors = []
    for k in range(len(body)):
        delta = np.zeros(4)
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, ref_anchor_quat[k])
        mujoco.mju_mulQuat(delta, anchor_quat_start[k], inv)
        yaw = np.arctan2(2 * (delta[0] * delta[3] + delta[1] * delta[2]), 1 - 2 * (delta[2] ** 2 + delta[3] ** 2))
        yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        delta_pos = np.array([anchor_pos_start[k, 0], anchor_pos_start[k, 1], ref[k, a, 2]])
        rel = np.zeros_like(ref[k])
        for b in range(len(ref[k])):
            mujoco.mju_rotVecQuat(rel[b], ref[k, b] - ref[k, a], yaw_quat)
        errors.append(np.linalg.norm(delta_pos + rel - body[k], axis=-1).mean())
    return np.mean(errors), np.max(errors)


def test_root_relative_mpkpe_matches_mjlab():
    rng = np.random.default_rng(1)
    T = 30
    body, ref = rng.normal(size=(T, 14, 3)), rng.normal(size=(T, 14, 3))
    quats = rng.normal(size=(3, T, 4))                  # tilted anchors: yaw(q_r q_ref^-1) != yaw(q_r) - yaw(q_ref)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    log = {"body_pos": body, "ref_body_pos": ref, "anchor_quat": quats[0], "ref_anchor_quat": quats[1],
           "anchor_pos_start": rng.normal(size=(T, 3)), "anchor_quat_start": quats[2]}
    m = body_tracking_metrics(log, anchor_index=7, strict_mpkpe_m=0.15, fell=False)
    want_mean, want_max = _mjlab_r_mpkpe(body, ref, log["anchor_pos_start"], quats[2], quats[1])
    assert m["r_mpkpe_m"] == pytest.approx(want_mean) and m["max_r_mpkpe_m"] == pytest.approx(want_max)
