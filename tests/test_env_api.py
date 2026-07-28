"""Gymnasium API contract and episode-structure invariants.

``check_env`` is the contract test: if it fails, no standard RL library will
drive this environment correctly regardless of how sensible the simulation is.

Requires the [dev]/[train] extras (gymnasium, numpy).
"""

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from ward_cgm_sim.config import SimConfig  # noqa: E402
from ward_cgm_sim.core.actions import N_ACTIONS, Action  # noqa: E402
from ward_cgm_sim.env import WardCGMTelemetryEnv, make_env  # noqa: E402


def test_passes_the_gymnasium_api_checker():
    env = WardCGMTelemetryEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_action_space_matches_the_documented_action_set():
    env = WardCGMTelemetryEnv()
    assert env.action_space.n == N_ACTIONS == 24
    assert len(env.action_labels) == N_ACTIONS
    env.close()


def test_observation_is_within_the_declared_space():
    env = WardCGMTelemetryEnv()
    obs, _info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "reset observation outside its own space"

    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, _r, terminated, truncated, _i = env.step(int(rng.integers(N_ACTIONS)))
        assert env.observation_space.contains(obs), "step observation outside its own space"
        if terminated or truncated:
            break
    env.close()


def test_observation_labels_line_up_with_observation_length():
    env = WardCGMTelemetryEnv()
    obs, _ = env.reset(seed=0)
    assert len(env.observation_labels) == len(obs)
    env.close()


def test_same_seed_reproduces_the_same_episode():
    def run(seed):
        env = WardCGMTelemetryEnv()
        env.reset(seed=seed)
        rewards = []
        for action in [Action.WAIT] * 40:
            _o, r, t, tr, _i = env.step(int(action))
            rewards.append(r)
            if t or tr:
                break
        env.close()
        return rewards

    first = run(7)
    assert len(first) > 5, "positive control: episode must actually run"
    assert first == run(7)
    assert first != run(8), "different seeds should not produce identical episodes"


def test_episode_lasts_one_twelve_hour_shift():
    """144 steps of 5 minutes each, unless it terminates early for cause."""
    cfg = SimConfig()
    assert cfg.steps_per_episode == 144
    assert cfg.minutes_per_step == 5

    env = WardCGMTelemetryEnv()
    env.reset(seed=1)
    steps = 0
    while True:
        _o, _r, terminated, truncated, info = env.step(int(Action.WAIT))
        steps += 1
        if terminated or truncated:
            break
    env.close()

    assert steps <= 144
    assert info["termination_reason"] in {
        "shift_end",
        "unsafe_overcrowding",
        "serious_adverse_event",
    }


def test_info_exposes_the_headline_safety_metric():
    env = WardCGMTelemetryEnv()
    env.reset(seed=2)
    info = None
    while True:
        _o, _r, terminated, truncated, info = env.step(int(Action.WAIT))
        if terminated or truncated:
            break
    env.close()

    kpi = info["kpi"]
    # Shifts without a serious incident is the key performance metric.
    assert kpi["incident_free_shift"] in (True, False)
    for key in (
        "time_below_range_steps",
        "severe_hypo_events",
        "serious_adverse_events",
        "discharge_delay_steps",
        "max_queue_length",
        "cohort_hypo_episodes",
    ):
        assert key in kpi, f"missing KPI: {key}"
    assert info["reward_components"], "reward breakdown should be populated"


def test_stepping_a_finished_episode_raises():
    env = WardCGMTelemetryEnv()
    env.reset(seed=3)
    while True:
        _o, _r, terminated, truncated, _i = env.step(int(Action.WAIT))
        if terminated or truncated:
            break
    with pytest.raises(RuntimeError):
        env.step(int(Action.WAIT))
    env.close()


def test_ansi_render_describes_the_shift():
    env = WardCGMTelemetryEnv(render_mode="ansi")
    env.reset(seed=4)
    env.step(int(Action.WAIT))
    text = env.render()
    assert "Shift" in text and "beds" in text
    env.close()


def test_make_env_toggles_the_telemetry_counterfactual():
    on = make_env(telemetry_enabled=True)
    off = make_env(telemetry_enabled=False)
    assert on.config.telemetry_enabled
    assert not off.config.telemetry_enabled

    # With telemetry off there is no dashboard, so no alarm can ever be visible.
    off.reset(seed=5)
    for _ in range(60):
        _o, _r, t, tr, _i = off.step(int(Action.CHECK_DASHBOARD))
        assert off.engine.visible_alarms() == []
        if t or tr:
            break
    on.close()
    off.close()


# --------------------------------------------------------------------------
# Render modes and lifecycle - the public surface a user of the env touches.
# --------------------------------------------------------------------------


def test_render_returns_nothing_when_no_mode_was_asked_for():
    env = WardCGMTelemetryEnv()
    env.reset(seed=1)
    assert env.render() is None
    env.close()


def test_ansi_render_describes_the_ward_in_text():
    env = WardCGMTelemetryEnv(render_mode="ansi")
    env.reset(seed=1)
    text = env.render()
    assert isinstance(text, str) and text.strip()
    lowered = text.lower()
    for expected in ("shift", "beds", "queue", "alarms"):
        assert expected in lowered, f"the ansi view never mentions {expected}"


def test_ansi_render_lists_a_live_alarm():
    """Positive control included: an empty board would satisfy any 'no alarm'
    assertion, so drive the ward until one actually fires."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = WardCGMTelemetryEnv(render_mode="ansi")
    env.reset(seed=3)
    engine = env.engine
    for _ in range(engine.cfg.steps_per_episode - 1):
        env.step(int(Action.CHECK_DASHBOARD))
        if engine.visible_alarms():
            text = env.render()
            assert "ALARM" in text
            assert "mmol/L" in text
            return
    pytest.skip("no alarm fired in this shift")


def test_rgb_array_render_returns_a_frame_of_the_right_shape():
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = WardCGMTelemetryEnv(render_mode="rgb_array")
    env.reset(seed=1)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[0] > 100 and frame.shape[1] > 100
    assert len({tuple(p) for row in frame[::11] for p in row[::11]}) > 5, (
        "the frame came back blank"
    )
    env.close()


def test_close_is_safe_to_call_twice_and_before_rendering():
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = WardCGMTelemetryEnv(render_mode="rgb_array")
    env.close()
    env.reset(seed=1)
    env.render()
    env.close()
    env.close()


def test_reset_accepts_a_replacement_config_through_options():
    env = WardCGMTelemetryEnv()
    env.reset(seed=1, options={"config": SimConfig(telemetry_enabled=False)})
    assert env.engine.cfg.telemetry_enabled is False
    assert env.config.telemetry_enabled is False


def test_the_same_seed_gives_the_same_episode_and_a_different_one_does_not():
    def run(seed):
        env = WardCGMTelemetryEnv()
        obs, _ = env.reset(seed=seed)
        rewards = []
        for action in range(12):
            _o, reward, _t, _tr, _i = env.step(action % 24)
            rewards.append(reward)
        return list(obs), rewards

    first = run(11)
    assert run(11) == first, "the same seed must replay identically"
    assert run(12) != first, "different seeds must diverge"
