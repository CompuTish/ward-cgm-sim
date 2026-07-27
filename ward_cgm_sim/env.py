"""Gymnasium-compatible wrapper around :class:`~ward_cgm_sim.core.engine.WardEngine`.

This module imports gymnasium and numpy and therefore must NEVER be vendored
into the browser build - see ``ward_cgm_sim/__init__.py`` for the import policy
and ``tests/test_web_bundle.py`` for the test that enforces it.

Formally the environment is a POMDP (S, A, T, R, Omega, O, gamma):

    S       full ward state: every patient's latent clinical state, sensor
            state, staff availability, queue and clock (``WardEngine``)
    A       Discrete(24), see ``core.actions.Action``
    T       stochastic transition implemented by ``WardEngine.step``
    R       ``core.rewards.RewardTracker``, weights in ``config.RewardConfig``
    Omega   the observation set produced by ``core.observations``
    O       the observation function: what is visible from where the agent is
            standing, plus what it has previously learned and recorded
    gamma   1.0 - finite horizon of 144 steps (one 12-hour shift)

ACADEMIC MODEL ONLY - not clinical decision support.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import SimConfig
from .core.actions import ACTION_LABELS, Action, N_ACTIONS
from .core.engine import WardEngine
from .core.observations import observation_labels, observation_size


class WardCGMTelemetryEnv(gym.Env):
    """A 12-hour ward shift managing CGM telemetry as routine care.

    The agent is a ward nurse or shift coordinator. There is no separate
    research team in this model: enrolment, alarm response, escalation, sensor
    troubleshooting and discharge work all compete for the same person's time,
    which is precisely the workflow question the simulator exists to probe.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        config: SimConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config or SimConfig()
        self.render_mode = render_mode
        self.engine = WardEngine(self.config)

        n_obs = observation_size(self.config.ward.n_beds)
        # -1 is the explicit "unknown" sentinel; everything else is normalised
        # into [0, 1.5] (queue fraction can exceed 1 before termination).
        self.observation_space = spaces.Box(
            low=-1.0, high=1.5, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_labels = observation_labels(self.config.ward.n_beds)
        self.action_labels = [ACTION_LABELS[Action(i)] for i in range(N_ACTIONS)]

        self._renderer = None

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options and "config" in options:
            self.config = options["config"]
            self.engine.cfg = self.config
        # Derive the engine's seed from gymnasium's seeded RNG so that
        # reset(seed=n) is reproducible and reset() alone still varies.
        engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        obs = self.engine.reset(engine_seed)
        info = self.engine.info()
        if self.render_mode == "human":
            self.render()
        return np.asarray(obs, dtype=np.float32), info

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.engine.step(int(action))
        if self.render_mode == "human":
            self.render()
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()

        from .render.pygame_renderer import WardRenderer

        if self._renderer is None:
            self._renderer = WardRenderer(
                self.engine,
                headless=(self.render_mode == "rgb_array"),
            )
        self._renderer.draw()
        if self.render_mode == "rgb_array":
            return self._renderer.to_rgb_array()
        self._renderer.flip()
        return None

    def _render_ansi(self) -> str:
        engine = self.engine
        minutes = engine.step_index * self.config.minutes_per_step
        lines = [
            f"Shift {minutes // 60:02d}:{minutes % 60:02d}  "
            f"beds {engine.flow.occupied_beds}/{engine.flow.n_beds}  "
            f"queue {engine.flow.queue_length}  "
            f"alarms {len(engine.visible_alarms())}",
            f"last: {engine.last_action_result}",
        ]
        for alarm in engine.visible_alarms():
            lines.append(
                f"  ALARM bed {alarm.bed:2d} {alarm.kind.value:12s} "
                f"{alarm.cgm_value:5.1f} mmol/L  age {alarm.age(engine.step_index)}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(telemetry_enabled: bool = True, **kwargs) -> WardCGMTelemetryEnv:
    """Convenience constructor for the CGM-on vs CGM-off comparison.

    ``telemetry_enabled=False`` removes the dashboard and all alarms, so the
    agent can only find a deteriorating patient by physically checking them.
    Running matched seeds with the flag on and off is the core experiment: see
    ``scripts/evaluate.py``.
    """
    config = kwargs.pop("config", None) or SimConfig()
    config.telemetry_enabled = telemetry_enabled
    return WardCGMTelemetryEnv(config=config, **kwargs)
