"""Uniform random baseline.

Useful as a floor: any policy worth reporting should beat a nurse who presses
buttons at random. Stdlib only - this ships to the browser build.
"""

import random

from ..core.actions import N_ACTIONS


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def act(self, engine) -> int:
        return self.rng.randrange(N_ACTIONS)
