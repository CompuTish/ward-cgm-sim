"""Pygame rendering. Stdlib + pygame only (ships to the browser build).

``WardRenderer`` is imported lazily by callers so that headless use of the
engine never needs a display.
"""

__all__ = ["WardRenderer", "SpriteSheet"]


def __getattr__(name):
    if name == "WardRenderer":
        from .pygame_renderer import WardRenderer

        return WardRenderer
    if name == "SpriteSheet":
        from .sprites import SpriteSheet

        return SpriteSheet
    raise AttributeError(name)
