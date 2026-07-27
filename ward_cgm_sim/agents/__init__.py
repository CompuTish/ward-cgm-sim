"""Baseline agents. Stdlib only - these ship to the browser build."""

from .random_agent import RandomAgent
from .rule_based import RuleBasedAgent

__all__ = ["RandomAgent", "RuleBasedAgent"]
