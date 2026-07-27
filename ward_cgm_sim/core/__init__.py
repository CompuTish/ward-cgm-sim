"""Core simulation package.

Pure Python standard library only. Everything here is safe to ship into the
WebAssembly (pygbag) browser build - no numpy, no gymnasium, no native
extensions. ``ward_cgm_sim.env`` deliberately lives outside this package for
exactly that reason.
"""

from .actions import ACTION_LABELS, Action, N_ACTIONS
from .alarms import Alarm, AlarmKind
from .engine import WardEngine
from .observations import observation_size
from .patient import DiabetesType, DischargeStage, EnrolmentStatus, Location, Specialty
from .ward_map import WardMap

__all__ = [
    "ACTION_LABELS",
    "Action",
    "Alarm",
    "AlarmKind",
    "DiabetesType",
    "DischargeStage",
    "EnrolmentStatus",
    "Location",
    "N_ACTIONS",
    "Specialty",
    "WardEngine",
    "WardMap",
    "observation_size",
]
