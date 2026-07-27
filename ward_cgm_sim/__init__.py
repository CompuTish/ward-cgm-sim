"""ward_cgm_sim - an academic POMDP simulator of inpatient CGM telemetry.

NOT CLINICAL DECISION SUPPORT. Every clinical pathway in this package is a
simplified, configurable, guideline-inspired placeholder for research modelling.

Import policy: this module must stay importable with the standard library plus
pygame-ce alone, because the same tree is vendored into the browser build.
``ward_cgm_sim.env`` (which subclasses ``gymnasium.Env`` and pulls in numpy) is
therefore NOT imported here - native entry points import it explicitly:

    from ward_cgm_sim.env import WardCGMTelemetryEnv
"""

from .config import SimConfig
from .core import Action, WardEngine

__version__ = "0.1.0"

__all__ = ["Action", "SimConfig", "WardEngine", "__version__"]
