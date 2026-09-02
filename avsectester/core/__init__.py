"""AVSecTester core: the minimal security-testing contract.

Six interfaces carry the whole framework:
  Frame        — one time-step of AV data (dataset & simulation share it)
  Environment  — sequential source of frames (reset/step); the dataset⇄sim bridge
  System       — the AV pipeline under test; fires attacks/defenses at its seams
  Attack       — offline prepare(data)→artifact + runtime apply(payload, ctx)   (Defense = runtime half)
  Metric       — score a clean vs attacked run
  Seam/Context — where an attack acts, and the per-call state it receives
"""

from .attack import Attack, Defense
from .context import Context
from .environment import Environment
from .frame import Frame
from .metric import Metric
from .runner import Result, run, run_experiment
from .seam import Seam
from .system import Outcome, System
from .threat_model import ThreatModel
from .trace import Trace

__all__ = [
    "Attack",
    "Context",
    "Defense",
    "Environment",
    "Frame",
    "Metric",
    "Outcome",
    "Result",
    "Seam",
    "System",
    "ThreatModel",
    "Trace",
    "run",
    "run_experiment",
]
