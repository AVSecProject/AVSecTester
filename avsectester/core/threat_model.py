"""Threat model — a first-class, validated entity (PROJECT.md 7.3, 4.1).

The threat model is what makes an attack *security-relevant* rather than generic noise.
Attacks declare the threat model they assume; the attack engine refuses to run when an
experiment would violate a declared constraint (assumption preservation, PROJECT.md 9.3).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Knowledge(str, Enum):
    """Attacker knowledge of the target system."""

    WHITEBOX = "whitebox"
    GRAYBOX = "graybox"
    BLACKBOX = "blackbox"


class AccessLevel(str, Enum):
    """Required system access for the attack to be delivered."""

    PHYSICAL_ENVIRONMENT = "physical_environment"  # place an object, shine a laser, etc.
    SENSOR = "sensor"                              # tamper with a sensor input stream
    NETWORK_V2X = "network_v2x"                    # inject/alter V2X messages
    SOFTWARE = "software"                          # dependency / config / parameter access
    MODEL = "model"                                # model weights / internals


class ThreatModel(BaseModel):
    """Adversary specification (PROJECT.md 7.3).

    TODO(phase1): richer capability/constraint typing (budgets, physical realism envelopes)
    and a ``compatible_with(attack, system)`` validator used by the attack engine.
    """

    goal: str = Field(..., description="Adversary objective in natural language.")
    knowledge: Knowledge = Knowledge.BLACKBOX
    access: list[AccessLevel] = Field(default_factory=list)
    target: str = Field(..., description="Targeted component/object/behavior.")

    # Capabilities & constraints (kept free-form for now; typed in Phase 1).
    capabilities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(
        default_factory=list,
        description="Physical/operational limits (e.g. max perturbation, LOS, timing).",
    )
    timing: str | None = Field(
        default=None, description="When the attack activates (trigger/window)."
    )
    success_criteria: str = Field(
        ..., description="Objective, checkable definition of attack success."
    )
