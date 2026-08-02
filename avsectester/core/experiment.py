"""The Security Experiment Specification (PROJECT.md 7).

A single machine-readable, reproducible contract that any backend can execute. This is
the object the agent layer generates from natural language and the framework validates
and runs.

The sub-configs use ``{"type": name, ...}`` dicts so they build directly from the
AVSecTester/avstack registries (registry.build(cfg)).
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, Field

from .threat_model import ThreatModel


class SystemSpec(BaseModel):
    """AV system under test (PROJECT.md 7.1)."""

    name: str
    # avstack pipeline config (SerialPipeline / MappedPipeline) or a named preset.
    pipeline: dict[str, Any] = Field(default_factory=dict)
    checkpoints: dict[str, str] = Field(default_factory=dict)
    access_level: str = "blackbox"


class ScenarioSpec(BaseModel):
    """Scenario / driving context (PROJECT.md 7.2)."""

    backend: dict[str, Any] = Field(
        ..., description="Backend build config, e.g. {'type': 'CarlaBackend', ...}."
    )
    map: str | None = None
    initial_conditions: dict[str, Any] = Field(default_factory=dict)
    participants: list[dict[str, Any]] = Field(default_factory=list)
    target_objects: list[dict[str, Any]] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [0])
    repetitions: int = 1


class AttackConfig(BaseModel):
    """Attack instantiation (PROJECT.md 7.4). ``spec`` builds from ATTACKS registry."""

    spec: dict[str, Any] = Field(..., description="{'type': attack_name, **params}")
    threat_model: ThreatModel


class DefenseConfig(BaseModel):
    """Defense instantiation (PROJECT.md 7.5). ``spec`` builds from DEFENSES registry."""

    spec: dict[str, Any] = Field(..., description="{'type': defense_name, **params}")


class EvaluationConfig(BaseModel):
    """Which security metrics to compute (PROJECT.md 7.6)."""

    metrics: list[dict[str, Any]] = Field(default_factory=list)
    repetitions: int = 1


class ReproducibilityInfo(BaseModel):
    """Provenance / repro (PROJECT.md 7.7). Auto-populated by the runner where possible."""

    software_versions: dict[str, str] = Field(default_factory=dict)
    container_image: str | None = None
    config_hash: str | None = None
    notes: str | None = None


class ExperimentSpec(BaseModel):
    """Top-level experiment contract."""

    name: str
    system: SystemSpec
    scenario: ScenarioSpec
    attack: AttackConfig | None = None
    defense: DefenseConfig | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    reproducibility: ReproducibilityInfo = Field(default_factory=ReproducibilityInfo)

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentSpec:
        with open(path, "r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)
