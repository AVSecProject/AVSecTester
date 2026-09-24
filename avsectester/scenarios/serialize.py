"""(De)serialize a :class:`ScenarioRequirement` to/from a plain dict (JSON-friendly).

This is the formal, machine-readable form of a requirement — the target an LLM emits when interpreting a
natural-language description (:mod:`avsectester.scenarios.nl`), and a stable on-disk format. Because
every :class:`~avsectester.scenarios.requirement.Constraint` is a dataclass, a constraint round-trips as
``{"kind": <ClassName>, <fields...>}``; :data:`CONSTRAINT_TYPES` maps the kind back to the class.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from avsectester.scenarios.requirement import (
    ClearLaneAhead,
    Constraint,
    DistanceRange,
    EgoMoving,
    ImageAreaFrac,
    InView,
    MinVisibility,
    ScenarioRequirement,
    TargetSpec,
    ViewpointRear,
)

#: The constraint vocabulary, keyed by the ``kind`` string used in the serialized form.
CONSTRAINT_TYPES: dict[str, type[Constraint]] = {
    c.__name__: c for c in (InView, DistanceRange, ImageAreaFrac, ViewpointRear,
                            MinVisibility, EgoMoving, ClearLaneAhead)
}


def constraint_to_dict(c: Constraint) -> dict[str, Any]:
    return {"kind": type(c).__name__, **dataclasses.asdict(c)}


def constraint_from_dict(d: dict[str, Any]) -> Constraint:
    d = dict(d)
    kind = d.pop("kind")
    if kind not in CONSTRAINT_TYPES:
        raise ValueError(f"unknown constraint kind {kind!r}; known: {sorted(CONSTRAINT_TYPES)}")
    return CONSTRAINT_TYPES[kind](**d)


def requirement_to_dict(req: ScenarioRequirement) -> dict[str, Any]:
    return {
        "name": req.name,
        "description": req.description,
        "target": dataclasses.asdict(req.target),
        "constraints": [constraint_to_dict(c) for c in req.constraints],
    }


def requirement_from_dict(d: dict[str, Any]) -> ScenarioRequirement:
    return ScenarioRequirement(
        name=d["name"],
        description=d.get("description", ""),
        target=TargetSpec(**d["target"]),
        constraints=[constraint_from_dict(c) for c in d.get("constraints", [])],
    )


def constraint_vocabulary() -> list[dict[str, Any]]:
    """Machine-readable description of each constraint kind + its fields/defaults + one-line doc — the
    schema fed to an LLM so it knows the exact vocabulary it may emit."""
    vocab = []
    for kind, cls in CONSTRAINT_TYPES.items():
        fields = {f.name: (None if f.default is dataclasses.MISSING else f.default)
                  for f in dataclasses.fields(cls)}
        doc = (cls.__doc__ or "").strip().split("\n", 1)[0]
        vocab.append({"kind": kind, "fields": fields, "doc": doc})
    return vocab
