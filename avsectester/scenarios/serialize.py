"""Dictionary serialization for requirements, filter groups and built-in insertions."""

from __future__ import annotations

import dataclasses
from typing import Any

from .filters import (
    All,
    Any as AnyFilter,
    Not,
    ClearLaneAhead,
    Constraint,
    DistanceRange,
    EgoMoving,
    ImageAreaFrac,
    InView,
    MinVisibility,
    ViewpointRear,
)
from .requirement import (
    InitialWindow,
    RoleSpec,
    ScenarioRequirement,
    TargetSpec,
)

CONSTRAINT_TYPES: dict[str, type[Constraint]] = {
    cls.__name__: cls
    for cls in (
        All,
        AnyFilter,
        Not,
        InView,
        DistanceRange,
        ImageAreaFrac,
        ViewpointRear,
        MinVisibility,
        EgoMoving,
        ClearLaneAhead,
    )
}


def register_constraint(cls: type[Constraint]) -> type[Constraint]:
    """Register a dataclass filter for serialization. Direct Python use needs no registry."""
    if not issubclass(cls, Constraint) or not dataclasses.is_dataclass(cls):
        raise TypeError("Registered filters must be Constraint dataclasses")
    name = cls.__name__
    if name in CONSTRAINT_TYPES and CONSTRAINT_TYPES[name] is not cls:
        raise ValueError(f"Constraint {name!r} is already registered")
    CONSTRAINT_TYPES[name] = cls
    return cls


def constraint_to_dict(constraint: Constraint) -> dict[str, Any]:
    if isinstance(constraint, (All, AnyFilter)):
        return {
            "kind": type(constraint).__name__,
            "filters": [constraint_to_dict(child) for child in constraint.filters],
        }
    if isinstance(constraint, Not):
        return {"kind": "Not", "filter": constraint_to_dict(constraint.filter)}
    return {"kind": type(constraint).__name__, **dataclasses.asdict(constraint)}


def constraint_from_dict(value: dict[str, Any]) -> Constraint:
    fields = dict(value)
    kind = fields.pop("kind")
    if kind not in CONSTRAINT_TYPES:
        raise ValueError(f"unknown constraint kind {kind!r}, known: {sorted(CONSTRAINT_TYPES)}")
    cls = CONSTRAINT_TYPES[kind]
    if cls in (All, AnyFilter):
        fields["filters"] = [constraint_from_dict(child) for child in fields.get("filters", [])]
    elif cls is Not:
        fields["filter"] = constraint_from_dict(fields["filter"])
    if "subjects" in fields:
        fields["subjects"] = tuple(fields["subjects"])
    return cls(**fields)


def insertion_to_dict(insertion):
    """Serialize a built-in planar asset. Custom asset objects remain usable through Python."""
    from avsectester.insertion import PlaneAsset

    if not isinstance(insertion.asset, PlaneAsset):
        raise TypeError(
            "Dictionary serialization supports PlaneAsset, use Python for custom assets"
        )
    asset = insertion.asset
    return {
        "id": insertion.id,
        "asset": {
            "kind": "PlaneAsset",
            "width_m": asset.width_m,
            "height_m": asset.height_m,
            "texture": asset.texture_path or asset.texture.tolist(),
        },
        "placement": {
            "kind": type(insertion.placement).__name__,
            **dataclasses.asdict(insertion.placement),
        },
        "orientation": dataclasses.asdict(insertion.orientation),
    }


def insertion_from_dict(value):
    import numpy as np
    from avsectester.insertion import (
        AttachedPlacement,
        Insertion,
        Orientation,
        PlaneAsset,
        WorldPlacement,
    )

    asset = dict(value["asset"])
    if asset.pop("kind") != "PlaneAsset":
        raise ValueError("Unknown serialized insertion asset")
    if not isinstance(asset["texture"], str):
        texture = np.asarray(asset["texture"])
        if not np.issubdtype(texture.dtype, np.integer) or np.any((texture < 0) | (texture > 255)):
            raise ValueError("Inline texture values must be uint8 integers")
        asset["texture"] = texture.astype(np.uint8)
    placement = dict(value["placement"])
    kind = placement.pop("kind")
    placements = {"AttachedPlacement": AttachedPlacement, "WorldPlacement": WorldPlacement}
    if kind not in placements:
        raise ValueError(f"Unknown placement kind {kind!r}")
    return Insertion(
        value["id"],
        PlaneAsset(**asset),
        placements[kind](**placement),
        Orientation(**value.get("orientation", {})),
    )


def requirement_to_dict(req: ScenarioRequirement) -> dict[str, Any]:
    return {
        "name": req.name,
        "description": req.description,
        "target": dataclasses.asdict(req.target) if req.target else None,
        "constraints": [constraint_to_dict(c) for c in req.constraints],
        "roles": {name: dataclasses.asdict(spec) for name, spec in req.roles.items()},
        "insertions": [insertion_to_dict(i) for i in req.insertions],
        "window": dataclasses.asdict(req.window),
        "max_bindings": req.max_bindings,
        "camera": req.camera,
    }


def requirement_from_dict(value: dict[str, Any]) -> ScenarioRequirement:
    return ScenarioRequirement(
        name=value["name"],
        description=value.get("description", ""),
        target=TargetSpec(**value["target"]) if value.get("target") is not None else None,
        constraints=[constraint_from_dict(c) for c in value.get("constraints", [])],
        roles={name: RoleSpec(**fields) for name, fields in value.get("roles", {}).items()},
        insertions=tuple(insertion_from_dict(i) for i in value.get("insertions", [])),
        window=InitialWindow(**value.get("window", {})),
        max_bindings=value.get("max_bindings", 1000),
        camera=value.get("camera"),
    )


def constraint_vocabulary() -> list[dict[str, Any]]:
    """Filter names, fields and defaults for the optional natural-language adapter."""
    result = []
    for kind, cls in CONSTRAINT_TYPES.items():
        fields = {}
        for field in dataclasses.fields(cls):
            if field.default is not dataclasses.MISSING:
                default = field.default
            elif field.default_factory is not dataclasses.MISSING:
                default = field.default_factory()
            else:
                default = None
            fields[field.name] = default
        result.append(
            {"kind": kind, "fields": fields, "doc": (cls.__doc__ or "").strip().split("\n", 1)[0]}
        )
    return result
