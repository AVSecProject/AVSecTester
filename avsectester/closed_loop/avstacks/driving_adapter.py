"""Driving-system component schemas, scorer capabilities and fresh instance bindings."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from avsectester.closed_loop.avstacks.base import AVStack


@dataclass(frozen=True)
class ComponentDescriptor:
    id: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    codecs: Mapping[str, str]
    differentiable_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScorerDescriptor(ComponentDescriptor):
    output_shape: tuple[int, ...] = ()
    direction: str = "minimize"

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("Scorer direction must be minimize or maximize")


class DrivingAdapter:
    """Explicit component/scorer bindings over a fresh AVStack factory.

    Callbacks receive (avstack, inputs), or (avstack, inputs, target) for scorers. Input
    schemas describe native argument conventions; adapters own conversion and warmup.
    Component evaluations and rollouts use fresh AVStack instances. Scoring reuses one
    AVStack per session to keep weights loaded; scorer callbacks must isolate each request
    from driving history. Separate sessions and rollouts never share that instance.
    """

    def __init__(
        self, factory: Callable[[int], AVStack], *,
        components: Sequence[tuple[ComponentDescriptor, Callable[..., Any]]] = (),
        scorers: Sequence[tuple[ScorerDescriptor, Callable[..., Any]]] = (),
    ) -> None:
        self.factory = factory
        self.component_bindings = self._bindings(components)
        self.scorer_bindings = self._bindings(scorers)

    @staticmethod
    def _bindings(entries: Sequence[Any]) -> dict[str, Any]:
        result = {}
        for descriptor, callback in entries:
            if not descriptor.id or descriptor.id in result or not callable(callback):
                raise ValueError("Component/scorer IDs must be unique and callbacks callable")
            if not descriptor.inputs or not descriptor.outputs or not descriptor.codecs:
                raise ValueError("Declare component input/output schemas and codecs")
            if set(descriptor.differentiable_inputs) - descriptor.inputs.keys():
                raise ValueError("Differentiable inputs must name declared inputs")
            result[descriptor.id] = (deepcopy(descriptor), callback)
        return result
