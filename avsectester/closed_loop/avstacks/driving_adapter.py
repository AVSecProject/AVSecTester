"""Driving-system component schemas, scorer capabilities and fresh instance bindings."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.validation import require_nonnegative_integer, require_positive_number, require_nonnegative_number


@dataclass(frozen=True)
class ComponentDescriptor:
    """Description of an evaluable driving-system component and its input/output formats."""
    id: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    codecs: Mapping[str, str]
    differentiable_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScorerDescriptor(ComponentDescriptor):
    """Describe a batch scorer returning one score per input; output_shape is the per-input shape."""
    output_shape: tuple[int, ...] = ()
    direction: str = "minimize"
    sensor_id: str | None = None
    decision_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("Scorer direction must be minimize or maximize")
        if self.sensor_id is not None and (not isinstance(self.sensor_id, str) or not self.sensor_id):
            raise ValueError("sensor_id must be a nonempty string or None")
        if self.decision_threshold is not None:
            require_nonnegative_number(self.decision_threshold, "decision_threshold")
            if self.decision_threshold > 1:
                raise ValueError("decision_threshold must be at most 1")


class DrivingAdapter:
    """Bind component callbacks ``(avstack, inputs)`` and scorer callbacks
    ``(avstack, images, targets)`` to a fresh AVStack factory.

    Input schemas describe native argument conventions; adapters own conversion and warmup.
    Component evaluations and rollouts use fresh AVStack instances. Scoring reuses one
    AVStack per session to keep weights loaded; scorer callbacks must isolate each request
    from driving history. Separate sessions and rollouts never share that instance.
    """

    def __init__(
        self, factory: Callable[[int], AVStack], *,
        components: Sequence[tuple[ComponentDescriptor, Callable[..., Any]]] = (),
        scorers: Sequence[tuple[ScorerDescriptor, Callable[..., Any]]] = (),
        past_steps: int = 0,
        decision_time_step: float | None = None,
    ) -> None:
        require_nonnegative_integer(past_steps, "past_steps")
        if decision_time_step is not None:
            require_positive_number(decision_time_step, "decision_time_step")
        self._decision_time_step = decision_time_step
        self.factory = factory
        self._past_steps = past_steps
        self.component_bindings = self._bindings(components)
        self.scorer_bindings = self._bindings(scorers)

    @property
    def past_steps(self) -> int:
        """Required recorded history, declared by the AVStack type or this adapter.

        Reading the declaration never constructs a driving system or loads model weights.
        A factory's ``avstack_type`` takes precedence when it is an AVStack subclass.
        """
        avstack_type = getattr(self.factory, "avstack_type", None)
        if isinstance(avstack_type, type) and issubclass(avstack_type, AVStack):
            value = avstack_type.past_steps
            require_nonnegative_integer(value, "past_steps")
            return value
        return self._past_steps

    @property
    def decision_time_step(self) -> float | None:
        """Required decision interval, read without constructing the driving system."""
        avstack_type = getattr(self.factory, "avstack_type", None)
        value = (avstack_type.decision_time_step
                 if isinstance(avstack_type, type) and issubclass(avstack_type, AVStack)
                 else self._decision_time_step)
        if value is not None:
            require_positive_number(value, "decision_time_step")
        return value

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


__all__ = [
    'ComponentDescriptor',
    'ScorerDescriptor',
    'DrivingAdapter',
]
