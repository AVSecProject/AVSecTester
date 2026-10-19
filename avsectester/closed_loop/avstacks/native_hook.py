"""Bridge explicitly registered avstack-library hooks into the framework hook lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from avsectester.closed_loop.hooks import CallInputs, Plugin, RunContext, StageContext


class NativeHook(Plugin):
    """Adapt a registered avstack hook's tuple convention to an ordered stage hook.

    Import the implementation's registration module explicitly before running. The native
    handler is built afresh on reset and released when its enclosing hook session closes.
    """

    def __init__(self, config: Mapping[str, Any], *, phase: str = "post") -> None:
        if phase not in {"pre", "post"}:
            raise ValueError("NativeHook phase must be 'pre' or 'post'")
        self.config = deepcopy(dict(config))
        self.phase = phase
        self.native: Any = None

    def reset(self, context: RunContext) -> None:
        from avstack.config import HOOKS

        self.native = HOOKS.build(deepcopy(self.config))

    def transform(self, value: Any, context: StageContext) -> Any:
        if self.phase == "pre":
            if not isinstance(value, CallInputs):
                raise TypeError("Native pre-hooks require CallInputs")
            args, kwargs = self.native(*value.args, **value.kwargs)
            return CallInputs(args=args, kwargs=kwargs)
        result = self.native(value)
        if not isinstance(result, tuple) or len(result) != 1:
            raise TypeError("Native post-hooks must return one output in a tuple")
        return result[0]

    def close(self) -> None:
        native, self.native = self.native, None
        close = getattr(native, "close", None)
        if close is not None:
            close()
