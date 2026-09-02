"""Registry construction.

Prefer avstack's ``Registry`` (so AVSecTester and avstack share one build/config system).
Fall back to a tiny compatible shim when avstack-core is not importable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:  # pragma: no cover - exercised only when the avstack stack is installed
    from avstack.config import Registry as _AvstackRegistry

    Registry = _AvstackRegistry
    _HAVE_AVSTACK = True
except Exception:  # noqa: BLE001 - avstack optional at import time
    _HAVE_AVSTACK = False

    class Registry:  # type: ignore[no-redef]
        """Minimal drop-in used when avstack is unavailable.

        Supports the subset AVSecTester relies on: ``register_module`` (as a decorator or
        direct call), ``get``, ``build`` (from a ``{"type": name, **kwargs}`` config dict),
        and ``__contains__``.
        """

        def __init__(self, name: str) -> None:
            self.name = name
            self._entries: dict[str, Callable[..., Any]] = {}

        def register_module(
            self,
            name: str | None = None,
            module: Callable[..., Any] | None = None,
        ):
            def _register(cls: Callable[..., Any]) -> Callable[..., Any]:
                key = name or getattr(cls, "__name__", None)
                if key is None:
                    raise ValueError("cannot infer registry key")
                self._entries[key] = cls
                return cls

            return _register(module) if module is not None else _register

        def get(self, key: str) -> Callable[..., Any]:
            if key not in self._entries:
                raise KeyError(f"{key!r} not registered in {self.name!r}")
            return self._entries[key]

        def build(self, cfg: dict[str, Any], **kwargs: Any) -> Any:
            cfg = dict(cfg)
            key = cfg.pop("type")
            return self.get(key)(**{**cfg, **kwargs})

        def __contains__(self, key: str) -> bool:
            return key in self._entries

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return f"Registry({self.name!r}, {list(self._entries)})"


# Security-layer registries owned by AVSecTester.
ATTACKS = Registry("attacks")
DEFENSES = Registry("defenses")
METRICS = Registry("metrics")
ENVIRONMENTS = Registry("environments")   # dataset / simulation frame sources
SYSTEMS = Registry("systems")             # AV pipelines under test

HAVE_AVSTACK = _HAVE_AVSTACK
