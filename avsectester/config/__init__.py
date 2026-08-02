"""AVSecTester registries.

We reuse avstack's OpenMMLab-style ``Registry`` so plugins compose the same way avstack
modules do (build-from-config, ``@register_module()`` decorators). Attacks/defenses/
monitors are additionally hook-compatible so they can be attached to avstack pipelines.

If avstack is not installed (e.g. docs-only or unit tests that don't touch the stack), we
fall back to a minimal local Registry with the same surface so imports never hard-fail.
"""

from .registry import (
    ATTACKS,
    BACKENDS,
    DEFENSES,
    HAVE_AVSTACK,
    METRICS,
    MONITORS,
    SEARCH,
    Registry,
)

__all__ = [
    "ATTACKS",
    "BACKENDS",
    "DEFENSES",
    "HAVE_AVSTACK",
    "METRICS",
    "MONITORS",
    "SEARCH",
    "Registry",
]
