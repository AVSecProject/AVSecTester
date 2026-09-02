"""AVSecTester registries.

We reuse avstack's OpenMMLab-style ``Registry`` so plugins compose the same way avstack
modules do (build-from-config, ``@register_module()`` decorators), with a minimal local
fallback so imports never hard-fail when avstack isn't installed.
"""

from .registry import (
    ATTACKS,
    DEFENSES,
    ENVIRONMENTS,
    HAVE_AVSTACK,
    METRICS,
    SYSTEMS,
    Registry,
)

__all__ = [
    "ATTACKS",
    "DEFENSES",
    "ENVIRONMENTS",
    "HAVE_AVSTACK",
    "METRICS",
    "SYSTEMS",
    "Registry",
]
