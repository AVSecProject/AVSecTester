"""Security metrics: score a clean vs attacked run (register with the METRICS registry)."""

from .impact import ImpactMetric

__all__ = ["ImpactMetric"]
