"""Seams — the statically-defined injection points of an AV pipeline.

A seam is just a *name* for a place an attack/defense can act. The AV pipeline (``System``)
knows how to fire attached plugins at each seam; a plugin declares which seams it uses and
implements ``apply`` — there is no separate "hook" object.
"""

from __future__ import annotations

from enum import Enum


class Seam(str, Enum):
    # raw sensor inputs
    RAW_LIDAR = "raw_lidar"
    RAW_CAMERA = "raw_camera"
    RAW_GPS = "raw_gps"
    # pipeline module boundaries
    PERCEPTION_INPUT = "perception_input"   # object-level detector input (GT passthrough)
    PERCEPTION_OUT = "perception_out"        # detector output (detections)
    LOCALIZATION_OUT = "localization_out"
    TRACKING_OUT = "tracking_out"
    PLANNING_OUT = "planning_out"
    CONTROL_OUT = "control_out"

    def __str__(self) -> str:  # so f"{seam}" is the bare name
        return self.value
