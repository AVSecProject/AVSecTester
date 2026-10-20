"""Insertion tools expose one public package API without circular initialization."""
import subprocess
import sys

import numpy as np


def test_public_asset_compositor_effect_and_extension_types_are_importable():
    from avsectester.attack.insertion import (
        ClassicHarmonizer, CuboidVisibilityEstimator, DepthVisibilityEstimator, Harmonizer,
        Insertion, InsertionAttackEffect, PatchCompositor, PlaneAsset, PlaneSurface,
        ResolvedInsertion, VisibilityEvidence, render_resolved, resolve_insertion,
    )
    from avsectester.attack.placement import WorldPlacement
    from avsectester.geometry.poses import ActorPose

    asset = PlaneAsset(np.full((2, 2, 4), 255, np.uint8), 1., 1.)
    insertion = Insertion("patch", asset, WorldPlacement((10., 0., 0.)))
    resolved = resolve_insertion(insertion, {}, ActorPose(np.eye(4)))
    assert isinstance(resolved, ResolvedInsertion)
    assert isinstance(resolved.planes()[0], PlaneSurface)
    assert isinstance(PatchCompositor().harmonizer, (Harmonizer, ClassicHarmonizer))
    assert isinstance(InsertionAttackEffect("front", insertion), InsertionAttackEffect)
    assert isinstance(CuboidVisibilityEstimator(), CuboidVisibilityEstimator)
    assert isinstance(DepthVisibilityEstimator(), DepthVisibilityEstimator)
    assert callable(render_resolved) and isinstance(VisibilityEvidence, type)


def test_public_package_can_be_initialized_before_its_effect_submodule():
    result = subprocess.run([
        sys.executable, "-c",
        "from avsectester.attack.insertion import InsertionAttackEffect, PatchCompositor; "
        "from avsectester.attack.insertion.effect import InsertionAttackEffect as Effect; "
        "assert InsertionAttackEffect is Effect; assert PatchCompositor is not None",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
