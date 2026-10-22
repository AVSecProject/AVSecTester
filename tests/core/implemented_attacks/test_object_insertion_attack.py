"""Object-only behavior: concrete assets and recorded nuScenes image insertion."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack import OfflineContext
from avsectester.implemented_attacks.object_insertion_attack import ObjectInsertionAttack
from avsectester.implemented_attacks.object_insertion_attack.assets import (
    PersonAsset, SignAsset, TrafficLightAsset,
)
from avsectester.attack.insertion import PatchCompositor, PlaneAsset
from avsectester.attack.insertion.placement import WorldPlacement
from avsectester.execution.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.execution.services import ClosedLoopServices
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from tests.fakes import ConstantAVStack


@pytest.mark.parametrize("make_asset,plane_count,width_m", [
    (lambda image: SignAsset(face=image), 2, .9),
    (lambda image: PersonAsset(image), 3, 1.4),
    (lambda image: PersonAsset(image, mode="standee"), 1, .875),
    (lambda image: TrafficLightAsset(width_m=2.4), 2, 2.4),
], ids=["sign", "poster", "standee", "traffic-light"])
def test_assets(make_asset, plane_count, width_m):
    image = np.full((32, 16, 4), [50, 80, 100, 255], np.uint8)
    asset = make_asset(image)
    planes = tuple(asset.planes())
    assert len(planes) == plane_count
    assert all(plane.texture.shape[2] == 4 and plane.texture.dtype == np.uint8 for plane in planes)
    assert np.linalg.norm(planes[-1].corners[1] - planes[-1].corners[0]) == pytest.approx(width_m)
    assert np.min(planes[-1].corners[:, 2]) >= 0.


def test_nuscenes_replay(nuscenes_dataset):
    dataset, simulator = nuscenes_dataset, RecordedReplaySimulator(camera_aliases={"CAM_FRONT": "front"})
    attack = ObjectInsertionAttack(decision_steps=2, max_segments=1,
        assets=PlaneAsset(np.full((8, 8, 4), [240, 20, 20, 255], np.uint8), 1., 1.),
        placement=WorldPlacement((5., 0., 1.)), compositor=PatchCompositor())
    with ClosedLoopServices(dataset, simulator) as services:
        segment, = attack.build_metadata(services.scenario("scene", seed=0))
        artifacts = attack.run_offline(OfflineContext(services, (segment,)))
    with pytest.raises(ValueError, match="missing keys: placement. Required keys: placement"):
        attack.create_runtime(replace(segment, user_data={}), artifacts)
    with pytest.raises(ValueError, match="missing keys: kind. Required keys: kind"):
        attack.create_runtime(replace(segment, user_data={"placement": {}}), artifacts)
    with pytest.raises(ValueError, match="missing keys: host. Required keys: kind, host"):
        attack.create_runtime(replace(segment, user_data={"placement": {"kind": "attached"}}), artifacts)
    with simulator.create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
            2, start=dataset.scenario_start("scene", segment.start),
            attacks=(attack.create_runtime(segment, artifacts),),
            recording=RecordingPolicy(("front",), ("raw", "model_input")),
        ))
    assert log.status == "completed"
    assert all(not np.array_equal(row.raw_observation.sensor_data["front"],
                                  row.model_observation.sensor_data["front"]) for row in log.steps)
    assert all(any(action.get("details", {}).get("rendered") is True for action in row.actions)
               for row in log.steps)


def test_sign_asset():
    sign = SignAsset()
    asset = sign.object_asset("stop_sign")
    assert asset.category == "stop_sign"
    np.testing.assert_allclose(asset.extent, [.06, .9, 2.4])
    assert sign.post_count == 1
    face_only = SignAsset(face=np.full((4, 4, 4), 255, np.uint8), post_count=0)
    assert len(face_only.planes()) == 1
    with pytest.raises(ValueError, match="post_count must be a nonnegative integer"):
        SignAsset(face=face_only.face, post_count=-1)
