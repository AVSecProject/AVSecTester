"""Scenario requirement DSL — the predicate selects the right target and enforces each constraint."""

import pytest
from avsectester.rendering.types import Visibility

import numpy as np
from dataclasses import replace

from avsectester.scenarios import CameraCalib, EgoState, FilterContext, ObjectGT, SceneGT
from avsectester.scenarios.filters import DistanceRange, ImageAreaFrac, ViewpointRear
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE

CAM = {"front": CameraCalib(name="front", width=800, height=600, cam_to_ego=np.eye(4))}


def _vehicle(track, x, box, yaw=0.0, vis=1.0, category="vehicle"):
    return ObjectGT(
        track_id=track,
        category=category,
        center=(x, 0.0, 0.0),
        extent=(4.5, 2.0, 1.5),
        yaw=yaw,
        box2d={"front": box},
        visibility=Visibility(vis, source="provided") if vis is not None else None,
    )


def _scene(objects, speed=5.0):
    return SceneGT(frame=0, t=0.0, ego=EgoState(speed=speed), cameras=CAM, objects=objects)


def test_requirement_matches_qualifying_vehicle_and_selects_nearest_ahead():
    near = _vehicle("near", 8.0, (330, 250, 470, 400))  # ~0.058 area frac, rear-facing, in view
    far = _vehicle("far", 12.0, (330, 250, 470, 400))  # also qualifies
    m = PHYSICAL_PATCH_HIDE_VEHICLE.match(_scene([far, near]))
    assert m is not None and m.target.track_id == "near" and m.camera == "front"


@pytest.mark.parametrize(
    "overrides",
    [
        {"x": 40.0},
        {"yaw": 3.0},
        {"vis": 0.2},
        {"box": (0, 0, 800, 600)},
        {"category": "pedestrian"},
    ],
    ids=["distance", "viewpoint", "visibility", "image-area", "category"],
)
def test_requirement_rejects_when_one_constraint_fails(overrides):
    settings = dict(track="v", x=8.0, box=(330, 250, 470, 400))
    settings.update(overrides)
    assert PHYSICAL_PATCH_HIDE_VEHICLE.match(_scene([_vehicle(**settings)])) is None


def test_individual_constraints():
    v = _vehicle("v", 10.0, (300, 200, 500, 400))
    scene = _scene([v])
    assert DistanceRange(4, 25).holds(scene, v) and not DistanceRange(0, 5).holds(scene, v)
    assert ViewpointRear(35).holds(scene, v) and not ViewpointRear(5).holds(
        scene, _vehicle("w", 10, (1, 1, 2, 2), yaw=1.0)
    )
    assert ImageAreaFrac(0.02, 0.5).holds(scene, v)  # 200*200 / (800*600) = 0.083


# ---- serialization + natural-language interpretation ----------------------------------------------
def test_requirement_serialize_roundtrip():
    from avsectester.scenarios.serialize import requirement_from_dict, requirement_to_dict

    d = requirement_to_dict(PHYSICAL_PATCH_HIDE_VEHICLE)
    assert d["target"]["category"] == "vehicle" and d["constraints"][0]["kind"] == "InView"
    back = requirement_from_dict(d)
    assert requirement_to_dict(back) == d  # exact round-trip through the constraint registry


@pytest.mark.parametrize("template", ["{}", "```json\n{}\n```", "Requirement: {}"])
def test_nl_interpret_with_stub_llm(template):
    import json

    from avsectester.scenarios.nl import build_prompt, interpret
    from avsectester.scenarios.serialize import requirement_to_dict

    target = requirement_to_dict(PHYSICAL_PATCH_HIDE_VEHICLE)

    def stub_llm(prompt: str) -> str:
        assert (
            "constraint kinds" in prompt and "DistanceRange" in prompt
        )  # grounded in the vocabulary
        return template.format(json.dumps(target))

    req = interpret("a vehicle directly ahead, close, rear facing us, unoccluded", stub_llm)
    # a scene that satisfies the hand-written requirement also satisfies the interpreted one
    assert req.match(_scene([_vehicle("near", 8.0, (330, 250, 470, 400))])) is not None
    assert "ImageAreaFrac" in build_prompt("x")  # the prompt lists the real constraint vocabulary


# ---- CARLA analytic ground truth + builder (offline: no CARLA) ------------------------------------
def test_predict_scene_gt_and_builder_are_offline():
    from avsectester.scenarios.carla_gt import predict_scene_gt
    from avsectester.scenarios.source import CarlaScenarioBuilder

    base = {
        "ego": {
            "sensors": [
                {"type": "CarlaRgbCamera", "image_size_x": 800, "image_size_y": 600, "fov": 90}
            ]
        }
    }
    scene = predict_scene_gt(base, gap=6.0, lateral=0.0, speed=5.0)
    # Analytic geometry cannot prove lack of occlusion in a real world.
    assert PHYSICAL_PATCH_HIDE_VEHICLE.evaluate(scene).status == "unknown"
    geometry_req = replace(
        PHYSICAL_PATCH_HIDE_VEHICLE, constraints=PHYSICAL_PATCH_HIDE_VEHICLE.constraints[:-1]
    )
    assert geometry_req.match(scene) is not None
    assert geometry_req.match(predict_scene_gt(base, gap=60.0, lateral=0.0)) is None

    insts = list(
        CarlaScenarioBuilder(
            base_scenario=base, samples=300, sample_scene=True, analytic_preview=True
        ).scenarios(geometry_req, limit=5)
    )
    assert len(insts) == 5  # the builder found qualifying lead placements, analytically
    for it in insts:
        assert 4.0 <= it.match.target.distance <= 25.0 and it.provenance["backend"] == "carla"


# ---- DatasetFilter over a stub dataset ------------------------------------------------------------
@pytest.mark.parametrize("limit,expected_frames", [(None, [0, 2]), (1, [0])])
def test_dataset_filter_keeps_qualifying_scenes_and_binds_each_factory(limit, expected_frames):
    from avsectester.scenarios.source import Dataset, DatasetFilter

    scenes = [
        _scene([_vehicle("v", 8.0, (330, 250, 470, 400))]),  # qualifies
        _scene([_vehicle("v", 60.0, (398, 298, 402, 302))]),  # too far
        _scene([_vehicle("v", 8.0, (330, 250, 470, 400))]),
    ]  # qualifies
    for i, sc in enumerate(scenes):
        sc.source = {"clip": "stub", "frame": i}

    class StubDataset(Dataset):
        def scenes(self):
            return iter(scenes)

        def make_backend(self, scene):
            return ("backend-for", scene.source["frame"])  # placeholder

    insts = list(DatasetFilter(StubDataset()).scenarios(PHYSICAL_PATCH_HIDE_VEHICLE, limit=limit))
    assert [it.provenance["frame"] for it in insts] == expected_frames
    assert [it.make_backend() for it in insts] == [
        ("backend-for", frame) for frame in expected_frames
    ]


@pytest.mark.parametrize(
    "response,error,match",
    [
        ("no usable response", ValueError, "no JSON object"),
        (
            '{"target": {"category": "vehicle"}, "constraints": [{"kind": "Unknown"}]}',
            ValueError,
            "unknown constraint kind",
        ),
    ],
)
def test_nl_interpret_rejects_invalid_requirements(response, error, match):
    from avsectester.scenarios.nl import interpret

    with pytest.raises(error, match=match):
        interpret("a lead vehicle", lambda prompt: response)


def test_filters_all_candidates_before_selecting_nearest():
    near = _vehicle("near", 5, (300, 200, 500, 400), vis=0.2)
    far = _vehicle("far", 10, (300, 200, 500, 400), vis=0.9)
    result = PHYSICAL_PATCH_HIDE_VEHICLE.evaluate(_scene([near, far]))
    assert result.match.target.track_id == "far"
    assert [c.status for c in result.candidates] == ["fail", "pass"]


def test_rear_angle_uses_camera_bearing_and_preserves_yaw_wrapping():
    vehicle = _vehicle("car", 10, (1, 1, 20, 20))
    scene = _scene([vehicle])
    camera = replace(CAM["front"], cam_to_ego=np.eye(4))
    camera.cam_to_ego[1, 3] = 10
    scene.cameras = {"front": camera}
    assert not ViewpointRear(35).holds(
        scene, vehicle
    )  # parallel heading, camera actually at 45 degrees
    assert ViewpointRear(46).holds(scene, vehicle)
    vehicle.yaw = -np.pi / 4 + 2 * np.pi
    assert ViewpointRear(1).holds(scene, vehicle)
    camera.cam_to_ego = None
    assert (
        ViewpointRear().evaluate(FilterContext(scene, bindings={"attacker": (vehicle,)})).status
        == "unknown"
    )


def test_unknown_visibility_and_wrong_camera_do_not_pass():
    from avsectester.scenarios import Visibility
    from avsectester.scenarios.filters import MinVisibility

    target = _vehicle("car", 8, (1, 1, 20, 20), vis=None)
    scene = _scene([target])
    assert MinVisibility().evaluate(FilterContext(scene, target=target)).status == "unknown"
    target.visibility = Visibility(1.0, "rendered_masks", camera="rear")
    assert (
        MinVisibility(camera="front").evaluate(FilterContext(scene, target=target)).status
        == "unknown"
    )
    target.visibility = Visibility(0.5, "nuscenes_label", label="2")
    assert MinVisibility(0.5).holds(scene, target)
    assert not MinVisibility(0.51).holds(scene, target)


def test_scene_only_and_placement_requirements_share_filters():
    from avsectester.scenarios import PlacementCandidate, ScenarioRequirement, TargetSpec
    from avsectester.scenarios.filters import EgoMoving, InView

    scene = _scene([])
    assert ScenarioRequirement("moving", constraints=[EgoMoving(2)]).match(scene).target is None
    scene.placements = [
        PlacementCandidate("sign", "sign", (10, 0, 0), box2d={"front": (-10, 20, 30, 60)})
    ]
    req = ScenarioRequirement("insert", TargetSpec("sign", kind="placement"), [InView("front")])
    assert req.match(scene).target.track_id == "sign"
    assert scene.placements[0].image_area_frac("front", CAM["front"]) == pytest.approx(
        30 * 40 / (800 * 600)
    )
    scene.placements[0].box2d["front"] = (-30, 20, -10, 60)
    assert req.match(scene) is None  # geometry does not reuse stale results after changing a box


def test_custom_filter_registration_roundtrip_and_exception_propagation(monkeypatch):
    from dataclasses import dataclass
    from avsectester.scenarios import (
        Constraint,
        FilterResult,
        ScenarioRequirement,
        register_constraint,
    )
    from avsectester.scenarios.serialize import (
        CONSTRAINT_TYPES,
        requirement_from_dict,
        requirement_to_dict,
    )

    @dataclass
    class SpeedBelow(Constraint):
        limit: float

        def evaluate(self, context):
            return FilterResult.from_bool(context.scene.ego.speed < self.limit, "speed bound")

    monkeypatch.setitem(CONSTRAINT_TYPES, "SpeedBelow", SpeedBelow)
    register_constraint(SpeedBelow)
    req = ScenarioRequirement("slow", constraints=[SpeedBelow(4)])
    restored = requirement_from_dict(requirement_to_dict(req))
    assert restored.match(_scene([], speed=3)) is not None
    assert restored.match(_scene([], speed=5)) is None

    class Broken(Constraint):
        def evaluate(self, context):
            raise RuntimeError("bad custom filter")

    with pytest.raises(RuntimeError, match="bad custom filter"):
        ScenarioRequirement("broken", constraints=[Broken()]).evaluate(_scene([]))


def test_dataset_filter_records_exclusions_and_preserves_selected_state(tmp_path):
    import json
    from avsectester.scenarios import Dataset, DatasetFilter

    class Cases(Dataset):
        def scenes(self):
            for i, visibility in enumerate([None, 0.2, 0.9]):
                scene = _scene([_vehicle("lead", 8, (330, 250, 470, 400), vis=visibility)])
                scene.frame, scene.t = i, i * 0.1
                yield scene

        def make_backend(self, scene):
            return scene.frame, scene.t

    source = DatasetFilter(Cases())
    cases = list(source.scenarios(PHYSICAL_PATCH_HIDE_VEHICLE))
    assert len(cases) == 1 and cases[0].make_backend() == (2, 0.2)
    report = tmp_path / "selection.json"
    source.save_selection(report)
    rows = json.loads(report.read_text())
    assert [r["status"] for r in rows] == ["unknown", "fail", "pass"]
    assert rows[-1]["selected_target"] == "lead"
    assert list(source.scenarios(PHYSICAL_PATCH_HIDE_VEHICLE, limit=0)) == []


def test_carla_builder_does_not_filter_against_fictitious_initial_speed():
    from avsectester.scenarios.source import CarlaScenarioBuilder
    from avsectester.scenarios import ScenarioRequirement
    from avsectester.scenarios.filters import EgoMoving

    with pytest.raises(ValueError, match="starts stationary"):
        CarlaScenarioBuilder(ego_speed=5)
    builder = CarlaScenarioBuilder(
        base_scenario={"ego": {"sensors": []}}, samples=1, analytic_preview=True
    )
    assert list(builder.scenarios(ScenarioRequirement("moving", constraints=[EgoMoving(1)]))) == []
    assert builder.selection_log[0]["status"] == "fail"
