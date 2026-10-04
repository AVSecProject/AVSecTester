"""Scenario requirement DSL — the predicate selects the right target and enforces each constraint."""

import pytest

from avsectester.scenarios import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.requirement import DistanceRange, ImageAreaFrac, ViewpointRear
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE

CAM = {"front": CameraCalib(name="front", width=800, height=600)}


def _vehicle(track, x, box, yaw=0.0, vis=1.0, category="vehicle"):
    return ObjectGT(
        track_id=track,
        category=category,
        center=(x, 0.0, 0.0),
        extent=(4.5, 2.0, 1.5),
        yaw=yaw,
        box2d={"front": box},
        visibility=vis,
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
    assert PHYSICAL_PATCH_HIDE_VEHICLE.match(scene) is not None  # a close lead qualifies
    assert PHYSICAL_PATCH_HIDE_VEHICLE.match(predict_scene_gt(base, gap=60.0, lateral=0.0)) is None

    insts = list(
        CarlaScenarioBuilder(base_scenario=base, samples=300).scenarios(
            PHYSICAL_PATCH_HIDE_VEHICLE, limit=5
        )
    )
    assert len(insts) == 5  # the builder found qualifying lead placements, analytically
    for it in insts:
        assert 4.0 <= it.target.target.distance <= 25.0 and it.provenance["backend"] == "carla"


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
