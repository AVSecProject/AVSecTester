"""Scenario requirement DSL — the predicate selects the right target and enforces each constraint."""

from avsectester.scenarios import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.requirement import DistanceRange, ImageAreaFrac, ViewpointRear
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE

CAM = {"front": CameraCalib(name="front", width=800, height=600)}


def _vehicle(track, x, box, yaw=0.0, vis=1.0, category="vehicle"):
    return ObjectGT(track_id=track, category=category, center=(x, 0.0, 0.0), extent=(4.5, 2.0, 1.5),
                    yaw=yaw, box2d={"front": box}, visibility=vis)


def _scene(objects, speed=5.0):
    return SceneGT(frame=0, t=0.0, ego=EgoState(speed=speed), cameras=CAM, objects=objects)


def test_requirement_matches_qualifying_vehicle_and_selects_nearest_ahead():
    near = _vehicle("near", 8.0, (330, 250, 470, 400))   # ~0.058 area frac, rear-facing, in view
    far = _vehicle("far", 60.0, (395, 295, 405, 305))    # too far, tiny box
    m = PHYSICAL_PATCH_HIDE_VEHICLE.match(_scene([far, near]))
    assert m is not None and m.target.track_id == "near" and m.camera == "front"


def test_requirement_rejects_when_a_constraint_fails():
    req = PHYSICAL_PATCH_HIDE_VEHICLE
    assert req.match(_scene([_vehicle("v", 40.0, (395, 295, 405, 305))])) is None  # DistanceRange fails
    assert req.match(_scene([_vehicle("v", 8.0, (330, 250, 470, 400), yaw=3.0)])) is None  # ViewpointRear
    assert req.match(_scene([_vehicle("v", 8.0, (330, 250, 470, 400), vis=0.2)])) is None  # MinVisibility
    assert req.match(_scene([_vehicle("v", 8.0, (0, 0, 800, 600))])) is None  # ImageAreaFrac too big
    assert req.match(_scene([_vehicle("v", 8.0, (330, 250, 470, 400), category="pedestrian")])) is None


def test_individual_constraints():
    v = _vehicle("v", 10.0, (300, 200, 500, 400))
    scene = _scene([v])
    assert DistanceRange(4, 25).holds(scene, v) and not DistanceRange(0, 5).holds(scene, v)
    assert ViewpointRear(35).holds(scene, v) and not ViewpointRear(5).holds(scene, _vehicle("w", 10, (1, 1, 2, 2), yaw=1.0))
    assert ImageAreaFrac(0.02, 0.5).holds(scene, v)  # 200*200 / (800*600) = 0.083


# ---- serialization + natural-language interpretation ----------------------------------------------
def test_requirement_serialize_roundtrip():
    from avsectester.scenarios.serialize import requirement_from_dict, requirement_to_dict

    d = requirement_to_dict(PHYSICAL_PATCH_HIDE_VEHICLE)
    assert d["target"]["category"] == "vehicle" and d["constraints"][0]["kind"] == "InView"
    back = requirement_from_dict(d)
    assert requirement_to_dict(back) == d  # exact round-trip through the constraint registry


def test_nl_interpret_with_stub_llm():
    import json

    from avsectester.scenarios.nl import build_prompt, interpret
    from avsectester.scenarios.serialize import requirement_to_dict

    target = requirement_to_dict(PHYSICAL_PATCH_HIDE_VEHICLE)

    def stub_llm(prompt: str) -> str:
        assert "constraint kinds" in prompt and "DistanceRange" in prompt  # grounded in the vocabulary
        return "```json\n" + json.dumps(target) + "\n```"  # LLM would emit this

    req = interpret("a vehicle directly ahead, close, rear facing us, unoccluded", stub_llm)
    # a scene that satisfies the hand-written requirement also satisfies the interpreted one
    assert req.match(_scene([_vehicle("near", 8.0, (330, 250, 470, 400))])) is not None
    assert "ImageAreaFrac" in build_prompt("x")  # the prompt lists the real constraint vocabulary


# ---- CARLA analytic ground truth + builder (offline: no CARLA) ------------------------------------
def test_predict_scene_gt_and_builder_are_offline():
    from avsectester.scenarios.carla_gt import predict_scene_gt
    from avsectester.scenarios.source import CarlaScenarioBuilder

    base = {"ego": {"sensors": [{"type": "CarlaRgbCamera", "image_size_x": 800,
                                 "image_size_y": 600, "fov": 90}]}}
    scene = predict_scene_gt(base, gap=6.0, lateral=0.0, speed=5.0)
    assert PHYSICAL_PATCH_HIDE_VEHICLE.match(scene) is not None  # a close lead qualifies
    assert PHYSICAL_PATCH_HIDE_VEHICLE.match(predict_scene_gt(base, gap=60.0, lateral=0.0)) is None

    insts = list(CarlaScenarioBuilder(base_scenario=base, samples=300).scenarios(
        PHYSICAL_PATCH_HIDE_VEHICLE, limit=5))
    assert len(insts) == 5  # the builder found qualifying lead placements, analytically
    for it in insts:
        assert 4.0 <= it.target.target.distance <= 25.0 and it.provenance["backend"] == "carla"


# ---- DatasetFilter over a stub dataset ------------------------------------------------------------
def test_dataset_filter_keeps_only_qualifying():
    from avsectester.scenarios.source import Dataset, DatasetFilter

    scenes = [_scene([_vehicle("v", 8.0, (330, 250, 470, 400))]),   # qualifies
              _scene([_vehicle("v", 60.0, (398, 298, 402, 302))]),  # too far
              _scene([_vehicle("v", 8.0, (330, 250, 470, 400))])]   # qualifies
    for i, sc in enumerate(scenes):
        sc.source = {"clip": "stub", "frame": i}

    class StubDataset(Dataset):
        def scenes(self):
            return iter(scenes)

        def make_backend(self, scene):
            return ("backend-for", scene.source["frame"])  # placeholder

    insts = list(DatasetFilter(StubDataset()).scenarios(PHYSICAL_PATCH_HIDE_VEHICLE))
    assert len(insts) == 2 and [it.provenance["frame"] for it in insts] == [0, 2]
    assert insts[0].make_backend() == ("backend-for", 0)  # lazy backend construction
