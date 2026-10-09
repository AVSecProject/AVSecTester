"""Initial selection contracts across role binding, native APIs and inserted geometry."""

from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import pytest

from avsectester.insertion import (
    Insertion,
    Orientation,
    PlaneAsset,
    WorldPlacement,
)
from avsectester.scenarios import (
    All,
    Any,
    Constraint,
    Dataset,
    DatasetFilter,
    FilterContext,
    FilterResult,
    InitialWindow,
    Not,
    RoleSpec,
    ScenarioRequirement,
    Visibility,
)
from avsectester.scenarios.filters import DistanceRange, InView, MinVisibility
from avsectester.scenarios.serialize import requirement_from_dict, requirement_to_dict
from avsectester.scenarios.source import CarlaScenarioBuilder


@dataclass
class Status(Constraint):
    status: str

    def evaluate(self, context):
        return FilterResult(self.status, self.status)


@pytest.mark.parametrize(
    "statuses,all_status,any_status",
    [
        (("pass", "unknown"), "unknown", "pass"),
        (("fail", "unknown"), "fail", "unknown"),
        (("pass", "fail"), "fail", "pass"),
        (("unknown", "unknown"), "unknown", "unknown"),
        ((), "pass", "fail"),
    ],
)
def test_groups_preserve_unknown(statuses, all_status, any_status, scene):
    ctx = FilterContext(scene())
    filters = [Status(status) for status in statuses]
    assert All(filters).evaluate(ctx).status == all_status
    assert Any(filters).evaluate(ctx).status == any_status
    assert Not(Status("unknown")).evaluate(ctx).status == "unknown"
    assert Not(Status("pass")).evaluate(ctx).status == "fail"
    assert Not(Status("fail")).evaluate(ctx).status == "pass"


@pytest.mark.parametrize("opaque_center,expected", [(False, "fail"), (True, "pass")])
def test_in_view_uses_opaque_pixels_not_the_texture_rectangle(opaque_center, expected, scene):
    texture = np.full((10, 100, 4), 255, np.uint8)
    if not opaque_center:
        texture[:, 5:-5, 3] = 0
    item = Insertion("wide", PlaneAsset(texture, 80, 2), WorldPlacement((10, 0, 1.5)))
    ctx = FilterContext(scene(), insertions=(item,))
    assert ctx.subjects()[0].image_area_frac("front", ctx.scene.cameras["front"]) > 0
    assert InView("front").evaluate(ctx).status == expected
    result = ScenarioRequirement("visible", insertions=(item,), camera="front").evaluate(ctx)
    assert result.status == expected


def test_in_view_is_independent_of_occlusion(scene):
    from avsectester.rendering.visibility import CuboidVisibilityEstimator

    item = Insertion(
        "hidden", PlaneAsset(np.full((4, 4, 4), 255, np.uint8), 1, 1), WorldPlacement((15, 0, 0.75))
    )
    ctx = FilterContext(
        scene(), insertions=(item,), visibility_estimator=CuboidVisibilityEstimator()
    )
    assert InView("front").evaluate(ctx).status == "pass"
    assert ctx.visibility(ctx.subjects()[0], "front").fraction == 0


@pytest.mark.parametrize("outcome", ["pass", "fail", "unknown", "error", "prepare_error"])
def test_dataset_filter_closes_owned_resources_on_all_exit_paths(outcome, scene):
    events = []

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(("close", self.name))

    shared = Resource("borrowed")

    class Records(Dataset):
        def scenes(self):
            yield scene()

        def initial_sequence(self, first, frames):
            return (first, scene(1))

        def context(self, frame):
            ctx = FilterContext(frame, backend=shared)
            ctx.renderer = ctx.own(Resource(frame.frame))
            events.append(("open", frame.frame))
            return ctx

        def make_backend(self, frame):
            return shared

    class Check(Constraint):
        def evaluate(self, context):
            assert ("close", context.frame_index) not in events
            if outcome == "error":
                raise RuntimeError("filter failure")
            return FilterResult(outcome, "test")

    def prepare(ctx):
        if outcome == "prepare_error":
            raise RuntimeError("preparation failure")
        # A replacement may own additional resources. Both scopes must be closed.
        replacement = FilterContext(ctx.scene, renderer=ctx.renderer)
        replacement.own(Resource(f"extra-{ctx.scene.frame}"))
        return replacement

    source = DatasetFilter(Records(), prepare_context=prepare)
    iterator = source.scenarios(
        ScenarioRequirement("resources", constraints=[Check()], window=InitialWindow(2))
    )
    if outcome in {"error", "prepare_error"}:
        with pytest.raises(RuntimeError):
            next(iterator)
    elif outcome == "pass":
        case = next(iterator)
        assert case.make_backend() is shared
        # No need to resume or close the iterator before candidate resources are released.
        assert ("close", 1) in events
    else:
        assert list(iterator) == []
    opened = [name for event, name in events if event == "open"]
    closed = [name for event, name in events if event == "close"]
    assert all(closed.count(name) == 1 for name in opened)
    assert "borrowed" not in closed
    if outcome != "prepare_error":
        assert all(closed.count(f"extra-{name}") == 1 for name in opened)


def test_derived_context_shares_resource_scope(scene):
    from unittest.mock import Mock

    resource = Mock()
    with FilterContext(scene()) as ctx:
        assert ctx.own(resource) is resource
        derived = ctx.derive(metadata={"custom": True})
        derived.close()
    resource.close.assert_called_once_with()


def test_manual_and_automatic_roles_never_replace_explicit_choices(scene):
    req = ScenarioRequirement(
        "roles",
        roles={
            "host": RoleSpec(ids=("b",)),
            "attacker": RoleSpec(count=2),
        },
        constraints=[DistanceRange(5, 15)],
    )
    result = req.evaluate(scene())
    assert result.match.binding_ids == {"host": ("b",), "attacker": ("a", "b")}
    assert req.evaluate(scene(distances=(8,), ids=("a",))).match is None
    too_far = scene(distances=(8, 40))
    assert req.evaluate(too_far).status == "fail"
    assert req.evaluate(too_far).candidates[0].bindings["host"] == ("b",)


def test_distance_does_not_silently_use_insertion_position(scene, patch):
    inserted = Insertion("sign", patch().asset, WorldPlacement((8, 0, 1)))
    req = ScenarioRequirement(
        "no attacker", insertions=(inserted,), constraints=[DistanceRange(1, 20)]
    )
    assert req.evaluate(scene()).status == "unknown"
    req.roles = {"attacker": RoleSpec(ids=("b",))}
    assert req.evaluate(scene(distances=(8, 30))).status == "fail"


def test_filter_uses_patch_visibility_without_host_visibility_and_ranks_eligible_hosts(scene, patch):
    calls = []

    def measure(ctx, insertion_id, camera):
        resolved = ctx.resolved_insertions[insertion_id]
        calls.append((ctx.bindings["host"][0].track_id, insertion_id, camera))
        return Visibility(0.9 if resolved.pose[0, 3] > 8 else 0.2, "custom", camera)

    req = ScenarioRequirement(
        "patch",
        roles={"host": RoleSpec()},
        insertions=(patch(),),
        constraints=[InView("front"), MinVisibility(0.7), MinVisibility(0.8)],
    )
    result = req.evaluate(FilterContext(scene(), visibility_provider=measure))
    assert result.match.binding_ids == {"host": ("b",)}
    assert [candidate.status for candidate in result.candidates] == ["fail", "pass"]
    assert result.match.bindings["host"][0].visibility.fraction == 0
    assert calls == [("a", "patch", "front"), ("b", "patch", "front")]


def test_multiple_independent_insertions_bind_indexed_hosts(scene, patch):
    small = patch("small", "hosts[0]")
    large = replace(
        patch("large", "hosts[1]"),
        asset=PlaneAsset(np.full((5, 5, 4), [127, 127, 127, 255], np.uint8), 1.2, 0.8),
        orientation=Orientation("face_victim"),
    )
    resolved = []

    class Inspect(Constraint):
        def evaluate(self, ctx):
            a, b = ctx.resolved_insertions.values()
            resolved.extend((a, b))
            assert a.asset.width_m == 0.6 and b.asset.width_m == 1.2
            return FilterResult("pass")

    req = ScenarioRequirement(
        "two",
        roles={"hosts": RoleSpec(count=2)},
        insertions=(small, large),
        constraints=[Inspect()],
    )
    match = req.match(scene())
    assert match.binding_ids["hosts"] == ("a", "b")
    assert resolved[0].pose[0, 3] < resolved[1].pose[0, 3]
    assert match.insertions == (small, large)


def test_initial_window_keeps_identity_instead_of_choosing_a_new_nearest_actor(scene):
    frames = (scene(0, (8, 12)), scene(1, (30, 8)), scene(2, (8, 12)))
    req = ScenarioRequirement(
        "window",
        roles={"attacker": RoleSpec()},
        constraints=[DistanceRange(5, 15)],
        window=InitialWindow(3),
    )
    result = req.evaluate(FilterContext(frames[0], sequence=frames))
    assert result.match.binding_ids["attacker"] == ("b",)
    assert result.match.scene.frame == 0
    assert result.match.window.frames == 3
    missing = scene(1, (8,), ids=("a",))
    result = req.evaluate(FilterContext(frames[0], sequence=(frames[0], missing, frames[2])))
    assert result.match.binding_ids["attacker"] == ("a",)
    assert result.candidates[1].status == "unknown"


def test_initial_window_missing_or_repeated_frames_cannot_pass(scene):
    initial = scene()
    req = ScenarioRequirement("window", window=InitialWindow(2))
    assert req.evaluate(initial).status == "unknown"
    with pytest.raises(ValueError, match="strictly increasing"):
        req.evaluate(FilterContext(initial, sequence=(initial, initial)))
    with pytest.raises(ValueError, match="selected scene"):
        req.evaluate(FilterContext(initial, sequence=(scene(1), scene(2))))


def test_live_sequence_evaluates_all_bindings_before_advancing_native_world(scene):
    class World:
        frame = 0

        def custom_unwrapped_api(self):
            return self.frame

    world = World()
    calls = []
    initial = scene()

    def frames():
        for index in range(3):
            world.frame = index
            yield FilterContext(scene(index), native={"world": world})
        pytest.fail("The selector advanced beyond the requested initial window")

    class NativeCondition(Constraint):
        def evaluate(self, ctx):
            assert ctx.native["world"] is world
            assert ctx.native["world"].custom_unwrapped_api() == ctx.scene.frame
            calls.append((ctx.scene.frame, ctx.bindings["attacker"][0].track_id))
            return FilterResult("pass")

    req = ScenarioRequirement(
        "live",
        roles={"attacker": RoleSpec()},
        constraints=[NativeCondition()],
        window=InitialWindow(3),
    )
    result = req.evaluate(FilterContext(initial, initial_contexts=frames))
    assert result.match.binding_ids["attacker"] == ("a",)
    assert calls == [(i, actor) for i in range(3) for actor in ("a", "b")]
    assert world.frame == 2


def test_custom_dataset_filter_gets_original_metadata_and_lazy_renderer(scene):
    raw_metadata = {"unstandardized": object()}
    render_calls, filter_calls = [], []

    class Renderer:
        def custom_render(self, value):
            render_calls.append(value)
            return "image"

    renderer = Renderer()

    class Records(Dataset):
        def scenes(self):
            yield scene()

        def initial_sequence(self, initial, frames):
            return tuple(scene(i) for i in range(frames))

        def context(self, state):
            return FilterContext(
                state,
                dataset=self,
                renderer=renderer,
                metadata=raw_metadata,
                native={"arbitrary": raw_metadata["unstandardized"]},
            )

        def make_backend(self, state):
            return (state.frame, state.t)

    dataset = Records()

    class Custom(Constraint):
        def evaluate(self, ctx):
            assert ctx.dataset is dataset and ctx.metadata is raw_metadata
            assert ctx.native["arbitrary"] is raw_metadata["unstandardized"]
            assert ctx.renderer.custom_render(ctx.scene.frame) == "image"
            filter_calls.append(ctx.scene.frame)
            return FilterResult("pass")

    cases = list(
        DatasetFilter(dataset).scenarios(
            ScenarioRequirement("custom", constraints=[Custom()], window=InitialWindow(2))
        )
    )
    assert filter_calls == render_calls == [0, 1]
    assert cases[0].make_backend() == (0, 0.0)
    assert filter_calls == [0, 1]  # Runtime construction does not invoke filters.


def test_carla_uses_prepared_actual_scene_and_closes_preview_before_execution(scene):
    lifecycle, configs = [], []

    @contextmanager
    def provider(config):
        configs.append(config)
        lifecycle.append("open")
        # The configured estimate is near, but the actual spawned actor is farther away.
        state = scene(distances=(40,), ids=("lead",))
        state.source = {"backend": "carla", "live": True}
        yield FilterContext(state, native={"make_backend": lambda: tuple(lifecycle)})
        lifecycle.append("closed")

    fixed = {"lead": {"gap": 8, "lateral": 0}, "ego": {}}
    req = ScenarioRequirement(
        "actual", roles={"attacker": RoleSpec(ids=("lead",))}, constraints=[DistanceRange(5, 15)]
    )
    builder = CarlaScenarioBuilder(fixed, candidate_provider=provider)
    assert list(builder.scenarios(req)) == []
    assert builder.selection_log[0]["status"] == "fail"
    req.constraints = [DistanceRange(30, 50)]
    case = next(builder.scenarios(req))
    assert case.make_backend()[-1] == "closed"
    assert configs[0]["lead"] == fixed["lead"]
    assert case.match.bindings["attacker"][0].distance == 40


def test_requirement_roundtrips_roles_window_groups_and_independent_patch_assets(scene, patch):
    import json

    first = patch("rear", "hosts[0]")
    second = replace(
        patch("side", "hosts[1]"),
        asset=PlaneAsset(np.full((8, 12, 4), [20, 180, 40, 255], np.uint8), 1.2, 0.8),
        orientation=Orientation("fixed_world", (10, 20, 170)),
    )
    req = ScenarioRequirement(
        "serialized",
        roles={"hosts": RoleSpec(ids=("a", "b"), count=2)},
        insertions=(first, second),
        window=InitialWindow(3),
        constraints=[
            All(
                [
                    InView("front"),
                    Any([MinVisibility(0.8, subjects=("rear",)), Not(InView("rear"))]),
                ]
            )
        ],
    )
    encoded = requirement_to_dict(req)
    restored = requirement_from_dict(json.loads(json.dumps(encoded)))
    assert requirement_to_dict(restored) == encoded
    assert [item.placement.host for item in restored.insertions] == ["hosts[0]", "hosts[1]"]
    assert restored.window.frames == 3
    state = scene()
    bindings = {"hosts": tuple(state.objects)}
    before = FilterContext(state, bindings=bindings, insertions=req.insertions).resolved_insertions
    after = FilterContext(state, bindings=bindings, insertions=restored.insertions).resolved_insertions
    for key in before:
        assert before[key].host_id == after[key].host_id
        np.testing.assert_array_equal(before[key].pose, after[key].pose)
        np.testing.assert_array_equal(before[key].asset.texture, after[key].asset.texture)
    assert not np.array_equal(after["rear"].asset.texture, after["side"].asset.texture)


def test_role_search_budget_reports_incomplete_search_instead_of_infeasible_case(scene):
    req = ScenarioRequirement(
        "budget",
        roles={"attacker": RoleSpec()},
        constraints=[DistanceRange(10, 15)],
        max_bindings=1,
    )
    result = req.evaluate(scene())
    assert result.status == "unknown" and "budget" in result.reason
    req.max_bindings = 2
    assert req.evaluate(scene()).match.binding_ids["attacker"] == ("b",)


def test_nuscenes_context_exposes_unmodified_sdk_records(scene, patch):
    from avsectester.scenarios.datasets.nuscenes import NuScenesDataset

    records = {
        ("sample", "sample-a"): {"data": {"CAM_FRONT": "camera-a"}, "extra_field": object()},
        ("sample_data", "camera-a"): {
            "calibrated_sensor_token": "calib-a",
            "ego_pose_token": "ego-a",
        },
        ("calibrated_sensor", "calib-a"): {"native_calibration_field": 7},
        ("ego_pose", "ego-a"): {"native_pose_field": 8},
    }

    class SDK:
        def get(self, table, token):
            return records[table, token]

        def custom_sdk_method(self):
            return "unrestricted"

    dataset = NuScenesDataset("unused")
    dataset._nusc = SDK()
    state = scene()
    state.source = {"sample_token": "sample-a"}
    context = dataset.context(state)
    assert context.dataset is dataset
    assert context.metadata["sample"] is records["sample", "sample-a"]
    assert context.native["nusc"] is dataset._nusc
    assert context.native["nusc"].custom_sdk_method() == "unrestricted"
    context = context.derive(insertions=(patch(host="a"),))
    subject = context.subjects()[0]
    assert context.visibility(subject, "front") is None


def test_missing_world_pose_is_unknown_in_selection_and_rejected_on_direct_resolution(scene, patch):
    state = scene()
    state.ego.pose = None
    req = ScenarioRequirement("pose", insertions=(patch(host="a"),), constraints=[InView("front")])
    assert req.evaluate(state).status == "unknown"
    with pytest.raises(ValueError, match="world-from-ego"):
        FilterContext(state, insertions=req.insertions).resolved_insertions


def test_insertions_require_actual_camera_overlap_even_without_explicit_visual_filter(scene, patch):
    asset = patch().asset
    outside = Insertion("outside", asset, WorldPlacement((10, 100, 1)), Orientation("face_victim"))
    req = ScenarioRequirement("required", insertions=(outside,))
    result = req.evaluate(scene())
    assert result.status == "fail"
    assert result.candidates[0].checks[0][0].endswith("InView[required]")
    req.insertions = (patch(host="a"),)
    assert req.evaluate(scene()).status == "pass"


def test_insertion_camera_selection_is_explicit_when_multiple_nonfront_cameras_exist(scene, patch):
    state = scene()
    calibration = state.cameras.pop("front")
    state.cameras = {
        "left": replace(calibration, name="left"),
        "right": replace(calibration, name="right"),
    }
    req = ScenarioRequirement("camera", insertions=(patch(host="a"),))
    assert req.evaluate(state).status == "unknown"
    req.camera = "left"
    result = req.evaluate(state)
    assert result.status == "pass" and result.match.camera == "left"


def test_nuscenes_box_adapter_preserves_full_orientation_for_attached_objects():
    from types import SimpleNamespace
    from avsectester.insertion import rotation_matrix
    from avsectester.scenarios.datasets.nuscenes import scene_from_cam_boxes

    rotation = rotation_matrix((20, 15, 35))
    center = np.array([0, 0, 10.0])
    corners = center[:, None] + np.array(
        [
            [-1, -1, -1, -1, 1, 1, 1, 1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [-1, 1, -1, 1, -1, 1, -1, 1],
        ]
    )
    box = SimpleNamespace(
        name="vehicle.car",
        token="car",
        center=center,
        wlh=(2, 4, 2),
        orientation=SimpleNamespace(rotate=lambda vector: rotation @ vector),
        corners=lambda: corners,
    )
    state = scene_from_cam_boxes([box], np.array([[80, 0, 80], [0, 80, 60], [0, 0, 1]]), 160, 120)
    target = state.objects[0]
    np.testing.assert_allclose(
        target.pose[:3, :3], state.cameras["front"].cam_to_ego[:3, :3] @ rotation
    )
    np.testing.assert_allclose(target.pose[:3, 3], target.center)
