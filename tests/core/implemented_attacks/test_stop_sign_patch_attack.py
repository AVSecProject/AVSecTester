"""Stop-sign selection, patched replica placement and real offline optimization."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from avsectester.attack import AttackSegment, OfflineContext, OfflineReport, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.attack.optimization import HideBelowThreshold, NES, PGD
from avsectester.execution.services import ClosedLoopServices, ScenarioView
from avsectester.closed_loop.avstacks import DrivingAdapter, ScorerDescriptor
from avsectester.closed_loop.avstacks.implemented.modules import (
    RuleBasedAVStack, YOLOXDetector, rule_based_driving_adapter,
)
from avsectester.closed_loop.simulators.ego_dynamics import TrajectoryFollower
from avsectester.closed_loop.simulators.nurec import NuRecSimulator
from avsectester.closed_loop.timing import HistoryRequirement, TimingConfiguration
from tests.fakes import (
    CAMERA, ConstantAVStack, DeceleratingAVStack, MemoryDataset, MemoryImageRenderer,
    MemorySimulator, StopApproachDataset, StopCommandAVStack, stop_approach_map,
)
from avsectester.implemented_attacks.stop_sign_patch_attack import StopSignPatchAttack
from avsectester.implemented_attacks.stop_sign_patch_attack.offline import (
    DetectorPatchOptimization, RolloutPatchOptimization,
)
from avsectester.rendering import ExposureHarmonizer
from avsectester.runlog import RunLog


@pytest.fixture
def approach_view():
    services = []

    def create(*, line_kind="stop_sign", facing=True, stopped=True, past_steps=15,
               line_offset_y=0., linked_sign=True, sign_x=60.):
        scenario_map = stop_approach_map(
            line_kind=line_kind, facing=facing, line_offset_y=line_offset_y,
            linked_sign=linked_sign, sign_x=sign_x,
        )
        dataset = StopApproachDataset(scenario_map, stopped=stopped)
        adapter = DrivingAdapter(lambda seed: ConstantAVStack(), history=HistoryRequirement(samples=past_steps + 1))
        scope = ClosedLoopServices(dataset, MemorySimulator(), adapter)
        scope.__enter__()
        services.append(scope)
        return ScenarioView("scene", scope, 0)

    yield create
    for scope in reversed(services):
        scope.close()


@pytest.fixture
def nurec_patch_services_and_segment(nurec_longitudinal_dataset):
    """Moving recorded camera, real binding, and image-independent candidate driver."""
    dataset = nurec_longitudinal_dataset
    simulator = NuRecSimulator(
        MemoryImageRenderer(CAMERA), TrajectoryFollower(), depth="none",
        route_generator="none", physics_update_mode="none", object_occlusion="geometry",
        time_step=.1,
    )
    adapter = DrivingAdapter(lambda seed: DeceleratingAVStack(), history=HistoryRequirement(samples=16))
    frames = tuple(dataset.frames("scene"))
    route = np.asarray([dataset.frame_data("scene", frame).ego.pose[:3, 3] for frame in frames])

    def make_segment(*, start=15, stop=30, line_x=28.):
        sign_pose = np.diag([-1., -1., 1., 1.])
        sign_pose[:3, 3] = (28., -1., 1.8)
        segment = AttackSegment("scene", frames[start], frames[stop], {
            "sign_id": "patch-sign", "sign_pose": sign_pose,
            "sign_extent": [.05, .8, .8], "sign_native_category": "MUTCD_R1_STOP",
            "sign_lane_ids": [], "stop_line_id": "patch-line",
            "stop_line_points": np.array([[line_x, -3., 0.], [line_x, 3., 0.]]),
            "stop_line_lane_ids": [], "recorded_stop_frame": 30,
            "recorded_stop_timestamp_s": frames[30].timestamp,
            "approach_distance_m": 28. - route[start, 0], "route_points": route,
            "line_route_distance_m": line_x,
        })
        StopSignPatchAttack().validate_segment(segment)
        return segment

    with ClosedLoopServices(dataset, simulator, adapter) as services:
        yield services, make_segment


@pytest.fixture
def replica_detector_model():
    """Create differentiable YOLOX outputs for replica pixels on a constant background."""
    class ReplicaDetector(torch.nn.Module):
        """Model boundary whose confidence differentiates the patch's red channel."""

        def forward(self, images):
            predictions = []
            for image in images:
                padding = (image - 114).abs().amax(dim=0) < 1e-6
                rows = torch.nonzero(~padding.all(dim=1), as_tuple=True)[0]
                columns = torch.nonzero(~padding.all(dim=0), as_tuple=True)[0]
                box = image.new_zeros((1, 4))
                score = image.sum() * 0
                if len(rows) and len(columns):
                    height, width = int(rows[-1]) + 1, int(columns[-1]) + 1
                    content = image[:, :height, :width]
                    changed = (content - 30).abs().amax(dim=0) > 1e-6
                    ys, xs = torch.nonzero(changed, as_tuple=True)
                    if len(xs):
                        x0, x1 = int(xs.min()), int(xs.max()) + 1
                        y0, y1 = int(ys.min()), int(ys.max()) + 1
                        score = .5 + .4 * content[2, y0:y1, x0:x1].mean() / 255
                        box = image.new_tensor([[(x0 + x1) / 2, (y0 + y1) / 2,
                                                 x1 - x0, y1 - y0]])
                classes = image.new_zeros((1, 80))
                classes[0, 11] = 1.
                predictions.append(torch.cat((box, score.reshape(1, 1), classes), dim=-1))
            return torch.stack(predictions)

    return ReplicaDetector()


@pytest.fixture
def nurec_detector_services_and_segment(nurec_patch_services_and_segment, replica_detector_model):
    """Real rule-based scoring binding over the shared synthetic recorded camera world."""
    services, make_segment = nurec_patch_services_and_segment

    def factory(seed):
        return RuleBasedAVStack(YOLOXDetector(
            "unused.pth", device="cpu", score_threshold=0., model=replica_detector_model,
        ), camera_id="front")

    factory.avstack_options = {
        "camera_id": "front", "detector": YOLOXDetector("unused.pth", device="cpu", score_threshold=0.),
    }
    adapter = rule_based_driving_adapter(factory)
    with ClosedLoopServices(services.dataset, services.simulator, adapter) as scope:
        yield scope, make_segment


def test_segment_selection(approach_view):
    scenario = approach_view()
    attack = StopSignPatchAttack()
    segment, = attack.build_metadata(scenario)
    assert segment.start.position == 15 and segment.stop.position == 78
    assert segment.user_data["recorded_stop_frame"] == 58
    assert segment.user_data["approach_distance_m"] == pytest.approx(45.)
    assert segment.user_data["line_route_distance_m"] == pytest.approx(60.)
    assert segment.user_data["stop_line_lane_ids"] == ["lane"]
    np.testing.assert_array_equal(segment.user_data["sign_pose"],
                                  scenario.scenario_map().traffic_controls["sign"].pose)
    assert tuple(segment.user_data["sign_extent"]) == (.05, .8, .8)
    later, = attack.build_metadata(approach_view(past_steps=25))
    assert later.start.position == 25
    offset, = attack.build_metadata(approach_view(line_offset_y=4.))
    assert offset.start.position == 15
    assert not attack.build_metadata(approach_view(line_kind="traffic_light"))
    assert not attack.build_metadata(approach_view(facing=False))
    assert not attack.build_metadata(approach_view(stopped=False))
    assert not attack.build_metadata(approach_view(linked_sign=False, sign_x=75.))
    short, = StopSignPatchAttack(approach_m=3., min_approach_m=0., hold_steps=1).build_metadata(scenario)
    assert (short.start.position, short.stop.position) == (57, 59)
    # History pushes this start to 10 m before the line, closer than the default 20 m minimum.
    late = approach_view(past_steps=50)
    assert not attack.build_metadata(late)
    assert any(reason.startswith("Approach starts closer to the line than min_approach_m")
               for reason in late.reasons)
    near, = StopSignPatchAttack(min_approach_m=5.).build_metadata(late)
    assert near.start.position == 50
    with pytest.raises(ValueError, match="approach_m"):
        StopSignPatchAttack(approach_m=1., min_approach_m=2.)
    missing = dict(segment.user_data)
    del missing["route_points"]
    with pytest.raises(ValueError, match="route_points"):
        attack.validate_segment(replace(segment, user_data=missing))


@pytest.fixture
def replica_approach(approach_view):
    scenario = approach_view()
    segment, = StopSignPatchAttack().build_metadata(scenario)
    return scenario, segment


def test_initial_replica_textures(replica_approach):
    scenario, segment = replica_approach
    attack = StopSignPatchAttack()
    context = OfflineContext(scenario.services, (segment,), seed=0)
    textures = {kind: StopSignPatchAttack(initial_patch=kind).run_offline(context)
                for kind in ("none", "white", "black", "low_frequency_noise")}
    canonical = attack.replica_placement(segment, textures["none"])[0].surfaces[0].nominal_texture
    np.testing.assert_array_equal(textures["none"], canonical)
    # The default bars cover only the red background, never the white border or letters.
    assert len(np.unique(canonical[attack.mask], axis=0)) == 1
    for texture in textures.values():
        np.testing.assert_array_equal(texture[..., 3], canonical[..., 3])
        np.testing.assert_array_equal(texture[~attack.mask], canonical[~attack.mask])
    noise = textures["low_frequency_noise"][..., :3] / 255.
    assert abs(noise[attack.mask].mean() - .5) < .02 and .22 < noise[attack.mask].std() < .25
    # Adjacent texels share low-frequency colors.
    neighbors = attack.mask[:, 1:] & attack.mask[:, :-1]
    assert np.abs(np.diff(noise, axis=1))[neighbors].mean() < .03
    np.testing.assert_array_equal(textures["white"][..., :3][attack.mask], 255)
    np.testing.assert_array_equal(textures["black"][..., :3][attack.mask], 0)
    np.testing.assert_array_equal(textures["low_frequency_noise"], StopSignPatchAttack(
        initial_patch="low_frequency_noise", noise_seed=0).run_offline(context))
    assert not np.array_equal(textures["low_frequency_noise"], StopSignPatchAttack(
        initial_patch="low_frequency_noise", noise_seed=1).run_offline(context))


def test_replica_artifacts(replica_approach, tmp_path):
    scenario, segment = replica_approach
    attack = StopSignPatchAttack(initial_patch="white")
    texture = attack.run_offline(OfflineContext(scenario.services, (segment,), seed=0))
    saved = tmp_path / "artifacts"
    attack.save_artifacts(texture, saved)
    np.testing.assert_array_equal(attack.load_artifacts(saved), texture)
    Image.fromarray(texture[..., :3]).save(saved / "deployed_texture.png")
    with pytest.raises(ValueError, match="RGBA"):
        attack.load_artifacts(saved)


def test_replica_mask(tmp_path):
    attack = StopSignPatchAttack()
    mask = np.zeros(attack.mask.shape, bool)
    mask[200:260, 100:400] = True
    path = tmp_path / "mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    np.testing.assert_array_equal(StopSignPatchAttack(mask_path=path).mask, mask)
    invalid = [
        (np.zeros(attack.mask.shape, np.uint8), "nonempty and binary"),
        (np.full((4, 4), 255, np.uint8), "resolution"),
    ]
    for pixels, message in invalid:
        Image.fromarray(pixels).save(path)
        with pytest.raises(ValueError, match=message):
            StopSignPatchAttack(mask_path=path)


def test_replica_placement(replica_approach):
    scenario, segment = replica_approach
    attack = StopSignPatchAttack(offset_m=.10, width_scale=1.)
    texture = attack.run_offline(OfflineContext(scenario.services, (segment,), seed=0))
    asset, pose = attack.replica_placement(segment, texture)
    assert asset.extent[1] == pytest.approx(.762)
    scaled = StopSignPatchAttack()
    assert scaled.replica_placement(segment, texture)[0].extent[1] == pytest.approx(.762 * 1.1)
    mapped = StopSignPatchAttack(face_size="map_width", width_scale=1.)
    assert mapped.replica_placement(segment, texture)[0].extent[1] == pytest.approx(.8)
    control = scenario.scenario_map().traffic_controls["sign"]
    face_center = asset.surfaces[0].corners.mean(axis=0)
    np.testing.assert_allclose((pose @ np.r_[face_center, 1.])[:3],
                               control.pose[:3, 3] + .10 * control.pose[:3, 0])
    # A pitched annotation has a tilted up axis and a nonhorizontal face normal.
    yaw, pitch = .5, .2
    normal = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.])
    tilted_pose = control.pose.copy()
    tilted_pose[:3, :3] = np.column_stack((normal, side, np.cross(normal, side)))
    tilted_segment = replace(segment, user_data={**segment.user_data, "sign_pose": tilted_pose})
    _, upright_pose = attack.replica_placement(tilted_segment, texture)
    expected_normal = np.array([np.cos(yaw), np.sin(yaw), 0.])
    np.testing.assert_allclose(upright_pose[:3, 2], [0., 0., 1.])
    np.testing.assert_allclose(upright_pose[:3, 0], expected_normal)
    np.testing.assert_allclose(upright_pose[:3, 1],
                               np.cross([0., 0., 1.], expected_normal))
    np.testing.assert_allclose(
        (upright_pose @ np.r_[face_center, 1.])[:3],
        control.pose[:3, 3] + attack.offset_m * expected_normal,
    )
    map_attack = StopSignPatchAttack(face_orientation="map", width_scale=attack.width_scale)
    _, map_pose = map_attack.replica_placement(tilted_segment, texture)
    np.testing.assert_allclose(map_pose[:3, :3], tilted_pose[:3, :3])
    np.testing.assert_allclose(
        (map_pose @ np.r_[face_center, 1.])[:3],
        control.pose[:3, 3] + map_attack.offset_m * normal,
    )
    with pytest.raises(ValueError, match="upright or map"):
        StopSignPatchAttack(face_orientation="tilted")
    with pytest.raises(ValueError, match="mutcd_standard or map_width"):
        StopSignPatchAttack(face_size="arbitrary")
    vertical_pose = tilted_pose.copy()
    vertical_pose[:3, :3] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    with pytest.raises(ValueError, match="horizontal normal"):
        attack.replica_placement(
            replace(segment, user_data={**segment.user_data, "sign_pose": vertical_pose}), texture,
        )


def test_create_replica_runtime(replica_approach):
    scenario, segment = replica_approach
    harmonizer = ExposureHarmonizer(reference="under")
    attack = StopSignPatchAttack(initial_patch="none", harmonizer=harmonizer)
    canonical = attack.run_offline(OfflineContext(scenario.services, (segment,), seed=0))
    candidate = canonical.copy()
    candidate[..., :3][attack.mask] = (35, 85, 135)
    runtime = attack.create_runtime(segment, candidate)
    action, = runtime.setup(OnlineContext(0, "setup", 0., segment.start.position))
    assert isinstance(action, InsertObject) and action.object_id == "stop_sign_replica"
    assert action.harmonizer is harmonizer and action.asset.category == "stop_sign"
    np.testing.assert_array_equal(action.asset.surfaces[0].texture, candidate)
    np.testing.assert_array_equal(action.asset.surfaces[0].nominal_texture, canonical)
    assert len(action.asset.surfaces) == 2
    front, back = action.asset.surfaces
    expected_back = front.corners[[1, 0, 3, 2]].copy()
    expected_back[:, 0] = -.003
    np.testing.assert_array_equal(back.corners, expected_back)
    np.testing.assert_array_equal(back.texture[..., :3], 160)
    np.testing.assert_array_equal(back.texture[..., 3], front.texture[:, ::-1, 3])
    assert back.nominal_texture is None
    with pytest.raises(ValueError, match="shape"):
        attack.create_runtime(segment, np.zeros((2, 2, 4), np.uint8))
    changed_alpha = candidate.copy()
    changed_alpha[0, 0, 3] ^= 255
    with pytest.raises(ValueError, match="alpha"):
        attack.create_runtime(segment, changed_alpha)
    outside = candidate.copy()
    outside[..., :3][~attack.mask & (outside[..., 3] > 0)] = 0
    with pytest.raises(ValueError, match="inside the patch mask"):
        attack.create_runtime(segment, outside)


@pytest.mark.parametrize("maximum,expected_starts", [(None, [15, 16]), (1, [15])])
def test_white_box_segment_limit(nurec_detector_services_and_segment, maximum, expected_starts):
    services, make_segment = nurec_detector_services_and_segment
    attack = StopSignPatchAttack(optimization=DetectorPatchOptimization(
        optimizer=PGD(epochs=1, batch_size=4, seed=0), min_visible_pixels=5,
        braking_deceleration_mps2=10., max_training_segments=maximum, samples_per_scene=4,
    ))
    context = OfflineContext(services, (make_segment(), make_segment(start=16)), seed=0)
    attack.run_offline(context)
    ranges = context.report.tables["training_segment_ranges"].rows
    assert list(dict.fromkeys(row["selected_start"] for row in ranges)) == expected_starts
    assert context.report.values["training_segments"].value == len(expected_starts)


def test_white_box_offline_phase(nurec_detector_services_and_segment):
    services, make_segment = nurec_detector_services_and_segment
    segment = make_segment()
    attack = StopSignPatchAttack(
        initial_patch="white", harmonizer=ExposureHarmonizer(reference="under"),
        optimization=DetectorPatchOptimization(
            optimizer=PGD(epochs=3, batch_size=3, seed=0), min_visible_pixels=5,
            braking_deceleration_mps2=10., samples_per_scene=4,
        ),
    )
    context = OfflineContext(services, (segment,), seed=0,
                             validation_metadata=(make_segment(start=23),))
    texture = attack.run_offline(context)
    face = StopSignPatchAttack(initial_patch="none").run_offline(
        OfflineContext(services, (segment,), seed=0))
    np.testing.assert_array_equal(texture[..., 3], face[..., 3])
    np.testing.assert_array_equal(texture[..., :3][~attack.mask], face[..., :3][~attack.mask])
    report = context.report
    assert set(report.tables) == {"training_segment_ranges", "rendered_frames", "frame_detection_scores",
                                  "texture_detection_summary", "detection_rate_vs_distance", "training_progress"}
    assert set(report.images) == {"initial_texture", "optimized_texture"}
    progress = report.tables["training_progress"].rows
    bins = sorted(name for name in progress[0] if name.startswith("detection_rate_"))
    assert bins and set(report.series) == {"total_loss", "detection_loss", "smoothness_loss", "detection_rate"}
    assert set(report.values) == {"train_frames", "training_segments", "mask_pixels", "patch_smoothness",
                                  "detected_frame_fraction", "max_confidence"}
    ranges = report.tables["training_segment_ranges"].rows
    assert ranges[0]["d_min_m"] == pytest.approx(5.)
    with pytest.raises(ValueError, match="training segment"):
        attack.run_offline(replace(context, metadata=()))
    with pytest.raises(ValueError, match="max_training_segments"):
        replace(attack.optimization, max_training_segments=0)
    for samples in (3, "some", True):
        with pytest.raises(ValueError, match="samples_per_scene"):
            replace(attack.optimization, samples_per_scene=samples)
    samples = [row for row in report.tables["rendered_frames"].rows
               if row["stage"] == "sample" and row["status"] == "used"]
    rows = report.tables["frame_detection_scores"].rows
    # Every used sample render trains. The diagnostics score the same frames.
    assert [(row["scene"], row["frame"]) for row in rows] == [(row["scene"], row["frame"]) for row in samples]
    assert {key for row in rows for key in row} == {
        "scene", "frame", "line_distance_m", "confidence_clean", "confidence_initial",
        "confidence_optimized"}
    for row in rows:
        assert row["confidence_optimized"] < row["confidence_initial"]
    assert len(rows) == 4 and len(rows) % attack.optimization.optimizer.batch_size != 0
    margin = attack.optimization.confidence_margin
    # Training-set scores are recorded at epoch zero and after every epoch.
    assert [row["epoch"] for row in progress] == [0, 1, 2, 3]
    for row in progress:
        assert row["total_loss"] == row["detection_loss"] + row["smoothness_loss"]
    assert list(progress[0])[:6] == ["epoch", "total_loss", "detection_loss", "smoothness_loss",
                                  "detection_rate", "max_confidence"]
    assert progress[0]["detection_loss"] == pytest.approx(np.mean([row["confidence_initial"] + margin for row in rows]))
    assert progress[-1]["detection_rate"] == report.values["detected_frame_fraction"].value
    assert progress[-1]["max_confidence"] == pytest.approx(report.values["max_confidence"].value)
    summary = report.tables["texture_detection_summary"].rows
    assert {row["texture"] for row in summary} == {"clean", "initial", "optimized"}
    assert set(summary[0]) == {"texture", "frames", "mean_confidence", "max_confidence", "detection_rate"}
    assert all(row["frames"] == len(rows) == report.values["train_frames"].value for row in summary)
    optimized = next(row for row in summary if row["texture"] == "optimized")
    assert report.values["max_confidence"].value == optimized["max_confidence"]
    assert report.values["detected_frame_fraction"].value == optimized["detection_rate"]
    assert set(ranges[0]) == {
        "scene", "selected_start", "approach_speed_mps", "d_min_m", "d_max_m",
        "search_first_frame", "search_last_frame", "status", "error"}
    rendered = report.tables["rendered_frames"].rows
    assert set(rendered[0]) == {
        "scene", "frame", "timestamp_s", "distance_m", "line_distance_m", "stage", "visible_pixels",
        "status", "reason", "occlusion_source", "target_box_left_px", "target_box_top_px",
        "target_box_right_px", "target_box_bottom_px", "clean_score"}
    assert {row["status"] for row in rendered} <= {"used", "skipped"}
    assert report.images["initial_texture"].description
    assert report.values["detected_frame_fraction"].unit == "ratio (0–1)"
    assert report.series["detection_rate"].y_unit == "ratio (0–1)"
    for name in ("total_loss", "detection_loss", "smoothness_loss"):
        assert report.series[name].y_unit is None
        assert report.series[name].x == [0, 1, 2, 3]
    for table in report.tables.values():
        assert table.description
        assert set(table.columns) == {key for row in table.rows for key in row}
    options = replace(attack.optimization, scorer_id="rule_based.detection_confidence")
    assert StopSignPatchAttack(optimization=options).run_offline(
        replace(context, metadata=(segment,), report=OfflineReport(None))).dtype == np.uint8
    descriptor = ScorerDescriptor("image-score", {"images": "BCHW float tensor [0,1]"},
                                  {"scores": "scalar confidence"}, {"scores": "tensor"}, ("images",),
                                  sensor_id="front", decision_threshold=0.)
    rejected = [
        ([], "exactly one"),
        ([replace(descriptor, sensor_id=None)], "sensor_id and decision_threshold"),
    ]
    for descriptors, message in rejected:
        adapter = DrivingAdapter(lambda seed: ConstantAVStack(),
                                 scorers=[(item, lambda avstack, images, targets: images.mean(dim=(1, 2, 3)))
                                          for item in descriptors])
        with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as scope:
            invalid = StopSignPatchAttack(optimization=replace(options, scorer_id=None))
            with pytest.raises(ValueError, match=message):
                invalid.run_offline(OfflineContext(scope, ()))


def test_white_box_training_range(nurec_detector_services_and_segment):
    services, make_segment = nurec_detector_services_and_segment
    # The line lies 3 m before the sign, so line and sign distances select different frames.
    context = OfflineContext(services, (make_segment(line_x=25.),), seed=0)
    attack = StopSignPatchAttack(initial_patch="white", optimization=DetectorPatchOptimization(
        optimizer=PGD(epochs=1, batch_size=4, seed=0), min_visible_pixels=5, braking_deceleration_mps2=10.))
    attack.run_offline(context)
    segment_range, = context.report.tables["training_segment_ranges"].rows
    rows = sorted((row for row in context.report.tables["rendered_frames"].rows
                   if row["stage"] == "sample" and row["status"] == "used"), key=lambda row: row["frame"])
    frames = [row["frame"] for row in rows]
    lines = [row["line_distance_m"] for row in rows]
    # Every eligible frame trains, from d_max down to the last frame where braking_deceleration_mps2
    # can still stop before the line.
    assert frames == list(range(frames[0], frames[-1] + 1))
    assert context.report.values["train_frames"].value == len(rows)
    assert segment_range["d_min_m"] <= min(lines) and max(lines) <= segment_range["d_max_m"]
    assert min(lines) - segment_range["d_min_m"] <= max(np.abs(np.diff(lines))) + 1e-9
    assert len(rows) > 8


def fit_replica(services, segment, initial_patch, optimizer):
    attack = StopSignPatchAttack(initial_patch=initial_patch, optimization=DetectorPatchOptimization(
        optimizer=optimizer, min_visible_pixels=5, braking_deceleration_mps2=10.,
        samples_per_scene=4,
    ))
    context = OfflineContext(services, (segment,), seed=0)
    return attack.run_offline(context), context.report, attack.optimization


def test_white_box_fixed_patch_scores(nurec_detector_services_and_segment):
    services, make_segment = nurec_detector_services_and_segment
    optimizer = PGD(epochs=3, batch_size=3, seed=0, step_size=0.)
    _, report, options = fit_replica(services, make_segment(), "low_frequency_noise", optimizer)
    descriptor, = services.capabilities.scorers
    objective = HideBelowThreshold(descriptor.decision_threshold, options.confidence_margin)
    initial = np.array([row["confidence_initial"]
                        for row in report.tables["frame_detection_scores"].rows])
    expected = float(objective.loss(initial).mean())
    progress = report.tables["training_progress"].rows
    assert [row["detection_loss"] for row in progress] == pytest.approx([expected] * 4)


def test_white_box_smoothness(nurec_detector_services_and_segment):
    services, make_segment = nurec_detector_services_and_segment
    segment = make_segment()
    optimizer = PGD(epochs=3, batch_size=3, seed=0, step_size=0., tv_weight=100.)
    _, uniform, _ = fit_replica(services, segment, "white", optimizer)
    assert uniform.values["patch_smoothness"].value == pytest.approx(0., abs=1e-6)
    _, noisy, _ = fit_replica(services, segment, "low_frequency_noise", optimizer)
    smoothness = noisy.values["patch_smoothness"].value
    assert smoothness > 0.
    progress = noisy.tables["training_progress"].rows
    assert progress[-1]["smoothness_loss"] == pytest.approx(optimizer.tv_weight * smoothness)

    ordinary, _, _ = fit_replica(
        services, segment, "low_frequency_noise", PGD(epochs=3, batch_size=3, seed=0))
    regularized, _, _ = fit_replica(
        services, segment, "low_frequency_noise",
        PGD(epochs=3, batch_size=3, seed=0, tv_weight=100.))
    assert not np.array_equal(ordinary, regularized)


@pytest.mark.parametrize("maximum,expected_segments", [
    pytest.param(None, 2, id="all-training"),
    pytest.param(1, 1, id="capped-training"),
])
@pytest.mark.parametrize("image_dependent,expected_unchanged", [
    pytest.param(False, True, id="image-independent"),
    pytest.param(True, False, id="image-dependent"),
])
def test_black_box_offline_phase(
    nurec_patch_services_and_segment, tmp_path, maximum, expected_segments,
    image_dependent, expected_unchanged,
):
    original_services, make_segment = nurec_patch_services_and_segment
    training = (make_segment(), make_segment(start=16))
    options = RolloutPatchOptimization(
        NES(steps=1, population_size=2, learning_rate=2000., seed=0, batch_size=2),
        max_training_segments=maximum, min_stop_command_step=2,
    )
    attack = StopSignPatchAttack(initial_patch="white", optimization=options)
    adapter = DrivingAdapter(lambda seed: StopCommandAVStack(image_dependent=image_dependent),
                             history=HistoryRequirement(samples=16))
    with ClosedLoopServices(original_services.dataset, original_services.simulator, adapter) as services:
        context = OfflineContext(
            services, training, seed=0, validation_metadata=(make_segment(start=23),),
            report=OfflineReport(tmp_path / "files"),
        )
        texture = attack.run_offline(context)
        report = context.report
        rows = report.tables["optimization_rollouts"].rows
        eligibility = report.tables["training_segment_eligibility"].rows
        assert len(eligibility) == expected_segments
        assert all(row["retained"] for row in eligibility)
        assert all(row["run_log"].startswith("eligibility_rollouts/") for row in eligibility)
        assert len(rows) == expected_segments * (1 + options.optimizer.steps
                                                 * (options.optimizer.population_size + 1))
        assert [row["query_index"] for row in rows] == [
            query for query in range(4) for _ in range(expected_segments)]
        assert report.values["training_segments"].value == expected_segments
        assert report.values["optimization_rollout_count"].value == len(rows)
        assert {row["status"] for row in rows} == {"completed"}
        for row in [*eligibility, *rows]:
            log = RunLog.read(report.directory / row["run_log"])
            assert log.status == "completed"
            assert row["outcome"] == "stop_command"
            assert len(log.steps) == row["decision_steps"] == row["stop_command_step"] + 1
            assert len(log.steps) < training[row["segment_index"]].decision_steps
            minima = []
            for step in log.steps:
                trajectory = step.produced_command.trajectory
                times = np.asarray([point[2] for point in trajectory]) / 1e6
                positions = np.asarray([point[0][:2] for point in trajectory])
                minima.append(np.min(np.linalg.norm(np.diff(positions, axis=0), axis=1)
                                     / np.diff(times)))
            assert all(value >= options.stop_speed_mps for value in minima[:-1])
            assert minima[-1] < options.stop_speed_mps
            sign = training[row["segment_index"]].user_data["sign_pose"]
            position = log.steps[-1].physical_pre.ego.pose[:3, 3]
            assert (position - sign[:3, 3]) @ sign[:3, 0] > 0
            assert row["stop_command_distance_m"] == pytest.approx(
                np.linalg.norm(position[:2] - sign[:2, 3]))
            assert row["elapsed_s"] > 0
        assert report.values["final_stop_command_distance_m"].unit == "m"
        assert report.values["final_stop_command_distance_m"].value == pytest.approx(
            np.mean([row["stop_command_distance_m"] for row in rows[-expected_segments:]]))
        assert all(row["run_log"].startswith("optimization_rollouts/") for row in rows)
        assert set(report.tables["optimization_rollouts"].columns) == set(rows[0])
        assert len(report.series["stop_command_distance_m"].y) == 2
        assert report.values["final_stop_command_distance_m"].value == pytest.approx(
            report.series["stop_command_distance_m"].y[-1])
        np.testing.assert_array_equal(texture, report.images["optimized_texture"].pixels)
        np.testing.assert_array_equal(texture[..., 3], report.images["initial_texture"].pixels[..., 3])
        np.testing.assert_array_equal(texture[~attack.mask],
                                      report.images["initial_texture"].pixels[~attack.mask])
        assert np.array_equal(texture, report.images["initial_texture"].pixels) == expected_unchanged


def test_black_box_force_gt(nurec_patch_services_and_segment, tmp_path):
    original, make_segment = nurec_patch_services_and_segment
    attack = StopSignPatchAttack(optimization=RolloutPatchOptimization(
        NES(steps=1, population_size=2, learning_rate=1., seed=0), min_stop_command_step=2))
    adapter = DrivingAdapter(lambda seed: StopCommandAVStack(stop_after=0),
                             history=HistoryRequirement(samples=16))
    with ClosedLoopServices(original.dataset, original.simulator, adapter,
            timing=TimingConfiguration(force_gt_duration_us=200000,
                                       skip_driver_during_force_gt=True)) as services:
        report = OfflineReport(tmp_path / "offline")
        attack.run_offline(OfflineContext(services, (make_segment(),), seed=0, report=report))
    rows = [*report.tables["training_segment_eligibility"].rows,
            *report.tables["optimization_rollouts"].rows]
    assert rows and all(row["stop_command_step"] == 2 for row in rows)
    for row in rows:
        log = RunLog.read(report.directory / row["run_log"])
        assert [step.force_gt for step in log.steps] == [True, True, False]
        assert all(step.produced_command is None for step in log.steps[:2])
        assert log.steps[-1].produced_command is not None


def test_black_box_eligibility_and_errors(nurec_patch_services_and_segment, tmp_path):
    original_services, make_segment = nurec_patch_services_and_segment
    options = RolloutPatchOptimization(
        NES(steps=1, population_size=2, seed=0), max_training_segments=1,
        min_stop_command_step=2,
    )
    attack = StopSignPatchAttack(initial_patch="white", optimization=options)
    adapter = DrivingAdapter(
        lambda seed: StopCommandAVStack(stop_by_start={15: None, 16: 0}), history=HistoryRequirement(samples=16))
    with ClosedLoopServices(original_services.dataset, original_services.simulator, adapter) as services:
        # Starting at 15 never stops: the full segment passes the sign, the short one ends first.
        context = OfflineContext(
            services, (make_segment(), make_segment(stop=20), make_segment(start=16),
                       make_segment(start=17)),
            report=OfflineReport(tmp_path / "files"),
        )
        texture = attack.run_offline(context)
        rows = context.report.tables["training_segment_eligibility"].rows
        assert [row["retained"] for row in rows] == [False, False, False, True]
        assert [row["outcome"] for row in rows] == [
            "passed_sign", "segment_ended", "stop_command", "stop_command"]
        assert [row["stop_command_step"] for row in rows] == [None, None, 0, 3]
        sign = context.metadata[0].user_data["sign_pose"]
        passed = RunLog.read(context.report.directory / rows[0]["run_log"])
        assert (passed.steps[-1].physical_post.ego.pose[:3, 3] - sign[:3, 3]) @ sign[:3, 0] <= 0
        assert rows[0]["stop_command_distance_m"] == 0
        assert rows[0]["decision_steps"] < context.metadata[0].decision_steps
        ended = RunLog.read(context.report.directory / rows[1]["run_log"])
        final = ended.steps[-1].physical_post.ego.pose[:3, 3]
        assert (final - sign[:3, 3]) @ sign[:3, 0] > 0
        assert rows[1]["decision_steps"] == context.metadata[1].decision_steps
        assert rows[1]["stop_command_distance_m"] == pytest.approx(np.hypot(28. - 20., 1.))
        assert "No stop command before passing" in rows[0]["reason"]
        assert "Segment ends before" in rows[1]["reason"]
        assert "before min_stop_command_step" in rows[2]["reason"]
        assert context.report.values["training_segments"].value == 1
        assert len(context.report.tables["optimization_rollouts"].rows) == 4
        np.testing.assert_array_equal(texture, context.report.images["optimized_texture"].pixels)
        in_memory = replace(context, report=OfflineReport(None))
        attack.run_offline(in_memory)
        assert all(row["run_log"] is None for row in in_memory.report.tables["optimization_rollouts"].rows)
        with pytest.raises(ValueError, match="training segment"):
            attack.run_offline(replace(context, metadata=()))
        rejected = replace(context, metadata=context.metadata[:3], report=OfflineReport(None))
        with pytest.raises(ValueError, match="No training segments have an eligible stop command"):
            attack.run_offline(rejected)
        assert all(not row["retained"] for row in rejected.report.tables["training_segment_eligibility"].rows)
    invalid_adapter = DrivingAdapter(
        lambda seed: StopCommandAVStack(invalid_timestamps=True), history=HistoryRequirement(samples=16))
    with ClosedLoopServices(original_services.dataset, original_services.simulator, invalid_adapter) as services:
        failed = OfflineContext(
            services, (make_segment(),), report=OfflineReport(tmp_path / "failed/files"))
        with pytest.raises(ValueError, match="timestamps must increase strictly"):
            attack.run_offline(failed)
        row, = failed.report.tables["training_segment_eligibility"].rows
        assert row["status"] == "failed"
        assert row["stop_command_distance_m"] is None
        assert "timestamps must increase strictly" in row["error"]
        log = RunLog.read(failed.report.directory / row["run_log"])
        assert log.status == "failed"
        assert row["decision_steps"] == len(log.steps) == 1
        assert "optimized_texture" not in failed.report.images
    with pytest.raises(ValueError, match="max_training_segments"):
        replace(options, max_training_segments=0)
    with pytest.raises(ValueError, match="stop_speed_mps"):
        replace(options, stop_speed_mps=0.)
    with pytest.raises(ValueError, match="min_stop_command_step"):
        replace(options, min_stop_command_step=-1)
    with pytest.raises(ValueError, match="cannot be optimized"):
        StopSignPatchAttack(initial_patch="none", optimization=options)
