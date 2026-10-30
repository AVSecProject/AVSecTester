"""Stop-sign selection, patched replica placement and real offline optimization."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from avsectester.attack import OfflineContext, OfflineReport, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.attack.optimization import HideBelowThreshold, NES, PGD
from avsectester.execution.services import ClosedLoopServices, RenderRequest
from avsectester.closed_loop.avstacks import DrivingAdapter, ScorerDescriptor
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, StopCommandAVStack
from avsectester.geometry.projection import project_box
from avsectester.implemented_attacks.stop_sign_patch_attack import StopSignPatchAttack
from avsectester.implemented_attacks.stop_sign_patch_attack.offline import (
    DetectorPatchOptimization, RolloutPatchOptimization,
)
from avsectester.rendering import ExposureHarmonizer
from avsectester.runlog import RunLog


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


def test_replica_deployment(approach_view, tmp_path):
    harmonizer = ExposureHarmonizer(reference="under")
    attack = StopSignPatchAttack(offset_m=.10, width_scale=1., harmonizer=harmonizer)
    scenario = approach_view()
    segment, = attack.build_metadata(scenario)
    context = OfflineContext(scenario.services, (segment,), seed=0)
    textures = {kind: StopSignPatchAttack(initial_patch=kind).run_offline(context)
                for kind in ("none", "white", "black", "low_frequency_noise")}
    canonical = attack.replica_placement(context.metadata[0], textures["none"])[0].surfaces[0].nominal_texture
    np.testing.assert_array_equal(textures["none"], canonical)
    assert sha256(textures["low_frequency_noise"].tobytes()).hexdigest() == (
        "9f44947df22b16eb5964c741c2711c3ebac127ef8a651f58e5de0ac95f9d8376")
    # The default bars cover only the red background, never the white border or letters.
    assert len(np.unique(canonical[attack.mask], axis=0)) == 1
    for texture in textures.values():
        np.testing.assert_array_equal(texture[..., 3], canonical[..., 3])
        np.testing.assert_array_equal(texture[~attack.mask], canonical[~attack.mask])
    noise = textures["low_frequency_noise"][..., :3] / 255.
    assert abs(noise[attack.mask].mean() - .5) < .02 and .22 < noise[attack.mask].std() < .25
    # Adjacent texels share low-frequency colors; independent per-texel noise differs by about .27.
    neighbors = attack.mask[:, 1:] & attack.mask[:, :-1]
    assert np.abs(np.diff(noise, axis=1))[neighbors].mean() < .03
    np.testing.assert_array_equal(textures["white"][..., :3][attack.mask], 255)
    np.testing.assert_array_equal(textures["black"][..., :3][attack.mask], 0)
    np.testing.assert_array_equal(textures["low_frequency_noise"], StopSignPatchAttack(
        initial_patch="low_frequency_noise", noise_seed=0).run_offline(context))
    assert not np.array_equal(textures["low_frequency_noise"], StopSignPatchAttack(
        initial_patch="low_frequency_noise", noise_seed=1).run_offline(context))
    # Artifacts persist as a lossless RGBA PNG, and loading validates the replica face.
    saved = tmp_path / "artifacts"
    attack.save_artifacts(textures["low_frequency_noise"], saved)
    assert [path.name for path in saved.iterdir()] == ["deployed_texture.png"]
    np.testing.assert_array_equal(attack.load_artifacts(saved), textures["low_frequency_noise"])
    Image.fromarray(textures["white"][..., :3]).save(saved / "deployed_texture.png")
    with pytest.raises(ValueError, match="RGBA"):
        attack.load_artifacts(saved)
    texture = textures["white"]
    mask = np.zeros(canonical.shape[:2], bool)
    mask[200:260, 100:400] = True
    path = tmp_path / "mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    np.testing.assert_array_equal(StopSignPatchAttack(mask_path=path).mask, mask)
    invalid = [
        (np.zeros(canonical.shape[:2], np.uint8), "nonempty and binary"),
        (np.full((4, 4), 255, np.uint8), "resolution"),
    ]
    for pixels, message in invalid:
        Image.fromarray(pixels).save(path)
        with pytest.raises(ValueError, match=message):
            StopSignPatchAttack(mask_path=path)
    candidate = texture.copy()
    candidate[..., :3][attack.mask] = (35, 85, 135)
    actions = attack.create_runtime(segment, candidate).setup(
        OnlineContext(0, "setup", 0., segment.start.position))
    assert len(actions) == 1 and isinstance(actions[0], InsertObject)
    action = actions[0]
    assert action.harmonizer is harmonizer
    assert action.object_id == "stop_sign_replica" and action.asset.category == "stop_sign"
    assert action.asset.extent[1] == pytest.approx(.762)
    assert StopSignPatchAttack().create_runtime(segment, candidate).setup(
        OnlineContext(0, "setup", 0., segment.start.position))[0].asset.extent[1] == pytest.approx(.762 * 1.1)
    mapped = StopSignPatchAttack(face_size="map_width", width_scale=1.)
    mapped_action, = mapped.create_runtime(segment, candidate).setup(
        OnlineContext(0, "setup", 0., segment.start.position))
    assert mapped_action.asset.extent[1] == pytest.approx(.8)
    np.testing.assert_array_equal(action.asset.surfaces[0].texture, candidate)
    face = canonical
    np.testing.assert_array_equal(action.asset.surfaces[0].nominal_texture, face)
    control = scenario.scenario_map().traffic_controls["sign"]
    face_center = action.asset.surfaces[0].corners.mean(axis=0)
    np.testing.assert_allclose((action.world_from_object @ np.r_[face_center, 1.])[:3],
                               control.pose[:3, 3] + .10 * control.pose[:3, 0])
    # A pitched annotation has a tilted up axis and a nonhorizontal face normal.
    yaw, pitch = .5, .2
    normal = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.])
    tilted_pose = control.pose.copy()
    tilted_pose[:3, :3] = np.column_stack((normal, side, np.cross(normal, side)))
    tilted_segment = replace(segment, user_data={**segment.user_data, "sign_pose": tilted_pose})
    upright_action, = attack.create_runtime(tilted_segment, candidate).setup(
        OnlineContext(0, "setup", 0., segment.start.position))
    expected_normal = np.array([np.cos(yaw), np.sin(yaw), 0.])
    np.testing.assert_allclose(upright_action.world_from_object[:3, 2], [0., 0., 1.])
    np.testing.assert_allclose(upright_action.world_from_object[:3, 0], expected_normal)
    np.testing.assert_allclose(upright_action.world_from_object[:3, 1],
                               np.cross([0., 0., 1.], expected_normal))
    np.testing.assert_allclose(
        (upright_action.world_from_object @ np.r_[face_center, 1.])[:3],
        control.pose[:3, 3] + attack.offset_m * expected_normal,
    )
    map_attack = StopSignPatchAttack(face_orientation="map", width_scale=attack.width_scale)
    map_action, = map_attack.create_runtime(tilted_segment, candidate).setup(
        OnlineContext(0, "setup", 0., segment.start.position))
    np.testing.assert_allclose(map_action.world_from_object[:3, :3], tilted_pose[:3, :3])
    np.testing.assert_allclose(
        (map_action.world_from_object @ np.r_[face_center, 1.])[:3],
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
            replace(segment, user_data={**segment.user_data, "sign_pose": vertical_pose}), candidate,
        )
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


def test_white_box_offline_phase(nurec_detector_services_and_segment):
    services, make_segment = nurec_detector_services_and_segment
    segment = make_segment()
    training = (segment, make_segment(start=16))
    for max_training_segments, expected_starts in ((None, [15, 16]), (1, [15])):
        attack = StopSignPatchAttack(
            initial_patch="white", harmonizer=ExposureHarmonizer(reference="under"),
            optimization=DetectorPatchOptimization(
                optimizer=PGD(epochs=3, batch_size=3, seed=0), min_visible_pixels=5,
                braking_deceleration_mps2=10., max_training_segments=max_training_segments,
                samples_per_scene=4,
            ),
        )
        context = OfflineContext(services, training, seed=0,
                                 validation_metadata=(make_segment(start=23),))
        texture = attack.run_offline(context)
        ranges = context.report.tables["training_segment_ranges"].rows
        assert list(dict.fromkeys(row["selected_start"] for row in ranges)) == expected_starts
        assert context.report.values["training_segments"].value == len(expected_starts)
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
    initial = report.images["initial_texture"].pixels
    assert report.images["initial_texture"].description
    assert report.values["detected_frame_fraction"].unit == "ratio (0–1)"
    assert report.series["detection_rate"].y_unit == "ratio (0–1)"
    for name in ("total_loss", "detection_loss", "smoothness_loss"):
        assert report.series[name].y_unit is None
        assert report.series[name].x == [0, 1, 2, 3]
    for table in report.tables.values():
        assert table.description
        assert set(table.columns) == {key for row in table.rows for key in row}
    asset, pose = attack.replica_placement(segment, initial)
    frames = tuple(services.dataset.frames(segment.scenario_id))
    with services.session(0) as session:
        for row in samples:
            rendered = session.render(RenderRequest(segment.scenario_id, frames[int(row["frame"])],
                "front", actions=[InsertObject("stop_sign_replica", asset, pose, harmonizer=attack.harmonizer)]))
            view = rendered.view
            expected = project_box(view.camera, view.cam_from_world, segment.user_data['sign_pose'], segment.user_data['sign_extent'])
            np.testing.assert_allclose([row[f"target_box_{coordinate}_px"] for coordinate in ("left", "top", "right", "bottom")], expected)
            ys, xs = np.nonzero(view.composition.plan.visible_mask)
            replica_bounds = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            assert not np.allclose(expected, replica_bounds)
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


def test_white_box_smoothness(nurec_detector_services_and_segment):
    services, make_segment = nurec_detector_services_and_segment
    context = OfflineContext(services, (make_segment(),), seed=0)

    def optimize(initial_patch, **pgd):
        attack = StopSignPatchAttack(initial_patch=initial_patch, optimization=DetectorPatchOptimization(
            optimizer=PGD(epochs=3, batch_size=3, seed=0, **pgd), min_visible_pixels=5,
            braking_deceleration_mps2=10., samples_per_scene=4))
        texture = attack.run_offline(context)
        if pgd.get("step_size") == 0.:
            descriptor, = services.capabilities.scorers
            objective = HideBelowThreshold(
                descriptor.decision_threshold, attack.optimization.confidence_margin)
            initial = np.array([row["confidence_initial"]
                                for row in context.report.tables["frame_detection_scores"].rows])
            expected = float(objective.loss(initial).mean())
            assert [row["detection_loss"] for row in context.report.tables["training_progress"].rows] == (
                pytest.approx([expected] * 4))
        smoothness = context.report.values["patch_smoothness"].value
        assert context.report.tables["training_progress"].rows[-1]["smoothness_loss"] == pytest.approx(
            attack.optimization.optimizer.tv_weight * smoothness)
        return texture, smoothness

    mask = StopSignPatchAttack().mask
    texture, smoothness = optimize("low_frequency_noise", step_size=0.)
    colors = texture[..., :3] / 255.
    squared = np.concatenate([(np.diff(colors, axis=0) ** 2).sum(-1)[mask[1:] & mask[:-1]],
                              (np.diff(colors, axis=1) ** 2).sum(-1)[mask[:, 1:] & mask[:, :-1]]])
    assert smoothness == pytest.approx((np.sqrt(squared + 1e-6) - 1e-3).mean(), rel=1e-4)
    # A uniform patch is perfectly smooth: its edge against the red face is the sticker outline.
    assert optimize("white", step_size=0.)[1] == pytest.approx(0., abs=1e-6)
    assert not np.array_equal(optimize("low_frequency_noise")[0],
                              optimize("low_frequency_noise", tv_weight=100.)[0])


@pytest.mark.parametrize("maximum,image_dependent,expected_unchanged", [
    pytest.param(None, False, True, id="all-training"),
    pytest.param(1, True, False, id="capped-image-dependent"),
])
def test_black_box_offline_phase(
    nurec_patch_services_and_segment, tmp_path, maximum, image_dependent, expected_unchanged,
):
    original_services, make_segment = nurec_patch_services_and_segment
    training = (make_segment(), make_segment(start=16))
    options = RolloutPatchOptimization(
        NES(steps=1, population_size=2, learning_rate=2000., seed=0, batch_size=2),
        max_training_segments=maximum, min_stop_command_step=2,
    )
    attack = StopSignPatchAttack(initial_patch="white", optimization=options)
    adapter = DrivingAdapter(lambda seed: StopCommandAVStack(image_dependent=image_dependent),
                             past_steps=15)
    with ClosedLoopServices(original_services.dataset, original_services.simulator, adapter) as services:
        context = OfflineContext(
            services, training, seed=0, validation_metadata=(make_segment(start=23),),
            report=OfflineReport(tmp_path / "files"),
        )
        texture = attack.run_offline(context)
        report = context.report
        rows = report.tables["optimization_rollouts"].rows
        eligibility = report.tables["training_segment_eligibility"].rows
        expected_segments = maximum or len(training)
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


def test_black_box_eligibility_and_errors(nurec_patch_services_and_segment, tmp_path):
    original_services, make_segment = nurec_patch_services_and_segment
    options = RolloutPatchOptimization(
        NES(steps=1, population_size=2, seed=0), max_training_segments=1,
        min_stop_command_step=2,
    )
    attack = StopSignPatchAttack(initial_patch="white", optimization=options)
    adapter = DrivingAdapter(
        lambda seed: StopCommandAVStack(stop_by_start={15: None, 16: 0}), past_steps=15)
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
        lambda seed: StopCommandAVStack(invalid_timestamps=True), past_steps=15)
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
