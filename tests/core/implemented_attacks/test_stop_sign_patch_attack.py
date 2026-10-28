"""Stop-sign selection, patched replica placement and real offline optimization."""
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from avsectester.attack import OfflineContext, OfflineReport, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.attack.optimization import NES, PGD
from avsectester.execution.services import ClosedLoopServices, RenderRequest
from avsectester.closed_loop.avstacks import DrivingAdapter, ScorerDescriptor
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, DeceleratingAVStack, ImageForwardAVStack
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
                optimizer=PGD(steps=3, batch=1, seed=0), min_visible_pixels=5,
                braking_deceleration_mps2=10., max_training_segments=max_training_segments,
                samples_per_scene=4,
            ),
        )
        context = OfflineContext(services, training, seed=0,
                                 validation_metadata=(make_segment(start=23),))
        texture = attack.run_offline(context)
        calibration = context.report.tables["calibration"].rows
        assert list(dict.fromkeys(row["selected_start"] for row in calibration)) == expected_starts
        assert context.report.values["training_segments"].value == len(expected_starts)
    face = StopSignPatchAttack(initial_patch="none").run_offline(
        OfflineContext(services, (segment,), seed=0))
    np.testing.assert_array_equal(texture[..., 3], face[..., 3])
    np.testing.assert_array_equal(texture[..., :3][~attack.mask], face[..., :3][~attack.mask])
    report = context.report
    assert set(report.tables) == {"calibration", "calibration_probes", "sample_manifest", "frame_scores",
                                  "texture_summary", "detection_rate_vs_distance", "training_progress"}
    assert set(report.images) == {"initial_texture", "optimized_texture"}
    progress = report.tables["training_progress"].rows
    bins = sorted(name for name in progress[0] if name.startswith("detection_rate_"))
    assert bins and set(report.series) == {"pgd_objective", "training_hinge_loss", "training_detection_rate",
                                           *(f"training_{name}" for name in bins)}
    assert set(report.values) == {"train_frames", "training_segments", "mask_pixels", "patch_smoothness",
                                  "detected_frame_fraction", "max_confidence"}
    calibration = report.tables["calibration"].rows
    assert calibration[0]["d_min_m"] == pytest.approx(5.)
    with pytest.raises(ValueError, match="training segment"):
        attack.run_offline(replace(context, metadata=()))
    with pytest.raises(ValueError, match="max_training_segments"):
        replace(attack.optimization, max_training_segments=0)
    for samples in (3, "some", True):
        with pytest.raises(ValueError, match="samples_per_scene"):
            replace(attack.optimization, samples_per_scene=samples)
    samples = report.tables["sample_manifest"].rows
    rows = report.tables["frame_scores"].rows
    # Every captured frame trains; the diagnostics score the same frames.
    assert [(row["scene"], row["frame"]) for row in rows] == [(row["scene"], row["frame"]) for row in samples]
    assert {key for row in rows for key in row} == {
        "scene", "frame", "line_distance_m", "confidence_clean", "confidence_replica", "confidence_initial",
        "confidence_optimized"}
    for row in rows:
        assert row["confidence_optimized"] < row["confidence_initial"]
    # The first PGD objective is the hinge of one sampled frame's initial-patch confidence; the
    # fixture's decision threshold is zero, so the hinge target is minus the margin.
    margin = attack.optimization.confidence_margin
    first = report.series["pgd_objective"].y[0]
    assert any(first == pytest.approx(row["confidence_initial"] + margin) for row in rows)
    # Training-set scores are recorded before the first update and after the last one.
    assert [row["step"] for row in progress] == [0, 3]
    assert progress[0]["hinge_loss"] == pytest.approx(np.mean([row["confidence_initial"] + margin for row in rows]))
    assert progress[-1]["detection_rate"] == report.values["detected_frame_fraction"].value
    assert progress[-1]["max_confidence"] == pytest.approx(report.values["max_confidence"].value)
    summary = report.tables["texture_summary"].rows
    assert {row["texture"] for row in summary} == {"clean", "replica", "initial", "optimized"}
    assert set(summary[0]) == {"texture", "frames", "mean_confidence", "max_confidence", "detection_rate"}
    assert all(row["frames"] == len(rows) == report.values["train_frames"].value for row in summary)
    optimized = next(row for row in summary if row["texture"] == "optimized")
    assert report.values["max_confidence"].value == optimized["max_confidence"]
    assert report.values["detected_frame_fraction"].value == optimized["detection_rate"]
    assert set(calibration[0]) == {
        "scene", "selected_start", "approach_speed_mps", "d_min_m", "d_max_m",
        "search_first_frame", "search_last_frame", "status", "error"}
    probes = report.tables["calibration_probes"].rows
    assert set(probes[0]) == {
        "scene", "frame", "timestamp_s", "distance_m", "line_distance_m", "stage", "visible_pixels",
        "status", "reason", "occlusion_source", "target_box_left_px", "target_box_top_px",
        "target_box_right_px", "target_box_bottom_px", "replica_score"}
    initial = report.images["initial_texture"].pixels
    assert report.images["initial_texture"].description
    assert report.values["detected_frame_fraction"].unit == "ratio (0–1)"
    assert report.series["pgd_objective"].y_unit == "ratio (0–1)"
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
    descriptor = ScorerDescriptor("image-score", {"image": "CHW float tensor [0,1]"},
                                  {"score": "scalar confidence"}, {"score": "tensor"}, ("image",),
                                  sensor_id="front", decision_threshold=0.)
    rejected = [
        ([], "exactly one"),
        ([replace(descriptor, sensor_id=None)], "sensor_id and decision_threshold"),
    ]
    for descriptors, message in rejected:
        adapter = DrivingAdapter(lambda seed: ConstantAVStack(),
                                 scorers=[(item, lambda avstack, image, target: image.sum())
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
        optimizer=PGD(steps=1, batch=1, seed=0), min_visible_pixels=5, braking_deceleration_mps2=10.))
    attack.run_offline(context)
    calibration, = context.report.tables["calibration"].rows
    rows = sorted(context.report.tables["sample_manifest"].rows, key=lambda row: row["frame"])
    frames = [row["frame"] for row in rows]
    lines = [row["line_distance_m"] for row in rows]
    # Every eligible frame trains, from d_max down to the last frame where braking_deceleration_mps2
    # can still stop before the line.
    assert frames == list(range(frames[0], frames[-1] + 1))
    assert context.report.values["train_frames"].value == len(rows)
    assert calibration["d_min_m"] <= min(lines) and max(lines) <= calibration["d_max_m"]
    assert min(lines) - calibration["d_min_m"] <= max(np.abs(np.diff(lines))) + 1e-9
    assert len(rows) > 8


def test_white_box_smoothness(nurec_detector_services_and_segment):
    services, make_segment = nurec_detector_services_and_segment
    context = OfflineContext(services, (make_segment(),), seed=0)

    def optimize(initial_patch, **pgd):
        attack = StopSignPatchAttack(initial_patch=initial_patch, optimization=DetectorPatchOptimization(
            optimizer=PGD(steps=3, batch=1, seed=0, **pgd), min_visible_pixels=5,
            braking_deceleration_mps2=10., samples_per_scene=4))
        texture = attack.run_offline(context)
        return texture, context.report.values["patch_smoothness"].value

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


@pytest.mark.parametrize("maximum,driver_type,expected_unchanged", [
    pytest.param(None, DeceleratingAVStack, True, id="all-training"),
    pytest.param(1, ImageForwardAVStack, False, id="capped-image-dependent"),
])
def test_black_box_offline_phase(nurec_patch_services_and_segment, tmp_path, maximum, driver_type, expected_unchanged):
    original_services, make_segment = nurec_patch_services_and_segment
    training = (make_segment(), make_segment(start=16))
    options = RolloutPatchOptimization(NES(steps=1, population_size=2, learning_rate=2000., seed=0),
                                      start_distances_m=(8.,), max_training_segments=maximum)
    attack = StopSignPatchAttack(initial_patch="white", optimization=options, min_approach_m=8.)
    adapter = DrivingAdapter(lambda seed: driver_type(), past_steps=15)
    with ClosedLoopServices(original_services.dataset, original_services.simulator, adapter) as services:
        context = OfflineContext(services, training, seed=0, validation_metadata=(make_segment(start=23),),
                                 report=OfflineReport(tmp_path / "files"))
        texture = attack.run_offline(context)
        report = context.report
        rows = report.tables["rollouts"].rows
        expected_segments = maximum or len(training)
        assert len(rows) == 4 * expected_segments
        assert report.values["training_segments"].value == expected_segments
        assert report.values["rollout_count"].value == len(rows)
        assert len(report.tables["starts"].rows) == expected_segments
        assert {row["status"] for row in rows} == {"completed"}
        for row in rows:
            log = RunLog.read(report.directory / row["run_log"])
            assert log.status == "completed" and len(log.steps) == 1
            trajectory = log.steps[0].produced_command.trajectory
            stamps = np.array([point[2] for point in trajectory])
            x = np.array([point[0][0] for point in trajectory])
            independent = np.interp(stamps[0] + options.horizon_s * 1e6, stamps, x) - x[0]
            assert row["forward_m"] == pytest.approx(independent)
        assert report.values["final_planned_forward_m"].unit == "m"
        assert report.values["final_planned_forward_m"].value == pytest.approx(
            np.mean([row["forward_m"] for row in rows[-expected_segments:]]))
        assert all(row["run_log"].startswith("rollouts/") for row in rows)
        assert set(report.tables["rollouts"].columns) == set(rows[0])
        assert len(report.series["planned_forward_m"].y) == 2
        np.testing.assert_array_equal(texture, report.images["optimized_texture"].pixels)
        assert report.series["planned_forward_m"].y[-1] >= report.series["planned_forward_m"].y[0]
        # The policy-dependent case selects a changed texture; the constant policy keeps its initial candidate.
        assert np.array_equal(texture, report.images["initial_texture"].pixels) == expected_unchanged


def test_black_box_offline_errors(nurec_patch_services_and_segment, tmp_path):
    services, make_segment = nurec_patch_services_and_segment
    options = RolloutPatchOptimization(
        NES(steps=1, population_size=2, seed=0),
        start_distances_m=(8.,), max_training_segments=1)
    attack = StopSignPatchAttack(
        initial_patch="white", optimization=options, min_approach_m=8.)
    context = OfflineContext(
        services, (make_segment(),), report=OfflineReport(tmp_path / "files"))
    in_memory = replace(context, report=OfflineReport(None))
    exported = attack.run_offline(in_memory)
    np.testing.assert_array_equal(exported, in_memory.report.images["optimized_texture"].pixels)
    assert all(row["run_log"] is None for row in in_memory.report.tables["rollouts"].rows)
    with pytest.raises(ValueError, match="training segment"):
        attack.run_offline(replace(context, metadata=()))
    failed = replace(context, report=OfflineReport(tmp_path / "failed/files"))
    invalid_horizon = StopSignPatchAttack(
        optimization=replace(options, horizon_s=6.), min_approach_m=8.)
    with pytest.raises(ValueError, match="does not cover the requested horizon"):
        invalid_horizon.run_offline(failed)
    rows = failed.report.tables["rollouts"].rows
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["forward_m"] is None
    assert "does not cover the requested horizon" in rows[0]["error"]
    log = RunLog.read(failed.report.directory / rows[0]["run_log"])
    assert log.steps and log.status == "completed"
    assert "optimized_texture" not in failed.report.images
    with pytest.raises(ValueError, match="max_training_segments"):
        replace(options, max_training_segments=0)
    with pytest.raises(ValueError, match="cannot be optimized"):
        StopSignPatchAttack(initial_patch="none", optimization=options, min_approach_m=8.)
    with pytest.raises(ValueError, match=r"start_distances_m must be within \[8.0, 50.0\] m"):
        StopSignPatchAttack(
            optimization=replace(options, start_distances_m=(7.,)), min_approach_m=8.)
    with pytest.raises(ValueError, match=r"start_distances_m must be within \[8.0, 50.0\] m"):
        StopSignPatchAttack(
            optimization=replace(options, start_distances_m=(51.,)), min_approach_m=8.)
