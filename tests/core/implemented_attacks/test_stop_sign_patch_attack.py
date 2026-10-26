"""Stop-sign selection, patched replica placement and real offline optimization."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from avsectester.attack import OfflineContext, OfflineReport, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.attack.optimization import AdversarialSample, NES, PGD
from avsectester.data_types import TargetSpec
from avsectester.execution.services import RenderRequest
from avsectester.geometry.cameras import PinholeCamera
from avsectester.geometry.projection import project_box
from avsectester.implemented_attacks.stop_sign_patch_attack import StopSignPatchAttack
from avsectester.implemented_attacks.stop_sign_patch_attack.assets import (
    DEFAULT_MASK_PATH, initial_patch_texture, read_patch_mask, replica_face, texture_from_colors,
)
from avsectester.implemented_attacks.stop_sign_patch_attack.detector import StopSignDetector
from avsectester.implemented_attacks.stop_sign_patch_attack.detector_optimization import (
    MaskedPatchPerturbation, TextureDataSource,
)
from avsectester.implemented_attacks.stop_sign_patch_attack.metabuilder import (
    control_from_user_data, route_progress, select_stop_approach,
)
from avsectester.implemented_attacks.stop_sign_patch_attack.offline import (
    DetectorPatchOptimization, RolloutPatchOptimization,
)
from avsectester.implemented_attacks.stop_sign_patch_attack.rollout_optimization import (
    MaskedTexturePerturbation, fixed_start_segments, planned_forward_distance,
)
from avsectester.rendering import ExposureHarmonizer
from avsectester.rendering.compositing import plan_composite
from avsectester.rendering.objects import ObjectAsset, PlacedObject
from avsectester.runlog import RunLog


def test_build_metadata(approach_view):
    scenario = approach_view()
    attack = StopSignPatchAttack()
    segment, = attack.build_metadata(scenario)
    assert segment.start.position == 15 and segment.stop.position == 78
    assert segment.user_data["recorded_stop_frame"] == 58
    assert segment.user_data["approach_distance_m"] == pytest.approx(45.)
    assert segment.user_data["line_route_distance_m"] == pytest.approx(60.)
    assert segment.user_data["stop_line_lane_ids"] == ["lane"]
    restored = control_from_user_data(segment.user_data)
    np.testing.assert_array_equal(restored.pose, scenario.scenario_map().traffic_controls["sign"].pose)
    assert restored.extent == (.05, .8, .8) and restored.lane_ids == ("lane",)
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
    with pytest.raises(ValueError, match="must not exceed approach_m"):
        StopSignPatchAttack(approach_m=10.)


def test_initial_patch_texture():
    mask = read_patch_mask(DEFAULT_MASK_PATH)
    face = replica_face()
    textures = {kind: initial_patch_texture(kind, mask, noise_std=.25, noise_seed=0)
                for kind in ("none", "white", "black", "gaussian_noise")}
    np.testing.assert_array_equal(textures["none"], face)
    np.testing.assert_array_equal(textures["white"][..., :3][mask], 255)
    np.testing.assert_array_equal(textures["black"][..., :3][mask], 0)
    for texture in textures.values():
        assert texture.dtype == np.uint8 and texture.shape == face.shape
        np.testing.assert_array_equal(texture[..., 3], face[..., 3])
        np.testing.assert_array_equal(texture[..., :3][~mask], face[..., :3][~mask])
    noise = textures["gaussian_noise"]
    np.testing.assert_array_equal(noise, initial_patch_texture("gaussian_noise", mask,
                                                            noise_std=.25, noise_seed=0))
    assert not np.array_equal(noise, initial_patch_texture("gaussian_noise", mask,
                                                          noise_std=.25, noise_seed=1))
    expected = np.rint(np.clip(.5 + .25 * np.random.default_rng(0).normal(
        size=(*mask.shape, 3)), 0., 1.) * 255).astype(np.uint8)
    np.testing.assert_array_equal(noise[..., :3][mask], expected[mask])
    with pytest.raises(ValueError, match="initial_patch"):
        StopSignPatchAttack(initial_patch="red")
    with pytest.raises(ValueError, match="cannot be optimized"):
        StopSignPatchAttack(initial_patch="none", optimization=RolloutPatchOptimization(
            NES(steps=1, population_size=2)))
    with pytest.raises(TypeError, match="optimization"):
        StopSignPatchAttack(optimization=object())


def test_create_runtime(approach_view):
    harmonizer = ExposureHarmonizer(reference="under")
    attack = StopSignPatchAttack(offset_m=.10, width_scale=1., harmonizer=harmonizer)
    scenario = approach_view()
    segment, = attack.build_metadata(scenario)
    texture = attack.run_offline(OfflineContext(scenario.services, (segment,), seed=0))
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
    np.testing.assert_array_equal(action.asset.surfaces[0].nominal_texture, replica_face())
    control = control_from_user_data(segment.user_data)
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
    unpatched = StopSignPatchAttack(initial_patch="none")
    np.testing.assert_array_equal(unpatched.run_offline(OfflineContext(scenario.services, (segment,))),
                                  replica_face())
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


@pytest.mark.parametrize(("max_training_segments", "expected_starts"), [
    (None, [15, 16]), (1, [15]), (10, [15, 16]),
])
def test_detector_patch_optimization(nurec_patch_services, replica_detector_model,
                                     max_training_segments, expected_starts):
    services, make_segment = nurec_patch_services
    attack = StopSignPatchAttack(initial_patch="white", harmonizer=ExposureHarmonizer(reference="under"),
                                optimization=DetectorPatchOptimization(
        detector=StopSignDetector(model=replica_detector_model),
        optimizer=PGD(steps=3, batch=1, seed=0),
        camera_id="front", min_visible_pixels=5, detection_threshold=0.,
        braking_deceleration_mps2=10., max_training_segments=max_training_segments, samples_per_scene=4,
    ))
    segment = make_segment()
    context = OfflineContext(services, (segment, make_segment(start=16)), seed=0,
                             validation_metadata=(make_segment(start=23),))
    texture = attack.run_offline(context)
    face = replica_face()
    np.testing.assert_array_equal(texture[..., 3], face[..., 3])
    np.testing.assert_array_equal(texture[..., :3][~attack.mask], face[..., :3][~attack.mask])
    report = context.report
    assert set(report.tables) == {"calibration", "calibration_probes", "sample_manifest", "heldout_scores",
                                  "heldout_summary", "detection_rate_vs_distance"}
    assert set(report.images) == {"initial_texture", "optimized_texture"}
    assert set(report.series) == {"pgd_confidence"}
    assert report.values["training_segments"].value == len(expected_starts)
    assert set(report.values) == {"train_frames", "heldout_frames", "training_segments",
                                  "mask_pixels", "final_confidence"}
    calibration = report.tables["calibration"].rows
    assert list(dict.fromkeys(row["selected_start"] for row in calibration)) == expected_starts
    assert calibration[0]["d_min_m"] == pytest.approx(5.)
    with pytest.raises(ValueError, match="training segment"):
        attack.run_offline(replace(context, metadata=()))
    with pytest.raises(ValueError, match="max_training_segments"):
        replace(attack.optimization, max_training_segments=0)
    samples = report.tables["sample_manifest"].rows
    train = {(row["scene"], row["frame"]) for row in samples if row["split"] == "train"}
    heldout = {(row["scene"], row["frame"]) for row in samples if row["split"] == "heldout"}
    assert train and heldout and not train.intersection(heldout)
    rows = report.tables["heldout_scores"].rows
    assert {key for row in rows for key in row} == {
        "scene", "frame", "distance_m", "confidence_clean", "confidence_replica", "confidence_initial", "confidence_optimized"}
    for row in rows:
        assert row["confidence_optimized"] < row["confidence_initial"]
    summary = report.tables["heldout_summary"].rows
    assert {row["texture"] for row in summary} == {"clean", "replica", "initial", "optimized"}
    assert set(summary[0]) == {"texture", "frames", "mean_confidence", "detection_rate"}
    assert all(row["frames"] == len(rows) == report.values["heldout_frames"].value for row in summary)
    assert set(calibration[0]) == {
        "scene", "selected_start", "approach_speed_mps", "d_min_m", "d_max_m",
        "search_first_frame", "search_last_frame", "status", "error"}
    probes = report.tables["calibration_probes"].rows
    assert set(probes[0]) == {
        "scene", "frame", "timestamp_s", "distance_m", "stage", "visible_pixels", "status", "reason",
        "occlusion_source", "target_box_left_px", "target_box_top_px", "target_box_right_px",
        "target_box_bottom_px", "replica_score"}
    initial = report.images["initial_texture"].pixels
    assert report.images["initial_texture"].description
    assert report.values["final_confidence"].unit == "ratio (0–1)"
    assert report.series["pgd_confidence"].y_unit == "ratio (0–1)"
    for table in report.tables.values():
        assert table.description
        assert set(table.columns) == {key for row in table.rows for key in row}
    asset, pose = attack.replica_placement(segment, initial)
    control = control_from_user_data(segment.user_data)
    frames = tuple(services.dataset.frames(segment.scenario_id))
    with services.session(0) as session:
        for row in samples:
            rendered = session.render(RenderRequest(segment.scenario_id, frames[int(row["frame"])],
                "front", actions=[InsertObject("stop_sign_replica", asset, pose, harmonizer=attack.harmonizer)]))
            view = rendered.view
            expected = project_box(view.camera, view.cam_from_world, control.pose, control.extent)
            np.testing.assert_allclose([row[f"target_box_{coordinate}_px"] for coordinate in ("left", "top", "right", "bottom")], expected)
            ys, xs = np.nonzero(view.composition.plan.visible_mask)
            replica_bounds = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            assert not np.allclose(expected, replica_bounds)


@pytest.mark.parametrize(("max_training_segments", "expected_segments"), [
    (None, 2), (1, 1), (10, 2),
])
def test_rollout_patch_optimization(nurec_patch_services, tmp_path,
                                    max_training_segments, expected_segments):
    services, make_segment = nurec_patch_services
    output = tmp_path / "rollout"
    options = RolloutPatchOptimization(
        NES(steps=1, population_size=2, seed=0),
        start_distances_m=(12., 8.), max_training_segments=max_training_segments,
    )
    attack = StopSignPatchAttack(initial_patch="white", optimization=options,
                                harmonizer=ExposureHarmonizer(reference="under"))
    training = (make_segment(), make_segment(start=16))
    context = OfflineContext(services, training, seed=0,
                             validation_metadata=(make_segment(start=23),),
                             report=OfflineReport(output / "files"))
    texture = attack.run_offline(context)
    report = context.report
    rows = report.tables["rollouts"].rows
    assert len(rows) == 8 * expected_segments
    assert {row["status"] for row in rows} == {"completed"}
    assert all(float(row["forward_m"]) == pytest.approx(21.) for row in rows)
    for row in rows:
        log = RunLog.read(report.directory / row["run_log"])
        assert log.status == "completed" and len(log.steps) == 1
        assert planned_forward_distance(log.steps[0].produced_command.trajectory) == pytest.approx(21.)
    assert report.values["training_segments"].value == expected_segments
    assert report.values["rollout_count"].value == 8 * expected_segments
    assert report.values["final_planned_forward_m"].value == pytest.approx(21.)
    assert report.values["final_planned_forward_m"].unit == "m"
    assert all(row["run_log"].startswith("rollouts/") for row in rows)
    assert set(report.tables["rollouts"].columns) == set(rows[0])
    assert len(report.tables["starts"].rows) == 2 * expected_segments
    assert {row["scenario_id"] for row in rows} == {training[0].scenario_id}
    assert len(report.series["planned_forward_m"].y) == 2
    assert report.series["planned_forward_m"].y == pytest.approx([21., 21.])
    initial = report.images["initial_texture"].pixels
    np.testing.assert_array_equal(texture, initial)
    with pytest.raises(ValueError, match="training segment"):
        attack.run_offline(replace(context, metadata=()))
    with pytest.raises(ValueError, match="max_training_segments"):
        replace(options, max_training_segments=0)
    failed_output = tmp_path / "failed"
    failed_context = replace(context, report=OfflineReport(failed_output / "files"))
    failed = StopSignPatchAttack(optimization=replace(options, horizon_s=6.),
                                harmonizer=attack.harmonizer)
    with pytest.raises(ValueError, match="does not cover the requested horizon"):
        failed.run_offline(failed_context)
    rows = failed_context.report.tables["rollouts"].rows
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert rows[0]["forward_m"] is None and "does not cover" in rows[0]["error"]
    assert RunLog.read(failed_context.report.directory / rows[0]["run_log"]).status == "completed"
    assert "optimized_texture" not in failed_context.report.images
    in_memory = replace(context, metadata=training[:1], report=OfflineReport(None))
    np.testing.assert_array_equal(attack.run_offline(in_memory), initial)
    assert all(row["run_log"] is None for row in in_memory.report.tables["rollouts"].rows)


def test_read_patch_mask(tmp_path):
    face = replica_face()
    mask = np.zeros(face.shape[:2], bool)
    mask[200:260, 100:400] = True
    path = tmp_path / "mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    np.testing.assert_array_equal(read_patch_mask(path), mask)
    corner = np.zeros(face.shape[:2], np.uint8)
    corner[0, 0] = 255
    invalid = {"empty": (np.zeros(face.shape[:2], np.uint8), "nonempty and binary"),
               "gray": (np.full(face.shape[:2], 128, np.uint8), "nonempty and binary"),
               "small": (np.full((4, 4), 255, np.uint8), "resolution"),
               "outside": (corner, "inside the replica face")}
    for name, (pixels, message) in invalid.items():
        destination = tmp_path / f"{name}.png"
        Image.fromarray(pixels).save(destination)
        with pytest.raises(ValueError, match=message):
            read_patch_mask(destination)


def test_texture_from_colors():
    original = np.full((2, 3, 4), (175, 30, 45, 128), np.uint8)
    mask = np.array([[True, False, True], [False, True, False]])
    colors = np.full((3, 2, 3), .25)
    texture = texture_from_colors(original, mask, colors)
    np.testing.assert_array_equal(texture[..., :3][mask], np.full((3, 3), 64))
    np.testing.assert_array_equal(texture[~mask], original[~mask])
    np.testing.assert_array_equal(texture[..., 3], original[..., 3])
    with pytest.raises(ValueError, match="finite RGB CHW"):
        texture_from_colors(original, mask, np.full_like(colors, np.nan))


def test_masked_patch_perturbation():
    original = np.full((4, 4, 4), (175, 30, 45, 255), np.uint8)
    original[0, 0, 3] = 0
    mask = np.zeros((4, 4), bool)
    mask[1:3] = True
    asset = ObjectAsset.from_panel(original, category="stop_sign", width_m=1.)
    pose = np.array([[0., 1., 0., 0.], [0., 0., -1., .5],
                     [-1., 0., 0., 4.], [0., 0., 0., 1.]])
    camera = PinholeCamera(np.array([[20., 0., 6.], [0., 20., 6.], [0., 0., 1.]]), 12, 12)
    plan = plan_composite(camera, np.eye(4), [PlacedObject(
        "stop_sign_replica", asset, pose, ExposureHarmonizer(reference="under"))])
    sample = AdversarialSample(torch.zeros(3, 12, 12), TargetSpec((3., 3., 9., 9.)), {"plan": plan})
    perturbation = MaskedPatchPerturbation(original, mask)
    colors = perturbation.init()
    image = perturbation.apply(sample, colors)
    image.sum().backward()
    assert colors.grad is not None and colors.grad.abs().sum() > 0
    assert image.shape == sample.x.shape and plan.visible_mask.any()
    projected = perturbation.project(colors * 4)
    assert torch.all(projected[:, mask] <= 1.)
    exported = perturbation.export(projected)
    np.testing.assert_array_equal(exported[~mask], original[~mask])
    np.testing.assert_array_equal(exported[..., 3], original[..., 3])
    with pytest.raises(ValueError, match="at least one pixel"):
        MaskedPatchPerturbation(original, np.zeros((4, 4), bool))


def test_texture_data_source():
    samples = [AdversarialSample(torch.full((3, 2, 2), value), TargetSpec((0., 0., 2., 2.)))
               for value in (.1, .9)]
    first, second = TextureDataSource(samples, seed=4), TextureDataSource(samples, seed=4)
    assert [sample.x.mean().item() for sample in first.sample(5)] == [
        sample.x.mean().item() for sample in second.sample(5)]
    with pytest.raises(ValueError, match="positive integer"):
        first.sample(0)


@pytest.mark.parametrize("epoch_us", [0, 100_000_000])
def test_planned_forward_distance(epoch_us):
    positions = [(5., 0., 0.), (7., 3., 0.), (13., 6., 0.), (19., 9., 0.)]
    trajectory = [(position, (1., 0., 0., 0.), timestamp) for position, timestamp in zip(
        positions, [epoch_us, epoch_us + 1_000_000, epoch_us + 2_500_000, epoch_us + 4_000_000])]
    assert planned_forward_distance(trajectory) == pytest.approx(10.)
    with pytest.raises(ValueError, match="does not cover the requested horizon"):
        planned_forward_distance(trajectory, horizon_s=4.1)
    unordered = [*trajectory[:2], (*trajectory[2][:2], trajectory[1][2]), trajectory[3]]
    with pytest.raises(ValueError, match="increase strictly"):
        planned_forward_distance(unordered)
    fractional = [(*trajectory[0][:2], float(trajectory[0][2])), *trajectory[1:]]
    with pytest.raises(ValueError, match="integer microseconds"):
        planned_forward_distance(fractional)


@pytest.mark.parametrize("past_steps,expected_starts", [(15, [20, 30, 40]), (25, [25, 30, 40])])
def test_fixed_start_segments(approach_view, past_steps, expected_starts):
    scenario = approach_view(past_steps=past_steps)
    approach = select_stop_approach(scenario)
    assert approach is not None
    segment = approach.to_segment()
    frames = scenario.frames
    with scenario.services.session(scenario.seed) as session:
        starts = fixed_start_segments(segment, frames, session.can_start_at)
        assert [item.start.position for item in starts] == expected_starts
        assert [item.stop for item in starts] == [frames[index + 1] for index in expected_starts]
        assert all(item.decision_steps == 1 and session.can_start_at(item.start.position)
                   for item in starts)
        short = replace(segment, start=frames[57], stop=frames[60])
        with pytest.raises(ValueError, match="Not enough distinct eligible start frames"):
            fixed_start_segments(short, frames, session.can_start_at)


def test_masked_texture_perturbation():
    original = np.full((2, 3, 4), (175, 30, 45, 128), np.uint8)
    mask = np.array([[True, False, True], [False, True, False]])
    perturbation = MaskedTexturePerturbation(original, mask)
    colors = perturbation.init()
    assert colors.shape == (3, 2, 3) and colors.dtype == np.float32
    np.testing.assert_array_equal(colors[:, mask],
                                  np.moveaxis(original[..., :3].astype(np.float32) / 255, -1, 0)[:, mask])
    proposal = colors.copy()
    proposal[0] = -1.
    proposal[2] = 2.
    projected = perturbation.project(proposal)
    assert np.all((projected[:, mask] >= 0.) & (projected[:, mask] <= 1.))
    np.testing.assert_array_equal(projected[:, ~mask], colors[:, ~mask])
    texture = perturbation.export(projected)
    assert texture.shape == original.shape and texture.dtype == np.uint8
    np.testing.assert_array_equal(perturbation.apply(None, projected), texture)
    with pytest.raises(ValueError, match="finite RGB CHW values"):
        perturbation.project(np.full_like(colors, np.nan))
    with pytest.raises(ValueError, match="at least one pixel"):
        MaskedTexturePerturbation(original, np.zeros_like(mask))


def test_route_progress():
    route = np.array([[0., 0., 0.], [10., 0., 0.], [10., 0., 0.], [10., 10., 0.]])
    points = np.array([[-2., 1., 0.], [5., 2., 0.], [12., 7., 0.], [10., 12., 0.]])
    np.testing.assert_allclose(route_progress(points, route), [-2., 5., 17., 22.])
    np.testing.assert_allclose(route_progress(np.array([[-2., 0., 0.], [12., 0., 0.]]),
                                              route[:2]), [-2., 12.])


def test_target_score(image_score_detector):
    model = image_score_detector(
        [[0., 0., 4., 4.], [0., 0., 4., 4.], [0., 0., 1., 1.], [0., 0., 4., 4.]],
        [13, 1, 13, 13], [.2, .94, .93, .7],
    )
    detector = StopSignDetector("unused.pth", model=model)
    image = torch.full((3, 8, 8), .5, requires_grad=True)
    score = detector.target_score(image, TargetSpec((0., 0., 4., 4.), "stop sign"))
    assert detector.stop_sign_label == 13
    assert score.item() == pytest.approx(.75)
    gradient, = torch.autograd.grad(score, image)
    torch.testing.assert_close(gradient, torch.full_like(image, .1 / image.numel()))
    assert model.roi_heads.score_thresh == 0.
    assert not model.training and not model.weight.requires_grad
    with pytest.raises(ValueError, match="finite float RGB"):
        detector.target_score(torch.ones(8, 8), TargetSpec((0., 0., 4., 4.)))
    with pytest.raises(ValueError, match="positive width and height"):
        detector.target_score(image, TargetSpec((4., 0., 0., 4.)))


def test_target_score_without_matching_detections(image_score_detector):
    image = torch.ones(3, 4, 4, requires_grad=True)
    empty = StopSignDetector("unused.pth", model=image_score_detector([], [], []))
    zero = empty.target_score(image, TargetSpec((0., 0., 2., 2.)))
    assert zero.item() == 0.
    gradient, = torch.autograd.grad(zero, image)
    torch.testing.assert_close(gradient, torch.zeros_like(image))
    unrelated = StopSignDetector("unused.pth", model=image_score_detector(
        [[0., 0., 2., 2.], [2., 2., 4., 4.]], [1, 13], [.9, .8],
    ))
    unmatched = unrelated.target_score(image, TargetSpec((0., 0., 2., 2.)))
    assert unmatched.item() == 0.
    gradient, = torch.autograd.grad(unmatched, image)
    torch.testing.assert_close(gradient, torch.zeros_like(image))
