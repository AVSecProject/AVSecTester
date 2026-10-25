"""Stop-sign selection, transparent patch placement and real offline optimization."""
import csv
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from PIL import Image

from avsectester.attack import OfflineContext, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.attack.optimization import AdversarialSample, NES, PGD
from avsectester.data_types import TargetSpec
from avsectester.execution.services import RenderRequest
from avsectester.geometry.cameras import PinholeCamera
from avsectester.geometry.projection import project_box
from avsectester.implemented_attacks.stop_sign_patch_attack import StopSignPatchAttack
from avsectester.implemented_attacks.stop_sign_patch_attack.assets import (
    initial_patch_texture, read_patch_mask, texture_from_colors,
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
    short, = StopSignPatchAttack(approach_m=3., hold_steps=1).build_metadata(scenario)
    assert (short.start.position, short.stop.position) == (57, 59)


def test_initial_patch_texture(tmp_path):
    mask = np.array([[True, False, True], [False, True, False]])
    white = initial_patch_texture("white", mask, noise_std=.25, noise_seed=0)
    black = initial_patch_texture("black", mask, noise_std=.25, noise_seed=0)
    noise = initial_patch_texture("gaussian_noise", mask, noise_std=.25, noise_seed=0)
    assert white.dtype == np.uint8 and white.shape == (*mask.shape, 4)
    np.testing.assert_array_equal(white[..., :3][mask], 255)
    np.testing.assert_array_equal(black[..., :3][mask], 0)
    for texture in (white, black, noise):
        np.testing.assert_array_equal(texture[..., 3], mask.astype(np.uint8) * 255)
        np.testing.assert_array_equal(texture[~mask], 0)
    np.testing.assert_array_equal(noise, initial_patch_texture("gaussian_noise", mask,
                                                            noise_std=.25, noise_seed=0))
    assert not np.array_equal(noise, initial_patch_texture("gaussian_noise", mask,
                                                          noise_std=.25, noise_seed=1))
    expected = np.rint(np.clip(.5 + .25 * np.random.default_rng(0).normal(
        size=(*mask.shape, 3)), 0., 1.) * 255).astype(np.uint8)
    np.testing.assert_array_equal(noise[..., :3][mask], expected[mask])
    with pytest.raises(ValueError, match="initial_patch"):
        StopSignPatchAttack(initial_patch="red")
    empty = tmp_path / "empty.png"
    Image.fromarray(np.zeros((3, 3), np.uint8)).save(empty)
    with pytest.raises(ValueError, match="nonempty"):
        StopSignPatchAttack(mask_path=empty)
    nonbinary = tmp_path / "nonbinary.png"
    Image.fromarray(np.full((3, 3), 128, np.uint8)).save(nonbinary)
    with pytest.raises(ValueError, match="binary"):
        StopSignPatchAttack(mask_path=nonbinary)
    with pytest.raises(TypeError, match="optimization"):
        StopSignPatchAttack(optimization=object())


def test_create_runtime(approach_view):
    attack = StopSignPatchAttack(offset_m=.10, width_scale=1.)
    scenario = approach_view()
    segment, = attack.build_metadata(scenario)
    texture = attack.run_offline(OfflineContext(scenario.services, (segment,), seed=0))
    candidate = texture.copy()
    candidate[candidate[..., 3] > 0, :3] = (35, 85, 135)
    actions = attack.create_runtime(segment, candidate).setup(
        OnlineContext(0, "setup", 0., segment.start.position))
    assert len(actions) == 1 and isinstance(actions[0], InsertObject)
    action = actions[0]
    assert action.object_id == "stop_sign_patch" and action.asset.category == "stop_sign_patch"
    assert action.asset.extent[1] == pytest.approx(.8)
    np.testing.assert_array_equal(action.asset.surfaces[0].texture, candidate)
    control = control_from_user_data(segment.user_data)
    face_center = action.asset.surfaces[0].corners.mean(axis=0)
    np.testing.assert_allclose((action.world_from_object @ np.r_[face_center, 1.])[:3],
                               control.pose[:3, 3] + .10 * control.pose[:3, 0])
    with pytest.raises(ValueError, match="shape|dimensions|size"):
        attack.create_runtime(segment, np.zeros((2, 2, 4), np.uint8))
    changed_alpha = candidate.copy()
    changed_alpha[0, 0, 3] ^= 255
    with pytest.raises(ValueError, match="alpha"):
        attack.create_runtime(segment, changed_alpha)


def test_detector_patch_optimization(nurec_patch_services, replica_detector_model, tmp_path):
    services, make_segment = nurec_patch_services
    output = tmp_path / "detector"
    attack = StopSignPatchAttack(initial_patch="white", optimization=DetectorPatchOptimization(
        detector=StopSignDetector(model=replica_detector_model),
        optimizer=PGD(steps=3, batch=1, seed=0), output_directory=output,
        camera_id="front", min_visible_pixels=5, detection_threshold=0.,
        braking_deceleration_mps2=10., scene_count=1, samples_per_scene=4,
    ))
    segment = make_segment()
    texture = attack.run_offline(OfflineContext(services, (segment,), seed=0))
    mask = read_patch_mask(attack.mask_path)
    np.testing.assert_array_equal(texture[..., 3], mask.astype(np.uint8) * 255)
    np.testing.assert_array_equal(texture[~mask], 0)
    assert all((output / name).is_file() for name in (
        "calibration.json", "sample_manifest.csv", "pgd_scores.csv", "heldout_scores.csv",
        "detection_rate_vs_distance.csv", "optimization.json", "initial_patch.png",
        "optimized_texture.png", "pgd_curve.png",
    ))
    calibration = json.loads((output / "calibration.json").read_text())
    assert calibration[0]["d_min_m"] == pytest.approx(5.)
    with (output / "sample_manifest.csv").open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    train = {(row["scene"], row["frame"]) for row in samples if row["split"] == "train"}
    heldout = {(row["scene"], row["frame"]) for row in samples if row["split"] == "heldout"}
    assert train and heldout and not train.intersection(heldout)
    with (output / "heldout_scores.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    scores = {(row["scene"], row["frame"], row["texture"]): float(row["score"]) for row in rows}
    for scene, frame in heldout:
        assert scores[scene, frame, "optimized"] < scores[scene, frame, "initial"]
    initial = np.asarray(Image.open(output / "initial_patch.png"))
    asset, pose = attack.patch_placement(segment, initial)
    control = control_from_user_data(segment.user_data)
    frames = tuple(services.dataset.frames(segment.scenario_id))
    with services.session(0) as session:
        for row in samples:
            rendered = session.render(RenderRequest(segment.scenario_id, frames[int(row["frame"])],
                "front", actions=[InsertObject("stop_sign_patch", asset, pose)]))
            view = rendered.view
            expected = project_box(view.camera, view.cam_from_world, control.pose, control.extent)
            np.testing.assert_allclose(json.loads(row["target_box"]), expected)
            ys, xs = np.nonzero(view.composition.plan.visible_mask)
            patch_bounds = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            assert not np.allclose(expected, patch_bounds)


def test_rollout_patch_optimization(nurec_patch_services, tmp_path):
    services, make_segment = nurec_patch_services
    output = tmp_path / "rollout"
    options = RolloutPatchOptimization(
        NES(steps=1, population_size=2, seed=0), output,
        start_distances_m=(12., 8.),
    )
    attack = StopSignPatchAttack(initial_patch="white", optimization=options)
    context = OfflineContext(services, (make_segment(),), seed=0)
    texture = attack.run_offline(context)
    with (output / "rollouts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["status"] for row in rows} == {"completed"}
    assert all(float(row["forward_m"]) == pytest.approx(21.) for row in rows)
    for row in rows:
        log = RunLog.read(row["run_log"])
        assert log.status == "completed" and len(log.steps) == 1
        assert planned_forward_distance(log.steps[0].produced_command.trajectory) == pytest.approx(21.)
    summary = json.loads((output / "optimization.json").read_text())
    assert len(summary["history"]) == 2
    assert summary["history"] == pytest.approx([-21., -21.])
    initial = np.asarray(Image.open(output / "initial_patch.png"))
    np.testing.assert_array_equal(texture, initial)
    with pytest.raises(ValueError, match="rollout directory must be empty"):
        attack.run_offline(context)
    failed_output = tmp_path / "failed"
    failed = StopSignPatchAttack(optimization=replace(options, output_directory=failed_output,
                                                     horizon_s=6.))
    with pytest.raises(ValueError, match="does not cover the requested horizon"):
        failed.run_offline(context)
    with (failed_output / "rollouts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert rows[0]["forward_m"] == "" and "does not cover" in rows[0]["error"]
    assert RunLog.read(rows[0]["run_log"]).status == "completed"
    assert not (failed_output / "optimization.json").exists()


def test_read_patch_mask(tmp_path):
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    path = tmp_path / "mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    np.testing.assert_array_equal(read_patch_mask(path), mask)
    for name, pixels in (("empty", np.zeros((4, 4), np.uint8)), ("gray", np.full((4, 4), 128, np.uint8))):
        invalid = tmp_path / f"{name}.png"
        Image.fromarray(pixels).save(invalid)
        with pytest.raises(ValueError, match="nonempty and binary"):
            read_patch_mask(invalid)


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
    plan = plan_composite(camera, np.eye(4), [PlacedObject("stop_sign_patch", asset, pose)])
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
