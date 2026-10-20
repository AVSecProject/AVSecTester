"""Service-bound optimization keeps calibrated wide-angle geometry and candidate lifecycles."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from avsectester.attack import AttackSegment
from avsectester.attack.runtime import AttackRuntime, SensorAttackEffect
from avsectester.attack.implemented_attacks.patch_attack.optimization_data import SessionDataSource, plane_sampling_grid
from avsectester.attack.optimization.interface import AdversarialSample, TargetSpec
from avsectester.attack.optimization.perturbations import PatchPerturbation
from avsectester.attack.implemented_attacks.patch_attack.offline import PatchOptimization, optimize_patch
from avsectester.attack.optimization import SessionScorer
from avsectester.attack.optimization.whitebox import PGD
from avsectester.attack.optimization.blackbox import NES
from avsectester.attack.implemented_attacks.patch_attack.attack import PatchAttack
from avsectester.attack.insertion import PlaneSurface, ResolvedInsertion, render_resolved
from avsectester.closed_loop.datatypes import Observation
from avsectester.closed_loop.services import ScorerDescriptor
from avsectester.closed_loop.simulator import SensorView
from avsectester.datasets.base import FrameReference
from avsectester.geometry.cameras import FThetaCamera, PinholeCamera
from avsectester.geometry.poses import ActorPose


def camera_view(camera):
    return SensorView("front-0", np.full((camera.height, camera.width, 3), 64, np.uint8),
                      camera, np.eye(4), {}, ActorPose(np.eye(4)), 0., 0)


@pytest.mark.parametrize("camera", [
    PinholeCamera(np.array([[25., 0., 24.], [0., 25., 16.], [0., 0., 1.]]), 48, 32),
    FThetaCamera(24., 16., (0., 25.), 48, 32, (0., 1/25)),
])
def test_calibrated_grid_matches_online_planar_pixels_and_remains_differentiable(camera):
    view = camera_view(camera)
    # An off-centre surface has visibly curved projected edges for f-theta.
    corners = np.array([[.4, -.8, 2.], [2.5, -.8, 2.], [2.5, .8, 2.], [.4, .8, 2.]])
    grid, mask = plane_sampling_grid(view, corners)
    texture = np.full((16, 16, 4), [255, 0, 0, 255], np.uint8)
    asset = SimpleNamespace(planes=lambda: (PlaneSurface(corners, texture),))
    resolved = ResolvedInsertion("sample", np.eye(4), asset)
    online, online_mask = render_resolved(view.data, camera, np.eye(4), resolved)
    # Texture boundaries can have fractional alpha. Stay inside the sampled texture for
    # the RGB comparison; geometric coverage itself must also match the online ray caster.
    interior = (np.abs(grid) < .8).all(axis=-1) & (mask[0] > 0)
    assert interior.sum() > 20
    sample = AdversarialSample(torch.as_tensor(view.data).permute(2, 0, 1).float()/255,
                       TargetSpec((0., 0., 48., 32.)), {"sampling_grid": grid, "mask": mask})
    delta = torch.tensor([1., 0., 0.])[:, None, None].expand(3, 16, 16).clone().requires_grad_()
    output = PatchPerturbation((16, 16), device="cpu").apply(sample, delta)
    np.testing.assert_allclose(output.detach().permute(1, 2, 0).numpy()[interior],
                               online[interior]/255, atol=1/255)
    assert np.all(online_mask[interior] == 255)
    output.sum().backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0
    np.testing.assert_array_equal(output.detach().permute(1, 2, 0).numpy()[mask[0] == 0],
                                  (view.data/255).astype(np.float32)[mask[0] == 0])


def test_calibrated_grid_marks_behind_camera_and_degenerate_planes():
    view = camera_view(PinholeCamera(np.eye(3), 4, 3))
    corners = np.array([[-1., -1., -2.], [1., -1., -2.], [1., 1., -2.], [-1., 1., -2.]])
    grid, mask = plane_sampling_grid(view, corners)
    assert not mask.any() and np.isfinite(grid).all()
    with pytest.raises(ValueError, match="nondegenerate"):
        plane_sampling_grid(view, np.zeros((4, 3)))


def test_session_scorer_only_demands_a_graph_when_gradients_are_enabled():
    flags = []
    class Session:
        def score(self, scorer_id, image, target, require_grad=False):
            flags.append(require_grad)
            return SimpleNamespace(value=image.mean())
    scorer = SessionScorer(Session(), "score", require_grad=True)
    image = torch.ones((3, 2, 2), requires_grad=True)
    scorer.target_score(image, TargetSpec((0, 0, 2, 2))).backward()
    with torch.no_grad():
        assert scorer.target_score(image, TargetSpec((0, 0, 2, 2))) == 1
    assert flags == [True, False]


class FakeSession:
    def __init__(self):
        self.closed = False
        self.requests = []
        self.frames = tuple(FrameReference("s", i, i, i*.1) for i in range(3))
        self.services = SimpleNamespace(dataset=SimpleNamespace(frames=lambda _: self.frames))
        matrix = np.eye(4)
        matrix[:3, :3] = [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]
        host = np.eye(4)
        host[0, 3] = 5.
        self.view = SensorView("front-0", np.full((24, 32, 3), 64, np.uint8),
            PinholeCamera(np.array([[25., 0., 16.], [0., 25., 12.], [0., 0., 1.]]), 32, 24),
            matrix, {"lead": ActorPose(host, (4., 2., 1.5))}, ActorPose(np.eye(4)), 0., 0)
        self.score_gradients = []
        self.fail_render = False

    def can_deploy_at(self, position):
        return True

    def scorers(self):
        return (ScorerDescriptor("score", {}, {}, {}, ("image",)),)

    def sensor_data(self, request):
        self.requests.append(("sensor", request))
        return SimpleNamespace(observation=Observation(request.frame.timestamp, request.frame.position,
            {"front-0": self.view.data.copy()}))

    def render(self, request):
        self.requests.append(("render", request))
        if request.effects and self.fail_render:
            raise RuntimeError("render failed")
        data = self.view.data.copy()
        for effect in request.effects:
            assert isinstance(effect, SensorAttackEffect)
            data = effect.apply(data, self.view)
        return SimpleNamespace(view=self.view, observation=Observation(request.frame.timestamp,
                               request.frame.position, {"front-0": data}))

    def score(self, scorer_id, image, target, require_grad=False):
        self.score_gradients.append(require_grad)
        return SimpleNamespace(value=image.mean())

    def rollout(self, request):
        self.requests.append(("rollout", request))
        return SimpleNamespace(status="completed", steps=[SimpleNamespace(physical_post=
            SimpleNamespace(ego_speed=3.)) for _ in range(request.decision_steps)])


class Services:
    def __init__(self, session):
        self.value = session
    @contextmanager
    def session(self, seed):
        try:
            yield self.value
        finally:
            self.value.closed = True


def patch_context(session, attack=None):
    attack = attack or PatchAttack(decision_steps=2, width_m=1., height_m=.5)
    segment = AttackSegment("s", session.frames[0], session.frames[2], roles={"host": ("lead",)},
                            placement=attack.placement_for("lead"))
    return attack, SimpleNamespace(services=Services(session), metadata=(segment,), seed=42, environment={"world": {}, "sensors": []})


def test_service_data_uses_one_capture_and_view_alias_with_occlusion_aware_grid():
    session = FakeSession()
    attack, context = patch_context(session)
    sample = SessionDataSource(session, attack, context.metadata, size=(4, 4), device="cpu",
                               brightness=(1., 1.), contrast=(1., 1.)).sample(1)[0]
    assert [kind for kind, _ in session.requests] == ["render"]
    assert sample.context["sampling_grid"].shape == (24, 32, 2)
    assert sample.context["mask"].any()
    assert sample.x.shape == (3, 24, 32)
    assert sample.target.box[2] > sample.target.box[0]


def test_pgd_runs_with_scorer_id_and_computes_final_score_without_gradient_requirement():
    session = FakeSession()
    attack, context = patch_context(session)
    artifact = optimize_patch(attack, context, PatchOptimization(PGD(steps=2, batch=1), device="cpu",
        size=(4, 4), scorer_id="score", brightness=(1., 1.), contrast=(1., 1.)))
    assert artifact.shape == (4, 4, 4) and artifact.dtype == np.uint8
    assert session.score_gradients == [True, True, False]
    assert session.closed


@pytest.mark.parametrize("failure", [False, True])
def test_nes_uses_runtime_factory_and_stops_each_candidate_on_success_or_render_failure(failure):
    session = FakeSession()
    session.fail_render = failure
    runtimes = []
    class Attack(PatchAttack):
        def create_runtime(self, segment, artifacts):
            parent = super().create_runtime(segment, artifacts)
            class Runtime(AttackRuntime):
                def __init__(self):
                    self.stopped = False
                def setup(self, context):
                    assert context.stage == "setup"
                    return parent.setup(context)
                def step(self, context):
                    assert context.stage == "step"
                    return parent.step(context)
                def stop(self, context):
                    self.stopped = True
            runtime = Runtime()
            runtimes.append(runtime)
            return runtime
    attack, context = patch_context(session, Attack(decision_steps=2))
    options = PatchOptimization(NES(steps=1, population_size=2), device="cpu", size=(4, 4),
                                scorer_id="score", validate_rollout=not failure)
    if failure:
        with pytest.raises(RuntimeError, match="render failed"):
            optimize_patch(attack, context, options)
    else:
        assert optimize_patch(attack, context, options).shape == (4, 4, 4)
        assert len(runtimes) == 4  # initial, antithetic pair, final evaluation
        assert session.requests[-1][0] == "rollout"
    assert runtimes and all(runtime.stopped for runtime in runtimes)
    assert session.closed


def test_scorer_converts_only_explicit_builtin_image_formats_and_keeps_gradients():
    received = []
    class Session:
        def score(self, scorer_id, image, target, require_grad=False):
            received.append(image)
            return SimpleNamespace(value=image.mean())
    session = Session()
    rgb = np.full((2, 4, 3), 128, np.uint8)
    black = SessionScorer(session, "inference", require_grad=False, input_format="HWC RGB uint8")
    black.target_score(rgb, TargetSpec((0, 0, 4, 2)))
    assert received[-1] is rgb
    white = SessionScorer(session, "rpn", require_grad=True,
                          input_format="CHW float tensor [0,1]", device="cpu")
    white.target_score(rgb, TargetSpec((0, 0, 4, 2)))
    assert received[-1].shape == (3, 2, 4) and received[-1].dtype == torch.float32
    torch.testing.assert_close(received[-1], torch.full((3, 2, 4), 128/255))
    delta = torch.ones((3, 2, 4), requires_grad=True)
    white.target_score(delta, TargetSpec((0, 0, 4, 2))).backward()
    assert delta.grad is not None
    with pytest.raises(ValueError, match="cannot preserve image gradients"):
        black.target_score(delta, TargetSpec((0, 0, 4, 2)))
    custom = SessionScorer(session, "user", require_grad=False, input_format="my custom protocol")
    custom.target_score(rgb, TargetSpec((0, 0, 4, 2)))
    assert received[-1] is rgb


def test_nes_default_inference_scorer_receives_actual_rendered_uint8_rgb():
    session = FakeSession()
    session.scorers = lambda: (ScorerDescriptor("mmdet.inference", {"image": "HWC RGB uint8"}, {}, {}),)
    def score(name, image, target, require_grad=False):
        assert name == "mmdet.inference"
        assert isinstance(image, np.ndarray) and image.dtype == np.uint8
        assert image.shape == (24, 32, 3) and not require_grad
        return SimpleNamespace(value=float(image.mean()/255))
    session.score = score
    attack, context = patch_context(session)
    result = optimize_patch(attack, context, PatchOptimization(NES(steps=1, population_size=2), size=(4, 4)))
    assert result.shape == (4, 4, 4) and session.closed


def test_default_target_is_full_host_box_instead_of_patch_footprint():
    session = FakeSession()
    attack, context = patch_context(session)
    sample = SessionDataSource(session, attack, context.metadata, size=(4, 4), device="cpu").sample(1)[0]
    ys, xs = np.nonzero(sample.context["mask"][0])
    x0, y0, x1, y1 = sample.target.box
    assert x0 < xs.min() and y0 < ys.min()
    assert x1 > xs.max() + 1 and y1 > ys.max() + 1


def test_optimizer_seed_is_separate_from_simulator_seed_and_rollout_uses_environment():
    session = FakeSession()
    seeds = []
    attack, context = patch_context(session)
    context.environment = {"world": {"weather": {"cloudiness": 42}}, "sensors": []}
    original = context.services.session
    @contextmanager
    def scoped(seed):
        seeds.append(seed)
        with original(seed) as value:
            yield value
    context.services.session = scoped
    optimize_patch(attack, context, PatchOptimization(NES(steps=1, population_size=2, seed=71),
                   size=(4, 4), scorer_id="score", validate_rollout=True))
    assert seeds == [42]
    assert all(request.environment["world"] == context.environment["world"]
               for operation, request in session.requests)
    request = session.requests[-1][1]
    assert request.seed == 42


def test_empty_metadata_returns_original_assets_without_opening_a_simulator():
    attack = PatchAttack(decision_steps=2)
    context = SimpleNamespace(metadata=(), services=None)
    assert optimize_patch(attack, context, PatchOptimization(PGD(steps=1))) is attack.assets


def test_nes_warm_start_uses_explicit_texture_and_preserves_asset_alpha():
    session = FakeSession()
    attack, context = patch_context(session)
    seen = []
    algorithm = NES(steps=1, population_size=2)
    def run(data, perturbation, scorer, objective):
        value = perturbation.init()
        seen.append(value.copy())
        return SimpleNamespace(export=perturbation.export(value))
    algorithm.run = run
    texture = np.full((4, 4, 4), [64, 128, 192, 255], np.uint8)
    output = optimize_patch(attack, context, PatchOptimization(algorithm, scorer_id="score",
                            size=(4, 4), init_texture=texture))
    np.testing.assert_allclose(seen[0][:, 0, 0], [64/255, 128/255, 192/255])
    np.testing.assert_array_equal(output[..., :3], texture[..., :3])


def test_rollout_validation_consumes_log_and_rejects_incomplete_run():
    session = FakeSession()
    session.rollout = lambda _: SimpleNamespace(status="failed", steps=[])
    attack, context = patch_context(session)
    with pytest.raises(RuntimeError, match="rollout did not complete: failed"):
        optimize_patch(attack, context, PatchOptimization(NES(steps=1, population_size=2),
                       scorer_id="score", size=(4, 4), validate_rollout=True))
    assert session.closed


def test_native_sampling_measures_actual_footprint_and_excludes_unchanged_background():
    from avsectester.attack.implemented_attacks.patch_attack.optimization_data import native_sampling_grid

    black = np.full((10, 12, 3), [20, 180, 20], np.uint8)  # green background is not a patch
    white = black.copy()
    black[2:8, 3:10] = 15
    white[2:8, 3:10] = 230
    white[4:6, 5:7] = black[4:6, 5:7]  # a foreground occluder / alpha cutout
    grid, mask = native_sampling_grid(black, white)
    expected = np.zeros((10, 12), bool)
    expected[2:8, 3:10] = True
    expected[4:6, 5:7] = False
    np.testing.assert_array_equal(mask[0] > 0, expected)
    np.testing.assert_allclose(grid[2, 3], [-1 + 1/7, -1 + 1/6])
    assert (grid[~expected] == 2).all()
    with pytest.raises(ValueError, match="no stable"):
        native_sampling_grid(black, black)
    broken = white.astype(float)
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite RGB"):
        native_sampling_grid(black, broken)


def test_native_probe_data_is_cached_for_the_same_origin_and_uses_online_factory():
    from dataclasses import replace
    from avsectester.attack.runtime import WorldAttackEffect

    session = FakeSession()
    textures = []
    class Effect(WorldAttackEffect):
        def __init__(self, texture):
            self.texture = texture
        def apply(self, simulator_runtime, resources):
            pass
    class Attack(PatchAttack):
        def create_runtime(self, segment, artifacts):
            class Runtime(AttackRuntime):
                def setup(self, context):
                    return (Effect(artifacts),)
            return Runtime()
    attack, context = patch_context(session, Attack(decision_steps=2, emissive=True))
    context.metadata = (replace(context.metadata[0], user_data={"deployment": "world"}),)
    def render(request):
        textures.append(request.effects[0].texture.copy())
        rgb = session.view.data.copy()
        rgb[4:8, 12:20] = request.effects[0].texture[0, 0, :3]
        return SimpleNamespace(view=session.view, observation=Observation(0., 0, {"front-0": rgb}))
    session.render = render
    source = SessionDataSource(session, attack, context.metadata, size=(4, 4), device="cpu",
                               at_segment_start=True)
    first, second = source.sample(1)[0], source.sample(1)[0]
    assert len(textures) == 2
    assert not textures[0][..., :3].any() and (textures[1][..., :3] == 255).all()
    np.testing.assert_array_equal(first.context["mask"], second.context["mask"])
    assert first.context["mask"].sum() == 32
    attack.prop = "different-mesh"
    source.sample(1)
    assert len(textures) == 4
    source.environment["world"] = {"weather": {"cloudiness": 30}}
    source.sample(1)
    assert len(textures) == 6


@pytest.mark.parametrize("options", [
    {"size": (0, 4)}, {"target_box": (3, 0, 1, 2)}, {"brightness": (2, 1)},
    {"contrast": (0, float("nan"))}, {"validate_rollout": 1}, {"scorer_id": ""},
])
def test_patch_optimization_options_are_checked_at_construction(options):
    with pytest.raises((TypeError, ValueError)):
        PatchOptimization(NES(steps=1, population_size=2), **options)


def test_default_device_preserves_algorithm_defaults():
    assert PatchOptimization(PGD(steps=1)).device == "cuda:0"
    assert PatchOptimization(NES(steps=1, population_size=2)).device == "cpu"


def test_old_algorithm_names_and_flat_dictionaries_are_rejected():
    with pytest.raises(TypeError, match="PGD or NES"):
        PatchOptimization("pgd")
    with pytest.raises(TypeError, match="PatchOptimization"):
        optimize_patch(None, None, {"algorithm": "pgd"})


def test_host_objective_requires_extent_without_inventing_zero_dimensions():
    from dataclasses import replace
    from avsectester.attack.implemented_attacks.patch_attack.optimization_data import projected_host_box
    session = FakeSession()
    view = replace(session.view, actors={"lead": ActorPose(np.eye(4), None)})
    with pytest.raises(ValueError, match="Optimization host 'lead' bounding-box extent is unavailable"):
        projected_host_box(view, "lead")


def test_pgd_rejects_nes_only_warm_start_at_construction():
    with pytest.raises(ValueError, match="init_texture is supported only by NES"):
        PatchOptimization(PGD(steps=1), init_texture=np.ones((2, 2, 4), np.uint8))


def test_physical_sampling_uses_framework_deployment_predicate_and_keeps_frame_zero():
    from dataclasses import replace

    session = FakeSession()
    session.frames = tuple(FrameReference("s", i, f"frame-{i}", i*.1) for i in range(8))
    session.can_deploy_at = lambda position: position == 0 or position >= 4
    attack, context = patch_context(session)
    segment = replace(context.metadata[0], stop=session.frames[-1], user_data={"deployment": "world"})
    source = SessionDataSource(session, attack, (segment,), size=(4, 4), device="cpu")
    seen = []
    def physical_capture(segment, frame):
        seen.append(frame.position)
        return (SimpleNamespace(view=session.view, observation=Observation(frame.timestamp,
                frame.position, {"front-0": session.view.data})),
                np.zeros((24, 32, 2), np.float32), np.ones((1, 24, 32), np.float32))
    source._physical_capture = physical_capture
    source.sample(50)
    assert set(seen) == {0, 4, 5, 6}
    assert not {1, 2, 3}.intersection(seen)


def test_render_candidate_context_uses_timeline_position_not_string_frame_id():
    from dataclasses import replace
    from avsectester.attack.implemented_attacks.patch_attack.offline import render_candidate

    session = FakeSession()
    attack, context = patch_context(session)
    frames_seen = []
    class Candidate(AttackRuntime):
        def setup(self, online):
            frames_seen.append(online.frame)
            return ()
    attack.create_runtime = lambda segment, artifacts: Candidate()
    frame = replace(session.frames[1], frame_id="recorded-token")
    render_candidate(session, attack, context.metadata[0], frame, attack.assets, {})
    assert frames_seen == [1]
