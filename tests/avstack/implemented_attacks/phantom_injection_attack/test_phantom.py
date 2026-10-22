"""A phantom's source-frame geometry and its action on the real modular driving pipeline."""
import numpy as np
import pytest

pytest.importorskip("avstack.config")

from avsectester.implemented_attacks.phantom_injection_attack import PhantomInjectionAttack
from avsectester.attack.runtime import OnlineContext
from avsectester.closed_loop.avstacks.modular import ModularAVStack
from avsectester.closed_loop.hooks import Hook, HookSet
from tests.fakes import MemorySimulator


def test_phantom_geometry(make_detections):
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame
    from avstack.geometry import transformations
    from avstack.modules.perception.object3d import Passthrough3DObjectDetector

    sensor = ReferenceFrame(np.array([20., -10., 1.6]),
                            transformations.transform_orientation([0, 0, np.pi / 2], "euler", "quat"),
                            GlobalOrigin3D)
    runtime = PhantomInjectionAttack(decision_steps=1, target_xyz=(8., 1., -1.4),
                                    extent=(4.5, 2., 1.7), score=.65, object_id=12345).create_runtime(None, None)
    action = runtime.step(OnlineContext(0, "step", 0., 0))[0]
    detector = Passthrough3DObjectDetector()
    detector.register_post_hook(lambda output: (action.apply(output, None),))
    for positions in ([], [(15., 4., -1.5)]):
        output = detector(make_detections(positions, reference=sensor, frame=7))
        phantom = output[-1]
        assert len(output) == len(positions) + 1 and phantom.ID == 12345
        assert phantom.obj_type == "Car" and phantom.score == .65
        np.testing.assert_allclose(phantom.position.x, [8., 1., -1.4])
        np.testing.assert_allclose(phantom.box.hwl, [1.7, 2., 4.5])
        np.testing.assert_allclose(phantom.position.change_reference(GlobalOrigin3D, inplace=False).x,
                                   [19., -2., .2], atol=1e-6)
    malformed = make_detections()
    malformed.source_reference = None
    with pytest.raises(ValueError, match="source_reference"):
        action.apply(malformed, None)


def test_phantom_braking(pipeline_config, native_observation):
    clean, attacked = ModularAVStack(pipeline_config), ModularAVStack(pipeline_config)
    attacked.instrument()
    runtime = PhantomInjectionAttack(decision_steps=10).create_runtime(None, None)
    action = runtime.step(OnlineContext(0, "step", 0., 0))[0]
    hooks = HookSet([Hook("perception.post", lambda output, context: action.apply(output, None))])
    with MemorySimulator().create_runtime() as simulator_runtime:
        with hooks.session(simulator_runtime, attacked).activate():
            simulator_runtime.reset()
            for frame in range(10):
                observation = native_observation(frame, [])
                if frame == 0:
                    clean.reset(observation)
                    attacked.reset(observation)
                clean_command = clean(observation)
                attacked_command = attacked(observation)
    assert clean_command.brake == 0. and attacked_command.brake > 0.
    assert len(attacked.detections()) == 1
    assert len(attacked.component_log()["tracking"]) == 1
    assert attacked.component_log()["planning"].top()[1].target_speed == 0.
    runtime.stop(OnlineContext(10, "stop", .5, 10))
