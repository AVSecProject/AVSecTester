"""The phantom attack is a real avstack HOOKS post-hook — verify it end-to-end at the hook level.

No CARLA/GPU needed: build a small detection list, attach PhantomInjection the way a pipeline would
(register_post_hook on a perception-like module), and confirm it appends exactly one fabricated
BoxDetection. This is the same hook the closed-loop demo attaches to MMDetObjectDetector3D.
"""

import numpy as np
import pytest

pytest.importorskip("avstack")

from avsectester.attacks import PhantomInjection


def _detections():
    pytest.importorskip("avstack")
    from avstack.datastructs import DataContainer
    from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position
    from avstack.modules.perception.detections import BoxDetection

    dets = []
    for i in range(3):
        pos = Position(np.array([10.0 + i, 0.0, 0.0]), GlobalOrigin3D)
        att = Attitude(np.quaternion(1), GlobalOrigin3D)
        box = Box3D(pos, att, [1.5, 1.8, 4.0], where_is_t="bottom")
        dets.append(
            BoxDetection(
                data=box,
                noise=np.ones(6),
                source_identifier="test",
                reference=GlobalOrigin3D,
                obj_type="Car",
                score=0.8,
            )
        )
    return DataContainer(0, 0.0, dets, "test", source_reference=GlobalOrigin3D)


def test_phantom_appends_one_detection():
    dets = _detections()
    n0 = len(dets)
    (out,) = PhantomInjection(target_xyz=(6.0, 0.0, -1.5))(dets)
    assert len(out) == n0 + 1
    phantom = out[-1]
    assert phantom.obj_type == "Car"
    assert float(phantom.score) == pytest.approx(0.9)


def test_phantom_registers_in_avstack_hooks():
    pytest.importorskip("avstack")
    from avstack.config import HOOKS

    assert "PhantomInjection" in HOOKS.module_dict
    hook = HOOKS.build({"type": "PhantomInjection"})
    assert isinstance(hook, PhantomInjection)


def test_phantom_post_hook_preserves_detections_and_sets_geometry(make_detections):
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame
    from avstack.geometry import transformations as tforms
    from avstack.modules.perception.object3d import Passthrough3DObjectDetector

    ref = ReferenceFrame(
        np.array([20.0, -10.0, 1.6]),
        tforms.transform_orientation([0, 0, np.pi / 2], "euler", "quat"),
        GlobalOrigin3D,
    )
    data = make_detections([(15, 4, -1.5), (25, -4, -1.5)], reference=ref, frame=7)
    original_positions = [det.position.x.copy() for det in data]
    detector = Passthrough3DObjectDetector()
    detector.register_post_hook(
        PhantomInjection(
            target_xyz=(8, 1, -1.4),
            extent=(1.7, 2.0, 4.5),
            score=0.65,
            oid=12345,
        )
    )

    output = detector(data)

    assert len(output) == 3
    assert output.frame == 7
    assert output.timestamp == pytest.approx(0.35)
    for original, retained, position in zip(data, output, original_positions):
        assert retained is original
        np.testing.assert_array_equal(original.position.x, position)
        assert retained.score == 0.8
    phantom = output[-1]
    assert phantom.ID == 12345
    assert phantom.obj_type == "Car"
    assert phantom.score == pytest.approx(0.65)
    assert phantom.reference is output.source_reference
    np.testing.assert_allclose(phantom.position.x, [8, 1, -1.4])
    np.testing.assert_allclose(phantom.box.hwl, [1.7, 2.0, 4.5])
    # For this pose, local forward points along global +y.
    np.testing.assert_allclose(
        phantom.position.change_reference(GlobalOrigin3D, inplace=False).x,
        [19, -2, 0.2],
        atol=1e-6,
    )


def test_phantom_handles_empty_detection_container(make_detections):
    from avstack.modules.perception.object3d import Passthrough3DObjectDetector

    detector = Passthrough3DObjectDetector()
    detector.register_post_hook(PhantomInjection())
    output = detector(make_detections(frame=4))
    assert len(output) == 1
    assert output.frame == 4
    assert output.timestamp == pytest.approx(0.2)
    np.testing.assert_allclose(output[0].position.x, [6, 0, -1.5])


def test_empty_output_keeps_phantom_in_sensor_frame(make_detections):
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame
    from avstack.geometry import transformations as tforms
    from avstack.modules.perception.object3d import Passthrough3DObjectDetector

    sensor_ref = ReferenceFrame(
        np.array([20.0, 10.0, 1.6]),
        tforms.transform_orientation([0, 0, np.pi / 2], "euler", "quat"),
        GlobalOrigin3D,
    )
    detector = Passthrough3DObjectDetector()
    detector.register_post_hook(PhantomInjection())
    populated = detector(make_detections([(30, 4, 0)], reference=sensor_ref, frame=0))
    empty = detector(make_detections(reference=sensor_ref, frame=1))
    expected = populated[-1].position.change_reference(GlobalOrigin3D, inplace=False).x
    actual = empty[-1].position.change_reference(GlobalOrigin3D, inplace=False).x
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("populated", [False, True])
def test_phantom_requires_explicit_source_reference(make_detections, populated):
    data = make_detections([(20, 0, 0)] if populated else [])
    data.source_reference = None
    original_count = len(data)

    with pytest.raises(ValueError, match="source_reference"):
        PhantomInjection()(data)

    assert len(data) == original_count


def test_phantom_uses_source_frame_when_existing_boxes_are_global(make_detections):
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame

    sensor_ref = ReferenceFrame(np.array([20.0, 10.0, 1.6]), np.quaternion(1), GlobalOrigin3D)
    data = make_detections([(10, 0, 0)], reference=sensor_ref)
    data[0].change_reference(GlobalOrigin3D, inplace=True)

    (output,) = PhantomInjection()(data)

    np.testing.assert_allclose(
        output[-1].position.change_reference(GlobalOrigin3D, inplace=False).x,
        [26, 10, 0.1],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "operation", ["copy", "filter", "apply", "add_list", "add_container", "encode"]
)
def test_detection_operations_preserve_source_frame_for_empty_outputs(make_detections, operation):
    import json

    from avstack.geometry import GlobalOrigin3D, ReferenceFrame
    from avstack.modules.perception.detections import DetectionContainerDecoder

    sensor_ref = ReferenceFrame(np.array([20.0, 10.0, 1.6]), np.quaternion(1), GlobalOrigin3D)
    data = make_detections(reference=sensor_ref)
    if operation == "copy":
        output = data.copy()
    elif operation == "filter":
        data = make_detections([(10, 0, 0)], reference=sensor_ref)
        output = data.filter(lambda detection: False)
    elif operation == "apply":
        output = data.apply_and_return(lambda detection: detection)
    elif operation == "add_list":
        output = data + []
    elif operation == "add_container":
        output = data + make_detections(reference=sensor_ref)
    else:
        output = json.loads(data.encode(), cls=DetectionContainerDecoder)

    assert len(output) == 0
    (attacked,) = PhantomInjection()(output)
    np.testing.assert_allclose(
        attacked[0].position.change_reference(GlobalOrigin3D, inplace=False).x,
        [26, 10, 0.1],
        atol=1e-6,
    )


def test_empty_lidar_outputs_follow_current_sensor_pose(monkeypatch):
    from avstack.calibration import LidarCalibration
    from avstack.datastructs import DataContainer
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame
    from avstack.geometry import transformations as tforms
    from avstack.modules.perception.base import _PerceptionAlgorithm
    from avstack.modules.perception.object3d import MMDetObjectDetector3D
    from avstack.sensors import LidarData

    # Exercise the neural detector's normal call and hook path without loading GPU weights.
    detector = MMDetObjectDetector3D.__new__(MMDetObjectDetector3D)
    _PerceptionAlgorithm.__init__(detector)

    def empty_inference(data, identifier, **kwargs):
        return DataContainer(data.frame, data.timestamp, [], identifier)

    monkeypatch.setattr(detector, "_execute", empty_inference)
    detector.register_post_hook(PhantomInjection())
    ego_ref = ReferenceFrame(np.array([20.0, 10.0, 0.0]), np.quaternion(1), GlobalOrigin3D)
    # A translated, rotated mount distinguishes sensor coordinates from ego coordinates.
    sensor_ref = ReferenceFrame(
        np.array([1.0, 0.0, 2.0]),
        tforms.transform_orientation([0, 0, np.pi / 2], "euler", "quat"),
        ego_ref,
    )
    calibration = LidarCalibration(sensor_ref)
    outputs = []
    poses = [((20, 10, 0), 0.0), ((30, 10, 0), np.pi / 2), ((30, 20, 0), np.pi)]
    expected_positions = [[21, 16, 0.5], [24, 11, 0.5], [29, 14, 0.5]]
    for frame, (xyz, yaw) in enumerate(poses):
        ego_ref.x = np.array(xyz, dtype=float)
        ego_ref.q = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
        cloud = LidarData(
            timestamp=frame * 0.05,
            frame=frame,
            data=np.zeros((0, 4)),
            calibration=calibration,
            source_ID=0,
            source_name="lidar",
        )
        output = detector(cloud)
        outputs.append(output)
        assert output.frame == frame
        assert output.timestamp == pytest.approx(frame * 0.05)
        np.testing.assert_allclose(output[0].position.x, [6, 0, -1.5])

    # Later changes to the live ego reference must not move earlier phantoms.
    for output, expected in zip(outputs, expected_positions):
        np.testing.assert_allclose(
            output[0].position.change_reference(GlobalOrigin3D, inplace=False).x,
            expected,
            atol=1e-6,
        )
