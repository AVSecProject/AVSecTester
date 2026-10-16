"""Run snapshots, typed persistence, recovery and dependency boundaries."""

from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest

from avsectester.metric import impact
from avsectester.plane import Control, FrameRecord, Observation, Trace, WorldSnapshot
from avsectester.runlog import CodecRegistry, Recorder, RunLog, StepRecord, read_table


class TaggedList(list):
    pass


def test_capture_namedtuple_preserves_field_access_and_detaches_children():
    Point = namedtuple("Point", "x y")
    source = Point([1], [2])
    snapshot = Recorder().capture(source)
    source.x.append(3)
    assert type(snapshot) is Point
    assert snapshot.x == [1] and snapshot.y == [2]


def test_capture_defaultdict_preserves_factory_and_detaches_values():
    source = defaultdict(list, items=[1])
    snapshot = Recorder().capture(source)
    source["items"].append(2)
    snapshot["missing"].append(3)
    assert type(snapshot) is defaultdict and snapshot.default_factory is list
    assert snapshot["items"] == [1] and "missing" not in source


def test_capture_ordered_dict_keeps_type_and_order():
    source = OrderedDict([("second", [2]), ("first", [1])])
    snapshot = Recorder().capture(source)
    source.move_to_end("second")
    source["first"].append(0)
    assert type(snapshot) is OrderedDict
    assert list(snapshot) == ["second", "first"] and snapshot["first"] == [1]


def test_capture_list_subclass_preserves_instance_attributes():
    source = TaggedList([[1]])
    source.label = "detections"
    snapshot = Recorder().capture(source)
    source[0].append(2)
    assert type(snapshot) is TaggedList and snapshot.label == "detections"
    assert snapshot == [[1]]


class TaggedDict(dict):
    pass


def test_capture_container_subclass_attributes_are_independent():
    tagged_list = TaggedList([1])
    tagged_list.meta = {"ids": [1]}
    tagged_dict = TaggedDict(value=1)
    tagged_dict.meta = {"ids": [1]}
    snapshots = Recorder().capture([tagged_list, tagged_dict])
    tagged_list.meta["ids"].append(2)
    tagged_dict.meta["ids"].append(2)
    assert [item.meta for item in snapshots] == [{"ids": [1]}, {"ids": [1]}]
    assert type(snapshots[1]) is TaggedDict and snapshots[1] == {"value": 1}


def test_capture_dict_subclass_cycle_uses_one_snapshot():
    source = defaultdict(list)
    source["self"] = source
    source["nested"] = [source]
    snapshot = Recorder().capture(source)
    assert snapshot is not source and type(snapshot) is defaultdict
    assert snapshot["self"] is snapshot and snapshot["nested"][0] is snapshot


def test_codec_snapshotter_reaches_defaultdict_children():
    class Native:
        def __init__(self, data):
            self.data = data

        def __deepcopy__(self, memo):
            raise TypeError("Use the snapshot codec")

    calls = []
    def snapshot_native(value, capture_child):
        calls.append(value)
        return Native(capture_child(value.data))

    registry = CodecRegistry().register(
        Native, lambda value: value.data, snapshot=snapshot_native,
    )
    native = Native([1])
    source = defaultdict(list, first=native, alias=native)
    snapshot = Recorder(codecs=registry).capture(source)
    native.data.append(2)
    assert calls == [native]
    assert type(snapshot) is defaultdict and snapshot.default_factory is list
    assert snapshot["first"] is snapshot["alias"]
    assert snapshot["first"].data == [1]


def test_tuple_subclasses_use_constructor_or_deepcopy_fallback():
    class Items(tuple):
        pass

    class Pair(tuple):
        def __new__(cls, first, second):
            return super().__new__(cls, (first, second))

        def __getnewargs__(self):
            return tuple(self)

    for source in (Items(([1], [2])), Pair([1], [2])):
        snapshot = Recorder().capture(source)
        source[0].append(3)
        assert type(snapshot) is type(source)
        assert snapshot == ([1], [2])


def test_container_subclasses_persist_as_plain_types(tmp_path):
    Point = namedtuple("Point", "x y")
    values = {
        "point": Point([1], [2]),
        "defaults": defaultdict(list, items=[1]),
        "ordered": OrderedDict([("second", [2]), ("first", [1])]),
        "tagged": TaggedList([[1]]),
    }
    values["tagged"].label = "memory only"
    with Recorder(tmp_path) as recorder:
        recorder.record_step(StepRecord(0, components=values))
    loaded = RunLog.read(tmp_path).steps[0].components
    assert loaded == values
    assert type(loaded["point"]) is tuple
    assert type(loaded["defaults"]) is dict
    assert type(loaded["ordered"]) is dict
    assert list(loaded["ordered"]) == ["second", "first"]
    assert type(loaded["tagged"]) is list


def record(step=0, *, image=None):
    return StepRecord(
        step=step,
        raw_observation=Observation(t=step * 0.05, frame=step, ego_speed=3.0),
        model_observation=Observation(t=step * 0.05 + 1.0, frame=step + 4, ego_speed=8.0),
        physical_pre=WorldSnapshot(step * 0.05, step, {"position": [1.0, 2.0]}, 3.0),
        physical_post=WorldSnapshot((step + 1) * 0.05, step + 1, {"position": [2.0, 2.0]}, 2.0),
        produced_command=Control(throttle=0.1, trajectory=[((1.0, 2.0, 3.0), (1., 0., 0., 0.), 50000)]),
        executed_command=Control(brake=0.2),
        components={"perception": [{"label": "car", "box": [1., 2., 3., 4.], "score": 0.9}]},
        actions=[{"stage": "command", "requested_t": step * 0.05, "applied_t": step * 0.05}],
        outputs=[{"name": "attack", "value": {"triggered": True, "image": image}}],
        telemetry={"n_detections": 1, "optional": None},
    )


def test_capture_is_immediate_and_keeps_physical_and_model_data_separate():
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    source = record(image=image)
    recorder = Recorder()
    recorder.record_step(source)
    source.components["perception"][0]["score"] = 0.0
    source.executed_command.brake = 1.0
    source.physical_pre.vehicle_state["position"][0] = 20.0
    image[:] = 0
    logged = recorder.finish().steps[0]
    assert logged.components["perception"][0]["score"] == 0.9
    assert logged.executed_command.brake == 0.2
    assert logged.physical_pre.vehicle_state["position"][0] == 1.0
    assert logged.raw_observation.ego_speed == 3.0
    assert logged.model_observation.ego_speed == 8.0
    np.testing.assert_array_equal(logged.outputs[0]["value"]["image"], np.arange(24).reshape(2, 4, 3))


def test_round_trip_nested_commands_arrays_and_timing(tmp_path):
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    recorder = Recorder(tmp_path, metadata={"treatment": "attack", "seed": 4}, payload_shard_bytes=17)
    recorder.record_event("initialization", initial_t=0.0, optional=None)
    recorder.record_step(record(image=image))
    recorder.finish(reason="horizon")
    result = RunLog.read(tmp_path)
    step = result.steps[0]
    assert result.status == "completed"
    assert result.metadata == {"treatment": "attack", "seed": 4}
    assert step.produced_command == record().produced_command
    assert step.executed_command == Control(brake=0.2)
    assert step.raw_observation.t == 0.0
    assert step.model_observation.t == 1.0
    assert step.physical_pre.vehicle_state == {"position": [1., 2.]}
    assert step.components == record().components
    assert step.telemetry == {"n_detections": 1, "optional": None}
    np.testing.assert_array_equal(step.outputs[0]["value"]["image"], image)
    assert all(path.stat().st_size <= 17 for path in (tmp_path / "payloads").glob("*.bin"))
    assert result.events[0].data == {"initial_t": 0.0, "optional": None}
    assert result.events[-1].data == {"status": "completed", "reason": "horizon"}
    table = read_table(tmp_path)
    assert pa.types.is_struct(table.schema.field("components").type)
    assert pa.types.is_list(table.schema.field("components").type.field("perception").type)
    assert pa.types.is_struct(table.schema.field("actions").type.value_type)


def test_schema_evolution_empty_then_nested_and_selected_columns(tmp_path):
    recorder = Recorder(tmp_path, chunk_steps=1)
    recorder.record_step(StepRecord(step=0))
    recorder.record_step(record(step=1))
    recorder.finish()
    restored = RunLog.read(tmp_path)
    assert restored.steps[0].components == {}
    assert restored.steps[0].actions == []
    assert restored.steps[0].physical_pre is None
    assert restored.steps[1].components == record().components
    selected = read_table(tmp_path, columns=["step", "executed_command"])
    assert selected.column_names == ["step", "executed_command"]
    assert selected["step"].to_pylist() == [0, 1]
    with pytest.raises(ValueError, match="Unknown RunLog columns"):
        read_table(tmp_path, columns=["not_a_column"])


def test_unknown_native_objects_require_codec_and_optional_decoder(tmp_path):
    @dataclass
    class NativeBox:
        label: str
        xyz: list[float]

    native = NativeBox("car", [1., 2., 3.])
    registry = CodecRegistry().register(
        NativeBox, lambda box: {"label": box.label, "xyz": box.xyz}, name="example.box.v1"
    )
    recorder = Recorder(tmp_path / "with_codec", codecs=registry)
    recorder.record_step(StepRecord(0, components={"perception": [native]}))
    native.xyz[0] = 40.0
    recorder.finish()
    assert isinstance(recorder.log.steps[0].components["perception"][0], NativeBox)
    expected = {"label": "car", "xyz": [1., 2., 3.]}
    assert RunLog.read(tmp_path / "with_codec").steps[0].components["perception"] == [expected]
    with_decoder = CodecRegistry().register(
        NativeBox, lambda box: {"label": box.label, "xyz": box.xyz},
        name="example.box.v1", decode=lambda value: NativeBox(**value),
    )
    assert RunLog.read(tmp_path / "with_codec", codecs=with_decoder).steps[0].components["perception"] == [NativeBox(**expected)]
    unsupported = Recorder(tmp_path / "unsupported")
    with pytest.raises(TypeError, match="register an adapter codec"):
        unsupported.record_step(StepRecord(0, components={"perception": [native]}))
    unsupported.abort()


def test_process_interruption_keeps_only_published_parts(tmp_path):
    script = '''
import os
import sys
import numpy as np
from avsectester.runlog import Recorder, StepRecord
recorder = Recorder(sys.argv[1], chunk_steps=2, payload_shard_bytes=64)
for i in range(3):
    recorder.record_step(StepRecord(i, outputs=[{"image": np.ones((10, 10), dtype=np.uint8)}]))
os._exit(17)
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert process.returncode == 17
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["last_durable_step"] == 1
    assert manifest["step_rows"] == 2
    assert manifest["step_parts"] == 1
    # An interrupted parquet write must not be confused with a published part.
    (tmp_path / "steps" / "part-00001.parquet.open").write_bytes(b"PAR1unfinished")
    result = RunLog.read(tmp_path)
    assert result.status == "incomplete"
    assert [step.step for step in result.steps] == [0, 1]
    np.testing.assert_array_equal(result.steps[1].outputs[0]["image"], np.ones((10, 10)))
    assert not any(event.kind == "termination" for event in result.events)


def test_payload_is_closed_before_parquet_is_published(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    original = pq.write_table
    writes = []

    def check(table, where, **kwargs):
        # Every payload reference already points to an existing completed shard.
        for row in table.to_pylist():
            for output in row.get("outputs", []):
                for chunk in output["image"]["chunks"]:
                    assert (tmp_path / chunk["path"]).exists()
        writes.append(Path(where).name)
        original(table, where, **kwargs)

    monkeypatch.setattr(pq, "write_table", check)
    with Recorder(tmp_path, chunk_steps=1) as recorder:
        recorder.record_step(StepRecord(0, outputs=[{"image": np.ones((2, 2))}]))
        assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "incomplete"
    assert writes == ["part-00000.parquet.open", "part-00000.parquet.open"]
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "completed"


def test_reader_rejects_unknown_schema_and_does_not_load_unselected_payloads(tmp_path):
    with Recorder(tmp_path) as recorder:
        recorder.record_step(record(image=np.ones((2, 2))))
    for path in (tmp_path / "payloads").glob("*.bin"):
        path.unlink()
    assert read_table(tmp_path, columns=["step"])["step"].to_pylist() == [0]
    assert RunLog.read(tmp_path, load_payloads=False).steps[0].outputs[0]["value"]["image"]["nbytes"] == 32
    with pytest.raises(FileNotFoundError):
        RunLog.read(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported RunLog schema"):
        RunLog.read(tmp_path)


def test_failed_run_and_empty_log(tmp_path):
    with pytest.raises(RuntimeError, match="driving failed"):
        with Recorder(tmp_path / "failed") as recorder:
            recorder.record_step(record())
            raise RuntimeError("driving failed")
    result = RunLog.read(tmp_path / "failed")
    assert result.status == "failed"
    assert result.events[-1].data["error"] == "driving failed"
    with Recorder(tmp_path / "empty"):
        pass
    assert RunLog.read(tmp_path / "empty").steps == []
    empty = read_table(tmp_path / "empty", columns=["step", "physical_post"])
    assert empty.num_rows == 0
    assert empty.column_names == ["step", "physical_post"]
    with pytest.raises(ValueError, match="Unknown RunLog columns"):
        read_table(tmp_path / "empty", columns=["nonexistent"])


def test_cancelled_run_has_distinct_status(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with Recorder(tmp_path) as recorder:
            recorder.record_step(record())
            raise KeyboardInterrupt()
    result = RunLog.read(tmp_path)
    assert result.status == "cancelled"
    assert result.events[-1].data["error_type"] == "KeyboardInterrupt"


def test_streaming_recorder_does_not_retain_entire_log(tmp_path):
    with pytest.raises(ValueError, match="requires a persistent output path"):
        Recorder(retain_steps=False)
    recorder = Recorder(tmp_path, retain_steps=False, chunk_steps=2)
    for step in range(8):
        recorder.record_step(record(step))
    recorder.finish()
    assert recorder.log.steps == []
    assert recorder.log.events == []
    assert len(RunLog.read(tmp_path).steps) == 8
    with pytest.raises(RuntimeError, match="already closed"):
        recorder.record_step(record(8))
    with pytest.raises(FileExistsError):
        Recorder(tmp_path)


def test_persisted_measurements_preserve_existing_impact_metric(tmp_path):
    def trace(log):
        return Trace(records=[FrameRecord(
            frame=step.physical_post.frame, t=step.physical_post.t,
            speed=step.physical_post.ego_speed,
            n_detections=step.telemetry["n_detections"],
            throttle=step.executed_command.throttle,
            brake=step.executed_command.brake, steer=step.executed_command.steer,
        ) for step in log.steps])

    logs = []
    for name, speeds in (("clean", [2., 3., 4.]), ("attack", [2., 1., 0.])):
        recorder = Recorder(tmp_path / name)
        for index, speed in enumerate(speeds):
            row = record(index)
            row.physical_post = WorldSnapshot((index + 1) * 0.05, index + 1, ego_speed=speed)
            row.executed_command = Control(throttle=0.5) if name == "clean" else Control(brake=0.8)
            recorder.record_step(row)
        logs.append(recorder.finish())
    before = impact(trace(logs[0]), trace(logs[1]))
    after = impact(trace(RunLog.read(tmp_path / "clean")), trace(RunLog.read(tmp_path / "attack")))
    assert before == after
    assert after.induced_stop


def test_import_in_memory_api_does_not_import_heavy_dependencies():
    script = '''
import sys
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyarrow', 'torch', 'avstack', 'carla', 'alpasim'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
from avsectester.runlog import Recorder, StepRecord
recorder = Recorder()
recorder.record_step(StepRecord(0, components={'test': [1, 2]}))
assert recorder.finish().steps[0].components == {'test': [1, 2]}
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def test_capture_tensor_detaches_graph():
    torch = pytest.importorskip("torch")
    source = torch.ones(3, requires_grad=True) * 2
    snapshot = Recorder().capture({"output": source})["output"]
    assert not snapshot.requires_grad
    assert snapshot.grad_fn is None
    with torch.no_grad():
        source[0] = 99
    assert snapshot[0].item() == 2


@pytest.mark.parametrize("array", [np.zeros((0, 3)), np.array(4), np.arange(12).reshape(3, 4).T])
def test_array_shapes_and_noncontiguous_views_round_trip(tmp_path, array):
    with Recorder(tmp_path) as recorder:
        recorder.record_step(StepRecord(0, outputs=[{"value": array}]))
    restored = RunLog.read(tmp_path).steps[0].outputs[0]["value"]
    assert restored.shape == array.shape
    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(restored, array)


def test_framework_dataclasses_are_data_without_importing_native_code(tmp_path):
    from avsectester.simulators.nurec import EgoPose
    pose = EgoPose(x=2.0, y=3.0, yaw=0.2, speed=4.0, t=0.05)
    with Recorder(tmp_path) as recorder:
        recorder.record_step(StepRecord(0, physical_post=WorldSnapshot(0.05, 1, pose, 4.0)))
    assert isinstance(recorder.log.steps[0].physical_post.vehicle_state, EgoPose)
    assert RunLog.read(tmp_path).steps[0].physical_post.vehicle_state == {
        "x": 2.0, "y": 3.0, "yaw": 0.2, "speed": 4.0, "t": 0.05,
    }


def test_nurec_runner_persists_and_restores_driving_log(tmp_path):
    from avsectester.backend import AVStack
    from avsectester.closed_loop import ClosedLoopRunner, RunRequest, trace_from_log
    from avsectester.simulators.nurec import NuRecBackend, StubRenderer

    class Driver(AVStack):
        def reset(self, observation):
            pass

        def __call__(self, observation):
            return Control(throttle=0.2)

    backend = NuRecBackend({}, renderer=StubRenderer(height=2, width=3))
    try:
        log = ClosedLoopRunner().run(backend, Driver(), RunRequest(3, output_path=tmp_path))
    finally:
        backend.close()
    restored = RunLog.read(tmp_path)
    assert restored.status == "completed"
    assert len(restored.steps) == 3
    assert trace_from_log(restored).records == trace_from_log(log).records
    assert restored.events[0].data["clock"]["frame"] == 0


def test_named_outputs_and_stage_actions_support_mixed_typed_values(tmp_path):
    from avsectester.attack_runtime import StageAction
    from avsectester.plane import InterventionOutput
    values = [1.5, {"host": 7, "distance": [2.0, None], "extra": {"ok": True}},
              [1, "changed", {"score": 0.5}], "finished", False, None, np.arange(3)]
    with Recorder(tmp_path, chunk_steps=1) as recorder:
        for index in range(2):
            recorder.record_step(StepRecord(
                index,
                outputs=[InterventionOutput(str(i), value, index, "step.end", index + 1, .05)
                         for i, value in enumerate(values)],
                actions=[StageAction("perception.post", {"score": 0.5}),
                         StageAction("command", Control(brake=0.2)),
                         StageAction("policy.post", None)],
            ))
    restored = RunLog.read(tmp_path)
    for step in restored.steps:
        assert [output.value for output in step.outputs[:-1]] == values[:-1]
        np.testing.assert_array_equal(step.outputs[-1].value, values[-1])
        assert step.actions == [
            {"stage": "perception.post", "value": {"score": 0.5}},
            {"stage": "command", "value": Control(brake=0.2)},
            {"stage": "policy.post", "value": None},
        ]
    value_type = read_table(tmp_path).schema.field("outputs").type.value_type.field("value").type
    assert pa.types.is_struct(value_type)
    assert pa.types.is_float64(value_type.field("number").type)
    assert pa.types.is_list(value_type.field("entries").type)


def test_explicit_snapshot_codec_detaches_noncopyable_native_sensor(tmp_path):
    class NativeSensor:
        def __init__(self, data, calibration):
            self.data, self.calibration = data, calibration

        def __deepcopy__(self, memo):
            raise TypeError("Native SDK buffer cannot be deep-copied")

    registry = CodecRegistry().register(
        NativeSensor,
        lambda value: {"data": value.data, "calibration": value.calibration},
        snapshot=lambda value, capture: NativeSensor(
            np.array(value.data, copy=True), capture(value.calibration)
        ),
    )
    sensor = NativeSensor(np.arange(4), {"position": [1., 2., 3.]})
    recorder = Recorder(tmp_path, codecs=registry)
    recorder.record_step(StepRecord(0, raw_observation=Observation(0, 0, {"lidar": sensor})))
    sensor.data[:] = 99
    sensor.calibration["position"][0] = 99
    recorder.finish()
    snapshot = recorder.log.steps[0].raw_observation.sensor_data["lidar"]
    assert isinstance(snapshot, NativeSensor)
    np.testing.assert_array_equal(snapshot.data, np.arange(4))
    assert snapshot.calibration["position"][0] == 1.
    restored = RunLog.read(tmp_path).steps[0].raw_observation.sensor_data["lidar"]
    np.testing.assert_array_equal(restored["data"], np.arange(4))
    assert restored["calibration"] == {"position": [1., 2., 3.]}


def test_dynamic_output_types_can_change_across_published_chunks(tmp_path):
    from avsectester.plane import InterventionOutput
    values = [1.2, {"distance": 2., "valid": True}, [None, "result", 4]]
    with Recorder(tmp_path, chunk_steps=1) as recorder:
        for step, value in enumerate(values):
            recorder.record_step(StepRecord(step, outputs=[
                InterventionOutput("objective", value, step, "step.end", step + 1, (step + 1) * .05),
            ]))
    restored = RunLog.read(tmp_path)
    assert [step.outputs[0].value for step in restored.steps] == values
