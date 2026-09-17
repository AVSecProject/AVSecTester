#!/usr/bin/env python
"""Verify the AlpaSimBackend relay over real gRPC, against AlpaSim's own EgodriverService protos.

No renderer / models / scene needed: a fake "runtime" (a gRPC client, exactly as AlpaSim's real
runtime is) streams a camera frame + egomotion and calls drive(); our AlpaSimBackend relays that into
run(backend, stack) and returns the stack's trajectory. This exercises the bridge + the AlpaSim<->
interface data mapping end to end. (The live neural runtime is a separate, heavier setup.)

Prereq: AlpaSim's compiled protos on PYTHONPATH. Clone NVlabs/alpasim and:
    cd alpasim/src/grpc && uv run compile-protos
    export PYTHONPATH=$PWD:/path/to/AVSecTester:$PYTHONPATH
    python scripts/alpasim_roundtrip.py
"""

import sys
import threading
import time

import grpc
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
from avsectester.alpasim import AlpaSimBackend
from avsectester.backend import AVStack, run
from avsectester.plane import Control

ENDPOINT = "127.0.0.1:50123"
FRAMES = 5


class StraightTrajectoryStack(AVStack):
    """A trajectory-output AV box: record what the Observation carried, then plan straight ahead."""

    def __init__(self):
        self.seen = []

    def __call__(self, obs):
        self.seen.append((round(obs.ego_speed, 3), sorted(obs.sensor_data)))
        return Control(trajectory=[((6.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 1_000_000)])


def fake_runtime():
    """Mimic AlpaSim's runtime client: submit sensors + egomotion, drive(), until told to stop."""
    channel = grpc.insecure_channel(ENDPOINT)
    stub = egodriver_pb2_grpc.EgodriverServiceStub(channel)
    grpc.channel_ready_future(channel).result(timeout=10)
    returned = 0
    for i in range(FRAMES + 2):
        stub.submit_image_observation(
            egodriver_pb2.RolloutCameraImage(
                session_uuid="s",
                camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
                    frame_start_us=i * 50_000,
                    frame_end_us=i * 50_000 + 40_000,
                    image_bytes=b"\xff\xd8fake-jpeg",
                    logical_id="camera_front_wide",
                ),
            )
        )
        stub.submit_egomotion_observation(
            egodriver_pb2.RolloutEgoTrajectory(
                session_uuid="s",
                trajectory=common_pb2.Trajectory(
                    poses=[
                        common_pb2.PoseAtTime(
                            pose=common_pb2.Pose(
                                vec=common_pb2.Vec3(x=float(i), y=0.0, z=0.0),
                                quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
                            ),
                            timestamp_us=i * 50_000,
                        )
                    ]
                ),
                dynamic_states=[
                    common_pb2.DynamicState(linear_velocity=common_pb2.Vec3(x=3.0, y=4.0, z=0.0))
                ],
            )
        )
        resp = stub.drive(
            egodriver_pb2.DriveRequest(session_uuid="s", time_now_us=i * 50_000, time_query_us=i)
        )
        if resp.terminate_session:
            break
        assert len(resp.trajectory.poses) == 1
        wp = resp.trajectory.poses[0]
        assert (wp.pose.vec.x, wp.timestamp_us) == (6.0, 1_000_000)
        returned += 1
    channel.close()
    return returned


def main() -> int:
    # launch_runtime=False: the fake runtime below is the "runtime"; we only serve the relay.
    backend = AlpaSimBackend({}, endpoint=ENDPOINT, launch_runtime=False)
    stack = StraightTrajectoryStack()
    result = {}

    def drive_our_side():
        result["trace"] = run(backend, stack, FRAMES)
        backend.close()  # terminates the last pending drive()

    ours = threading.Thread(target=drive_our_side)
    ours.start()
    time.sleep(0.5)  # let the relay gRPC server bind
    returned = fake_runtime()
    ours.join(timeout=15)

    trace = result["trace"]
    speeds = [s for s, _ in stack.seen]
    print(f"frames driven (Trace): {len(trace.records)}")
    print(f"trajectories returned to the runtime: {returned}")
    print(f"ego_speed seen by the stack: {speeds}")
    print(f"sensors seen by the stack: {stack.seen[0][1] if stack.seen else None}")
    ok = (
        len(trace.records) == FRAMES
        and returned >= FRAMES
        and all(abs(s - 5.0) < 1e-6 for s in speeds)  # |(3,4,0)| = 5
        and stack.seen[0][1] == ["camera_front_wide"]
    )
    print("ROUND-TRIP:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
