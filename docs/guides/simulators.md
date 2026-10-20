# Simulators

> **Audience:** Experiment authors and simulator implementers.
>
> **Outcome:** Select a simulator or implement its configuration/simulator-runtime boundary.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Simulator and simulator runtime

A Simulator stores reusable configuration and capability declarations. `create_runtime(seed)`
creates a fresh SimulatorRuntime owning one live world, clock, sensors and native connections. Its
creator closes it, including after errors. ClosedLoopRunner uses but does not own caller resources;
Experiment and ClosedLoopServices create and close their own resources.

SimulatorRuntime `reset(start, initialization=None, *, deployment=None, world=None)` initializes a
world and captures the first observation. `observe()` returns a copy of that capture without
advancing, rendering or running hooks. `step(control)` advances one decision and captures again.
`state()` returns independent physical state, available even at setup and step hooks.

Only world settings go into reset. The complete `{world, sensors}` envelope is resolved by the
experiment, runner and services; sensor transforms run after capture.

## CARLA

`CarlaSimulator(client, settle_iters=100, *, simulator_version="0.9.16", depth_camera=None)` accepts
simulator-runtime connection and timing settings. Scene content belongs in
`CarlaDataset(recipes=...)`, not the simulator. Each reset requires a ScenarioStart carrying its
recipe. Use matching CARLA 0.9.16 client/server packages and a dedicated server.

### Scene configuration

CarlaRecipe holds map name, ego and NPC spawns, sensors, explicit motion, time_step and horizon. The
simulator client chooses address, port, traffic-manager port, synchronous operation and rate.
CarlaSimulator.client_config holds these reusable settings; the live simulator runtime owns client.
An explicit create_runtime(seed) overrides client.seed for that run; absent an explicit run seed,
the configured seed is retained. traffic_manager_seed is fixed when explicitly supplied; otherwise
it is derived from the resolved run seed. Seeds must be nonnegative integers, and RPC/TM ports must
be integers from 1 through 65535. The client rate must agree with recipe time_step. PrescribedMotion
selects constant_control, per-step controls or an N+1 trajectory for each actor.

### Selected-origin initialization

The simulator runtime initializes base actors and sensors, replays prescribed motion without hooks
to the deployment window, applies simulator.setup effects and runs the shared deployment ticks.
Capture ticks then make world effects visible in the selected frame. The selected boundary is
preserved; a positive boundary earlier than the required window is rejected. At position zero,
actors are held while initialization ticks execute.

Clean treatments use the same tick budget. Setup does not count as a driving decision. Actual
initial states, actor identities and sensor times are recorded. No replay deviation threshold,
selection revalidation or automatic retry is performed. Before decision zero, ego is released from
recipe motion so driving commands control it.

Set depth_camera to a configured RGB sensor alias for aligned depth and a pinhole camera view. The
simulator runtime constructs SensorView directly from native camera geometry and captured data.

## NuRec

NuRecSimulator takes `config`, a Renderer template and an ego-dynamics template. Each simulator
runtime copies them, so connections, loaded scenes and dynamics mutations are isolated. Set
`config.dt` to NuRecDataset.time_step; this mismatch is rejected before simulation starts.

NuRecRenderer connects to the reconstruction service and renders at requested times and ego poses.
StubRenderer provides synthetic images for offline tests. Actor tracks are recorded, not reactive.
SimulatorRuntime.restore immediately captures its restored state; later observe calls do not render
again.

KinematicBicycle consumes actuator controls. TrajectoryFollower consumes trajectory controls and
follows them ideally. Choose compatible AVStack commands; there is no automatic conversion.

**Not yet implemented.** Reactive other-actor traffic and a physical trajectory controller. See
[TODO](../development/TODO.md#vehicle-dynamics).

## Recorded playback

RecordedReplaySimulator reads a NuScenesDataset ScenarioStart and advances through recorded
FrameData. It is open-loop: commands cannot change ego or traffic motion. N decisions require N+1
frames. Sensor environments and effects can change model inputs but do not turn playback into
closed-loop simulation.

## Write a custom simulator adapter

Declare dataset_kinds, supported_world_settings, command_modes, decision_time_step,
supported_stages, can_render, capture_ticks and identity on Simulator. Validate start compatibility
and world-setting values before creating native resources. Values depending on live actors are
checked during reset.

Implement native operations on SimulatorRuntime; do not override reset or step. The base class owns
capture lifecycle, stage timing, reset isolation and deployment windows. This complete example runs
offline:

<!-- runnable -->
```python
import numpy as np
from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.datasets import ActorState

class LineWorld(Simulator):
    @property
    def decision_time_step(self):
        return 0.1

    @property
    def command_modes(self):
        return frozenset({"actuator"})

    def create_runtime(self, seed=None):
        return LineRuntime(self)

class LineRuntime(SimulatorRuntime):
    def _initialize_world(self, start, initialization, environment):
        self.position = 0.0
        self.speed = 0.0
        self.index = 0

    def _advance_native(self, command):
        self.speed = max(0.0, self.speed + command.throttle - command.brake)
        self.position += self.speed * self.decision_time_step
        self.index += 1

    def _ego(self):
        pose = np.eye(4)
        pose[0, 3] = self.position
        return ActorState(pose, (self.speed, 0, 0), "vehicle", (4, 2, 1.5))

    def _observe(self):
        return Observation(self.index * 0.1, self.index,
                           vehicle_state=self._ego(), ego_speed=self.speed)

    def _state(self):
        return WorldSnapshot(self.index * 0.1, self.index,
                             ego_speed=self.speed, ego=self._ego())

class Cruise(AVStack):
    def __call__(self, observation):
        return Control(throttle=0.2)

with LineWorld().create_runtime() as simulator_runtime:
    log = ClosedLoopRunner().run(simulator_runtime, Cruise(), RunRequest(3))
assert log.status == "completed"
assert len(log.steps) == 3
assert log.steps[-1].physical_post.ego_speed > 0
```

Set can_render=True for image insertion. A rendering simulator declares camera_sensors and
implements sensor_geometry(sensor) without copying images, plus sensor_view(sensor), supplying
canonical camera projection, data, camera transform, actor/ego poses with optional cuboid extents
and measured depth. The simulator provides evidence; concrete attack code owns insertion rendering.

## Next steps

Read [Runtime](../reference/runtime.md), [Scenarios](scenarios.md) and [Python
API](../reference/python-api.md#simulator-api).
