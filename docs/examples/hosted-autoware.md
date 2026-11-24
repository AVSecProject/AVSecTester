# Hosted Autoware

> **Audience:** Users running CARLA experiments with Autoware.
>
> **Outcome:** Compare hosted Autoware runs and save messages from selected ROS 2 topics.
>
> **Prerequisites:** [Installation](../getting-started/installation.md#hosted-autoware),
> including the pinned image, model data, ROS 2 network and Autoware assets.

## Check the prerequisites

Run these checks from the checkout root. Replace `AUTOWARE_DATA` with your model data directory.

```bash
docker image inspect --format '{{.Id}}' \
  ghcr.io/autowarefoundation/autoware@sha256:1db2988caa5dc5bae41548a8d556d0b57f55eb4dd63685426d94fd9c7a786034
(cd AUTOWARE_DATA/ml_models/lidar_centerpoint && ls pts_voxel_encoder_centerpoint_tiny.onnx \
  pts_backbone_neck_head_centerpoint_tiny.onnx centerpoint_tiny_ml_package.param.yaml \
  detection_class_remapper.param.yaml)
ls data/carla_autoware/map_tools/maps/Town10HD data/carla_autoware/postprocess/autoware_demo.json
ls data/carla_ros2
docker network inspect --format '{{json .Labels}}' "avsectester-ros2-$(id -u)"
grep -E '^(CARLA_SERVER_ARGS|CARLA_RPC_PORT|FASTRTPS_DEFAULT_PROFILES_FILE|CARLA_FASTDDS_PROFILE_HOST)=' .env
docker compose ps carla-server
docker inspect --format '{{range .Mounts}}{{println .Destination}}{{end}}' \
  "$(docker compose ps -q carla-server)"
```

The image check prints its ID. The four CenterPoint files, the Town10HD map, the camera profile and
the DDS profiles exist. The network labels show your numeric user ID. `.env` contains
`CARLA_RPC_PORT` and the three lines printed by the network script. The server is running, and its
mounts include `/dds/fastdds_profile.xml` and the `autoware_demo.json` camera profile, which the ROS
2 and Autoware overlays add. If a check fails, return to [Native ROS
2](../getting-started/installation.md#native-ros-2) or [Hosted
Autoware](../getting-started/installation.md#hosted-autoware) in Installation. This configuration
uses `Town10HD_Opt` only, so it does not need the AdditionalMaps package.

## Run the supplied comparison

Copy [carla_autoware.yaml](../../configs/carla_autoware.yaml) and edit the copy.

```bash
cp configs/carla_autoware.yaml configs/my_autoware.yaml
```

Set the host GPU, ROS domain and model data path under
`simulator.options.external_driving_system.options`. Use the same domain as the CARLA server and set
`runs.output` to `../results/my_autoware`. If you changed `CARLA_RPC_PORT`, set
`simulator.options.client.connect_port` to the same value. On a shared host, also choose an unused
`simulator.options.client.traffic_manager_port`. CARLA and Autoware can share one GPU with enough
free memory. The experiment service runs no model in this configuration, so `CARLA_EXPERIMENT_GPU`
can be any GPU.

The Town10HD configuration runs Autoware in LiDAR mode. The clean treatment leaves a stationary
vehicle outside ego's route. The relocation treatment moves it into ego's lane instantly at
the configured decision using `SetActorState`. Each treatment runs five times by default.

Each run starts a new Autoware container and simulates 4800 decisions of 0.05 s in real time
after Autoware becomes ready. The configuration has ten runs. Check a reduced copy first. Copy
`configs/my_autoware.yaml` to `configs/my_autoware_check.yaml` and in the copy set
`runs.repeats: 1`, keep only `clean` in `runs.treatments`, set `runs.decision_steps: 200`, set
`runs.output` to `../results/my_autoware_check` and set `startup_timeout_s: 1200` under
`simulator.options.external_driving_system.options`, because the first launch builds TensorRT
engines in the model directory. Then run the check copy:

```bash
docker compose exec carla-experiment avsectester run configs/my_autoware_check.yaml
```

The check passes when the command exits with code 0 and reports no failed runs. Its run's
`manifest.json`, in `results/my_autoware_check/runs/segment-00000/seed-0/clean`, then has status
`completed`. If readiness times out, raise `startup_timeout_s`
and run again with a new `runs.output`. Run the full copy only if you need its results:

```bash
docker compose exec carla-experiment avsectester run configs/my_autoware.yaml
```

Open `results/my_autoware/evaluation_report.html` to compare outcomes and within-treatment
variation. The [output layout](../reference/cli.md#output-layout) gives the paths to RunLogs and
evaluation files.

## Observe, modify and record topics

Save these settings as `configs/hosted-topics.yaml`. The example observes the IMU topic and
modifies the planning trajectory. It saves message contents for both topics.

```yaml
simulator:
  type: CarlaSimulator
  options:
    ego_control: ros2
    realtime_factor: 1.0
    external_driving_system:
      type: AutowareDrivingSystem
      options:
        observed_topics:
        - /sensing/imu/imu_data
        modified_topics:
        - /planning/trajectory
hooks:
- stage: /sensing/imu/imu_data
  type: my_project.hosted_topics:ObserveImu
- stage: /planning/trajectory
  type: my_project.hosted_topics:StationaryTrajectory
runs:
  output: ../results/my_autoware_topics
  recording:
    topics:
    - /sensing/imu/imu_data
    - /planning/trajectory
```

Combine the settings with your configured copy. This writes a complete second experiment YAML
and keeps the supplied recipe and sensor rig in one place. Run from the checkout root:

```bash
docker compose exec -T carla-experiment python - <<'PYTHON'
from pathlib import Path
import yaml


def merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


base = yaml.safe_load(Path("configs/my_autoware.yaml").read_text())
merge(base, yaml.safe_load(Path("configs/hosted-topics.yaml").read_text()))
Path("configs/my_autoware_topics.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
PYTHON
```

Save this module as `my_project/hosted_topics.py`, with `my_project/` directly in the checkout
root. The experiment service imports the two hook classes from the mounted checkout.

```python
from dataclasses import replace


class ObserveImu:
    def transform(self, message, context):
        context.record_output("imu", message.fields["angular_velocity"]["z"])
        return message


class StationaryTrajectory:
    def transform(self, message, context):
        points = message.fields["points"].copy()
        points["longitudinal_velocity_mps"] = 0.0
        return replace(message, fields={**message.fields, "points": points})
```

The IMU hook records yaw rate and returns the input object unchanged. The planning hook sends
a replacement trajectory with zero speed on every point. Both hooks apply to every treatment.
Use treatment-specific hooks or an attack for a comparison that varies the message changes.

To try the hooks on a short run first, change the snippet's base to
`configs/my_autoware_check.yaml` and its output file name, and give the result its own
`runs.output`.

Repeat the run command with `configs/my_autoware_topics.yaml`. To read message records, pass
the saved run directory containing `manifest.json` to `RunLog.read`. Run the snippet in the
experiment service, for example after `docker compose exec carla-experiment python`. Repeated
runs have a `repeat-R` directory, and single runs omit it.

```python
from avsectester.runlog import RunLog

log = RunLog.read("results/my_autoware_topics/runs/segment-00000/seed-0/repeat-0/clean")
for record in log.messages:
    if record.topic == "/planning/trajectory":
        forwarded = record.forwarded if record.modified else record.original
        print(record.step, record.stamp_ns, forwarded.fields["points"]["longitudinal_velocity_mps"])
```

Observation can subscribe to topics carried by DDS that have a unique ROS type. Read-only hooks
receive observed messages, and RunLog can save their payloads. Attack runtimes do not receive
observed messages, and neither `SensorAttackAction` nor `AVStackAttackAction` can target them.
In-process links that bypass DDS are outside this interface.
Modification is available for the [supported topics](../guides/simulators.md#hosted-ros-topics).
Forwarding adds latency, so compare treatments with the same forwarding settings.

## Add sensor noise

The supplied sensors are noise-free. Sensor noise is part of the environment that every treatment
shares, so set it in the recipe rather than in an attack. In your copy, add CARLA's noise attributes
to the `blueprint_attributes` of the matching entry under
`dataset.options.recipes.town10hd_autoware.options.scenario.ego.ros_rig`. These keys add gyro and
accelerometer noise to the `imu` entry:

```yaml
noise_gyro_stddev_z: 0.001
noise_accel_stddev_x: 0.1
```

[ROS rig sensor noise](../reference/configuration.md#ros-rig-sensor-noise) lists the supported
attributes, the GNSS limitation and how noise seeds are chosen.

## If a run fails

The CLI prints each failed run directory and its error. The run's `manifest.json` records its
status, and its `failure` event records the error. Each run also has a session directory under
`work_directory` with `autoware.log`, `adapter.log`, `adapter-events.jsonl`, the relay logs and the
other process logs. The termination event lists these paths under `external_driving_system.logs`,
also for failed runs.

| Error | Check |
|---|---|
| An Autoware readiness step failed or timed out | The error names the step, the last ADAPI state, recent process output and the session directory. `adapter-events.jsonl` shows the readiness timeline. On the first run, the TensorRT engine build can cause a timeout |
| `SensorAlignmentError` before the origin | The CARLA server's GPU is heavily loaded. Choose a less loaded GPU |
| A CARLA connection error | The server is not running, or `connect_port` differs from `CARLA_RPC_PORT` |

The framework stops each Autoware container when its run ends. After an interrupted
experiment, list your remaining containers with
`docker ps -a --filter name=avsectester-autoware-$(id -u)-` and remove only those.

Two servers that share a ROS domain publish into each other's topics. Use a domain nobody else
uses on the host.

## Next steps

- [Hosted ROS topics](../guides/simulators.md#hosted-ros-topics) lists the supported topics.
- [Message stages](../reference/runtime.md#ros-message-stages) describes hook order and timing.
- [Evaluate experiments](../guides/evaluation.md) explains repeated-run comparisons.
