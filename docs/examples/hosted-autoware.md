# Hosted Autoware

> **Audience:** Users running CARLA experiments with Autoware.
>
> **Outcome:** Compare hosted Autoware runs and save messages from selected ROS 2 topics.
>
> **Prerequisites:** [Installation](../getting-started/installation.md#hosted-autoware),
> including the pinned image, model data, ROS 2 network and Autoware assets.

## Run the supplied comparison

Copy [carla_autoware.yaml](../../configs/carla_autoware.yaml) and edit the copy.

```bash
cp configs/carla_autoware.yaml configs/my_autoware.yaml
```

Set the host GPU, ROS domain and model data path under
`simulator.options.external_driving_system.options`. Use the same domain as the CARLA server
and set `runs.output` to `../results/my_autoware`.

The Town10HD configuration runs Autoware in LiDAR mode. The clean treatment leaves a stationary
vehicle outside ego's route. The relocation treatment moves it into ego's lane instantly at
the configured decision using `SetActorState`. Each treatment runs five times by default.

Run with the same Compose overlays you used to start CARLA.

```bash
docker compose -f docker-compose.yml -f docker-compose.ros2.yml -f docker-compose.autoware.yml \
  exec carla-experiment avsectester run configs/my_autoware.yaml
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
  type: hosted_topics:ObserveImu
- stage: /planning/trajectory
  type: hosted_topics:StationaryTrajectory
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

Save this module as `hosted_topics.py` at the checkout root. The experiment service imports
the two hook classes from the mounted checkout.

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

Repeat the run command with `configs/my_autoware_topics.yaml`. To read message records, pass
the saved run directory containing `manifest.json` to `RunLog.read`.

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

The supplied sensors are noise-free. Sensor noise is part of the environment that every
treatment shares, so set it in the recipe rather than in an attack. In your copy, add CARLA's
noise attributes to the `blueprint_attributes` of the matching entry under
`scenario.ego.ros_rig`. These keys add gyro and accelerometer noise to the `imu` entry:

```yaml
noise_gyro_stddev_z: 0.001
noise_accel_stddev_x: 0.1
```

[ROS rig sensor noise](../reference/configuration.md#ros-rig-sensor-noise) lists the supported
attributes, the GNSS limitation and how noise seeds are chosen.

## Next steps

- [Hosted ROS topics](../guides/simulators.md#hosted-ros-topics) lists the supported topics.
- [Message stages](../reference/runtime.md#ros-message-stages) describes hook order and timing.
- [Evaluate experiments](../guides/evaluation.md) explains repeated-run comparisons.
