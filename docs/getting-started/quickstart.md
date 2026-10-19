# Quickstart

> **Audience:** New users who want to see an experiment run before reading the details.  
> **Outcome:** Run a clean/attack comparison, read its output and know where to go next.  
> **Prerequisites:** [Installation](installation.md).


## Run a CARLA comparison

This section needs the [Docker](installation.md#docker) installation, built and started as in
[Build and start](installation.md#build-and-start). From the repository root:

```bash
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40
```

`configs/carla_scenario.yaml` describes one CARLA scene. The ego vehicle carries a LiDAR sensor
and is driven by the modular driving system: a CARLA-trained PointPillars detector, a tracker, a
forward-collision planner with a 6 m/s target speed, and PID control. Six other vehicles drive in
the scene. The configuration enables one attack, `PhantomInjection`, which appends a fabricated
car detection to the detector's output. No pixels or LiDAR points change.

The command runs in `pair` mode with the `attack` [treatment](../concepts/architecture.md#terminology),
the defaults when neither the YAML nor the command line selects others. It prepares one scenario,
runs a clean baseline and then an attacked run from the same origin, and compares the two runs.
`--frames 40` sets 40 driving steps per run. Each run reloads the CARLA server's world, so use a
server dedicated to the experiment.

With the [local full installation](installation.md#local-full-installation) and a running CARLA
server, run the same experiment on the host. `--gpu 1` selects the perception device:

```bash
avsectester run configs/carla_scenario.yaml --frames 40 --gpu 1
```

## Read the output

An attack pair prints lines in this format. Values in angle brackets depend on the scene, the
model and the run length:

```text
[scenario] seed=0
[pair] 40 frames per run, treatment=attack
[clean] mean_detections=<n> peak_speed=<m/s> final_speed=<m/s> brake_frames=<n> outputs=<n>
[attacked] mean_detections=<n> peak_speed=<m/s> final_speed=<m/s> brake_frames=<n> outputs=<n>
clean:    peak_speed=<m/s>  final_speed=<m/s>  brake_frames=<n>
attacked: final_speed=<m/s>  brake_frames=<n>  overdrive_frames=<n> (kept moving while clean was stopped)
=> <verdict>
```

- `[scenario] seed=` is the CARLA seed used for random choices in both runs. See
  [Random choices and seeds](../guides/simulators.md#random-choices-and-seeds).
- `[pair]` repeats the mode, the driving steps per run and the treatment.
- One summary line per run follows. Speeds are the ego vehicle's physical speeds in m/s.
  `mean_detections` is the mean number of perception detections per driving step.
  `brake_frames` counts driving steps whose executed command had zero throttle and positive
  brake. `outputs` counts the intervention outputs recorded during the run.
- The two comparison lines contrast the clean run with the attacked run. `overdrive_frames`
  counts driving steps in which the clean ego had stopped while the attacked ego kept moving.
- The line starting with `=>` is the verdict under the supplied stop-based criterion. An attack
  succeeds when it causes a stop that the clean run did not make, or prevents a stop that the
  clean run made. The result is inconclusive when the clean ego never started driving.

The verdict is one possible outcome, not a guaranteed one. The criterion and its thresholds are
explained in [Driving-impact metric](../guides/evaluation.md#driving-impact-metric). Every output
line is listed in [Output](../reference/cli.md#output). The exit code also reports the outcome, so
scripts can distinguish a successful, failed or inconclusive attack from an error. See
[Exit codes](../reference/cli.md#exit-codes).

## Save a plot

Add `--plot` to save a figure of the comparison. The figure shows the ego speed and the brake
command over the driving steps of both runs. For an attack pair, the verdict is its caption:

```bash
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40 \
  --plot results/impact.png
```

The CLI prints `[output] wrote driving-impact figure: /app/results/impact.png`. Relative paths
are resolved in the container's working directory, `/app`. Compose mounts only `models/`, so copy
the figure to the host:

```bash
mkdir -p results
docker compose cp avsectester:/app/results/impact.png results/impact.png
```

`--plot` works only in `pair` mode. With `--mode single`, the command stops with
`Configuration error: --plot requires pair mode`. The plot needs the `viz` extra, which the Docker
image includes. On the host, the same option writes the figure relative to the current directory.
To plot traces from Python, see [Plot a comparison](../guides/evaluation.md#plot-a-comparison).

## Try other modes and treatments

`--mode` and `--treatment` override the YAML keys `mode` and `treatment`. `single` runs one
treatment without a comparison. `pair` runs a clean baseline and the selected treatment. The
treatments are `clean`, `attack`, `defense` and `attack-defense`, the
[combined treatment](../concepts/architecture.md#terminology). Which interventions each one
enables is described in [Treatments and modes](../guides/experiments.md#treatments-and-modes).

With the supplied configuration:

```bash
# One attacked run, without a comparison or verdict
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --mode single --treatment attack
# One clean run
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --mode single --treatment clean
# Two clean runs from the same origin, compared without a verdict
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --mode pair --treatment clean
```

A single run prints only its summary line. A clean pair prints `[pair] Baseline comparison.` and
labels its second run `baseline`.

The supplied configuration contains an attack only. `--treatment defense` and
`--treatment attack-defense` need a configured defense. Without one, the command stops before
any run with `Configuration error: No enabled intervention configured for ['defense']`. To add a
defense, declare it under `interventions` with `kind: defense`, as described in
[Package and configure a defense](../guides/defenses.md#package-and-configure-a-defense). In the
output, the combined treatment is labeled `attack+defense`.

## Run offline without a GPU

This example runs a clean drive without CARLA, a GPU or model weights. `NuRecSimulator` with
`StubRenderer` renders black camera frames and moves the ego vehicle with an in-process kinematic
model. A minimal `AVStack` reads the reported speed and returns actuator commands. `run` executes
the driving steps and returns a `Trace`.

Save this code as `offline_drive.py`:

<!-- runnable -->
```python
from avsectester.closed_loop.avstack import AVStack
from avsectester.backend import run
from avsectester.closed_loop.datatypes import Control, Observation
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer


class CruiseControl(AVStack):
    """Accelerate until the reported speed reaches a target, then hold it."""

    def __init__(self, target_speed: float = 5.0) -> None:
        self.target_speed = target_speed

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset({"actuator"})

    def __call__(self, observation: Observation) -> Control:
        if observation.ego_speed < self.target_speed:
            return Control(throttle=0.5)
        return Control()


backend = NuRecSimulator({"dt": 0.1}, renderer=StubRenderer())
try:
    trace = run(backend, CruiseControl(), frames=50)
finally:
    backend.close()  # run() leaves the backend open. The caller closes it.

for record in trace.records[::10]:
    # record.frame is the driving-step index. Speed is the physical speed after the step.
    print(f"step={record.frame} t={record.t:.1f}s speed={record.speed:.2f} throttle={record.throttle}")
print(f"peak_speed={trace.peak_speed:.2f} final_speed={trace.final_speed:.2f}")
```

Run it in the [local core](installation.md#local-core-installation) environment:

```bash
python offline_drive.py
```

Or save it in the repository root and run it with the Docker image. Mounting the checkout at
`/app` makes the container use your current code rather than the code copied into the image when
it was built:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/app -w /app -e HOME=/tmp \
  -e NVIDIA_VISIBLE_DEVICES=void --entrypoint python avsectester:latest offline_drive.py
```

The output is:

```text
step=0 t=0.1s speed=0.15 throttle=0.5
step=10 t=1.1s speed=1.65 throttle=0.5
step=20 t=2.1s speed=3.15 throttle=0.5
step=30 t=3.1s speed=4.65 throttle=0.5
step=40 t=4.1s speed=5.10 throttle=0.0
peak_speed=5.10 final_speed=5.10
```

The default dynamics turn half throttle into 1.5 m/s² of acceleration. Once the reported speed
passes 5 m/s, the zero command holds the speed. Each record holds the physical state after one
driving step and the command executed in it. Reading traces is described in
[Driving traces](../guides/evaluation.md#driving-traces).

To change what the driving system sees, register a handler with a `Runtime` and pass it to `run`.
Continue with [Write your first attack](../guides/attacks.md#write-your-first-attack).

## Next steps

- [Run experiments](../guides/experiments.md): entry points, treatments and modes, and outputs.
- [Implement an attack](../guides/attacks.md): write, package and configure your own attack.
- [Evaluate results](../guides/evaluation.md): read traces, verdicts and intervention outputs.
- [Architecture](../concepts/architecture.md): the components and terms behind these runs.
