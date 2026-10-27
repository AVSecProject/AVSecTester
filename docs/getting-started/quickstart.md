# Quickstart

> **Audience:** Users running their first experiment.
>
> **Outcome:** Run a comparison and open its saved reports.
>
> **Prerequisites:** [Installation](installation.md), including data and model weights.

## Run a CARLA comparison

Select the available CARLA server and experiment GPUs in the Compose settings described in
[Installation](installation.md). Copy the configuration first:

```bash
cp configs/carla_phantom.yaml configs/my_carla.yaml
```

Edit `configs/my_carla.yaml`: `simulator.options.client.connect_ip` and
`simulator.options.client.connect_port` select the endpoint;
`avstack.options.pipeline_cfg.perception.gpu` selects the perception GPU.

```bash
docker compose up -d carla-server carla-experiment
docker compose exec carla-experiment avsectester run configs/my_carla.yaml
```

The command prepares the configured attacks, executes treatments and evaluates saved evidence.
Workflow and resume behavior are described in [Experiments](../guides/experiments.md).

## Run a NuRec comparison

Place NuRec 26.01 scenes under `data/nurec/26.01_release` and prepare the COCO weights listed in
[Installation](installation.md#coco-stop-sign-detector-weights).
The supplied `split.train_segments: 5` requires at least six selected approaches.
Copy the configuration within `configs/`, then edit the copy before running it. Set
`NUREC_GPU` in `.env`; `simulator.options.renderer.options.endpoint` selects NRE, and the
top-level `device` selects the detector device inside the experiment container.

```bash
cp configs/nurec_stop_sign_patch_modular_2601.yaml configs/my_stop_sign.yaml
docker compose run --rm nurec-experiment run configs/my_stop_sign.yaml
```

The command manages NRE automatically. Service reuse and GPU options are described in the
[CLI reference](../reference/cli.md#automatic-local-services). See
[Stop-sign patch](../examples/stop-sign-patch.md) for white-box attack and black-box attack
configurations, and [Supplied examples](../examples/README.md) for the full configuration list.

## Read the reports

Open `offline_report.html` and `evaluation_report.html` under `runs.output`. The output is in
the mounted checkout on the host. Structured JSON/CSV and per-run evidence are described in the
[Output layout](../reference/cli.md#output-layout).

## Reuse preparation

Copy a configuration, then edit `configs/carla_phantom_reuse.yaml` to set `runs.output`
to `../results/carla_phantom_reused`:

```bash
cp configs/my_carla.yaml configs/carla_phantom_reuse.yaml
```

Reuse preparation with the edited copy:

```bash
docker compose exec carla-experiment avsectester run \
  configs/carla_phantom_reuse.yaml --prepared results/carla_phantom
```

For NuRec, use the same command through `nurec-experiment`:

```bash
docker compose run --rm nurec-experiment run CONFIG.yaml --prepared RESULTS_DIRECTORY
```

See [Saved-preparation compatibility](../reference/configuration.md#saved-preparation-compatibility).
To continue interrupted preparation or execution, use `--resume`:

```bash
docker compose run --rm nurec-experiment run CONFIG.yaml --resume
```

## Re-evaluate saved evidence

Evaluation reads the saved experiment outputs:

```bash
docker compose exec carla-experiment avsectester evaluate configs/my_carla.yaml
docker compose run --rm nurec-experiment evaluate \
  configs/my_stop_sign.yaml
```

See [Evaluation](../guides/evaluation.md) for metrics and comparison groups.

## Stop CARLA

When finished with CARLA commands, stop its services:

```bash
docker compose down
```

## Next steps

Choose a [supplied configuration](../examples/README.md), read
[Writing attacks](../guides/attacks.md), or configure a
[driving system](../guides/driving-systems.md).
