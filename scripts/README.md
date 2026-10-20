# Repository tools

Run experiments from the repository root with the supplied configurations:

```bash
avsectester run configs/carla_phantom.yaml
avsectester run configs/carla_patch.yaml
avsectester run configs/carla_patch_optimized.yaml
avsectester run configs/carla_object_insertion.yaml
avsectester run configs/nurec_speed_feedback.yaml
avsectester run configs/nurec_object_insertion.yaml
avsectester run configs/nurec_patch.yaml
avsectester run configs/nuscenes_object_insertion.yaml
```

Before running, set the server, device, model and dataset locations in the selected configuration.
Paths are relative to that configuration file. CARLA uses a dedicated synchronous server and the
Modular or camera driving policy. NuRec uses a running neural renderer and Alpamayo in its driver
environment. nuScenes uses recorded images; commands do not alter recorded trajectories. Its
configuration uses the supported Faster R-CNN/nuScenes detector pair and requires the checkpoint
downloaded by scripts/preparation/fetch_models.sh.

The command prepares metadata and artifacts, executes clean and attacked treatments, and writes
RunLog records and evaluation under the configured output. Existing `runs/` directories are
rejected. Reuse saved preparation with `--prepared DIRECTORY`; recompute metrics after the simulator
closes with `avsectester evaluate EXPERIMENT.yaml`. Experiment and optimization settings belong in
the YAML files. Fixed and optimized patches use separate configurations, each with one attack.

## Preparation

- [fetch_models.sh](preparation/fetch_models.sh) downloads the perception models used by the
  supplied driving configurations.
- [extract_person_cutouts.py](preparation/extract_person_cutouts.py) extracts reusable person
  assets using SAM and nuScenes annotations.

See [installation](../docs/getting-started/installation.md) for environments and model locations.

## Validation and visualization

Run Python tools as modules, for example:

```bash
python -m scripts.validation.patch_hide_probe --help
python -m scripts.validation.validate_carla_visibility --help
python -m scripts.visualization.visualize_scene_labels --help
python -m scripts.visualization.visualize_augmentations --help
```

- [patch_hide_probe.py](validation/patch_hide_probe.py) compares patch sizes and harmonization
  on a CARLA SensorView using the same insertion effect used by attacks.
- [validate_carla_visibility.py](validation/validate_carla_visibility.py) measures visibility
  with simulator-owned RGB/depth captures and full-silhouette reference masks on a dedicated CARLA
  world. It does not construct sensor actors, queues or depth decoders.
- [visualize_scene_labels.py](visualization/visualize_scene_labels.py) projects recorded
  FrameData actors onto their camera images, using timestamps to match video frames.
- [visualize_augmentations.py](visualization/visualize_augmentations.py) applies sensor transforms
  to one image and writes a labelled comparison figure.

These tools write files only when invoked. They do not run as part of configuration preflight.
