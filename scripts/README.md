# Repository tools

> **Audience:** Experiment users and maintainers.
>
> **Outcome:** Prepare assets and inspect recorded inputs or results.
>
> **Prerequisites:** [Installation](../docs/getting-started/installation.md).

## Preparation

-  [write_compose_env.sh](preparation/write_compose_env.sh) writes local Compose settings.
-  [fetch_models.sh](preparation/fetch_models.sh) downloads avstack's CARLA and
  nuScenes checkpoints.
-  [export_carla_maps.py](preparation/export_carla_maps.py) exports OpenDRIVE maps; follow
  [CARLA semantic maps](../docs/getting-started/installation.md#carla-semantic-maps).
-  [extract_person_cutouts.py](preparation/extract_person_cutouts.py) prepares RGBA person
  assets; follow [Optional assets](../docs/getting-started/installation.md#optional-assets).

## Validation and visualization

[validate_nurec_depth.py](validation/validate_nurec_depth.py) checks ground-mesh alignment
against recorded rig and road points without NRE. Supply `--endpoint` to add rendered
depth-edge overlays.

```bash
docker compose run --rm --entrypoint python nurec-experiment -m scripts.validation.validate_nurec_depth \
  --usdz data/nurec/023b7fcc-671c-40e3-9bd2-c66b0b073fbc/023b7fcc-671c-40e3-9bd2-c66b0b073fbc.usdz --output results/nurec-depth-validation
```

The `--nuscenes DATAROOT` mode of `visualize_scene_labels` and Albumentations run in
`carla-experiment`. Its `--usdz` mode runs in `nurec-experiment`:

```bash
docker compose exec carla-experiment python -m scripts.visualization.visualize_scene_labels --help
docker compose run --rm --entrypoint python nurec-experiment \
  -m scripts.visualization.visualize_scene_labels --help
docker compose exec carla-experiment python -m scripts.visualization.visualize_augmentations --implementation albumentations --help
```

[patch_hide_probe.py](validation/patch_hide_probe.py) and
[validate_carla_visibility.py](validation/validate_carla_visibility.py) inspect CARLA insertion
and visibility. [plot_stop_line_approaches.py](visualization/plot_stop_line_approaches.py) plots
saved run evidence; see the [Stop-sign patch example](../docs/examples/stop-sign-patch.md).

## API reference generation

[write_api_reference.py](development/write_api_reference.py) generates public names, signatures
and descriptions from source; hand-written constraints and links remain outside
generated markers.

```bash
docker compose run --rm --entrypoint python nurec-experiment -m scripts.development.write_api_reference
```

## Next steps

-  [Installation](../docs/getting-started/installation.md)
-  [Supplied examples](../docs/examples/README.md)
