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
-  [export_carla_maps.py](preparation/export_carla_maps.py) exports OpenDRIVE maps. Follow
  [CARLA semantic maps](../docs/getting-started/installation.md#carla-semantic-maps).
-  [extract_person_cutouts.py](preparation/extract_person_cutouts.py) prepares RGBA person
  assets. Follow [Optional assets](../docs/getting-started/installation.md#optional-assets).

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
saved run evidence. See the [Stop-sign patch example](../docs/examples/stop-sign-patch.md).

## API reference generation

[write_api_reference.py](development/write_api_reference.py) generates public names, signatures
and descriptions from source. Hand-written constraints and links remain outside
generated markers.

```bash
docker compose run --rm --entrypoint python nurec-experiment -m scripts.development.write_api_reference
```

## Project website

[hooks.py](website/hooks.py) publishes docs/ unchanged under /docs/ beside the homepage in
`website/content/`. It rewrites links that leave docs/ to GitHub at the built commit and removes the
table-cell pipe escapes that only GitHub needs. The [TODO](../docs/development/TODO.md) stays on
GitHub only. Site links to it go to GitHub. [check_site.py](website/check_site.py) checks a
built site. The [avsecproject.github.io](https://github.com/AVSecProject/avsecproject.github.io)
repository rebuilds the site from main every hour, or on a manual run, and serves it at
https://avsecproject.github.io/. Build and check it in a separate virtual environment:

```bash
python -m pip install -r website/requirements.txt
mkdocs build --strict -f website/mkdocs.yml
python -m scripts.website.check_site website/site
```

### Website media

[make_media.py](website/make_media.py) turns saved experiment outputs into the homepage media in
`website/content/assets/media/`. It reads runs through the documented output layout and the public
`avsectester.runlog` and `avsectester.evaluation.visualization` interfaces. Each media file
comes from a supplied configuration with the changes listed below. The derived YAML files are
not tracked.

All runs record the front camera's raw and model-input images at every decision and use a
separate NRE renderer port or CARLA server port. Videos play at 8 frames per second, slower than
the decisions.

| Media | Source configuration and changes | Run (segment, seed, treatments) | Command (`python -m scripts.website.make_media ...`) |
|---|---|---|---|
| `stop-sign-white-box` | `nurec_stop_sign_patch_modular_2601.yaml`, treatments `clean` (attack `clean`) and `noise_detector_pgd` only | 15 (scene `clipgt-d2fcf4b2-2a4b-47e3-a2ec-a9e3372c3a8a`), 0, `clean` and `noise_detector_pgd` | `compare OUTPUT --segment 15 --attacked noise_detector_pgd --clean-label "Clean" --attacked-label "White-box attack (PGD)" --fps 8 --poster-step 22 --name stop-sign-white-box` |
| `stop-sign-patch` | `nurec_stop_sign_patch_alpamayo_2601.yaml`, one scene in `usdz_paths`, no `split`, treatments `white` and `white_unharmonized` (the `white` attack without its harmonizer) | 0 (scene `clipgt-7b186d92-d6f5-4e27-a324-d705d1fad7e1`), 0, `white_unharmonized` | `frame OUTPUT --segment 0 --treatment white_unharmonized --step 60 --crop 800 360 1760 900 --width 640 --name stop-sign-patch` |
| `stop-sign-patch-harmonized` | The same run | 0, 0, `white` | `frame OUTPUT --segment 0 --treatment white --step 60 --crop 800 360 1760 900 --width 640 --name stop-sign-patch-harmonized` |
| `image-insertion` | `carla_object_insertion.yaml`, sign at `position_m: [12.0, -4.0, 0.0]`, `settle_iters: 1000` | 0, 0, `object` | `frame OUTPUT --segment 0 --treatment object --camera camera-0-0 --step 19 --crop 360 164 760 389 --width 640 --name image-insertion` |
| `image-insertion-harmonized` | The same run. Attack `object_harmonized` adds `compositor: PatchCompositor(ExposureHarmonizer(), soften: 0.6)` | 0, 0, `object_harmonized` | `frame OUTPUT --segment 0 --treatment object_harmonized --camera camera-0-0 --step 19 --crop 360 164 760 389 --width 640 --name image-insertion-harmonized` |
| `physical-patch` | `nurec_patch.yaml` | 0, 0, `patch` | `frame OUTPUT --segment 0 --treatment patch --step 19 --crop 360 610 1000 970 --width 640 --name physical-patch` |
| `physical-patch-harmonized` | `nurec_patch.yaml`, adds `compositor: PatchCompositor(ExposureHarmonizer(), soften: 0.6)` | 0, 0, `patch` | `frame OUTPUT --segment 0 --treatment patch --step 19 --crop 360 610 1000 970 --width 640 --name physical-patch-harmonized` |
| `stop-line-approach` | Runs of `stop-sign-white-box` | Same segment, seed and treatments | `stop-line CONFIG --segment 15 --attacked noise_detector_pgd --clean-label "Clean" --attacked-label "White-box attack (PGD)" --name stop-line-approach` |
| `evaluation-report` | `evaluation_report.html` of the `stop-sign-white-box` run | Report header and experiment setup | Browser screenshot at 1440 px width, top 1380 px without the scrollbar, then `still SCREENSHOT --name evaluation-report` |

The stop-sign runs use the local NuRec 26.01 release. `stop-sign-white-box` shows a test unit whose
clean run stops before the line without a collision and whose attacked run crosses it.
The stop-sign patch frames show a decision where the replica sign is visible beside a real stop
sign of the scene. The object insertion and physical patch frames show the last decision of their
runs. The CARLA sign is placed closer than in the supplied configuration so that it is legible in
the 800x600 camera.

## Next steps

-  [Installation](../docs/getting-started/installation.md)
-  [Supplied examples](../docs/examples/README.md)
