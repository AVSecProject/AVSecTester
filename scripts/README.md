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

## Project website

[hooks.py](website/hooks.py) publishes docs/ unchanged under /docs/ beside the homepage in
`website/content/`. It rewrites links that leave docs/ to GitHub at the built commit and removes the
table-cell pipe escapes that only GitHub needs. The [TODO](../docs/development/TODO.md) stays on
GitHub only; site links to it go to GitHub. [check_site.py](website/check_site.py) checks a
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
comes from a supplied configuration with the changes listed below; the derived YAML files are
not tracked.

All runs record the front camera's raw and model-input images at every decision and use a
separate renderer port. Videos play at 8 frames per second, slower than the 10 Hz decisions.

| Media | Source configuration and changes | Run (segment, seed, treatments) | Command (`python -m scripts.website.make_media ...`) |
|---|---|---|---|
| `stop-sign-white-box` | `nurec_stop_sign_patch_modular_2601.yaml`; treatments `clean` (attack `no_patch`) and `white_detector_pgd` only; `metadata.load` of the first seven selected units in dataset order, of which the split keeps five for training | 0 (scene `clipgt-07981e6a-22dd-4796-ad2f-1252037ecd28`), 0, `clean` and `white_detector_pgd` | `compare OUTPUT --segment 0 --attacked white_detector_pgd --clean-label "clean (unpatched replica)" --attacked-label "white-box attack (algorithm used: PGD)" --fps 8 --poster-step 28 --name stop-sign-white-box` |
| `stop-sign-black-box` | `nurec_stop_sign_patch_alpamayo_2601.yaml`; treatments `clean` and `white_rollout_nes` only; first seven units that AlpamayoAVStack can start; NES `steps: 3`, `population_size: 4`, `start_distances_m: [30]`, `max_training_segments: 1` (reduced budget for illustration) | 0 (same scene as `stop-sign-white-box`), 0, `white_rollout_nes` | `overlay OUTPUT --segment 0 --treatment white_rollout_nes --label "black-box attack (algorithm used: NES)" --fps 8 --poster-step 40 --name stop-sign-black-box` |
| `image-insertion` | `nurec_object_insertion.yaml` | 0, 0, `object` | `frame OUTPUT --segment 0 --treatment object --step 19 --crop 990 560 1630 920 --width 640 --name image-insertion` |
| `image-insertion-harmonized` | `nurec_object_insertion.yaml`; adds `compositor: PatchCompositor(ExposureHarmonizer(), soften: 0.6)` | 0, 0, `object` | `frame OUTPUT --segment 0 --treatment object --step 19 --crop 990 560 1630 920 --width 640 --name image-insertion-harmonized` |
| `physical-patch` | `nurec_patch.yaml` | 0, 0, `patch` | `frame OUTPUT --segment 0 --treatment patch --step 19 --crop 360 610 1000 970 --width 640 --name physical-patch` |
| `physical-patch-harmonized` | `nurec_patch.yaml`; adds `compositor: PatchCompositor(ExposureHarmonizer(), soften: 0.6)` | 0, 0, `patch` | `frame OUTPUT --segment 0 --treatment patch --step 19 --crop 360 610 1000 970 --width 640 --name physical-patch-harmonized` |
| `stop-line-approach` | Runs of `stop-sign-white-box` | Same segment, seed and treatments | `stop-line CONFIG --segment 0 --attacked white_detector_pgd --clean-label "clean (unpatched replica)" --attacked-label "white-box attack (algorithm used: PGD)" --name stop-line-approach` |
| `evaluation-report` | `evaluation_report.html` of the `stop-sign-white-box` run | Report header and experiment setup | Browser screenshot at 1440 px width, top 1380 px without the scrollbar, then `still SCREENSHOT --name evaluation-report` |

Stop-sign selections run on the local NuRec 26.01 release before the first seven units are saved
with `save_metadata`. `stop-sign-white-box` shows the test unit whose clean and attacked runs
differ most clearly; `stop-sign-black-box` shows the same scene. The insertion and patch frames
show the last decision of their single run.

## Next steps

-  [Installation](../docs/getting-started/installation.md)
-  [Supplied examples](../docs/examples/README.md)
