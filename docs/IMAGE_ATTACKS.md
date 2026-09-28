# Image attacks — realistic object insertion into rendered camera frames

Camera-facing attacks from the *natural / adversarial physical object* families in
[`PROJECT.md`](PROJECT.md), simulated by inserting the object into each rendered frame (NuRec
neural reconstruction or CARLA) rather than into the world. The attack chooses the **payload**
(what) and the **target** (where); the insertion itself (geometry + harmonization) is shared
simulation code in `avsectester/simulators/patch_insertion.py`.

| Attack | Payload | Placement | Module / demo |
|---|---|---|---|
| Physical patch | checkerboard / optimized texture | lead vehicle rear (detector quad) | `attacks/physical_patch.py`, `scripts/nurec_patch_demo.py` |
| **Fake STOP sign — roadside** | MUTCD R1-1 face + post | fixed world position on the shoulder | `attacks/sign_spoof.py`, `scripts/nurec_object_demo.py --mode roadside` |
| **Fake STOP sign — on vehicle** | MUTCD R1-1 face | lead vehicle rear (ego-lane detector quad) | `attacks/sign_spoof.py`, `scripts/nurec_object_demo.py --mode vehicle` |
| **Phantom person — standee** | life-size cut-out of a real pedestrian | fixed world position on the shoulder | `attacks/person_poster.py`, `--object standee --mode roadside` |
| **Phantom person — billboard** | the person printed on a poster board on two legs | roadside, or a poster on the lead vehicle's rear | `attacks/person_poster.py`, `--object billboard --mode roadside vehicle` |

Person cut-outs come from nuScenes camera images, segmented with SAM from the annotated 2-D boxes
(`scripts/extract_person_cutouts.py`). nuScenes is CC BY-NC-SA 4.0, so the cut-outs are not in the
repo; the demo takes one with `--asset`. The STOP face is public domain and ships in
`avsectester/assets/signs/`.

## Two ways to place an object

- **Image-anchored** (`detector_quad` → `warp_patch`): a 2-D detector box gives a quad on the target
  surface; the object is homography-warped onto it. No depth or camera model needed, so it works on
  any imagery, but it only follows what the detector finds. `pick="lane"` keeps the object on the
  in-lane lead vehicle when a larger car overtakes, and `hold_quad` bridges missed detections.
- **World-anchored** (`render_plane`): the object is a textured 3-D rectangle at a fixed scene
  position, rendered through the real camera — `camera_models.FThetaCamera` for NuRec's 120° f-theta
  fisheye, `PinholeCamera` for pinhole cameras. Each pixel is ray-cast onto the plane, which is exact
  under lens distortion (a 4-corner homography is not), and the texture is pre-filtered to its
  on-screen size. The object keeps correct perspective and scale as the ego moves.
  `NuRecRenderer.camera_model()` / `cam_from_world(pose)` expose the camera for this.

Both become a `perturb(Observation)` via `patch_insertion.frame_perturbation`, so the AV stack
perceives the inserted object in closed loop (not only the visualization).

## Harmonization

| Harmonizer | What it does | On a STOP sign |
|---|---|---|
| none | alpha paste | too bright and saturated for an overcast scene; stands out |
| `ClassicHarmonizer()` (default) | Lab mean/std transfer + Poisson blend | **destroys the red**: the sign turns into grey "STOP" lettering with a halo |
| `ClassicHarmonizer(preserve_chroma=True, blend="feather")` | exposure gain on lightness only + feathered edge | keeps hue and legend contrast; dims to the scene |
| `PCTNetHarmonizer()` (libcom) | learned color transform | natural, darker red |

Colour-transfer harmonizers suit textures whose hue does not matter (a patch), but wash out objects
whose colour carries meaning. Every harmonizer here estimates the lighting from the pixels around the
object, so a sign on a dark truck comes out too dark: a dark *surface* is read as dark *light*.

### Does perception see it?

`--eval` runs the COCO Faster R-CNN (torchvision) on every clean and attacked frame and records the best
`stop sign` score for a box on the inserted sign. Scene `clipgt-01d503d4`, 50 frames at 3.7 m/s:

| Placement | none | classic | chroma | libcom |
|---|---|---|---|---|
| roadside (x=28 m, y=-6.5 m) | 50/50 | **0/50** | 50/50 | 50/50 |
| lead vehicle rear | 48/50 | **0/50** | 47/50 | 47/50 |

Phantom person (cut-out `person_001`, COCO `person` class):

| Object | none | classic | chroma | libcom |
|---|---|---|---|---|
| standee, roadside (x=25 m, y=-3.2 m) | 50/50 | **4/50** | 50/50 | 50/50 |
| billboard, roadside (x=28 m, y=-7 m) | 50/50 | 50/50 | 50/50 | 50/50 |
| poster on lead vehicle rear | 47/50 | 48/50 | 48/50 | 48/50 |

Frames with score >= 0.5, out of the frames where the object is in view; the clean runs score 0
throughout. A printed person is detected as a pedestrian in nearly every frame, which is the
phantom attack. The choice of harmonizer decides whether the attack works at all: the default classic
harmonizer removes the sign's colour, and with it the detection; it likewise washes a person
silhouette into the background (the standee), while a poster's own paper and frame shield the figure.

### Across real scenes (nuScenes)

`scripts/nuscenes_object_demo.py` places the same three objects in the ego frame of real nuScenes
`CAM_FRONT` photos (pinhole camera from the image's calibration) and harmonizes with PCTNet. A spot is
used only if its footprint overlaps no annotated object. On 8 val images (Boston and Singapore, 6 day,
2 night) the COCO detector finds the STOP sign, the standee and the billboard person in 8/8 images
each, and nothing at those spots in the clean images. Seen in the grid: an unlit white poster stays
too bright at night, and a spot can land in a traffic lane because no map is consulted.

### Driving impact (Alpamayo-1.5, closed loop)

`scripts/alpamayo_attack_demo.py` drives the real Alpamayo-1.5-10B through the NuRec scene for 60
frames (6 s) clean and with the roadside STOP sign (PCTNet), same inference seeds. With the sign 28 m
ahead / 6.5 m right the final speed is 4.59 vs 4.59 m/s; 30 m ahead / 4 m right, 4.65 vs 4.57 m/s. No
driving impact: the policy follows the truck ahead throughout and its reasoning never mentions the
sign. In the near placement the attacked run's reasoning calls the lead vehicle "stopped" 7 times
(0 clean), a possible perception shift that one run cannot confirm. Once the ego leaves the recorded
path, NuRec renders visible artifacts on neighbouring vehicles.

## Running

With an `nre-ga` server serving a NuRec scene (see [`SETUP.md`](SETUP.md) §4b):

```bash
python scripts/nurec_object_demo.py --endpoint 127.0.0.1:50051 --object stop \
    --mode roadside vehicle --harmonizer none classic chroma libcom --frames 50 --eval

# the three objects on real nuScenes photos
python scripts/nuscenes_object_demo.py --nuscenes <nuscenes root> \
    --asset <assets>/pedestrians/person_001.png --n 8 --harmonizer libcom --eval

# person cut-outs (once), then a standee / billboard
python scripts/extract_person_cutouts.py --nuscenes <nuscenes root> --out <assets>/pedestrians
python scripts/nurec_object_demo.py --endpoint 127.0.0.1:50051 --object billboard \
    --asset <assets>/pedestrians/person_001.png --mode roadside vehicle --eval
```

Writes `tmp/nurec_<object>/<mode>_<harmonizer>/`: `side_by_side.gif` (clean | attacked), `filmstrip.png`,
`zoom_XXXX.png` crops, and the attacked frames; with `--eval` also `perception_eval.json` and
`perception_<mode>.png`. Roadside placement: `--x/--y` (scene metres; start
pose is the origin, x forward, y left), `--yaw`, `--size`, `--mount`, `--ground-z`; each object has
its own default position.

## Known limitations

- No occlusion: an inserted object is always drawn on top, even if a vehicle passes in front of it.
- No shadows, specular reflection or retro-reflectivity; lighting comes only from harmonization.
- The world-anchored ground height is a parameter (`--ground-z`), not read from the scene, and
  placements do not consult a map (drivable area, sidewalk).
- No light sources: an unlit object is only as dark as its surroundings suggest (a white poster at
  night stays too bright).
- Scene `clipgt-01d503d4` only; single-seed closed-loop runs. Perception is scored with a COCO
  detector, not the stack under test.
