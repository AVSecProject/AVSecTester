# Image attacks — realistic object insertion into rendered camera frames

Camera-facing attacks from the *natural / adversarial physical object* families in
[`PROJECT.md`](PROJECT.md), simulated by inserting the object into each rendered frame (NuRec
neural reconstruction or CARLA) rather than into the world. The attack chooses the **payload**
(what) and the **target** (where); the insertion itself (geometry + harmonization) is shared
simulation code in `avsectester/simulators/patch_insertion.py`.

| Attack | Payload | Placement | Module / demo |
|---|---|---|---|
| Physical patch | checkerboard / optimized texture | lead vehicle rear (detector quad) | `attacks/physical_patch.py`, `scripts/nurec_patch_demo.py` |
| **Fake STOP sign — roadside** | MUTCD R1-1 face + post | fixed world position on the shoulder | `attacks/sign_spoof.py`, `scripts/nurec_sign_demo.py --mode roadside` |
| **Fake STOP sign — on vehicle** | MUTCD R1-1 face | lead vehicle rear (ego-lane detector quad) | `attacks/sign_spoof.py`, `scripts/nurec_sign_demo.py --mode vehicle` |

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

## Running

With an `nre-ga` server serving a NuRec scene (see [`SETUP.md`](SETUP.md) §4b):

```bash
python scripts/nurec_sign_demo.py --endpoint 127.0.0.1:50051 \
    --mode roadside vehicle --harmonizer none classic chroma libcom --frames 50
```

Writes `tmp/nurec_sign/<mode>_<harmonizer>/`: `side_by_side.gif` (clean | attacked), `filmstrip.png`,
`zoom_XXXX.png` crops, and the attacked frames. Roadside placement: `--x/--y` (scene metres; start
pose is the origin, x forward, y left), `--yaw`, `--size`, `--mount`, `--ground-z`.

## Known limitations

- No occlusion: an inserted object is always drawn on top, even if a vehicle passes in front of it.
- No shadows, specular reflection or retro-reflectivity; lighting comes only from harmonization.
- The world-anchored ground height is a parameter (`--ground-z`), not read from the scene.
- Scene `clipgt-01d503d4` only; no driving-impact evaluation yet.
