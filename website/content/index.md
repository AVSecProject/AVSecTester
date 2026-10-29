---
template: home.html
hide: [navigation, toc]
---

## See it run

<figure class="avs-figure">
  <video class="avs-video" muted loop playsinline autoplay preload="metadata"
         poster="assets/media/stop-sign-white-box.webp"
         aria-label="Side-by-side video of a clean run and a white-box attack run of the same stop-sign approach in a NuRec scene, with the rule-based driving system's stop-sign detections and a speed plot">
    <source src="assets/media/stop-sign-white-box.mp4" type="video/mp4">
  </video>
  <figcaption>In this example run, the rule-based driving system detects the replica stop sign from far
  away in the clean run (left) and stops before the line. With a patch from a white-box attack
  (algorithm used: PGD) on its detector (right), it first detects the sign too close to stop and
  crosses the stop line.</figcaption>
</figure>

## Why closed loop

A patch can hide a stop sign in most camera frames and the vehicle can still stop: one detection
at the right distance can be enough. Per-frame detection scores would call that attack a success.
The driving outcome shows that it failed. AVSecTester keeps the driving system in the loop. Its
commands move the simulated vehicle, the next camera frame follows from that motion, and the
attack acts again at every decision.

- Measure driving outcomes such as stopping, braking, stop-line distance and collisions.
- Compare clean and attacked drives of the same segment, started from the same state and seed.
- Keep the vehicle's physical state separate from what the driving system received.

## What you can do with it

AVSecTester is built for people who develop and evaluate autonomous-vehicle security: researchers
who design attacks and defenses, and engineers who test driving systems against them.

<div class="avs-grid avs-grid--four" markdown>

<div class="avs-card" markdown>

### Test a new attack

Does your attack change how the vehicle drives, not just a detection score? Write it in Python or
start from a supplied attack: choose where it applies, optimize it offline and act at every
decision. Each test segment runs clean and attacked from the same start.

[Attacks](docs/guides/attacks.md){ .avs-card__link }

</div>

<div class="avs-card" markdown>

### Stress-test a driving system

How does your driving system behave under known attacks? Wrap it as an `AVStack`. A
`DrivingAdapter` exposes its components to attacks. Run supplied attacks over many drives and get
paired reports of stopping, braking, stop-line and collision outcomes.

[Driving systems](docs/guides/driving-systems.md){ .avs-card__link }

</div>

<div class="avs-card" markdown>

### Optimize an attack

How strong can an attack become with gradients, or with queries only? Optimize artifacts with a
white-box attack (algorithm used: PGD) through the system's own detector, or a black-box attack
(algorithm used: NES) through driving rollouts, then test them on held-out segments.

[Optimization](docs/guides/optimization.md){ .avs-card__link }

</div>

<div class="avs-card" markdown>

### Record and replay runs

Every run saves the camera input the driving system received, its outputs and the vehicle's
physical state. Measure, compare, replay and export frames later without running the simulator
again.

[Evaluation](docs/guides/evaluation.md){ .avs-card__link }

</div>

</div>

Defense evaluation is planned. See [Defenses](docs/guides/defenses.md).

## How it works

<div class="avs-steps" markdown>

1. **Choose where to drive.** Use real-world drives reconstructed with NuRec or CARLA scenarios,
   and a supplied driving system or your own.
2. **Choose or write an attack.** It selects the segments it applies to, prepares its artifacts in
   an offline phase and acts on the world, the sensors or the driving system in the online phase.
3. **Run and compare.** One command prepares the attack, runs clean and attacked treatments of
   every test segment in closed loop and writes metrics and an HTML report.

</div>

```yaml
# configs/nurec_stop_sign_patch_modular_2601.yaml, abridged
dataset:
  type: NuRecDataset            # reconstructed real-world drives
simulator:
  type: NuRecSimulator          # commands move the vehicle and the next frame follows
avstack:
  type: RuleBasedAVStack        # or AlpamayoAVStack, or your own AVStack
attacks:
  noise_detector_pgd:
    type: StopSignPatchAttack   # or your own Attack
    options:
      initial_patch: low_frequency_noise
      optimization: ...         # white-box attack (algorithm used: PGD)
runs:
  treatments:
    - {name: clean, attack_name: no_patch}
    - {name: noise_detector_pgd, attack_name: noise_detector_pgd}
```

```bash
docker compose run --rm nurec-experiment run configs/nurec_stop_sign_patch_modular_2601.yaml
```

## Architecture

<figure class="avs-figure" markdown>
[![Framework diagram. User configuration feeds a dataset, an attack and the closed loop. Inside the attack, the metabuilder produces metadata, the offline phase produces artifacts and the online phase acts on the closed loop, where a simulator sends observations to an AV stack and receives commands. Run logs go to evaluation.](docs/assets/architecture.png)](docs/assets/architecture.png)
</figure>

## Supported components

<div class="avs-components" markdown>

**Closed-loop simulators:** NuRec (reconstructed real-world drives) · CARLA<br>
**Driving systems:** modular · end-to-end<br>
[Supported configurations](docs/concepts/architecture.md#supported-configurations)

</div>

## Examples

<div class="avs-grid avs-grid--wide" markdown>

<div class="avs-card" markdown>

### Stop-sign patch attack

<div class="avs-pair" markdown>
<figure markdown>
![Replica stop sign with a white patch at a NuRec intersection, brighter and more saturated than a real stop sign nearby](assets/media/stop-sign-patch.webp)
<figcaption>Without harmonization</figcaption>
</figure>
<figure markdown>
![The same replica stop sign, matched to the exposure of the scene and of the real stop sign nearby](assets/media/stop-sign-patch-harmonized.webp)
<figcaption>With harmonization</figcaption>
</figure>
</div>

In these example runs, a replica stop sign with a white patch is placed over the reconstructed
sign in a NuRec drive while Alpamayo drives. The two images show the same decision of runs that
differ only in compositing: without harmonization, and with exposure matching by
`ExposureHarmonizer`. The sign on the left of each image is a real stop sign in the scene.

[Stop-sign patch attack](docs/examples/stop-sign-patch.md){ .avs-card__link }

</div>

<div class="avs-card" markdown>

### Object insertion attack

<div class="avs-pair" markdown>
<figure markdown>
![Stop sign inserted beside a CARLA road, brighter and sharper than the surrounding scene](assets/media/image-insertion.webp)
<figcaption>Without harmonization</figcaption>
</figure>
<figure markdown>
![The same inserted stop sign, matched to the exposure of the surrounding CARLA scene](assets/media/image-insertion-harmonized.webp)
<figcaption>With harmonization</figcaption>
</figure>
</div>

In this example in CARLA, a stop sign is inserted into the front-camera image while a
camera-based forward-collision system drives. Only the camera image changes. The simulated world
contains no sign. The two images show the same decision of treatments that differ only in
compositing: without harmonization, and with exposure matching by `ExposureHarmonizer` and slight
edge softening.

[Object insertion attack](docs/examples/image-insertion.md){ .avs-card__link }

</div>

<div class="avs-card" markdown>

### Physical patch attack

<div class="avs-pair" markdown>
<figure markdown>
![Checkerboard patch on the rear of a white car on a highway at dusk, brighter than the car](assets/media/physical-patch.webp)
<figcaption>Without harmonization</figcaption>
</figure>
<figure markdown>
![The same checkerboard patch, darkened to the exposure of the car it is placed on](assets/media/physical-patch-harmonized.webp)
<figcaption>With harmonization</figcaption>
</figure>
</div>

In these example runs, a checkerboard patch is composited onto the rear of a recorded vehicle in
the NuRec front-camera image while Alpamayo drives. The two images show the same decision of runs
that differ only in compositing: without harmonization, and with exposure matching by
`ExposureHarmonizer` and slight edge softening.

[Physical patch attack](docs/examples/physical-patch.md){ .avs-card__link }

</div>

</div>

## Evaluation

An experiment output contains the selected metadata, the offline reports, a `RunLog` for every
run and an evaluation report in HTML and structured form. Metrics are computed from the saved logs
without a simulator.

<div class="avs-grid avs-grid--wide" markdown>

<figure class="avs-figure" markdown>
![Speed against signed distance to the stop line for the clean run and the white-box attack run of the example segment](assets/media/stop-line-approach.png)
<figcaption>Speed against distance to the stop line, plotted from the saved logs of the runs shown at the
top of this page.</figcaption>
</figure>

<figure class="avs-figure" markdown>
![Top of the HTML evaluation report with its contents, reading guide and experiment setup](assets/media/evaluation-report.webp)
<figcaption>The HTML evaluation report written for the same experiment.</figcaption>
</figure>

</div>

[Evaluation](docs/guides/evaluation.md)

## Before you try

- **Requirements.** Linux with NVIDIA GPUs, Docker Compose and the NVIDIA Container Toolkit.
  Experiments run in the supplied Docker images. NuRec scenes, NRE images and model weights are
  downloaded separately and need NGC and Hugging Face access. See
  [Installation](docs/getting-started/installation.md).
- **Status.** AVSecTester is an early research release. The supported simulator and
  driving-system combinations are listed in
  [Supported configurations](docs/concepts/architecture.md#supported-configurations).

## Responsible use

AVSecTester is intended for research, education and authorized security testing of autonomous
driving systems. Its attacks run in simulation and on recorded or reconstructed data.

- Do not use the attacks, or physical artifacts produced with them, against vehicles, road
  infrastructure or systems that you do not own or are not authorized to test, and never on
  public roads.
- Follow applicable laws and the terms of the datasets and models you use.
- If you find a vulnerability in a real system, report it to the affected vendor before publishing
  it.
- Results in simulation do not establish that a real vehicle is safe or unsafe. The software is
  provided under the MIT License, without warranty.

<div class="avs-attribution" markdown>

[MIT License](https://github.com/AVSecProject/AVSecTester/blob/main/LICENSE) ·
[GitHub](https://github.com/AVSecProject/AVSecTester)

Media on this page are rendered from the NVIDIA PhysicalAI Autonomous Vehicles NuRec dataset
(NVIDIA Corporation, NVIDIA Autonomous Vehicle Dataset License Agreement) and the CARLA simulator.
The Alpamayo examples use NVIDIA Alpamayo 1.5 for research under the NVIDIA License
(non-commercial use).

</div>
