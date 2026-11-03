# Vehicle insertion

> **Audience:** Users inserting a custom vehicle mesh into a reconstructed scene.
>
> **Outcome:** Load a mesh from configuration and insert it into NuRec camera and LiDAR captures.
>
> **Prerequisites:** [Installation](../getting-started/installation.md), NuRec 26.04 scene archives
> and the YOLOX weights used by the supplied rule-based driving system.

## Configuration

Run the supplied configuration from the repository root:

```bash
docker compose run --rm nurec-experiment run configs/nurec_vehicle_insertion_modular_2604.yaml
```

The configuration uses `NuRecSimulator` with `object_insertion: scene`, the front-wide camera
and `lidar_gt_top_p128`. Its `modular` name refers to `RuleBasedAVStack`. Service startup follows
the [CLI service lifecycle](../reference/cli.md#automatic-local-services).

The attack constructs its required mesh asset explicitly:

```yaml
attacks:
  vehicle:
    type: VehicleInsertionAttack
    options:
      asset:
        type: avsectester.rendering:load_mesh_asset
        options:
          path: ../avsectester/implemented_attacks/vehicle_insertion_attack/models/real_car_painted.ply
          category: vehicle
          scale: 1.0
          forward_axis: "+x"
          up_axis: "+z"
          origin: file
      harmonizer:
        type: avsectester.rendering.harmonizers:ExposureHarmonizer
        options: {reference: ring}
      distance_m: 25.0
      lateral_offset_m: 0.0
      decision_steps: 20
      max_segments: 1
```

The path resolves relative to the configuration file. `origin: file` keeps the model's own
origin, with the wheels resting on z = 0. The file already contains face colors, so no fallback
`color` is supplied. For other formats, unit conversions and axes, see
[Mesh assets](../reference/configuration.md#mesh-assets).

The supplied `real_car.ply` is a white low-polygon body without wheels.
`real_car_painted.ply` scales it to 4.2 m × 1.9 m × 1.6 m and adds body paint, dark window
panes, lights, a plate and four wheels, with simple shading baked into its colors.
[paint_demo_vehicle.py](../../scripts/preparation/paint_demo_vehicle.py) regenerates it from
`real_car.ply`. Both are illustrative assets. Replace the asset specification to use another
model. The attack has no default model.

## Selection and placement

The metabuilder uses the framework's segment helper and checks each start through the public
closed-loop services. Starts without sufficient recorded history or deployment support are
excluded.

For each segment start, it accumulates distance along the recorded ego trajectory and selects
the first recorded pose that reaches `distance_m`. The mesh keeps that pose's rotation,
including road slope. `lateral_offset_m` moves it along the selected pose's left axis, with
positive values to the left. A start is skipped when its remaining recording cannot reach the
requested distance. `max_segments` optionally limits the accepted segments in each scenario.

The world pose is stored in `AttackSegment.user_data`. The offline phase has no preparation
work. The online phase returns one `InsertObject` action during setup, making the vehicle
visible in the first capture. Its world pose stays fixed while ego moves.

## Treatments and evidence

The configuration compares an unattacked `clean` treatment with `vehicle_insertion`. Both use
the same selected start, decision count and seed. The enabled treatment inserts the mesh through
the simulator, so it appears in the configured camera and LiDAR captures.

The configuration uses `ExposureHarmonizer` with the `ring` reference. It scales the rendered
vehicle's exposure toward the surrounding background and keeps the vehicle's colors. The
attack's `harmonizer` option defaults to `null`, which renders the model's colors unchanged.
Lighting, shadows and reflections are not simulated. Insertion capabilities, visibility and
physics limits are defined in [Simulators](../guides/simulators.md#inserted-world-objects).

`RuleBasedAVStack` follows its configured stop-sign policy. Its perception and planning scope is
described in [Driving systems](../guides/driving-systems.md#nurec-rule-based-driving-system).

The configuration records camera and point-cloud payloads as both `raw` and `model_input` at
each decision. Use those captures, insertion receipts and object-visibility evidence to inspect
deployment. Driving and trajectory metrics compare the treatments. `DrivingMetrics` includes
the inserted vehicle in its bounding-box distance and collision measurements, as described in
[Built-in metrics](../guides/evaluation.md#built-in-metrics). Read
[Evaluation](../guides/evaluation.md) for saved records and reports.

## Next steps

Read [Writing attacks](../guides/attacks.md),
[Mesh assets](../reference/configuration.md#mesh-assets) and [Evaluation](../guides/evaluation.md).
