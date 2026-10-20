# Phantom injection

> **Audience:** Users testing modular perception attacks.
>
> **Outcome:** Insert a synthetic detection through the Attack interface.
>
> **Prerequisites:** CARLA with ModularAVStack and [Installation](../getting-started/installation.md).

## Configuration

```bash
avsectester run configs/carla_phantom.yaml
```

PhantomInjectionAttack is defined in
avsectester.attack.implemented_attacks.phantom_injection_attack.attack. It takes explicit
decision_steps and optional max_segments, target_xyz, object_type, score, extent, object_id and
keep_tail. It needs no image asset or host role.

```yaml
type: PhantomInjectionAttack
options:
  decision_steps: 20
  target_xyz: [6.0, 0.0, -0.7]
  object_type: Car
  score: 0.9
  max_segments: 2
```

## Workflow

The metabuilder partitions decision windows. The default offline phase produces no artifact. Each
runtime returns an AVStackAttackEffect at perception.post; the synthetic object reaches tracking,
planning and control in the same decision. The physical world contains no such object. The effect
constructs the added detection directly; it does not register a separate native attack hook. Native
avstack detection classes are imported only when their output format is used. Constructing
ModularAVStack does not globally register attack hooks. Phantom extent uses full (length, width,
height), consistently with canonical Detection.extent. target_xyz is the box center in the
perception source coordinate system (metres), not its bottom. For the supplied LiDAR setup, z=-0.7
preserves the original vehicle-sized phantom bottom.

## Evidence

Inspect StepRecord.detections, components, commands and effect receipts, then compare clean and
attacked DrivingMetrics, DetectionMetrics and StopImpact. A changed detection count alone does not
establish driving impact.

## Next steps

Read [Runtime](../reference/runtime.md) and [Evaluation](../guides/evaluation.md).
