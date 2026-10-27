# Phantom injection

> **Audience:** Users testing modular perception inputs.
>
> **Outcome:** Run `PhantomInjectionAttack` and inspect its recorded effect on driving.
>
> **Prerequisites:** [Installation](../getting-started/installation.md) with CARLA.

## Configuration

```bash
docker compose exec carla-experiment avsectester run configs/carla_phantom.yaml
```

The supplied attack configuration is:

```yaml
type: PhantomInjectionAttack
options:
  decision_steps: 20
  max_segments: 2
  target_xyz: [6.0, 0.0, -0.7]
  object_type: Car
  score: 0.9
  extent: [4.0, 1.8, 1.6]
```

Constructor parameters and coordinate conventions are in the
[Python API](../reference/python-api.md).

## Offline phase

The attack selects decision windows. Its offline phase returns no artifact; selection and
deployment budgets follow [Writing attacks](../guides/attacks.md).

## Online phase

Each runtime inserts a synthetic perception detection at `perception.post`. The physical world
contains no corresponding actor. The same decision then continues through the driving system;
stage order is defined in [Runtime](../reference/runtime.md).

## Evidence

Inspect the recorded detections, action receipts and commands, and use the configured metrics in
[Evaluation](../guides/evaluation.md). The [example overview](README.md) explains how to
interpret execution and attack impact.

## Next steps

Read [Driving systems](../guides/driving-systems.md) and [Runtime](../reference/runtime.md).
