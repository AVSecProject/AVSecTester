# Attack implementations

> **Audience:** Attack implementers.
>
> **Outcome:** Place an attack in a package with explicit public exports.
>
> **Prerequisites:** [Attack guide](../../docs/guides/attacks.md).

Prefer a user-owned package and reference it with `type: my_project.attacks:MyAttack`. The
framework's supplied attacks use the class name converted to `snake_case` as their directory
name: `PatchAttack` belongs in `patch_attack/`. Discovery by class name depends on this rule.

A supplied attack package exports only its `Attack` subclass through `__init__.py` and `__all__`.
Its implementation modules keep selection, offline preparation, runtime actions and assets
separate. An offline phase or an online phase may be a no-op.

The [attack guide](../../docs/guides/attacks.md) describes phase contracts and deployment tick
budgets. The [architecture](../../docs/concepts/architecture.md#dependency-layers) describes
dependency rules. Concrete attacks do not import one another.

Public modules declare `__all__`. Internal modules and helpers are not public imports. See
[tests](../../tests/README.md#writing-tests) for component contract tests.

## Next steps

-  [Public API](../../docs/reference/python-api.md)
-  [Supplied configurations](../../docs/examples/README.md)
