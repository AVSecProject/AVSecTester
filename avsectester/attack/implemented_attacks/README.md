# Attack implementations

Built-in and user-defined attacks use the same structure and discovery rule. Place a user
attack here in a folder named after the class in snake_case. For example, `MyNNAttack` lives
in `my_nn_attack/`, whose `__init__.py` exports only `MyNNAttack`. Configuration can then use
`type: MyNNAttack`; an external package can use `type: package.module:MyNNAttack`.
Consecutive uppercase letters stay together until the final capital starts a word: `MyNNAttack`
becomes `my_nn_attack`. A capital after a digit starts a word, so `Attack2D` becomes `attack2_d`.

Each folder contains:

- `attack.py`: the `Attack` subclass, its validated parameters and three entry points;
- `metabuilder.py`: `build_metadata` selection logic;
- `offline.py`: optional optimization; omit it when there is no offline work;
- `online.py`: the `AttackRuntime` and its actions;
- `assets.py`: optional attack-specific assets;
- `__init__.py`: exports only the attack class.

A phase may use additional modules in the same folder when it needs them, for example
`patch_attack/optimization_data.py` for the patch optimization data source.

Core interfaces do not import insertion tools, optimization tools or concrete attacks. Tools
use core interfaces and never import another tool or a concrete attack. Concrete attacks use
core interfaces and tools, and never import another attack. Placement types belong to the
core in `attack/placement.py`; insertion and optimization are user-facing framework tools.

`PatchAttack` ranks candidate hosts independently within each selected time window. It keeps
`hosts_per_window` candidates and records their host binding, score and zero-based rank. Its
`max_windows` parameter limits time windows, not the number of host candidates.
