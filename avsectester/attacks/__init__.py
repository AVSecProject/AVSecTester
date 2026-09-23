"""Attacks — grouped by the *seam* in the sim<->stack loop they exploit, not one class hierarchy.

Four families, from deepest (inside the stack) to shallowest (on the render):

  * **Pipeline hook** — a callable registered in avstack's ``HOOKS`` registry and attached to a
    module's pre/post hooks, so it composes with a real pipeline with no parallel machinery.
    ``PhantomInjection`` appends a fabricated ``BoxDetection`` to the detector output, propagating a
    phantom obstacle detection -> track -> an unsafe stop.

  * **World-level physical patch** (``physical_patch``) — for CARLA: attach + paint a panel on a
    target vehicle so the ego's camera renders it in-scene (applied at ``CarlaBackend.reset``).

  * **Sensor-plane composite** (``patch_composite``) — backend-agnostic: warp a patch onto the target
    surface in the *rendered* frame and harmonize it to the scene, never baking it into the world /
    reconstruction. One insert path for CARLA and NuRec (paired with a per-backend quad projector and
    :func:`avsectester.simulators.viz.composite_view`).

  * **Optimization** (``optim``) — the algorithm layer (PGD white-box, NES black-box) that *produces*
    an adversarial patch/perturbation against a scorer; not tied to any one threat model.

Only ``PhantomInjection`` needs avstack; it is imported lazily (PEP 562) so importing this package —
or the pure ``physical_patch`` / ``patch_composite`` helpers — does not pull it in. The CARLA path
imports the ``phantom`` submodule explicitly (in :mod:`avsectester.scenario`) to run its registration.
"""

__all__ = ["PhantomInjection"]


def __getattr__(name: str):  # PEP 562: import the avstack-dependent hook only on demand
    if name == "PhantomInjection":
        from .phantom import PhantomInjection

        return PhantomInjection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
