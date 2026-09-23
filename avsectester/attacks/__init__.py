"""Attacks — grouped by the *seam* in the sim<->stack loop they exploit, not one class hierarchy.

Three families, from deepest (inside the stack) to shallowest (on the render):

  * **Pipeline hook** — a callable registered in avstack's ``HOOKS`` registry and attached to a
    module's pre/post hooks, so it composes with a real pipeline with no parallel machinery.
    ``PhantomInjection`` appends a fabricated ``BoxDetection`` to the detector output, propagating a
    phantom obstacle detection -> track -> an unsafe stop.

  * **Physical patch** (``physical_patch``) — put an adversarial patch on a target surface (the lead
    vehicle's rear). One attack, two render substrates: *world-level* (CARLA: attach + paint a panel on
    the vehicle so the camera renders it in-scene) and *sensor-level* (composite it onto the rendered
    frame). The attack owns only the payload (which texture) + the target; the realistic-insertion
    mechanics (warp + harmonize) live in :mod:`avsectester.simulators.patch_insertion` because that is
    a simulation/rendering concern, wired per backend by ``carla.lead_rear_quad`` /
    ``viz.detector_quad`` + ``viz.composite_view`` / ``carla.camera_patch_perturbation``.

  * **Optimization** (``optim``) — the algorithm layer (PGD white-box, NES black-box) that *produces*
    an adversarial patch/perturbation against a scorer; not tied to any one threat model.

Only ``PhantomInjection`` needs avstack; it is imported lazily (PEP 562) so importing this package —
or the pure ``physical_patch`` helpers — does not pull it in. The CARLA path imports the ``phantom``
submodule explicitly (in :mod:`avsectester.scenario`) to run its registration.
"""

__all__ = ["PhantomInjection"]


def __getattr__(name: str):  # PEP 562: import the avstack-dependent hook only on demand
    if name == "PhantomInjection":
        from .phantom import PhantomInjection

        return PhantomInjection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
