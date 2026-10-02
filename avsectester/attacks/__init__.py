"""Attack families and shared optimization tools.

Each family owns its attack payload and target; simulation helpers own rendering and compositing:

  * **Pipeline hooks** (``pipeline``) — callables registered in avstack's ``HOOKS`` registry and
    attached to a module's pre/post hooks, so they compose with a real pipeline.
    ``PhantomInjection`` appends a fabricated ``BoxDetection`` to the detector output, propagating a
    phantom obstacle detection -> track -> an unsafe stop.

  * **Physical patches** (``patch``) — put an adversarial patch on a target surface (the lead
    vehicle's rear). One attack, two render substrates: *world-level* (CARLA: attach + paint a panel on
    the vehicle so the camera renders it in-scene) and *sensor-level* (composite it onto the rendered
    frame). The attack owns only the payload (which texture) + the target; the realistic-insertion
    mechanics (warp + harmonize) live in :mod:`avsectester.simulators.patch_insertion` because that is
    a simulation/rendering concern, wired per backend by ``carla.lead_rear_quad`` /
    ``patch_insertion.detector_quad`` + ``patch_insertion.composite_view`` / ``carla.camera_patch_perturbation``.

  * **Object insertion** (``object_insertion``) — insert standard traffic signs, pedestrian images
    or traffic lights at invalid locations in camera frames, using shared simulation helpers.

  * **Optimization** (``optim``) — shared algorithms (PGD white-box, NES black-box) that *produce*
    an adversarial patch/perturbation against a scorer; not tied to any one threat model.

Only ``PhantomInjection`` needs avstack; it is imported lazily (PEP 562) so importing this package —
or the pure image helpers — does not pull it in. The modular stack imports ``pipeline.phantom``
explicitly (in :mod:`avsectester.stacks.modular`) to run its registration.
"""

__all__ = ["PhantomInjection"]


def __getattr__(name: str):  # PEP 562: import the avstack-dependent hook only on demand
    if name == "PhantomInjection":
        from .pipeline.phantom import PhantomInjection

        return PhantomInjection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
