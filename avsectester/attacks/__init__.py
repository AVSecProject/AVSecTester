"""Attacks — each is an avstack ``HOOKS`` hook attached to a pipeline stage.

An attack in AVSecTester is not a bespoke class hierarchy; it is a callable registered in avstack's
``HOOKS`` registry and attached to a module's pre/post hooks (via config or ``register_post_hook``).
That is the same mechanism avstack uses for any hook, so an attack composes with a real pipeline
without any parallel machinery.

  ``PhantomInjection`` — appends a fabricated ``BoxDetection`` to the detector output, so a phantom
  obstacle propagates detection -> track -> an unsafe stop (an avstack ``HOOKS`` hook, needs avstack).

  ``physical_patch`` — a *world-level* patch attack for CARLA (attach + paint a panel on a vehicle);
  its pixel helpers are pure numpy/PIL, so this submodule imports without avstack/carla.

``PhantomInjection`` is imported lazily (PEP 562) so importing this package — or the pure
``physical_patch`` helpers — does not pull in avstack. The CARLA path imports the ``phantom`` submodule
explicitly (in :mod:`avsectester.scenario`) to run its ``HOOKS`` registration.
"""

__all__ = ["PhantomInjection"]


def __getattr__(name: str):  # PEP 562: import the avstack-dependent hook only on demand
    if name == "PhantomInjection":
        from .phantom import PhantomInjection

        return PhantomInjection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
