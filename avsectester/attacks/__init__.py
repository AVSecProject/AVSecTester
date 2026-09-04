"""Attacks — each is an avstack ``HOOKS`` hook attached to a pipeline stage.

An attack in AVSecTester is not a bespoke class hierarchy; it is a callable registered in avstack's
``HOOKS`` registry and attached to a module's pre/post hooks (via config or ``register_post_hook``).
That is the same mechanism avstack uses for any hook, so an attack composes with a real pipeline
without any parallel machinery.

  ``PhantomInjection`` — appends a fabricated ``BoxDetection`` to the detector output, so a phantom
  obstacle propagates detection -> track -> an unsafe stop.
"""

from .phantom import PhantomInjection

__all__ = ["PhantomInjection"]
