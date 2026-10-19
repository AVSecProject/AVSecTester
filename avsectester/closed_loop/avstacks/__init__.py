"""AV stacks — the framework's ``AVStack`` implementations (the driving systems under test).

Not to be confused with the third-party avstack library under ``third_party/``, which
``ModularAVStack`` uses to build its pipeline.

``ModularAVStack`` (avstack modular pipeline) and ``AlpamayoAVStack`` (end-to-end Alpamayo) both
lazy-import their heavy deps, so this package imports without avstack/torch.
"""

from .alpamayo import AlpamayoAVStack
from .modular import ModularAVStack

__all__ = ["AlpamayoAVStack", "ModularAVStack"]
