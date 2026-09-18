"""AV stacks — ``AVStack`` implementations (the box under test).

``ModularAVStack`` (avstack modular pipeline) and ``AlpamayoAVStack`` (end-to-end Alpamayo) both
lazy-import their heavy deps, so this package imports without avstack/torch.
"""

from .alpamayo import AlpamayoAVStack
from .modular import ModularAVStack

__all__ = ["AlpamayoAVStack", "ModularAVStack"]
