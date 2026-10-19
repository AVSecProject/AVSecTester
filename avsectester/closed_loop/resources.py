"""Nonblocking process-local ownership of simulator worlds.

CARLA clients connected to one endpoint share a mutable world. A caller must also reserve
that server externally when running multiple processes; this guard coordinates framework
scopes and runners within one process, without claiming to lock external simulator clients.
"""

from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from threading import Lock

_lock = Lock()
_owners: dict[Hashable, tuple[object, int]] = {}


@contextmanager
def reserve(key: Hashable, owner: object) -> Iterator[None]:
    with _lock:
        previous, depth = _owners.get(key, (owner, 0))
        if previous is not owner:
            raise RuntimeError(f"Simulator resource is already in use: {key!r}")
        _owners[key] = (owner, depth + 1)
    try:
        yield
    finally:
        with _lock:
            _, depth = _owners[key]
            if depth == 1:
                del _owners[key]
            else:
                _owners[key] = (owner, depth - 1)


def simulator_key(simulator: object) -> Hashable:
    return getattr(simulator, "resource_key", ("simulator", id(simulator)))
