"""A driving-system placeholder for a simulator controlled by an external process."""

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import Observation


class ExternalAVStack(AVStack):
    """Represent a driving system that controls ego through the simulator from outside the framework.

    Requires a simulator whose ``command_modes`` is empty. Vehicle motion and control are
    supplied by the external driving system, while the framework records the run.
    """

    provides_detections = False
    command_modes = frozenset()

    def __call__(self, observation: Observation) -> None:
        """Accept the decision input without producing a driving command."""
        return None


__all__ = ['ExternalAVStack']
