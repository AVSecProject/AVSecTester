"""Apply object insertion to the declared camera on each decision."""
from avsectester.attack.runtime import AttackRuntime
from avsectester.attack.insertion.action import InsertionAttackAction


class ObjectInsertionAttackRuntime(AttackRuntime):
    """Apply one selected object placement in its configured camera."""

    def __init__(self, sensor, insertion, compositor):
        self.sensor, self.insertion, self.compositor = sensor, insertion, compositor

    def step(self, context):
        return (InsertionAttackAction(self.sensor, self.insertion, compositor=self.compositor),)
