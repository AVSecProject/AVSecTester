"""Apply object insertion to the declared camera on each decision."""
from avsectester.attack.runtime import AttackRuntime
from avsectester.attack.insertion.effect import InsertionAttackEffect


class ObjectInsertionAttackRuntime(AttackRuntime):
    """Apply one selected object placement in its configured camera."""

    def __init__(self, sensor, insertion, compositor):
        self.sensor, self.insertion, self.compositor = sensor, insertion, compositor

    def step(self, context):
        return (InsertionAttackEffect(self.sensor, self.insertion, compositor=self.compositor),)
