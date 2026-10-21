"""AVStack-native phantom replacement at perception.post."""
import numpy as np

from avsectester.attack.runtime import AttackRuntime, AVStackAttackAction


class PhantomInjectionAttackRuntime(AttackRuntime):
    """Append one configured phantom at the AVStack perception stage."""

    def __init__(self, attack):
        self.attack = attack

    def step(self, context):
        return (AVStackAttackAction("perception.post", self.inject),)

    def inject(self, detections, context):
        reference = getattr(detections, "source_reference", None)
        if reference is None:
            raise ValueError("Phantom injection requires detections.source_reference")
        from avstack.geometry import Attitude, Box3D, Position
        from avstack.modules.perception.detections import BoxDetection

        attack = self.attack
        length, width, height = attack.extent
        box = Box3D(Position(np.asarray(attack.target_xyz), reference),
                    Attitude(np.quaternion(1), reference), [height, width, length], where_is_t="center")
        phantom = BoxDetection(data=box,
            noise=np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) ** 2,
            source_identifier=getattr(detections, "source_identifier", "phantom"),
            reference=reference, obj_type=attack.object_type, score=attack.score)
        phantom.ID = attack.object_id
        detections.append(phantom)
        return detections
