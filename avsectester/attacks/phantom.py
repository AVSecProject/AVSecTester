"""Phantom-injection attack as a native avstack post-hook.

This is the whole attack interface: a callable registered in avstack's ``HOOKS`` registry that
avstack attaches to a module's ``post_hooks`` (via config or ``module.register_post_hook(...)``).
It runs on the detector's output and appends a fabricated ``BoxDetection`` — so a phantom
obstacle propagates detection -> track -> an unsafe stop, using avstack's own hook mechanism.
No parallel seam/system machinery: the attack IS an avstack hook.

avstack post-hook contract (``_apply_post_hooks``): the hook is called on the module's return
value and must return it re-splatted as a 1-tuple, which avstack unwraps back to a bare value.
"""

from __future__ import annotations

from typing import Any

from avstack.config import HOOKS


@HOOKS.register_module()
class PhantomInjection:
    def __init__(
        self,
        target_xyz: tuple[float, float, float] = (6.0, 0.0, -1.5),  # [fwd, left, up] in detection frame
        obj_type: str = "Car",
        score: float = 0.9,
        extent: tuple[float, float, float] = (1.6, 1.8, 4.0),       # h, w, l
        oid: int = 90002,
    ) -> None:
        self.target_xyz = target_xyz
        self.obj_type = obj_type
        self.score = score
        self.extent = extent
        self.oid = oid

    def __call__(self, detections: Any) -> tuple[Any]:
        import numpy as np
        from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position
        from avstack.modules.perception.detections import BoxDetection

        ref = detections[0].box.reference if len(detections) > 0 else GlobalOrigin3D
        pos = Position(np.asarray(self.target_xyz, dtype=float), ref)
        att = Attitude(np.quaternion(1), ref)
        box = Box3D(pos, att, list(self.extent), where_is_t="bottom")
        phantom = BoxDetection(
            data=box,
            noise=np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) ** 2,
            source_identifier=getattr(detections, "source_identifier", "phantom"),
            reference=ref,
            obj_type=self.obj_type,
            score=self.score,
        )
        phantom.ID = self.oid
        detections.append(phantom)
        return (detections,)  # 1-tuple per avstack's post-hook contract
