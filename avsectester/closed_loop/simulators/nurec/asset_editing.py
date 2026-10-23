"""NRE scene asset-editing messages, defined from NVIDIA's public NuRec gRPC API reference.

AlpaSim's Apache-2.0 sensorsim.proto omits ``SensorsimService.edit_assets``. The descriptors below
repeat the documented message fields, numbers and types from
https://docs.nvidia.com/nurec/api/nre.grpc.protos.sensorsim.html and
https://docs.nvidia.com/nurec/api/nre.grpc.protos.common.html, so requests are wire-compatible
with the NRE service. Only the fields this framework sends or reads are declared;
``EditAssetsRequest.replace`` (field 2) is omitted.
"""

from __future__ import annotations

from functools import cache
from typing import Any

EDIT_ASSETS_METHOD = "/nre.grpc.protos.sensorsim.SensorsimService/edit_assets"

_COMMON = {
    "Vec3": [("x", 1, "float"), ("y", 2, "float"), ("z", 3, "float")],
    "Quat": [("w", 1, "float"), ("x", 2, "float"), ("y", 3, "float"), ("z", 4, "float")],
    "Pose": [("vec", 1, ".nre.grpc.protos.common.Vec3"), ("quat", 2, ".nre.grpc.protos.common.Quat")],
    "PoseAtTime": [("pose", 1, ".nre.grpc.protos.common.Pose"), ("timestamp_us", 2, "fixed64")],
    "Trajectory": [("poses", 1, "repeated .nre.grpc.protos.common.PoseAtTime")],
    "AABB": [("size_x", 1, "float"), ("size_y", 2, "float"), ("size_z", 3, "float")],
}
_SENSORSIM = {
    "DynamicObjectTrack": [
        ("id", 1, "string"), ("semantic_class", 2, "string"),
        ("trajectory", 3, ".nre.grpc.protos.common.Trajectory"),
        ("object_size", 4, ".nre.grpc.protos.common.AABB"), ("asset_id", 5, "string"),
    ],
    "EditAssetsRequest": [
        ("scene_id", 1, "string"), ("insert", 3, "repeated .nre.grpc.protos.sensorsim.DynamicObjectTrack"),
    ],
    "EditAssetsResponse": [("success", 1, "bool"), ("message", 2, "string")],
}


@cache
def asset_editing_messages() -> dict[str, Any]:
    """Message classes ``DynamicObjectTrack``, ``EditAssetsRequest`` and ``EditAssetsResponse``."""
    from google.protobuf import descriptor_pb2, descriptor_pool
    from google.protobuf.message_factory import GetMessageClass

    field = descriptor_pb2.FieldDescriptorProto
    scalars = {"float": field.TYPE_FLOAT, "fixed64": field.TYPE_FIXED64,
               "string": field.TYPE_STRING, "bool": field.TYPE_BOOL}
    pool = descriptor_pool.DescriptorPool()
    for package, declarations, dependency in (
            ("nre.grpc.protos.common", _COMMON, []),
            ("nre.grpc.protos.sensorsim", _SENSORSIM, ["avsectester/nre_common.proto"])):
        file = descriptor_pb2.FileDescriptorProto(
            name=f"avsectester/nre_{package.rsplit('.', 1)[1]}.proto", package=package,
            syntax="proto3", dependency=dependency)
        for name, fields in declarations.items():
            message = file.message_type.add(name=name)
            for field_name, number, kind in fields:
                repeated = kind.startswith("repeated ")
                kind = kind.removeprefix("repeated ")
                declared = message.field.add(
                    name=field_name, number=number,
                    label=field.LABEL_REPEATED if repeated else field.LABEL_OPTIONAL)
                if kind in scalars:
                    declared.type = scalars[kind]
                else:
                    declared.type, declared.type_name = field.TYPE_MESSAGE, kind
        pool.Add(file)
    return {name: GetMessageClass(pool.FindMessageTypeByName(f"nre.grpc.protos.sensorsim.{name}"))
            for name in _SENSORSIM}
