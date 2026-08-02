"""Perception smoke test: confirm the compiled mmcv/mmdet3d CUDA ops run on this GPU.

Validates the riskiest part of the full-stack install on Ada (sm_89) GPUs: the mmdet3d
CUDA ops are compiled to sm_86 cubin (nvcc 11.5 can't target sm_89 directly) and must
run forward-compatibly on the L40S. Run inside the `avsec` conda env.
"""
import sys

import torch
from mmcv.ops import Voxelization, furthest_point_sample


def main() -> int:
    if not torch.cuda.is_available():
        print("PERCEPTION CUDA-OP SMOKE: FAIL -> CUDA not available")
        return 1
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)

    span = torch.tensor([70.0, 80.0, 4.0, 1.0], device=dev)
    off = torch.tensor([35.0, 40.0, 2.0, 0.0], device=dev)
    pts = torch.rand(200000, 4, device=dev) * span - off

    vox = Voxelization(
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[-35, -40, -2, 35, 40, 2],
        max_num_points=32,
        max_voxels=20000,
    )
    voxels, coors, _num = vox(pts)
    idx = furthest_point_sample(pts[:, :3].unsqueeze(0).contiguous(), 512)

    print(f"GPU: {name} (compute capability {cc[0]}.{cc[1]})")
    print(f"  Voxelization -> voxels {tuple(voxels.shape)}, coors {tuple(coors.shape)}")
    print(f"  furthest_point_sample -> {tuple(idx.shape)}")
    ok = voxels.shape[0] > 0 and idx.shape[-1] == 512
    print("PERCEPTION CUDA-OP SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
