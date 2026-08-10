"""GPU backend detection. Do not use this to gate `.cuda()`/`torch.cuda.*` calls —
those remain correct under ROCm's HIP-as-CUDA-namespace PyTorch build. Use it only
where behavior must actually diverge between NVIDIA and AMD (e.g. capability
queries, kernel selection, COLMAP GPU-SIFT availability).
"""

import functools
import re

import torch


def is_rocm() -> bool:
    return torch.version.hip is not None


def is_cuda() -> bool:
    return torch.version.cuda is not None


def gpu_backend() -> str:
    if is_rocm():
        return "rocm"
    if is_cuda():
        return "cuda"
    return "cpu"


@functools.lru_cache(maxsize=1)
def get_gpu_arch() -> str | None:
    """Return the GCN/RDNA/CDNA target (e.g. 'gfx1151') on ROCm, else None."""
    if not is_rocm() or not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    gcn_arch_name = getattr(props, "gcnArchName", "")
    match = re.match(r"(gfx[0-9a-fA-F]+)", gcn_arch_name)
    return match.group(1) if match else None


def is_rdna() -> bool:
    """RDNA (wave32, consumer/workstation: gfx11xx, gfx12xx) vs CDNA (wave64, Instinct: gfx9xx)."""
    arch = get_gpu_arch()
    if arch is None:
        return False
    return arch.startswith("gfx11") or arch.startswith("gfx12")
