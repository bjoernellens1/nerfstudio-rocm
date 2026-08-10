"""Per-backend capability declarations, so Nerfstudio can choose tcnn vs. torch
implementations per-component instead of an all-or-nothing switch.

Values below reflect the state of the pinned `tiny-rocm-nn` commit recorded in
dependencies/rocm-lock.toml as of the initial fork (verified by inspecting that
repo directly, not assumed from documentation): SphericalHarmonics is present
(include/tiny-cuda-nn/encodings/spherical_harmonics.h, wired into encoding.cpp).
CutlassMLP has no ROCm equivalent. The upstream tiny-rocm-nn README targets
CDNA/MFMA (gfx90a, gfx942) explicitly; RDNA3.5/gfx1151 (this machine) is
untested by that project, so fully_fused_mlp is marked untested rather than
False — verify empirically before trusting it, see ROCM.md.
"""

from dataclasses import dataclass, field

from rocm.backend import gpu_backend, is_rdna


@dataclass(frozen=True)
class TcnnCapabilities:
    hash_grid: bool
    frequency: bool
    spherical_harmonics: bool
    fully_fused_mlp: bool
    cutlass_mlp: bool
    supported_widths: frozenset[int] = field(default_factory=frozenset)
    verified_on_this_arch: bool = False


CUDA_CAPS = TcnnCapabilities(
    hash_grid=True,
    frequency=True,
    spherical_harmonics=True,
    fully_fused_mlp=True,
    cutlass_mlp=True,
    supported_widths=frozenset({16, 32, 64, 128}),
    verified_on_this_arch=True,
)

# tiny-rocm-nn as pinned: HashGrid/Frequency/SH ported, FullyFusedMLP ported but
# only verified on MI300X (gfx942) upstream, not on RDNA (gfx11xx/gfx12xx).
ROCM_CDNA_CAPS = TcnnCapabilities(
    hash_grid=True,
    frequency=True,
    spherical_harmonics=True,
    fully_fused_mlp=True,
    cutlass_mlp=False,
    supported_widths=frozenset({64, 128}),
    verified_on_this_arch=True,
)

ROCM_RDNA_CAPS = TcnnCapabilities(
    hash_grid=True,
    frequency=True,
    spherical_harmonics=True,
    fully_fused_mlp=True,
    cutlass_mlp=False,
    supported_widths=frozenset({64, 128}),
    verified_on_this_arch=False,
)

CPU_CAPS = TcnnCapabilities(
    hash_grid=False,
    frequency=False,
    spherical_harmonics=False,
    fully_fused_mlp=False,
    cutlass_mlp=False,
)


def current_capabilities() -> TcnnCapabilities:
    backend = gpu_backend()
    if backend == "cuda":
        return CUDA_CAPS
    if backend == "rocm":
        return ROCM_RDNA_CAPS if is_rdna() else ROCM_CDNA_CAPS
    return CPU_CAPS
