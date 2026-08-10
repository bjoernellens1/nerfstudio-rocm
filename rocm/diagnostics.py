"""`ns-rocm-info` — dump the ROCm/CUDA stack and pinned dependency commits for
bug reports and CI logs.
"""

import importlib.metadata
import pathlib
import tomllib

import torch

from rocm.backend import gpu_backend, get_gpu_arch, is_rocm, is_cuda
from rocm.capabilities import current_capabilities

LOCK_FILE = pathlib.Path(__file__).resolve().parent.parent / "dependencies" / "rocm-lock.toml"


def _pinned_deps() -> dict:
    if not LOCK_FILE.exists():
        return {}
    with open(LOCK_FILE, "rb") as f:
        return tomllib.load(f)


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    backend = gpu_backend()
    print(f"nerfstudio-rocm diagnostics")
    print(f"  backend:        {backend}")
    print(f"  torch:          {torch.__version__}")
    if is_rocm():
        print(f"  hip:            {torch.version.hip}")
    if is_cuda():
        print(f"  cuda:           {torch.version.cuda}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu:            {torch.cuda.get_device_name(0)}")
        print(f"  gfx arch:       {get_gpu_arch()}")

    caps = current_capabilities()
    print("tcnn capabilities:")
    for field_name in ("hash_grid", "frequency", "spherical_harmonics", "fully_fused_mlp", "cutlass_mlp"):
        print(f"  {field_name:24s}{getattr(caps, field_name)}")
    print(f"  supported_widths       {sorted(caps.supported_widths)}")
    print(f"  verified_on_this_arch  {caps.verified_on_this_arch}")

    print("installed packages:")
    for pkg in ("nerfstudio", "gsplat", "nerfacc", "tinycudann"):
        print(f"  {pkg:24s}{_pkg_version(pkg)}")

    print("pinned dependency commits (dependencies/rocm-lock.toml):")
    for name, entry in _pinned_deps().items():
        if isinstance(entry, dict) and "commit" in entry:
            print(f"  {name:24s}{entry.get('repo', '?')} @ {entry['commit']}")


if __name__ == "__main__":
    main()
