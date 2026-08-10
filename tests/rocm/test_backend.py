"""CPU-runnable checks for rocm/backend.py and rocm/capabilities.py.
GPU-dependent behavior (actual ROCm/CUDA detection) is exercised by
scripts/smoke-test.sh on a real ROCm host, not here.
"""

from rocm.backend import gpu_backend, is_cuda, is_rocm
from rocm.capabilities import current_capabilities


def test_backend_reports_cpu_without_gpu():
    # This suite runs on GitHub-hosted CPU runners; a torch build with GPU
    # support may still be installed without a GPU present, so only assert
    # internal consistency, not a specific backend.
    backend = gpu_backend()
    assert backend in ("cpu", "cuda", "rocm")
    assert is_rocm() == (backend == "rocm")
    assert is_cuda() == (backend == "cuda")


def test_capabilities_match_backend():
    caps = current_capabilities()
    if gpu_backend() == "cpu":
        assert caps.hash_grid is False
        assert caps.cutlass_mlp is False
