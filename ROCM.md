# ROCm port status

`nerfstudio-rocm` is a fork of [`nerfstudio-project/nerfstudio`](https://github.com/nerfstudio-project/nerfstudio),
kept as close to upstream as possible. GPU-specific work lives in separate
forks of Nerfstudio's three accelerated dependencies, each a real fork of its
own upstream so fixes can be merged back there:

```
                     nerfstudio-rocm
                           │
              mostly upstream Nerfstudio
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
   tiny-rocm-nn      nerfacc-rocm       gsplat (rocm fork)
          │                │                 │
          └────────────────┼─────────────────┘
                           ▼
                    PyTorch ROCm / HIP
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        RDNA (gfx11xx/gfx12xx)    CDNA (gfx90a/gfx942)
        e.g. Radeon 8060S/gfx1151   e.g. MI250X/MI300X
```

| Repo | Fork of | Status |
|---|---|---|
| [nerfstudio-rocm](https://github.com/bjoernellens1/nerfstudio-rocm) | [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | this repo |
| [gsplat](https://github.com/bjoernellens1/gsplat) | [AMD-Ecosystem/gsplat](https://github.com/AMD-Ecosystem/gsplat) (→ [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)) | **verified**: gfx1151 + gfx1100 forward/backward tested, `release/1.5.3b2` |
| [nerfacc-rocm](https://github.com/bjoernellens1/nerfacc-rocm) | [nerfstudio-project/nerfacc](https://github.com/nerfstudio-project/nerfacc) | **unported**, fresh fork; `AMD-Ecosystem/nerfacc` exists as a reference (hardcodes `gfx942`, not merged wholesale) |
| [tiny-rocm-nn](https://github.com/bjoernellens1/tiny-rocm-nn) | [ZJLi2013/tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) | **partially ported** — see below |
| [colmap](https://github.com/bjoernellens1/colmap) | [colmap/colmap](https://github.com/colmap/colmap) | `patch_match_stereo` HIP support merged; feature extraction/matching CPU-only on AMD (see §COLMAP) |

Pinned commits for all of the above: [`dependencies/rocm-lock.toml`](dependencies/rocm-lock.toml).
Run `python -m rocm.diagnostics` (`ns-rocm-info`) inside the container to print the live stack + pins.

## tiny-rocm-nn: what's actually there

Verified directly against the pinned commit (not assumed from its README, which
undersells current state in one place and doesn't mention RDNA support at all):

- ✅ `HashGrid`, `Frequency`, `OneBlob` encodings — ported
- ✅ `SphericalHarmonics` encoding — ported (`include/tiny-cuda-nn/encodings/spherical_harmonics.h`), wired into `encoding.cpp`. The original porting-plan assumption that this was missing was **stale**; Nerfstudio's `SHEncoding` with `implementation="tcnn"` should work as-is.
- ✅ `FullyFusedMLP` — ported, PyTorch binding builds and passes a GPU smoke test on **MI300X (gfx942, CDNA)**
- ❌ `CutlassMLP` — no ROCm equivalent; widths outside `{64, 128}` and `n_hidden_layers` requiring Cutlass fall back to `implementation="torch"`
- ⚠️ **RDNA / gfx1151 (this project's primary dev target, the Radeon 8060S)** — upstream `tiny-rocm-nn`'s README explicitly scopes itself to "AMD GPUs with Matrix Core support (CDNA/RDNA3)" and MFMA/rocWMMA, and its documented benchmarks are MI300X-only. Nothing in the repo confirms wave32 correctness. `rocm/capabilities.py` marks `ROCM_RDNA_CAPS.verified_on_this_arch = False` for exactly this reason — treat any RDNA usage as unverified until `scripts/smoke-test.sh` (or a dedicated tcnn-parity test) actually runs green on gfx1151.

`rocm/capabilities.py` exposes these per-component so Nerfstudio can mix
tcnn and torch implementations rather than an all-or-nothing switch.

## nerfacc-rocm: not started

Fresh fork of `nerfstudio-project/nerfacc`. `AMD-Ecosystem/nerfacc` has prior
ROCm work but hardcodes `--offload-arch=gfx942` in `setup.py` — treat it as a
reference to port logic from, not a remote to merge (see remote `amd` in the
local clone). Porting plan: architecture selection via `PYTORCH_ROCM_ARCH` /
inferred from PyTorch instead of the hardcoded arch, then validate ray
marching / occupancy grid / sampling / transmittance kernels against the
Python reference before touching Nerfstudio's Instant-NGP integration.

## COLMAP

Nerfstudio's COLMAP wrapper's `gpu=True` path uses CUDA SIFT extraction/matching,
which has no ROCm equivalent. On AMD, `rocm/backend.py`'s `is_rocm()` should
gate this off in the data-processing scripts so COLMAP falls back to CPU
SIFT rather than silently failing — **not yet wired in**, tracked as follow-up.
`patch_match_stereo` (dense reconstruction) *is* HIP-accelerated in the pinned
`colmap` fork already.

## Phase 1 (this session): pure-Torch baseline

Goal: `vanilla-nerf`, `mipnerf`, and `nerfacto --pipeline.model.implementation torch`
running end-to-end on gfx1151 with no accelerated dependency required, as the
reference baseline everything else gets compared against.

Acceptance criteria (per `scripts/smoke-test.sh`, 20 iterations each):
- training starts without error
- loss decreases over the run
- backward pass completes
- no NaN/Inf losses
- checkpoint save succeeds

See `tests/rocm/logs/` (gitignored, produced by the smoke test) for the actual
run output backing the status claim in this file's changelog / PR description
— don't trust this doc's claims about Phase 1 without checking those logs.

## Not in scope here

Full method parity (Instant-NGP, Splatfacto-MCMC, NeuS, etc.), performance
tuning, and multi-GPU are follow-on work, tracked as GitHub issues on this
repo once Phase 1 is confirmed green.
