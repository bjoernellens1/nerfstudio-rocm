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
| [gsplat](https://github.com/bjoernellens1/gsplat) | [AMD-Ecosystem/gsplat](https://github.com/AMD-Ecosystem/gsplat) (→ [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)) | **verified on gfx1151**: builds cleanly under `docker build` (hard dependency in `docker/Dockerfile.rocm`); full pytest suite 114 passed / 1 skipped / 0 failed (87 unrelated failures pre-existing in the suite are all attributable to nerfacc having no ROCm support, not gsplat, see [#5](https://github.com/bjoernellens1/gsplat/issues/5)); Splatfacto trained 7000 iterations on the bonsai (mip-nerf360) scene end-to-end with no NaN/Inf and confirmed densification |
| [nerfacc-rocm](https://github.com/bjoernellens1/nerfacc-rocm) | [nerfstudio-project/nerfacc](https://github.com/nerfstudio-project/nerfacc) | **unported**, fresh fork; `AMD-Ecosystem/nerfacc` exists as a reference (hardcodes `gfx942`, not merged wholesale) |
| [tiny-rocm-nn](https://github.com/bjoernellens1/tiny-rocm-nn) | [ZJLi2013/tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) | **partially ported** — see below |
| [colmap](https://github.com/bjoernellens1/colmap) | [colmap/colmap](https://github.com/colmap/colmap) | `patch_match_stereo` HIP support merged; feature extraction/matching CPU-only on AMD (see §COLMAP) |

Pinned commits for all of the above: [`dependencies/rocm-lock.toml`](dependencies/rocm-lock.toml).
Run `python -m rocm.diagnostics` (`ns-rocm-info`) inside the container to print the live stack + pins.

## Known build issues

gsplat's two `docker build`-time issues are both fixed: the vendored glm submodule not reaching hipcc's include path ([gsplat#3](https://github.com/bjoernellens1/gsplat/issues/3), fixed in [42d17c9](https://github.com/bjoernellens1/gsplat/commit/42d17c9)), and `get_rocm_arch()` ignoring `PYTORCH_ROCM_ARCH` and silently falling back to `gfx942` since `docker build` has no GPU device access to run `rocminfo` ([gsplat#2](https://github.com/bjoernellens1/gsplat/issues/2), fixed in [4515618](https://github.com/bjoernellens1/gsplat/commit/4515618)). `docker/Dockerfile.rocm` now installs gsplat as a hard, non-fallback build step.

- **nerfstudio's `gsplat==1.4.0` pin is intentionally not carried over**: the ROCm fork builds/publishes under the distribution name `amd_gsplat` (not `gsplat`, so it can coexist with upstream on PyPI) even though it installs into the same `gsplat` import namespace. Because the distribution names differ, pip's resolver doesn't recognize the already-installed ROCm fork as satisfying a `gsplat==X` pin — keeping that pin in `pyproject.toml` caused `pip install -e .` to silently re-fetch stock CUDA-only `gsplat` from PyPI afterwards and overwrite the ROCm build's files in `site-packages/gsplat/`. This was caught empirically while verifying this task (`import gsplat` reported version `1.4.0`, not the fork's `1.5.3+<commit>`, after a full image build). Fixed by dropping the pin from `pyproject.toml`'s dependency list; see the comment there for details.

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

## Phase 1: pure-Torch baseline — ✅ verified on gfx1151

`vanilla-nerf`, `mipnerf`, and `nerfacto --pipeline.model.implementation torch`
all completed 20 training iterations end-to-end on a real Radeon 8060S
(gfx1151), via `scripts/smoke-test.sh` against `tests/data/lego_test`, with no
accelerated dependency (gsplat/nerfacc/tiny-rocm-nn) required. Each run ended
with Nerfstudio's own "Training Finished" banner and a saved checkpoint; no
tracebacks, no NaNs. Re-run with `scripts/smoke-test.sh` inside the container
to reproduce; logs land in `tests/rocm/logs/` (gitignored).

Two genuine, ROCm-unrelated bugs were found and fixed along the way (both
reproduce on any platform, not just ROCm):
- Pillow ≥12 breaks `nerfstudio/data/utils/data_utils.py`'s `pil_to_numpy()`,
  which calls a private `PIL.Image._getencoder(...).setimage()` API whose
  arity changed. Nerfstudio's open-ended `Pillow>=10.3.0` pin lets this happen
  on a fresh install. Pinned to `Pillow<12` in `docker/Dockerfile.rocm` for
  now; worth reporting/fixing upstream in nerfstudio itself.
- `ns-train`'s dataparser subcommand (e.g. `blender-data`) must be the very
  last CLI token — tyro hands off argument parsing to it, so any `--flag`
  placed after it silently gets consumed by the dataparser's own parser
  instead of erroring. Not a nerfstudio bug, just an easy CLI-ordering trap;
  `scripts/smoke-test.sh`'s `run()` now enforces the correct order.

## Not in scope here

Full method parity (Instant-NGP, Splatfacto-MCMC, NeuS, etc.), performance
tuning, and multi-GPU are follow-on work, tracked as GitHub issues on this
repo once Phase 1 is confirmed green.
