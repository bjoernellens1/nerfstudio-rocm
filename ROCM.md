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
| [gsplat](https://github.com/bjoernellens1/gsplat) | [AMD-Ecosystem/gsplat](https://github.com/AMD-Ecosystem/gsplat) (→ [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)) | **verified on gfx1151**: builds cleanly under `docker build` (hard dependency in `docker/Dockerfile.rocm`); full pytest suite 114 passed / 1 skipped / **87 failed** — all 87 attributable to the test-only `nerfacc` dependency having no ROCm support at the time (measured before `nerfacc-rocm` existed; worth re-running now that a verified ROCm nerfacc is available, see the nerfacc-rocm row below), zero gsplat kernel failures (see [#5](https://github.com/bjoernellens1/gsplat/issues/5)); Splatfacto trained 7000 iterations on the bonsai (mip-nerf360) scene end-to-end with no NaN/Inf and confirmed densification |
| [nerfacc-rocm](https://github.com/bjoernellens1/nerfacc-rocm) | [AMD-Ecosystem/nerfacc](https://github.com/AMD-Ecosystem/nerfacc) (→ [nerfstudio-project/nerfacc](https://github.com/nerfstudio-project/nerfacc)) | **verified on gfx1151**: builds cleanly under `docker build` (hard dependency in `docker/Dockerfile.rocm`); own pytest suite 23 passed / 0 failed (21/23 real exercised coverage; see below); Instant-NGP trained 5000 iterations on the bonsai (mip-nerf360) scene end-to-end with no NaN/Inf, healthy loss/PSNR trend, and a final-checkpoint occupancy grid 29.07% occupied confirming `OccGridEstimator` correctly prunes empty space |
| [tiny-rocm-nn](https://github.com/bjoernellens1/tiny-rocm-nn) | [ZJLi2013/tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) | **verified on gfx1151** — see below |
| [colmap](https://github.com/bjoernellens1/colmap) | [colmap/colmap](https://github.com/colmap/colmap) | **full HIP acceleration** (GPU SIFT, patch_match_stereo, CASPAR bundle adjustment) — `hip-integration` branch, hard dependency in `docker/Dockerfile.rocm` (see §COLMAP) |

Pinned commits for all of the above: [`dependencies/rocm-lock.toml`](dependencies/rocm-lock.toml).
Run `python -m rocm.diagnostics` (`ns-rocm-info`) inside the container to print the live stack + pins.

## Known build issues

gsplat's two `docker build`-time issues are both fixed: the vendored glm submodule not reaching hipcc's include path ([gsplat#3](https://github.com/bjoernellens1/gsplat/issues/3), fixed in [42d17c9](https://github.com/bjoernellens1/gsplat/commit/42d17c9)), and `get_rocm_arch()` ignoring `PYTORCH_ROCM_ARCH` and silently falling back to `gfx942` since `docker build` has no GPU device access to run `rocminfo` ([gsplat#2](https://github.com/bjoernellens1/gsplat/issues/2), fixed in [4515618](https://github.com/bjoernellens1/gsplat/commit/4515618)). `docker/Dockerfile.rocm` now installs gsplat as a hard, non-fallback build step.

- **nerfstudio's `gsplat==1.4.0` pin is intentionally not carried over**: the ROCm fork builds/publishes under the distribution name `amd_gsplat` (not `gsplat`, so it can coexist with upstream on PyPI) even though it installs into the same `gsplat` import namespace. Because the distribution names differ, pip's resolver doesn't recognize the already-installed ROCm fork as satisfying a `gsplat==X` pin — keeping that pin in `pyproject.toml` caused `pip install -e .` to silently re-fetch stock CUDA-only `gsplat` from PyPI afterwards and overwrite the ROCm build's files in `site-packages/gsplat/`. This was caught empirically while verifying this task (`import gsplat` reported version `1.4.0`, not the fork's `1.5.3+<commit>`, after a full image build). Fixed by dropping the pin from `pyproject.toml`'s dependency list; see the comment there for details.

- **nerfstudio's `nerfacc==0.5.2` pin has the identical problem, and was fixed the identical way**: the ROCm fork builds/publishes under the distribution name `amd_nerfacc` (not `nerfacc`) while still installing into the same `nerfacc` import namespace. Keeping the `nerfacc==0.5.2` pin caused `pip install -e .` to silently re-fetch stock CUDA-only `nerfacc` from PyPI afterwards and overwrite the ROCm build's files in `site-packages/nerfacc/` — confirmed empirically (`import nerfacc` reported `0.5.2`, not the fork's `0.5.3`, after a full image build before the fix). Fixed by dropping the pin from `pyproject.toml`'s dependency list; a rebuild afterwards confirmed `import nerfacc` correctly reports `0.5.3`.

- **nerfacc's docker-built kernel artifact hasn't had a live on-device GPU re-run since the last verification pass**: the Instant-NGP training run and pytest suite (see "nerfacc-rocm: verified on gfx1151" below) were completed successfully, but a follow-up live-GPU kernel smoke test of that same docker-built artifact hit failures root-caused to external GPU contention on the dev machine (confirmed via a control test against a stock, unmodified image, which failed the same way) — not a defect in the nerfacc-rocm build itself. Tracked as a follow-up: re-run the on-device GPU kernel test once the GPU is free of contention to close this out.

- **FIXED: `nerfstudio/utils/eval_utils.py:62`'s `torch.load(load_path, map_location="cpu")` broke under torch's post-2.6 `weights_only=True` default**, unrelated to ROCm — `pyproject.toml` pins `torch==2.7.1` (affected). Loading a trained nerfstudio checkpoint through the stock `eval_setup()` helper (used by `ns-eval`/`ns-render`) raised `_pickle.UnpicklingError: Weights only load failed ... Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default`, because nerfstudio checkpoints pickle a bare numpy scalar in their state dict. Found while verifying tiny-rocm-nn's Instant-NGP integration (`ns-train instant-ngp` itself trains and checkpoints fine; only the *loading* path was affected). Fixed by passing `weights_only=False` explicitly at all three affected `torch.load` call sites (`eval_utils.py`, and `engine/trainer.py`'s two checkpoint-resume call sites, same bug class) — these load nerfstudio's own trusted checkpoint output, not third-party input, so this is safe. Verified via a synthetic repro matching the exact failure signature (a bare numpy scalar embedded in a pickled state dict): fails under the torch default, succeeds with the fix.

## tiny-rocm-nn: what's actually there

Verified directly against the pinned commit (not assumed from its README, which
undersells current state in one place and doesn't mention RDNA support at all).
Upstream `tiny-rocm-nn`'s README explicitly scopes itself to "AMD GPUs with
Matrix Core support (CDNA/RDNA3)" and MFMA/rocWMMA, with documented benchmarks
MI300X-only — nothing in the repo itself confirmed wave32/RDNA correctness, so
it was verified independently rather than assumed:

- ✅ `HashGrid`, `Frequency`, `OneBlob` encodings — ported
- ✅ `SphericalHarmonics` encoding — ported (`include/tiny-cuda-nn/encodings/spherical_harmonics.h`), wired into `encoding.cpp`. The original porting-plan assumption that this was missing was **stale**; Nerfstudio's `SHEncoding` with `implementation="tcnn"` works as-is.
- ✅ `FullyFusedMLP` — ported, verified on both **MI300X (gfx942, CDNA)** and, as of this section, **gfx1151 (RDNA3.5)**.
- ❌ `CutlassMLP` — no ROCm equivalent; widths outside `{64, 128}` and `n_hidden_layers` requiring Cutlass fall back to `implementation="torch"`.
- ✅ **RDNA / gfx1151 (this project's primary dev target, the Radeon 8060S) — verified, `ROCM_RDNA_CAPS.verified_on_this_arch = True`.** Build fix: ROCm 7.x deprecated `hipblasDatatype_t` in favor of `hipDataType`/`hipblasComputeType_t`, breaking the build; ported the type-compatibility fix from `kodai731/tiny-rocm-nn`'s `rocm-gfx1100` branch (6 lines, `include/tiny-cuda-nn/cublas_matmul.h` only) onto `bjoernellens1/tiny-rocm-nn`'s `rocm-gfx1151-validation` branch, pinned commit in `dependencies/rocm-lock.toml`. **tiny-rocm-nn vendors `dependencies/fmt` as a git submodule that a plain `git clone` does not populate — the build fails on a missing `format.cc` without `--recursive`**; `docker/Dockerfile.rocm`'s clone step uses `--recursive` for exactly this reason.

  Numerical correctness (`tests/rocm/tcnn_wave32_correctness.py`, run against a real Radeon
  8060S confirmed to be wave/warp size 32, batch granularity 256): each of `HashEncoding`
  (HashGrid), `SHEncoding` (SphericalHarmonics), and `MLP` (`FullyFusedMLP`, `otype` asserted
  to be `FullyFusedMLP` not `CutlassMLP`) was checked forward and backward against an
  independent PyTorch reference — a verbatim transcription of tcnn's own C++ semantics fed
  tcnn's own parameter buffer, not nerfstudio's differently-shaped torch encodings — across a
  batch-size sweep (`1, 31, 33, 127, 128, 129, 255, 256, 257, 4096`) straddling the 32/64 lane
  boundaries and the 256 batch granularity:
  - `HashEncoding`: forward error sits at the fp16 rounding floor at every batch size (worst
    relative error 1.5e-03); backward d/dparams relative error 6.6e-04 to 1.05e-03; the
    layout-free partition-of-unity invariant (uniform grid params -> output must be exactly
    0.5) holds to 7.32e-04, i.e. 1 fp16 ULP.
  - `SHEncoding`: forward output is **bit-identical** to the fp16 rounding of the exact fp32
    result (`|tcnn-b| = 0` at most batch sizes); backward relative error 2.2e-04 to 2.5e-04.
  - `FullyFusedMLP`: for the ReLU-activation MLP, forward relative error <=2.15e-03 across all
    ten batch sizes, at roughly 2.5-4.6x the derived fp16 error floor. Backward weight-gradient
    relative error 1.2e-03 on kink-safe rows. One apparent all-rows backward mismatch at bs=128
    with `activation=ReLU` (3/128 rows, up to 11.9% relative error) was root-caused, not waved
    away: a seed sweep showed the disagreeing row *indices* move freely with the input seed
    (including one seed with zero disagreement), which a wave32 lane bug cannot produce since
    lane mapping doesn't depend on input values. A smooth-activation (`Sigmoid`) control run
    against the same all-rows check is the more persuasive evidence: it sits at or near the
    fp16 floor (0.85x-3.5x the floor across the four backward checks, vs. the ReLU all-rows
    failure's 325x the floor: `|tcnn-a|` 2.058e-02 against a 6.323e-05 floor). Conclusion: an
    inherent fp16 ReLU-kink sensitivity (identical on NVIDIA stock tiny-cuda-nn), not a
    gfx1151/wave32 defect.
  - Bitwise-deterministic forward across 3 repeated runs for all three components (backward
    determinism not separately re-verified, expected non-deterministic for `HashGrid` by
    construction since its backward scatter-adds into the parameter table via atomics).

  End-to-end integration (`ns-train instant-ngp`, bonsai scene from mip-nerf360, 5000
  iterations, tcnn's `implementation="tcnn"` genuinely active — confirmed via live pipeline
  module introspection finding real `tinycudann.modules.{Encoding,Network,NetworkWithInputEncoding}`
  instances, not just the absence of a fallback warning in the log): zero NaN/Inf across all
  20 logged TensorBoard scalar series over the full run; train loss 0.1196 -> 0.001537, train
  PSNR 9.34 -> 28.13 dB, eval PSNR 22.22 -> 26.55 dB. This **beats** the torch-fallback
  Instant-NGP baseline on the same scene (train loss 0.0044, train PSNR 23.4 dB, see the
  nerfacc-rocm section below) — lower final loss, higher final PSNR, no instability anywhere
  in either run.

  No wave32-lane/warp defect of the kind found in gsplat's early history exists in any
  nerfstudio-reachable tiny-rocm-nn code path. Scope note: only `HashGrid`, `SphericalHarmonics`,
  `FullyFusedMLP` (width 64), and `NetworkWithInputEncoding` with an `Identity` encoding were
  independently numerically verified — `CutlassMLP`, other `n_neurons` widths, other encodings
  (`Frequency`, `TriangleWave`, `OneBlob`, `Composite`), and second-order gradients were not.
  Full evidence and methodology: `tests/rocm/tcnn_wave32_correctness.py` — the numbers cited
  above are transcribed directly from a run of that script (`tests/rocm/logs/` is gitignored
  and not part of the durable record; re-run the script against real hardware to reproduce).

`tiny-rocm-nn` now installs as a hard, non-fallback build step in
`docker/Dockerfile.rocm` (same pattern as gsplat/nerfacc), pointing at the
`rocm-gfx1151-validation` branch.

`rocm/capabilities.py` exposes these per-component so Nerfstudio can mix
tcnn and torch implementations rather than an all-or-nothing switch.

## nerfacc-rocm: verified on gfx1151

Re-forked from `AMD-Ecosystem/nerfacc`'s prior ROCm work (which hardcoded
`--offload-arch=gfx942` and a module-level `IS_ROCM = True` regardless of the
actual installed PyTorch build). Both bugs were fixed in
[`bjoernellens1/nerfacc-rocm`](https://github.com/bjoernellens1/nerfacc-rocm)
`release/0.5.3`, same shape as the equivalent fixes already applied to the
`gsplat` fork: `IS_ROCM` is now derived from `torch.version.hip`, and arch
selection checks `PYTORCH_ROCM_ARCH` first, falling back to `gfx942` only if
unset.

Validation performed (not just a build smoke test):
- The fork's own pytest suite: **23 passed / 0 failed**, run twice from a
  fresh clone/build for a determinism check (identical both times). 21/23 is
  real exercised coverage — the other 2 (`test_vdb.py`) short-circuit on a
  missing optional `fvdb` dependency unrelated to ROCm. All
  occupancy-grid/traversal, PDF/importance-sampling, rendering (including the
  backward pass), and scan tests pass in full.
- A structural (not just empirical) check for the wave32/wave64-risk
  hypothesis: nerfacc's HIP kernels contain no warp-level intrinsics
  (`__shfl`/`__ballot`/`__syncwarp`/etc.) anywhere; the one `cub::`-based fast
  path (`scan_cub.cu`) is preprocessed out entirely under HIP, so the one
  lane-width-sensitive code path is structurally absent from the ROCm build,
  not merely untested.
- An end-to-end `ns-train instant-ngp` run on the bonsai (mip-nerf360) scene,
  5000 iterations: train loss 0.114→0.0044 (26x), train PSNR 9.6→23.4 dB,
  eval PSNR 15.2→20.6 dB (peak 22.8), no NaN/Inf anywhere. The final
  checkpoint's occupancy grid is 29.07% occupied — direct evidence
  `OccGridEstimator` is actively pruning empty space on gfx1151, not just
  running without crashing.

Now installs as a hard, non-fallback build step in `docker/Dockerfile.rocm`
(same pattern as gsplat).

## COLMAP: full HIP acceleration

`docker/Dockerfile.rocm` now builds `bjoernellens1/colmap`'s `hip-integration`
branch from source (`-DCUDA_ENABLED=OFF -DHIP_ENABLED=ON -DCASPAR_ENABLED=ON
-DCMAKE_HIP_ARCHITECTURES=${PYTORCH_ROCM_ARCH}`), replacing the previous
`caspar-opencv-support` pin. This branch has all three of COLMAP's GPU-heavy
paths HIP-accelerated: GPU SIFT feature extraction/matching, `patch_match_stereo`
dense stereo, and CASPAR bundle adjustment (including native `OPENCV`
camera-model support). The previous `caspar-opencv-support` branch only had
`patch_match_stereo`; `hip-integration` is a strict superset that additionally
absorbed OpenCV-Caspar support and fixed a real NaN bug in Caspar's BA that
`caspar-opencv-support` still has unpatched (see below).

**Where this verification comes from.** This exact branch/commit and build
recipe is already proven end-to-end on real gfx1151 hardware — not by this
repo, but by an independent sibling project, `rosbag-colmap-pipeline`
(`colmap-rgbd-gt` / `gttool`), which vendors the same `hip-integration` source
as its own default local Docker build and has run its full production
pipeline (`gttool run-colmap`, not just raw `colmap` CLI calls) against a
613-frame real RGB-D scene: 613/613 frames registered, genuine GPU-side work
confirmed in logs throughout feature extraction/matching/global positioning
(not a silent CPU fallback), and Caspar-HIP `OPENCV`-model bundle adjustment
converging correctly after a real bug was found, root-caused, and fixed. See
that repo's `docs/local-hip-run.md` for the full verification writeup.

**The bug that was found and fixed** (`bjoernellens1/colmap` commit
`b5ead5a9`, now part of `hip-integration`): two of Caspar's generated
`OPENCV`-camera-model score kernels declared their per-thread squared-residual
accumulator once per invocation but only ever assigned it inside a
problem-size guard; `SumStore()` then read that accumulator unconditionally
for every thread in the launched block, so any "padding lane" thread (the
common case for real problem sizes) read an uninitialized register —
undefined behavior that reliably decoded as NaN for OpenCV's larger,
more register-pressured kernel, silently corrupting the whole reduction from
iteration 0. Fixed by explicitly zeroing the accumulator for out-of-range
threads before the reduction. Verified on a 613-image/54458-point real
problem: genuine 70-90-iteration convergence (score `4.5487e5 → ~4.4298e5`)
instead of a false 3-iteration bail-out with unchanged output. The same
uninitialized-accumulator pattern likely exists in every other generated
score kernel across every camera model — currently unpatched (numerically
silent so far, not proven safe by construction) and flagged as an open risk,
not blanket-patched without per-model verification.

**A second, independent bug was found and fixed in this repo while wiring
this in**: `nerfstudio/process_data/colmap_utils.py` used the CLI option names
`--SiftExtraction.use_gpu`/`--SiftMatching.use_gpu`, which upstream COLMAP
renamed to `--FeatureExtraction.use_gpu`/`--FeatureMatching.use_gpu` in 3.13.0
(part of generalizing feature extraction beyond SIFT) without keeping the old
names as aliases. `hip-integration` builds COLMAP 4.2.0.dev0, well past that
rename — the old flag names would have caused `feature_extractor`/
`*_matcher` to error out, meaning GPU SIFT would never actually have been
reachable through nerfstudio's own COLMAP invocation even with a fully
HIP-accelerated binary. Not ROCm-specific (would break on any recent CUDA
COLMAP too), but only surfaced now because this is the first time this repo
built a COLMAP recent enough to hit it. Fixed with a version gate using the
existing `get_colmap_version()` check already present in that file.

**Build-time verification done in this repo**: `docker build -f
docker/Dockerfile.rocm` completes successfully (no GPU device access needed
to compile). The resulting binary reports `COLMAP 4.2.0.dev0 (Commit
807af94b ... with HIP)` — confirming the exact pinned, attribution-clean
commit and that HIP support actually compiled in, not silently disabled.
`feature_extractor -h` confirms `--FeatureExtraction.use_gpu` is the real,
present flag (validating the `colmap_utils.py` fix above against the actual
binary, not just the source), and `bundle_adjuster -h` confirms the `CASPAR`
backend and its options are present.

**What's not yet independently verified in *this* repo specifically**: actual
runtime GPU execution — a real `ns-process-data`/`colmap feature_extractor
--gpu` run producing correct features/matches on this repo's own container.
The `rosbag-colmap-pipeline` verification above is strong evidence the same
source/build recipe works at runtime too, but this repo hasn't run that
check itself yet (blocked on a transient host GPU issue at the time this was
wired in — tracked as a follow-up, not a defect in the approach). Do the same
rigor as the other dependencies once GPU access is available: run a real
`ns-process-data images --gpu` (or equivalent COLMAP invocation) against a
real image set end-to-end.

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

Instant-NGP is now verified end-to-end (see "nerfacc-rocm: verified on
gfx1151" above). Remaining method parity (Splatfacto-MCMC, NeuS, etc.),
performance tuning, and multi-GPU are follow-on work, tracked as GitHub
issues on this repo.
