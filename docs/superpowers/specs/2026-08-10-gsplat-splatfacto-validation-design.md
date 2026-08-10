# gsplat validation + Splatfacto training on gfx1151

## Context

`nerfstudio-rocm` Phase 1 (pure-Torch methods: vanilla-nerf, mipnerf,
nerfacto with `implementation=torch`) is verified working end-to-end on a
real Radeon 8060S (gfx1151). This is the first sub-project of the larger
porting plan's Phase 2: get the `gsplat` accelerated dependency actually
validated on real hardware and Splatfacto (3D Gaussian Splatting) training
end-to-end, rather than relying on documentation claims.

`bjoernellens1/gsplat` (fork of `AMD-Ecosystem/gsplat`, pinned at
`release/1.5.3b2` in `dependencies/rocm-lock.toml`) is documented as
"verified" on gfx1151/gfx1100, but that verification predates this fork and
was never re-confirmed here. Last session, attempting to build gsplat inside
`docker/Dockerfile.rocm` surfaced two real, filed bugs
([gsplat#2](https://github.com/bjoernellens1/gsplat/issues/2),
[gsplat#3](https://github.com/bjoernellens1/gsplat/issues/3)) and the
Dockerfile currently treats the gsplat install as best-effort (falls through
silently on failure) rather than a hard requirement. This sub-project closes
that gap: get gsplat's own test suite green on this exact hardware, then
prove the pipeline works by training Splatfacto on a real multi-view scene.

The original porting-plan doc marks this as difficulty 2-3/5, low/medium
risk — the easiest of the three accelerated dependencies, and the natural
first sub-project to de-risk before attempting nerfacc or tiny-rocm-nn.

## Goal / success criteria

1. gsplat's own pytest suite (`test_basic.py`, `test_rasterization.py`,
   `test_2dgs.py`, `test_compression.py`, `test_ftheta.py`,
   `test_strategy.py`) passes on gfx1151 with real GPU devices attached.
   `_test_distributed.py` (multi-GPU) is explicitly out of scope.
2. `ns-train splatfacto` runs 7000 iterations on the `bonsai` mip-nerf360
   scene without crashing, with the Gaussian count growing (densification
   working) and loss/PSNR trending in the right direction, no NaN/Inf.
3. gsplat is a hard (not best-effort/fallback) build step in
   `docker/Dockerfile.rocm`, and its pinned commit in
   `dependencies/rocm-lock.toml` reflects whatever fix was needed.
4. `ROCM.md`'s gsplat status line reflects the real, current, re-verified
   state — not the pre-existing "verified" claim inherited from before this
   fork existed.
5. A `splatfacto` smoke test exists under `tests/rocm/` and is wired into
   `.github/workflows/tests-rocm.yml` (manual-trigger, self-hosted-runner
   job — no such runner exists yet, so this stays dormant until one does,
   same pattern as Phase 1's methods).

## Step 1: root-cause the gsplat build failure

This blocks everything else, so it comes first.

**Known state:** `docker/Dockerfile.rocm` currently clones gsplat's
`release/1.5.3b2`, `git submodule update --init --recursive` (populates the
vendored `gsplat/cuda/csrc/third_party/glm` submodule), then
`pip install --no-build-isolation .` — and this fails during `docker build`
for two independent, already-diagnosed reasons:

- **gsplat#2**: `setup.py`'s `get_rocm_arch()` shells out to `rocminfo` and
  ignores `PYTORCH_ROCM_ARCH`; `docker build` has no GPU device access, so
  this silently falls back to `gfx942` instead of erroring or respecting the
  intended target. This is purely a *build-time-vs-runtime* problem — it
  should not reproduce when installing inside a *running* container with
  `--device=/dev/kfd --device=/dev/dri` attached (`rocminfo` works there).
- **gsplat#3**: the compiled `hipcc` command for `.hip` sources was missing
  the `-I.../gsplat/cuda/csrc/third_party/glm` flag that `setup.py`'s
  `include_dirs` list should have supplied, so the compiler picked up
  `/usr/include/glm` (from the `libglm-dev` apt package we had installed at
  the time) instead — a non-HIP-aware GLM, causing template resolution
  failures on `mat3 * vec3` and similar operators used inside `.hip`
  kernels. We since removed `libglm-dev` from the Dockerfile's apt-get step
  as part of making the gsplat step best-effort, but **never re-tested
  whether that alone resolves #3** — it's plausible the include path issue
  is real regardless of what system GLM exists (the flag was simply absent
  from the logged command), in which case removing `libglm-dev` just
  changes the failure mode (from wrong-glm-found to no-glm-found) rather
  than fixing it.

**Plan:**
1. Rebuild the current `docker/Dockerfile.rocm` image (no `libglm-dev`), run
   gsplat's install step inside a *running* container with real GPU devices
   attached (via `scripts/install-rocm-deps.sh`, which already targets
   runtime install for exactly this reason), and see what actually happens
   now on both fronts (arch detection and glm include path).
2. If #2 is resolved by runtime GPU access (expected) but #3 still
   reproduces, fix `setup.py`'s `include_dirs` handling for the HIP build
   path directly in the gsplat fork — this needs actual debugging of why
   `CUDAExtension`'s `include_dirs` isn't reaching the hipified `.hip`
   compile command (a torch/hipify interaction bug, scope per gsplat#3).
   Push the fix to `bjoernellens1/gsplat` as a proper commit (not a
   workaround here), since this needs to be reliable for CI later and is
   the kind of fix worth eventually upstreaming to `AMD-Ecosystem/gsplat`.
3. Once gsplat installs and imports cleanly
   (`python -c "from gsplat.cuda._backend import _C; print(_C)"` succeeds),
   move to Step 2.

## Step 2: gsplat test suite

Run gsplat's full pytest suite (minus `_test_distributed.py`) inside the
ROCm container with real GPU devices attached, targeting gfx1151. Triage and
fix any failures found — expect possible RDNA-specific issues (wave32 vs
wave64, atomics, cooperative-group differences) per the original plan's risk
assessment, even though this fork's docs claim gfx1151 was already verified;
treat that claim as unconfirmed until this run actually happens.

Record pass/fail per test file. A test that's skipped for a legitimate
architectural reason (e.g. requires a feature genuinely unavailable on
RDNA) is acceptable if documented — but no silent skips.

## Step 3: Splatfacto training on bonsai

- Data: `/home/bjoern/Downloads/mipnerf360_v2_dataset/bonsai` — already has
  COLMAP `sparse/0/{cameras,images,points3D}.bin`. Use the `images_4` or
  `images_8` (downsampled) directory for reasonable iteration speed; decide
  which based on observed iteration time once training starts (target: keep
  7000 iterations to a practical wall-clock time, adjust downsample factor
  if `images_4` proves too slow).
- Command: `ns-train splatfacto --data <path> --max-num-iterations 7000`
  inside the nerfstudio-rocm ROCm container (same container pattern as
  Phase 1's `scripts/smoke-test.sh`, extended or run ad hoc for this
  longer/one-off validation run — doesn't need to block on a permanent
  script yet).
- Watch for: training starts, loss decreases over the run, Gaussian count
  in the model grows (densification is working — Splatfacto starts from a
  sparse point cloud and adds Gaussians during training), no NaN/Inf losses,
  checkpoint saves. A rendered output image comparison is a nice-to-have,
  not a hard requirement for this sub-project.
- Note: this machine may have other GPU jobs running concurrently (as seen
  during Phase 1 — a `splatograph` training job caused significant slowdown
  but not failure). Expect and tolerate slower wall-clock time from
  contention; only treat genuine hangs/errors as failures.

## Step 4: integrate back into nerfstudio-rocm

- `docker/Dockerfile.rocm`: change gsplat's install step from best-effort
  (`|| echo "..."`) to a hard `RUN` step once Step 1 is resolved. Keep
  nerfacc and tiny-rocm-nn best-effort (unrelated to this sub-project, still
  unported).
- `dependencies/rocm-lock.toml`: update the gsplat commit pin if Step 1
  required new commits, and update its `status` field to reflect what was
  actually verified this session (test suite pass rate, Splatfacto result)
  rather than the prior generic "verified" claim.
- `ROCM.md`: update the gsplat row in the fork-status table and the
  "Known build issues" section (remove or update the gsplat#2/#3 entry once
  resolved).
- Add `tests/rocm/README.md`-documented `splatfacto` case to
  `.github/workflows/tests-rocm.yml`'s smoke-test list (it's already
  manual-trigger/self-hosted-runner-gated, matching Phase 1's pattern — no
  runner exists yet, so this doesn't run automatically, just documents
  intent for when one does).

## Out of scope

- Splatfacto-MCMC, Splatfacto-big (variant configs) — natural quick
  follow-ons once plain Splatfacto works, not part of this sub-project.
- Multi-GPU / distributed training.
- Performance benchmarking against CUDA reference hardware.
- nerfacc and tiny-rocm-nn porting — separate sub-projects per the
  decomposition already agreed with the user.

## Verification

- `pytest tests/` inside gsplat (excluding `_test_distributed.py`) — record
  full pass/fail output.
- `python -c "from gsplat.cuda._backend import _C; print(_C)"` succeeds
  inside the built `nerfstudio-rocm` ROCm image (i.e. the Dockerfile's hard
  gsplat step actually works, not just a manually-debugged one-off).
- `ns-train splatfacto --data <bonsai path> --max-num-iterations 7000`
  completes with Nerfstudio's own "Training Finished" banner, a saved
  checkpoint, and no NaN/Inf in the loss log.
- `git log` on `bjoernellens1/gsplat` shows the actual fix commit(s) for
  gsplat#3 if needed, and the corresponding GitHub issue(s) closed with a
  comment explaining the fix.
