# nerfacc validation + Instant-NGP training on gfx1151

## Context

This is the second sub-project of the porting plan's Phase 2 (following gsplat
validation + Splatfacto, completed and merged to `main` earlier). `nerfacc` is
the second of the three accelerated dependencies `nerfstudio-rocm` needs —
it's used by Nerfstudio's `nerfacc-data`-adjacent methods, most notably
Instant-NGP, for occupancy-grid-accelerated ray marching and sampling.

`bjoernellens1/nerfacc-rocm` currently exists as a completely unported,
fresh GitHub fork of `nerfstudio-project/nerfacc` (zero ROCm code — plain
CUDA-only `setup.py`). Meanwhile `AMD-Ecosystem/nerfacc` (referenced as
remote `amd` in the local checkout, same non-fork-relationship pattern as
`AMD-Ecosystem/gsplat` was for the gsplat sub-project) already has a real,
working ROCm port on its `release/0.5.3` branch — close to the
`nerfacc==0.5.2` version Nerstudio's `pyproject.toml` expects.

Diffing `release/0.5.3` against our fork's tip shows the actual port is
small: a `LICENSE` rename, a `README.md` trim, a 7-line `#ifndef USE_ROCM`
guard in `nerfacc/cuda/csrc/include/utils_math.cuh` (avoiding duplicate
`float2`/`float4` operator overloads that HIP's own headers already define),
and `setup.py` (the bulk of the diff — HIP build wiring). `nerfacc`'s test
suite (`tests/test_camera.py`, `test_grid.py`, `test_pack.py`, `test_pdf.py`,
`test_rendering.py`, `test_scan.py`, `test_vdb.py`) is byte-identical to
upstream — no ROCm-specific test rewrites exist or are needed.

Critically, `nerfacc`'s `setup.py` has the *same two bug patterns* gsplat's
did before this session's earlier fixes:
- `IS_ROCM = True` is a hardcoded module-level constant, not conditioned on
  `torch.version.hip` — it will "compile as ROCm" even on a CUDA build.
- `hipcc_flags` hardcodes `--offload-arch=gfx942` (CDNA/Instinct), the same
  bug as gsplat#2, which had a proven fix in this session (check
  `PYTORCH_ROCM_ARCH` env var first, fall back to `rocminfo`/hardcoded
  default only if unset).

Unlike gsplat, nothing in nerfacc's CUDA kernels uses `rocprim` warp-level
reduction primitives (`rocprim::warp_reduce`, `rocprim_warpSum`,
`cg::tiled_partition`) — grepping the port's CUDA sources found no such
usage. This means the wave32-vs-wave64 class of bug that was gsplat's
hardest problem (Task 2b, porting `AMD-Ecosystem/gsplat#17`) is not expected
to apply here. This is a hypothesis to verify empirically during the test
suite run, not an assumption to skip verification on.

## Decisions from brainstorming

- **Fork strategy**: delete `bjoernellens1/nerfacc-rocm` (currently has
  nothing of value beyond unmodified upstream history plus one banner
  commit) and re-fork `AMD-Ecosystem/nerfacc` fresh, renamed to
  `nerfacc-rocm`. This matches gsplat's actual fork lineage
  (`bjoernellens1/gsplat` is a real fork of `AMD-Ecosystem/gsplat`, not of
  `nerfstudio-project/gsplat` directly) and gets the working ROCm port as
  day-one content instead of porting individual patches onto a from-scratch
  base.
- Base branch: `release/0.5.3` (closest to nerfstudio's pinned
  `nerfacc==0.5.2`), not `main` (AMD-Ecosystem/nerfacc's `main` branch is
  unversioned/untested against nerfstudio's actual pin).

## Goal / success criteria

1. `bjoernellens1/nerfacc-rocm` exists as a public GitHub fork of
   `AMD-Ecosystem/nerfacc`, based on `release/0.5.3`, with a local checkout
   remote layout mirroring the gsplat pattern (`origin` = own fork,
   `upstream` = `nerfstudio-project/nerfacc` for reference/potential future
   PRs, since there's no direct fork-merge path to it).
2. Both known bug patterns fixed and verified: `IS_ROCM` conditioned on
   `torch.version.hip` (not a hardcoded `True`), and arch selection checking
   `PYTORCH_ROCM_ARCH` before falling back — same pattern as the gsplat#2
   fix, applied here.
3. `pip install --no-build-isolation .` succeeds and `import nerfacc`
   succeeds on real gfx1151 hardware.
4. nerfacc's own pytest suite (`test_camera.py`, `test_grid.py`,
   `test_pack.py`, `test_pdf.py`, `test_rendering.py`, `test_scan.py`,
   `test_vdb.py`) runs on gfx1151; failures are triaged (genuine kernel bug
   vs. environment issue vs. already-known) the same way gsplat's suite was.
   Do not assume the "no wave32 risk" hypothesis holds without this run.
5. `ns-train instant-ngp` trains successfully end-to-end on gfx1151 inside
   the `nerfstudio-rocm` ROCm container, using the fixed nerfacc build —
   this is nerfacc's actual integration point in Nerfstudio, the way
   Splatfacto was gsplat's.
6. nerfacc becomes a hard (non-fallback) build-time dependency in
   `docker/Dockerfile.rocm`, `dependencies/rocm-lock.toml` and `ROCM.md`
   updated to reflect verified state, matching gsplat's Task 5 pattern.

## Step 1: Repo setup

1. `gh repo delete bjoernellens1/nerfacc-rocm` (confirm — destructive, but
   the repo currently has nothing beyond unmodified upstream history and one
   banner commit, already established as acceptable to lose).
2. `gh repo fork AMD-Ecosystem/nerfacc --fork-name nerfacc-rocm --clone=false`
3. Local checkout: clone fresh, add `upstream` remote pointing to
   `nerfstudio-project/nerfacc`. Check out `release/0.5.3` and set it as the
   working branch (the fork's default branch may be `main` — verify and
   switch if so, matching gsplat's `release/1.5.3b2` pattern of using a
   specific stable branch rather than the default).
4. Update `nerfstudio-rocm`'s `dependencies/rocm-lock.toml` `[nerfacc]`
   section's `repo` field to the new fork URL (it currently points at the
   old, about-to-be-deleted `nerfacc-rocm` fork).

## Step 2: Fix the two known bug patterns

In the new fork's `setup.py`:
- Replace `IS_ROCM = True` with a check against `torch.version.hip is not
  None` (import `torch` at the point this is evaluated — check where in the
  file this needs to happen, since the current unconditional `True` is set
  at module level before any torch import).
- Apply the `PYTORCH_ROCM_ARCH`-first pattern to wherever
  `--offload-arch=gfx942` is hardcoded in `hipcc_flags` — same shape as the
  gsplat#2 fix (`bjoernellens1/gsplat` commit `4515618`): check the env var,
  split on comma/semicolon if multi-arch, take the first, fall back to the
  existing hardcoded default only if unset.

Verify: build succeeds with real GPU devices attached, targeting gfx1151
specifically (not silently compiling for gfx942).

## Step 3: Validate

Run nerfacc's own pytest suite on gfx1151 with real GPU devices attached.
Triage any failures the same way gsplat's Task 3 did: classify each as a
genuine kernel bug, a test-environment issue, or an already-known limitation
— fix what's fixable and scoped, document what isn't, don't block
indefinitely on out-of-scope work (e.g. if something depends on gsplat or
another still-partially-working dependency).

## Step 4: Instant-NGP end-to-end

Train Instant-NGP inside the `nerfstudio-rocm` ROCm container on gfx1151,
using the fixed nerfacc build. Needs a suitable dataset — Instant-NGP's
default dataparser is `InstantNGPDataParserConfig` (blender-format-adjacent,
per the original porting plan's method table), not COLMAP — determine the
right dataset during implementation (may reuse `tests/data/lego_test` from
Phase 1, or need a proper multi-view scene; decide once the actual
dataparser requirements are confirmed against nerfstudio's source, the way
Task 4 of the gsplat plan had to empirically correct its dataparser
command). Watch for the same class of "did it complete AND is the quality
believable" signal the gsplat plan used (occupancy grid updates producing
sane sampling behavior, no NaN/Inf, reasonable PSNR trend) — not just "did
it finish."

## Step 5: Integrate into nerfstudio-rocm

- `docker/Dockerfile.rocm`: nerfacc's install step goes from best-effort
  (`|| echo "..."`) to a hard `RUN` step, matching gsplat's Task 5.
- `dependencies/rocm-lock.toml`: update `[nerfacc]`'s `repo`, `branch`,
  `commit`, and `status` fields to reflect the new fork and verified state.
- `ROCM.md`: update the nerfacc row in the fork-status table and the
  architecture diagram's dependency list if the fork relationship changed
  its description.

## Out of scope

- `tiny-rocm-nn` RDNA porting — separate sub-project, not started.
- Wiring nerfacc into other Instant-NGP-adjacent methods beyond the base
  `instant-ngp` config (e.g. `instant-ngp-bounded`) — a natural follow-on
  once the base method works, not required for this sub-project's success
  criteria.
- Performance tuning / benchmarking against CUDA.
- Upstreaming fixes back to `AMD-Ecosystem/nerfacc` or
  `nerfstudio-project/nerfacc` — evaluate after validation, same as gsplat's
  PR #17 comment was posted only after the fix was proven working here.

## Verification

- `python -c "import nerfacc; print(nerfacc.__version__)"` succeeds inside
  the built ROCm container, reports a version consistent with the
  `release/0.5.3` fork, not silently falling back to a stock PyPI
  CUDA-only install (apply the same empirical-verification lesson from
  gsplat's Task 5 `gsplat==1.4.0` pin discovery — don't trust that pip
  "succeeded," actually check what got installed).
- `pytest tests/` inside the nerfacc checkout — record full pass/fail
  output, same rigor as gsplat's Task 3.
- `ns-train instant-ngp --max-num-iterations <N>` completes with
  Nerfstudio's own "Training Finished" banner, no NaN/Inf, and a training
  quality signal that looks healthy (specific criteria to be pinned down
  once Step 4 determines the actual dataset/dataparser).
- `git log` on `bjoernellens1/nerfacc-rocm` shows the two bug-pattern fixes
  as real commits, matching the no-AI-attribution rule established and
  enforced throughout this session's work (verify via `git log --format=%B`
  before considering any commit or GitHub comment complete).
