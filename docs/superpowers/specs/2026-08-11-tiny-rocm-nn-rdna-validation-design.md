# tiny-rocm-nn RDNA/gfx1151 validation

## Context

This is the third and final sub-project of the porting plan's Phase 2, following
gsplat + Splatfacto and nerfacc + Instant-NGP (both completed and merged to
`main`). `tiny-cuda-nn`/`tiny-rocm-nn` is Nerfstudio's fast hash-grid encoding
and fully-fused MLP library, wired in via `nerfstudio.utils.external.TCNN_EXISTS`
(set by `import tinycudann` succeeding) and used by `NerfactoField`
(`nerfstudio/fields/nerfacto_field.py`), which both `nerfacto` and
`instant-ngp` build on. `NerfactoField.__init__`'s `implementation` parameter
defaults to `"tcnn"`; when `TCNN_EXISTS` is `False` it silently prints a speed
warning and falls back to the pure-PyTorch path per-component (confirmed by
reading `nerfstudio/field_components/encodings.py:123-131` and the analogous
blocks in `mlp.py`) — this is why the nerfacc sub-project's Instant-NGP
training run succeeded without tiny-rocm-nn: it silently used the torch
fallback throughout. There is no special CLI flag needed for the integration
test in this sub-project — once `tinycudann` is importable, `ns-train
instant-ngp`/`nerfacto` automatically prefer the tcnn path by default.

`bjoernellens1/tiny-rocm-nn` is a fork of `ZJLi2013/tiny-rocm-nn` (itself an
independent ROCm reimplementation of `NVlabs/tiny-cuda-nn`, not a GitHub fork
of it — see `feedback_fork_existing_ports` memory for why we build on this
base rather than a fresh port). Our fork is currently one commit ahead of
`upstream/main` (a README banner only) — otherwise unmodified.

`ROCM.md`'s existing inventory (verified accurate by direct repo inspection
during this spec's research, not assumed from the README): `HashGrid`,
`Frequency`, `OneBlob`, and `SphericalHarmonics` encodings are ported;
`FullyFusedMLP` is ported and CDNA(gfx942)-verified upstream; `CutlassMLP` has
no ROCm equivalent. RDNA/gfx1151 is explicitly unverified — `rocm/capabilities.py`
marks `ROCM_RDNA_CAPS.verified_on_this_arch = False`.

### Research findings that shape this spec's scope

A pre-brainstorming research pass (required before scoping any GPU-kernel
work, per this project's established pattern of checking for prior art
first — see `feedback_fork_existing_ports`) found:

1. **Prior art exists and is directly relevant.** `kodai731/tiny-rocm-nn`
   (one of only 3 forks of `ZJLi2013/tiny-rocm-nn`) has a branch
   `rocm-gfx1100` (gfx1100 = RDNA3, same wave-size family as our gfx1151
   target) with one commit: "arrange to compile on gfx1100 and rocm 7.2",
   touching exactly `include/tiny-cuda-nn/cublas_matmul.h`. It replaces the
   ROCm-7.x-deprecated `hipblasDatatype_t` type with `hipDataType`/
   `hipblasComputeType_t` and updates the corresponding `hipblasGemmEx` call
   sites — a hipBLAS API-version compatibility fix, **not an RDNA-kernel
   rewrite**. No other RDNA/gfx1151/wave32-relevant fork, PR, or issue was
   found anywhere (`gh search prs`/`issues`/`code` all came back empty for
   these terms against `ZJLi2013/tiny-rocm-nn` and `NVlabs/tiny-cuda-nn`; no
   `AMD-Ecosystem` org repo exists for either).

2. **The actual current build failure was reproduced live on real gfx1151
   hardware**, inside `localhost/nerfstudio-rocm:phase3-nerfacc` (confirmed
   real GPU, ROCm 7.2.2): compilation fails in exactly
   `include/tiny-cuda-nn/cublas_matmul.h` with `error: unknown type name
   'hipblasDatatype_t'` and `no matching function for call to
   'hipblasGemmEx'` — the identical file and API mismatch `kodai731`'s
   branch already fixes, for the identical ROCm version. This is a
   build-time hipBLAS compatibility issue, not RDNA-specific — it would fail
   identically on any ROCm 7.2 target, including CDNA. The RDNA/wave32
   correctness question (whether `FullyFusedMLP`'s warp-shuffle-based kernels
   actually work on wave32 hardware) is **not yet reachable** — the build
   never gets past this compile error to test it.

3. **A second occurrence of the hardcoded-arch bug pattern was found** during
   this spec's own verification pass (not by the research fork):
   `bindings/torch/setup.py:53` reads
   `os.environ.get("PYTORCH_ROCM_ARCH", "gfx942")` — same shape as gsplat#2
   and nerfacc's `setup.py`, third occurrence of this exact bug class in this
   project. Also, `test/CMakeLists.txt` hardcodes
   `--offload-arch=gfx942` in three C++ test targets — lower priority (test
   binaries only, not the PyTorch bindings Nerfstudio actually loads), noted
   for awareness but not necessarily required to fix for this sub-project's
   goal.

Given this, **the scope here is "port the known fix + verify," not "write
RDNA kernels from scratch."** The open, genuinely unresolved question this
sub-project exists to answer is: once the build compiles, does
`FullyFusedMLP`'s warp-shuffle-based implementation actually produce correct
results on wave32 (RDNA) hardware, or does it silently corrupt results the
way gsplat's wave32 bug did before its fix? Unlike gsplat (which had a known,
portable upstream fix in `AMD-Ecosystem/gsplat#17`) and nerfacc (which
structurally has no warp intrinsics at all), no existing fix for a
`FullyFusedMLP`-on-RDNA correctness problem is known to exist anywhere. If
one is needed and none is found, **the bounded, legitimate success outcome
for this sub-project is determining and documenting whether it works** —
not an open-ended kernel rewrite. `ROCM_RDNA_CAPS.verified_on_this_arch`
must end up reflecting whatever is actually found, true or false.

## Decisions

- **Base the port on `kodai731`'s fix**, hand-verified (not blindly
  cherry-picked) against our own ROCm 7.2 target and checked for any
  gfx1100-specific hardcoding before use.
- **Fold the `bindings/torch/setup.py` arch-hardcoding fix into the same
  task** as the hipBLAS fix (same shape as gsplat#2/nerfacc's fix: check
  `PYTORCH_ROCM_ARCH` first, fall back to the existing hardcoded default only
  if unset) rather than discovering it as a separate blocker later.
- **The Dockerfile step stays best-effort (not hard) unless the wave32
  correctness check and the Instant-NGP integration test both come out
  clean.** Do not reflexively copy gsplat/nerfacc's "make it a hard
  dependency" pattern — it's only correct here if tiny-rocm-nn actually
  works on gfx1151, which is the open question this sub-project answers.
- **No explicit CLI implementation flag needed for the integration test** —
  `NerfactoField` (used by `nerfacto` and `instant-ngp`) already defaults to
  `implementation="tcnn"`; once `tinycudann` is importable
  (`nerfstudio.utils.external.TCNN_EXISTS` becomes `True`), Nerfstudio uses it
  automatically. The test is to confirm no `print_tcnn_speed_warning` /
  fallback-to-torch messages appear in the training log, i.e. tcnn is
  genuinely exercised, not silently bypassed the way nerfacc's Task 4
  accidentally was.

## Goal / success criteria

1. `bjoernellens1/tiny-rocm-nn` has the hipBLAS API-compat fix ported from
   `kodai731`'s branch (or an equivalent fix, independently verified against
   our ROCm 7.2 target) and the `bindings/torch/setup.py` arch-hardcoding bug
   fixed, both as real commits.
2. The PyTorch bindings (`bindings/torch/setup.py`) build successfully with
   real GPU devices attached, targeting gfx1151 specifically (verify via the
   same embedded-fatbinary-arch-grep technique used for nerfacc's Task 5, not
   just "the build succeeded").
3. `import tinycudann` succeeds inside the container, and
   `nerfstudio.utils.external.TCNN_EXISTS` is `True`.
4. The RDNA/wave32 correctness question is answered with direct evidence, not
   assumed: run tiny-rocm-nn's own test suite if one exists and is runnable
   with real GPU devices, AND run a targeted numerical check — compare
   `FullyFusedMLP`/`HashGrid`/`SphericalHarmonics` outputs (forward, and
   backward gradients) between the tcnn implementation and Nerfstudio's own
   pure-PyTorch reference implementation of the same encoding/MLP, on
   gfx1151, for a small representative input. This directly targets the
   wave32-correctness question in a way "did training not crash" cannot.
5. `ns-train instant-ngp` (or `nerfacto`, whichever gives a cleaner signal —
   decide during implementation) trains end-to-end on gfx1151 with tcnn
   genuinely active (no fallback warnings in the log), with a training
   quality signal at least as healthy as nerfacc's torch-fallback baseline
   run (comparable loss/PSNR trend, no NaN/Inf) — if step 4 finds a
   correctness problem, this step's expected outcome changes accordingly
   (e.g., testing only the encoding path with a torch MLP) rather than being
   run blindly.
6. `rocm/capabilities.py`'s `ROCM_RDNA_CAPS.verified_on_this_arch` and
   `ROCM.md`'s tiny-rocm-nn section are updated to state the real, verified
   outcome — whichever way it goes. `dependencies/rocm-lock.toml` updated.
   `docker/Dockerfile.rocm`'s tiny-rocm-nn step becomes a hard dependency
   ONLY if steps 4-5 came out clean; otherwise it stays best-effort with an
   accurate comment explaining the known limitation.

## Step 1: Port the build fix

In `/home/bjoern/git/tiny-rocm-nn` (on a new branch off `main`):
- Add `kodai731`'s fork as a reference remote, inspect the `rocm-gfx1100`
  branch's single commit to `include/tiny-cuda-nn/cublas_matmul.h`, and
  port the fix (cherry-pick if it applies cleanly, hand-port otherwise).
  Verify the diff doesn't hardcode `gfx1100` anywhere before using it.
- Fix `bindings/torch/setup.py:53`'s hardcoded `gfx942` fallback default the
  same way gsplat#2/nerfacc were fixed: read `PYTORCH_ROCM_ARCH` first, keep
  `gfx942` only as the last-resort fallback if the env var is unset.
- Grep the whole repo (not just these two files) for `gfx[0-9]`,
  `--offload-arch`, `hipblasDatatype_t` to check for any other occurrences of
  either bug class before moving on.

## Step 2: Build on gfx1151

Build `bindings/torch/` with real GPU devices attached inside the project's
ROCm container, targeting gfx1151. Expect new errors after the hipBLAS one
clears — triage each as a genuine bug vs. environment issue vs. already-known
limitation, same rigor as nerfacc's Task 3. Verify the compiled artifact's
embedded fatbinary contains `gfx1151` and not a stray `gfx942`, same
technique as nerfacc's Task 5.

## Step 3: Wave32/RDNA correctness verification

This is the core open question. Two parts:
- Run tiny-rocm-nn's own test suite (check `test/` — note it's currently
  CMake/C++ tests with their own hardcoded `gfx942` in `CMakeLists.txt`;
  decide whether to fix and run these C++ tests too, or whether the PyTorch
  binding-level check below is sufficient coverage — use judgment based on
  what's actually runnable and meaningful within scope).
- Write a small, targeted script (not a full training run) that constructs
  the same `HashGrid`/`SphericalHarmonics` encoding and `FullyFusedMLP` via
  both `tinycudann` and Nerfstudio's pure-PyTorch fallback path, feeds
  identical small input, and compares forward outputs and backward gradients
  numerically (e.g. `torch.allclose` with a documented, justified tolerance —
  tcnn's half-precision internals mean exact equality isn't the right bar;
  pick a tolerance and justify it, don't just eyeball it). This is the direct
  evidence gsplat's wave32 bug class would show up in — a subtly wrong
  computation, not a crash.

Document the outcome honestly either way. If a real correctness problem is
found, do not attempt an open-ended kernel fix in this sub-project — document
the specific failure mode, mark `ROCM_RDNA_CAPS.verified_on_this_arch =
False` with the reason, and treat that as this sub-project's legitimate,
bounded completion.

## Step 4: Instant-NGP/Nerfacto integration test

Only if Step 3 finds tiny-rocm-nn numerically correct on gfx1151: train
`ns-train instant-ngp` (or `nerfacto`, decide based on which gives a cleaner
signal) on the same `bonsai` scene used for nerfacc's validation, confirming
via the training log that tcnn is genuinely active (no
`print_tcnn_speed_warning` messages) and that training quality is healthy
(comparable to or better than the nerfacc sub-project's torch-fallback
baseline: loss/PSNR trend, no NaN/Inf).

If Step 3 finds a correctness problem, adapt this step's goal — e.g. test
only the parts of tiny-rocm-nn confirmed correct (the encodings, with
Nerfstudio's torch MLP), rather than running a training job with a known-bad
component.

## Step 5: Integrate into nerfstudio-rocm

- `dependencies/rocm-lock.toml`: update `[tiny_rocm_nn]` (branch/commit,
  real verified status).
- `ROCM.md`: rewrite the "tiny-rocm-nn: what's actually there" section with
  the real, direct-evidence outcome of Steps 3-4 — including if the outcome
  is "does not work correctly on RDNA, here's why."
- `rocm/capabilities.py`: `ROCM_RDNA_CAPS.verified_on_this_arch` set to
  match the real finding.
- `docker/Dockerfile.rocm`: tiny-rocm-nn's build step becomes hard
  (non-fallback) ONLY if Steps 3-4 are clean; otherwise it stays best-effort,
  with its comment updated to state the specific, verified reason (not the
  old generic "not yet ported" text) — e.g. "builds and imports, but
  FullyFusedMLP produces numerically incorrect gradients on RDNA/wave32; see
  ROCM.md."
- Check `pyproject.toml` for the same distribution-name-vs-import-name pin
  risk gsplat/nerfacc had (search for a `tinycudann==` pin) — verify
  empirically inside a real build, don't assume either way.

## Out of scope

- Writing new RDNA-specific kernels (warp-shuffle rewrites, wave32-aware
  reductions) if Step 3 finds a genuine correctness problem — that becomes a
  separate, explicitly-scoped follow-up, not silently absorbed into this
  plan.
- `CutlassMLP` — already known to have no ROCm equivalent; out of scope
  unless this sub-project's research turns up new prior art (unlikely,
  not found in the initial research pass).
- Fixing `test/CMakeLists.txt`'s hardcoded `gfx942` in the C++ test targets,
  unless Step 3 determines running those tests is necessary for the
  correctness verification.
- Performance tuning/benchmarking against CUDA tiny-cuda-nn.
- Upstreaming fixes to `ZJLi2013/tiny-rocm-nn` or crediting/notifying
  `kodai731` — evaluate after validation succeeds, same pattern as gsplat's
  PR #17 comment (posted only once the fix was proven working here).

## Verification

- `python -c "import tinycudann; print(tinycudann.__version__ if hasattr(tinycudann, '__version__') else 'ok')"`
  succeeds inside the built container.
- Embedded fatbinary of the built `.so` contains `gfx1151`, not `gfx942`.
- The forward/backward numerical comparison script's output (documented
  tolerance, pass/fail per component) — the load-bearing evidence for the
  wave32-correctness question.
- If Step 4 runs: `ns-train instant-ngp`/`nerfacto` completes with
  Nerfstudio's "Training Finished" banner, no NaN/Inf, and the training log
  shows no tcnn-fallback warnings.
- `git log --format=%B` on every new commit in `bjoernellens1/tiny-rocm-nn`
  and `nerfstudio-rocm` shows no AI/Claude/assistant attribution — this
  project's zero-exception rule, verified explicitly before considering any
  task complete.
