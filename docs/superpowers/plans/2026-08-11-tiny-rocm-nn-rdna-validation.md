# tiny-rocm-nn RDNA/gfx1151 validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the known hipBLAS-compat fix into `bjoernellens1/tiny-rocm-nn`, get it building on gfx1151, and determine — with direct numerical evidence, not just "did training crash" — whether tiny-rocm-nn's `FullyFusedMLP` produces correct results on RDNA/wave32 hardware. Integrate into `nerfstudio-rocm` reflecting whatever is actually found.

**Architecture:** `bjoernellens1/tiny-rocm-nn` (fork of `ZJLi2013/tiny-rocm-nn`) gets two small fixes ported (a hipBLAS API-version fix already proven by `kodai731`'s fork, plus the same `PYTORCH_ROCM_ARCH`-hardcoding bug class fixed twice already this project). The PyTorch bindings (`bindings/torch/`) build against real gfx1151 hardware. A standalone numerical comparison script (not a full training run) directly tests wave32 correctness by comparing tcnn's outputs against Nerfstudio's own pure-PyTorch reference implementation. Only if that comes back clean does an `ns-train instant-ngp` integration test and a "hard dependency" Dockerfile step follow — otherwise the sub-project's bounded, legitimate outcome is documenting the specific failure and leaving the Dockerfile step best-effort.

**Tech Stack:** C++/CUDA-via-HIP (hipify'd), CMake, PyTorch C++ extension (`bindings/torch/setup.py`), Python/pytest, Docker/podman, ROCm 7.2.2, gfx1151 (Radeon 8060S).

## Global Constraints

- Never mention Claude/AI/assistant/agent authorship in any commit message or GitHub comment, in any repo — zero exceptions, established and enforced throughout this project.
- All ROCm build/test/verification work happens inside the project's ROCm container with real GPU devices attached (`--device=/dev/kfd --device=/dev/dri --group-add video`) — this project is Docker-first; the host has CUDA-only PyTorch and no usable GPU for this work.
- Every claim of "it works" needs empirical, in-container verification with real hardware — this project has repeatedly found silent failures (arch hardcoding, pin conflicts, dist-name mismatches) that only surfaced under actual verification, not code inspection alone.
- Base branch for `bjoernellens1/tiny-rocm-nn` work: create a new branch off `main` (the fork's current tip, one commit ahead of `upstream/main`).
- Do NOT attempt to write new RDNA-specific kernels (warp-shuffle rewrites, wave32-aware reductions) if Task 3 finds a correctness problem — document the failure and stop; that becomes separate, explicitly-scoped future work.
- `docker/Dockerfile.rocm`'s tiny-rocm-nn step becomes a hard (non-fallback) dependency ONLY if Tasks 3 and 4 both come out clean. Otherwise it stays best-effort with an accurate, specific comment.

---

### Task 1: Port the hipBLAS fix and the arch-hardcoding fix

**Files:**
- Modify (in `/home/bjoern/git/tiny-rocm-nn`): `include/tiny-cuda-nn/cublas_matmul.h`
- Modify (in `/home/bjoern/git/tiny-rocm-nn`): `bindings/torch/setup.py`

**Interfaces:**
- Consumes: `kodai731/tiny-rocm-nn`'s `rocm-gfx1100` branch (external reference, add as a git remote to inspect/cherry-pick from).
- Produces: a `tiny-rocm-nn` branch (off `main`) that compiles under ROCm 7.2's hipBLAS headers, with `PYTORCH_ROCM_ARCH` respected during PyTorch-binding builds — later tasks build on this branch.

- [ ] **Step 1: Add the reference remote and inspect the fix**

```bash
cd /home/bjoern/git/tiny-rocm-nn
git remote add kodai731 https://github.com/kodai731/tiny-rocm-nn.git
git fetch kodai731
git log kodai731/rocm-gfx1100 -1 -p -- include/tiny-cuda-nn/cublas_matmul.h
```

Read the full patch. Confirm it only touches `hipblasDatatype_t`/`hipblasGemmEx`-related type names and call sites (replacing deprecated ROCm-7.x types with `hipDataType`/`hipblasComputeType_t`), and does **not** hardcode `gfx1100` or any other arch string anywhere in the diff. If it does, note the discrepancy and hand-port only the type-compatibility parts.

- [ ] **Step 2: Create the working branch and apply the fix**

```bash
cd /home/bjoern/git/tiny-rocm-nn
git checkout main
git checkout -b rocm-gfx1151-validation
git cherry-pick kodai731/rocm-gfx1100
```

If the cherry-pick applies cleanly, keep it (but re-verify the resulting diff still matches what you read in Step 1 — a clean cherry-pick can still carry unwanted content if `kodai731`'s branch has diverged elsewhere). If it conflicts, resolve by hand-porting just the `include/tiny-cuda-nn/cublas_matmul.h` hunk, keeping everything else from our `main`.

- [ ] **Step 3: Verify no other hipBLAS/hipblasDatatype_t occurrences were missed**

```bash
grep -rn "hipblasDatatype_t" --include="*.h" --include="*.cu" --include="*.cpp" .
```

Expected: no matches (all occurrences fixed). If any remain, apply the same type-name fix pattern from Step 1's patch to them too, in the same commit.

- [ ] **Step 4: Fix the arch-hardcoding bug in `bindings/torch/setup.py`**

Current state (confirmed during spec research):
```python
rocm_arch = os.environ.get("PYTORCH_ROCM_ARCH", "gfx942")
```

This is the third occurrence of this exact bug class in this project (after gsplat#2 and nerfacc's `setup.py`). Fix using the same shape as those two fixes: keep reading `PYTORCH_ROCM_ARCH` first (already correct), but verify the code path that consumes `rocm_arch` afterward doesn't silently ignore it or re-hardcode `gfx942` elsewhere in the same file — read the full file (`cat bindings/torch/setup.py`) before concluding this one `os.environ.get(...)` line is the only fix needed; if the variable is used correctly everywhere after this line, no further code change is needed here beyond confirming the default fallback value (`"gfx942"`, used only when the env var is genuinely unset) is intentional and matches the existing project convention (it does — same fallback gsplat/nerfacc use). If you find the value is NOT actually threaded through to the compiler flags correctly (e.g. a different hardcoded flag downstream), fix that too, in this same commit.

- [ ] **Step 5: Commit**

```bash
cd /home/bjoern/git/tiny-rocm-nn
git add include/tiny-cuda-nn/cublas_matmul.h bindings/torch/setup.py
git commit -m "$(cat <<'EOF'
Fix hipBLAS API-version compatibility and verify PYTORCH_ROCM_ARCH handling

ROCm 7.x deprecated hipblasDatatype_t in favor of hipDataType and
hipblasComputeType_t, breaking the build under recent ROCm. Ports the
type-compatibility fix (same shape as kodai731/tiny-rocm-nn's
rocm-gfx1100 branch, verified against our own ROCm 7.2 target rather
than assumed compatible).
EOF
)"
git push origin rocm-gfx1151-validation
```

(Push to `bjoernellens1/tiny-rocm-nn`'s `origin` — this is a separate repo from `nerfstudio-rocm`, pushing here is not the same as pushing the main `nerfstudio-rocm` worktree branch.)

---

### Task 2: Build the PyTorch bindings on gfx1151

**Files:**
- No new files — this task builds and verifies `bindings/torch/` from Task 1's branch inside the ROCm container.

**Interfaces:**
- Consumes: `bjoernellens1/tiny-rocm-nn`'s `rocm-gfx1151-validation` branch (Task 1's output).
- Produces: a verified-buildable wheel/install and a confirmed-correct-arch compiled artifact, for Task 3 to test against.

- [ ] **Step 1: Build inside the ROCm container with real GPU devices**

Reuse the existing ROCm image built during the nerfacc sub-project if still present (`podman images | grep nerfstudio-rocm`), otherwise build one from `docker/Dockerfile.rocm` in the `nerfstudio-rocm` worktree first.

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --entrypoint bash <image> -c '
git clone --branch rocm-gfx1151-validation https://github.com/bjoernellens1/tiny-rocm-nn.git /tmp/tiny-rocm-nn
cd /tmp/tiny-rocm-nn/bindings/torch
PYTORCH_ROCM_ARCH=gfx1151 CUDA_HOME=/opt/rocm python setup.py bdist_wheel 2>&1 | tee /tmp/build.log
'
```

(Adjust the exact build invocation if `setup.py bdist_wheel` isn't the right entry point — check `bindings/torch/setup.py`'s `if __name__ == "__main__"` block or any `README`/`DOCUMENTATION.md` build instructions in the repo first; use whatever mechanism the project actually documents/expects.)

- [ ] **Step 2: Triage any new errors after the hipBLAS one clears**

Expect the build to progress further than before but possibly hit new errors — different HIP/ROCm-7.2 API mismatches, missing headers, etc. For each: classify as (a) another instance of a known bug pattern from this project (hardcoded arch, deprecated API, glm/hipify-style header issue) and fix directly, (b) a genuine new ROCm compatibility issue requiring investigation, or (c) an environment/sandbox issue unrelated to the code (like the earlier `podman build` DNS quirk found in nerfacc's Task 5) — document and work around, no code change needed. Do not spend more than a reasonable, bounded effort chasing environment issues; if truly blocked by something outside the code, document it clearly as BLOCKED with specifics and let the controller decide whether to unblock it (e.g. host-level fix) or adjust scope.

- [ ] **Step 3: Verify the compiled artifact targets gfx1151, not gfx942**

Same technique as nerfacc's Task 5:
```bash
podman run --rm <built-image-or-installed-wheel-context> sh -c \
  "grep -ao 'gfx[0-9a-z]*' <path-to-compiled-.so> | sort -u"
```
Expected: only `gfx1151` appears, confirming `PYTORCH_ROCM_ARCH` was honored, not silently defaulting to `gfx942`.

- [ ] **Step 4: Verify `import tinycudann` succeeds**

```bash
podman run --rm <image-with-wheel-installed> python -c \
  "import tinycudann; print('import ok')"
```

Also verify via `python -c "from nerfstudio.utils.external import TCNN_EXISTS; print(TCNN_EXISTS)"` inside a container that also has `nerfstudio-rocm` installed (the nerfacc sub-project's built images already do) — this confirms Nerfstudio's own detection logic picks it up, not just a bare `import tinycudann`.

---

### Task 3: Wave32/RDNA correctness verification (the core open question)

**Files:**
- Create: `tests/rocm/tcnn_wave32_correctness.py` (in `nerfstudio-rocm`) — standalone numerical comparison script, not a pytest file (no CI runner has GPU access yet, matching this project's existing `tests/rocm/` convention of manually-run verification scripts).

**Interfaces:**
- Consumes: Task 2's built/installed `tinycudann`, plus Nerfstudio's own pure-PyTorch reference implementations (`nerfstudio/field_components/encodings.py`'s `HashEncoding`/`SHEncoding` torch paths, `nerfstudio/field_components/mlp.py`'s torch `MLP`).
- Produces: a documented, direct-evidence answer to "does tiny-rocm-nn produce numerically correct results on gfx1151" — this is what Task 4 and Task 5's `ROCM_RDNA_CAPS.verified_on_this_arch` value depend on.

- [ ] **Step 1: Run tiny-rocm-nn's own test suite if runnable**

Check `/home/bjoern/git/tiny-rocm-nn/test/` — it's CMake/C++ tests with their own hardcoded `--offload-arch=gfx942` in `test/CMakeLists.txt` (confirmed during spec research, out of this plan's required scope to fix). Attempt to build and run them with `-DCMAKE_HIP_ARCHITECTURES=gfx1151` overriding the hardcoded flag if CMake allows a command-line override; if it doesn't cleanly override (the hardcoded `target_compile_options` calls may need editing to test), use your judgment: either make the minimal edit needed to point at gfx1151 for this verification run only (not necessarily committing it, since it's out of the plan's stated scope — note the tradeoff in your report), or skip this step and rely on Step 2's PyTorch-level check as sufficient coverage. Document which choice you made and why.

- [ ] **Step 2: Write the numerical comparison script**

Create `tests/rocm/tcnn_wave32_correctness.py` in the `nerfstudio-rocm` worktree. The script must:

```python
"""Compares tinycudann (tcnn) outputs against Nerfstudio's pure-PyTorch
reference implementation for the same encoding/MLP, to directly test
whether tcnn's wave32 (RDNA) kernels produce numerically correct results —
not just whether training runs without crashing.
"""
import torch

from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.utils.external import TCNN_EXISTS

assert TCNN_EXISTS, "tinycudann must be importable for this check to be meaningful"

device = torch.device("cuda")
torch.manual_seed(0)

# --- HashEncoding: tcnn vs torch ---
in_dim = 3
num_levels = 16
features_per_level = 2
enc_tcnn = HashEncoding(
    num_levels=num_levels, features_per_level=features_per_level,
    implementation="tcnn",
).to(device)
enc_torch = HashEncoding(
    num_levels=num_levels, features_per_level=features_per_level,
    implementation="torch",
).to(device)
# Copy tcnn's parameters into the torch version so we're comparing the
# same learned/initialized weights, not different random inits.
# (Inspect HashEncoding's tcnn_encoding vs torch-path parameter names/shapes
# directly — nerfstudio/field_components/encodings.py — and copy correctly;
# tcnn stores params half-precision/differently-shaped than the torch path,
# so this copy must be done carefully. If a clean 1:1 parameter copy isn't
# feasible given tcnn's internal layout, fall back to comparing statistical
# properties of the OUTPUT distribution over many random inputs instead of
# exact per-element matching — document which approach was used and why.)

x = torch.rand(4096, in_dim, device=device)
out_tcnn = enc_tcnn(x)
out_torch = enc_torch(x)

# Document and justify the tolerance used — tcnn's internals are commonly
# half-precision, so exact equality is not the right bar.
print("HashEncoding forward max abs diff:", (out_tcnn.float() - out_torch.float()).abs().max().item())

# Repeat the same forward+backward comparison pattern for:
# - SHEncoding (implementation="tcnn" vs "torch")
# - MLP (FullyFusedMLP path vs torch path) — this is the component most
#   likely to show a wave32 problem, per the spec's risk analysis; prioritize
#   getting this comparison right even if the encoding comparisons above end
#   up using the statistical-property fallback.
# For each: also run .backward() on a scalar reduction of the output and
# compare input .grad — a wave32 bug could plausibly corrupt gradients
# without corrupting the forward pass, or vice versa; both need checking.
```

Fill in the full script for all three components (`HashEncoding`, `SHEncoding`, `MLP`/`FullyFusedMLP`), both forward and backward comparisons, with a printed pass/fail verdict per component against a documented, justified tolerance. Read `nerfstudio/field_components/encodings.py` and `nerfstudio/field_components/mlp.py` in full before writing the comparisons — the exact constructor arguments and parameter-copying feasibility must be verified against the real code, not guessed.

- [ ] **Step 3: Run the script inside the container on real gfx1151 hardware**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v <worktree>:/workspace/nerfstudio-rocm <image-with-tinycudann-installed> \
  python /workspace/nerfstudio-rocm/tests/rocm/tcnn_wave32_correctness.py 2>&1 | tee tests/rocm/logs/tcnn-wave32-correctness.log
```

- [ ] **Step 4: Document the outcome honestly**

Write the verdict (per-component pass/fail, with the actual numbers) into the task report. If any component fails the correctness check: this is the sub-project's answer — do NOT attempt to fix the underlying kernel in this task or plan. Document the specific failure mode (which component, forward or backward, magnitude of the discrepancy) precisely enough that a future, separately-scoped effort could act on it.

---

### Task 4: Instant-NGP/Nerfacto integration test (conditional on Task 3)

**Files:**
- No new files — a training run + log, same pattern as nerfacc's Task 4.

**Interfaces:**
- Consumes: Task 3's correctness verdict — this task's exact shape depends on that outcome.
- Produces: an end-to-end training validation log for `tests/rocm/logs/`.

- [ ] **Step 1: Decide the test's scope based on Task 3's outcome**

If Task 3 found tiny-rocm-nn numerically correct across `HashEncoding`, `SHEncoding`, and `MLP`/`FullyFusedMLP`: proceed to Step 2 as written (full tcnn-active training run).

If Task 3 found a correctness problem in `FullyFusedMLP` specifically but the encodings (`HashEncoding`/`SHEncoding`) passed: adapt this task to test only the encoding path with tcnn active and the MLP forced to `implementation="torch"` (check whether `NerfactoField`'s constructor allows mixing implementations per-component, or whether a small, temporary local change to force this combination is needed for the test only — do not commit a permanent Nerfstudio code change for this without flagging it to the controller first, since `nerfstudio-rocm` aims to keep its diff from upstream small).

If Task 3 found the encodings themselves broken: this task's goal cannot be meaningfully attempted — report BLOCKED-BY-DESIGN (not a subagent failure) with a one-line explanation, and let Task 5 proceed straight to documenting the negative outcome.

- [ ] **Step 2: Run the training command**

Same dataset/command shape as nerfacc's Task 4 (reuse for direct comparability):

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360:ro \
  -v <nerfstudio-rocm worktree>:/workspace/nerfstudio-rocm \
  --entrypoint bash <image-with-tinycudann-installed> -c '
cd /workspace/nerfstudio-rocm
pip install --no-build-isolation -e . -q
ns-train instant-ngp \
  --data /data/mipnerf360/bonsai \
  --max-num-iterations <N, pick based on a trial run same as nerfacc Task 4> \
  --viewer.quit-on-train-completion True --vis tensorboard \
  colmap --images-path images --colmap-path sparse/0 --downscale-factor 4 --downscale-rounding-mode round
' 2>&1 | tee tests/rocm/logs/tcnn-instant-ngp.log
```

- [ ] **Step 3: Verify tcnn was genuinely active, not silently bypassed**

```bash
grep -i "tcnn\|speed warning\|fallback" tests/rocm/logs/tcnn-instant-ngp.log
```

Confirm no `print_tcnn_speed_warning` messages appear (which would mean it silently fell back to torch despite `tinycudann` being installed — the same trap nerfacc's Task 4 accidentally fell into, this task must NOT repeat that mistake unnoticed).

- [ ] **Step 4: Verify training health**

Same rigor as nerfacc's Task 4: "Training Finished" banner, no NaN/Inf across TensorBoard scalar series, loss/PSNR trend comparable to or better than nerfacc's torch-fallback baseline run (train loss 0.114→0.0044, train PSNR 9.6→23.4dB over 5000 iterations, for reference).

---

### Task 5: Integrate into nerfstudio-rocm, reflecting the real outcome

**Files:**
- Modify: `dependencies/rocm-lock.toml`
- Modify: `ROCM.md`
- Modify: `rocm/capabilities.py`
- Modify: `docker/Dockerfile.rocm`
- Modify: `pyproject.toml` (only if a pin-conflict risk is found — see Step 4 below)

**Interfaces:**
- Consumes: Tasks 1-4's fixes, build, and verification results.
- Produces: `nerfstudio-rocm`'s documented, accurate state for tiny-rocm-nn — whichever way the verification went.

- [ ] **Step 1: Update `dependencies/rocm-lock.toml`**

```toml
[tiny_rocm_nn]
repo = "https://github.com/bjoernellens1/tiny-rocm-nn"
branch = "rocm-gfx1151-validation"
commit = "<paste final HEAD commit hash from Task 1>"
status = "<accurate summary of Tasks 2-4's real findings — builds on gfx1151: yes/no; wave32 correctness: pass/fail per component with specifics; Instant-NGP/Nerfacto integration: verified/not-applicable>"
```

- [ ] **Step 2: Update `ROCM.md`'s "tiny-rocm-nn: what's actually there" section**

Rewrite with the real, direct-evidence outcome — including if the outcome is negative (e.g. "FullyFusedMLP produces numerically incorrect gradients on RDNA/wave32, verified via `tests/rocm/tcnn_wave32_correctness.py`; do not use `implementation=\"tcnn\"` for the MLP on this hardware until fixed"). Follow the existing section's style (concrete, direct-evidence-cited, no aspirational language) — see the current text as the template for tone/detail level (it already does this well for `SphericalHarmonics`'s "the plan's assumption was stale" correction; extend the same honesty here).

- [ ] **Step 3: Update `rocm/capabilities.py`**

Set `ROCM_RDNA_CAPS.verified_on_this_arch` to match Task 3's real finding (`True` only if ALL of `HashGrid`/`SphericalHarmonics`/`FullyFusedMLP` passed the correctness check; `False` otherwise, with any per-component fields the dataclass already supports updated individually if it supports finer granularity — check the actual dataclass fields in `rocm/capabilities.py` before writing this, don't assume a single boolean is the only field to touch).

- [ ] **Step 4: Update `docker/Dockerfile.rocm`**

If Tasks 3-4 are fully clean: change tiny-rocm-nn's step from best-effort to hard, matching gsplat/nerfacc's pattern, pointing at the `rocm-gfx1151-validation` branch. Also check `pyproject.toml` for a `tinycudann==` pin causing the same distribution-name-vs-import-name silent-overwrite risk gsplat/nerfacc had — verify empirically inside a real build (`import tinycudann` version check after a full image build), fix if it reproduces, following the exact same pattern as those two fixes.

If Tasks 3-4 found a correctness problem: leave the step best-effort, but rewrite its comment to state the specific, verified reason (not generic "not yet ported" text) — e.g. reference `ROCM.md`'s new section directly.

- [ ] **Step 5: Commit**

```bash
cd <nerfstudio-rocm worktree>
git add dependencies/rocm-lock.toml ROCM.md rocm/capabilities.py docker/Dockerfile.rocm tests/rocm/tcnn_wave32_correctness.py tests/rocm/logs/.gitkeep 2>/dev/null
git add pyproject.toml  # only if Step 4 touched it
git commit -m "$(cat <<'EOF'
Integrate tiny-rocm-nn RDNA/gfx1151 findings: <one-line real outcome>

<2-3 sentences summarizing what was verified and the actual result,
written to match whichever outcome Tasks 3-4 actually found>
EOF
)"
```

(Do NOT push to `main` directly — this happens on a worktree feature branch; pushing/merging follows the SDD workflow's finishing step, outside this task.)

## Global Constraints (repeated for task-level visibility)

- Never mention Claude/AI/assistant/agent authorship in any commit message, in either `tiny-rocm-nn` or `nerfstudio-rocm`.
- No open-ended RDNA kernel rewriting if Task 3 finds a correctness problem — document and stop.
- Every "it works" claim needs real in-container, real-GPU verification.
