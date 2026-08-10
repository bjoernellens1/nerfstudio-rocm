# gsplat validation + Splatfacto training on gfx1151 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get gsplat's own test suite green on real gfx1151 hardware, then train Splatfacto to a 7000-iteration checkpoint on a real multi-view scene, and make both a hard (non-fallback) part of `nerfstudio-rocm`.

**Architecture:** All work happens inside the existing `nerfstudio-rocm` ROCm Docker image (`docker/Dockerfile.rocm`, built from public `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`), run with real GPU devices attached (`--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host`). `bjoernellens1/gsplat` (`release/1.5.3b2`) is installed from source inside the running container, not baked into the image at build time, until Task 5 makes it a hard build-time step once proven reliable.

**Tech Stack:** Podman (this machine uses podman, not docker — `docker` and `podman` commands are aliased/compatible here), PyTorch 2.10.0+rocm7.2.2, gsplat (HIP/hipcc), Nerfstudio, pytest.

## Global Constraints

- Target architecture: gfx1151 (Radeon 8060S / Strix Halo) — this machine's actual GPU, verified via `rocminfo` and `torch.cuda.get_device_properties(0).gcnArchName`.
- gsplat pinned branch: `release/1.5.3b2`, repo `https://github.com/bjoernellens1/gsplat`.
- Dataset: `/home/bjoern/Downloads/mipnerf360_v2_dataset/bonsai` (already has COLMAP `sparse/0/{cameras,images,points3D}.bin`).
- Any fix to gsplat's build system goes into the `bjoernellens1/gsplat` repo (separate git checkout at `/home/bjoern/git/gsplat`) as a real commit — not worked around only in `nerfstudio-rocm`.
- `_test_distributed.py` (gsplat's multi-GPU test file) is explicitly excluded from all test runs in this plan.
- Do not modify `dependencies/rocm-lock.toml`'s `nerfacc` or `tiny_rocm_nn` sections — those stay unported, out of scope.

---

### Task 1: Diagnose current gsplat build state with real GPU devices attached

**Files:**
- None modified. This task produces a diagnostic report only (written to `/tmp/gsplat-diag-report.md` for the next task to reference, not committed).

**Interfaces:**
- Consumes: existing `localhost/nerfstudio-rocm:phase1-pillowfix` image (already built, has nerfstudio + Pillow<12 fix, no gsplat baked in) OR rebuild fresh from current `docker/Dockerfile.rocm` if that image no longer exists locally.
- Produces: a diagnostic report answering (a) does `get_rocm_arch()` correctly detect gfx1151 at runtime, (b) does the glm include-path bug (gsplat#3) still reproduce now that `libglm-dev` is no longer installed in the image.

- [ ] **Step 1: Confirm the base image exists, rebuild if not**

Run: `podman images | grep nerfstudio-rocm`

If `localhost/nerfstudio-rocm:phase1-pillowfix` is not listed, rebuild it:
```bash
cd /home/bjoern/git/nerfstudio-rocm
podman build -f docker/Dockerfile.rocm -t nerfstudio-rocm:phase1 --build-arg PYTORCH_ROCM_ARCH=gfx1151 .
CID=$(podman run -d --entrypoint sleep localhost/nerfstudio-rocm:phase1 300)
podman exec $CID pip install -q "Pillow<12"
podman commit $CID localhost/nerfstudio-rocm:phase1-pillowfix
podman stop $CID
```

- [ ] **Step 2: Run gsplat's install step manually inside a live container with GPU devices attached, verbose**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
set -x
git clone --branch release/1.5.3b2 https://github.com/bjoernellens1/gsplat.git /tmp/gsplat
cd /tmp/gsplat
git submodule update --init --recursive
rocminfo | grep -m1 -oE "gfx[0-9a-fA-F]+"
CUDA_HOME=/opt/rocm python -m pip install -v --no-build-isolation . 2>&1 | tee /tmp/gsplat-build.log
' 2>&1 | tee /tmp/gsplat-diag-report.md
```

Expected: this either succeeds (gsplat installs cleanly) or fails with a specific hipcc compile error. Either outcome is a valid result for this diagnostic task — do not attempt fixes yet.

- [ ] **Step 3: Extract the actual hipcc invocation and check for the glm include flag**

```bash
grep -n "third_party/glm\|hipcc.*IntersectTile\|error:" /tmp/gsplat-diag-report.md | head -40
```

Record in `/tmp/gsplat-diag-report.md` (append manually or via `echo >>`) one of:
- "RESOLVED: gsplat installed successfully with no libglm-dev present, arch correctly detected as gfx1151" — skip Task 2 entirely, proceed to Task 3.
- "STILL FAILING: <paste the specific error and whether -I.../third_party/glm is present in the failing hipcc command>" — proceed to Task 2.

- [ ] **Step 4: Verify the import works if build succeeded**

If Step 2 succeeded, confirm the extension actually loads:
```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
pip install --no-build-isolation /tmp/gsplat 2>&1 | tail -5 || true
python -c "from gsplat.cuda._backend import _C; print(_C)"
'
```
Expected: prints a module path, no ImportError. If this fails even though the pip install "succeeded," append that finding to the report and treat it as still-failing for Task 2 purposes.

---

### Task 2: Fix the gsplat HIP build (only if Task 1 found it still failing)

**Skip this task entirely if Task 1's Step 3 concluded "RESOLVED."**

**Files:**
- Modify: `/home/bjoern/git/gsplat/setup.py` (local checkout of `bjoernellens1/gsplat`, branch `release/1.5.3b2`)

**Interfaces:**
- Consumes: the specific failing hipcc command and `include_dirs` construction logic from `setup.py` (around the `include_dirs = [...]` list and `CUDAExtension(...)` call found in Task 1).
- Produces: a gsplat commit that makes `CUDA_HOME=/opt/rocm python -m pip install --no-build-isolation .` succeed inside the container, pushed to `bjoernellens1/gsplat`.

This is a debugging task with an unknown root cause going in — use the **superpowers:systematic-debugging** skill's approach (reproduce reliably, isolate the minimal failing case, form a hypothesis, test it) rather than guessing at a fix.

- [ ] **Step 1: Reproduce minimally and dump the actual include_dirs setup.py computes**

```bash
cd /home/bjoern/git/gsplat
git checkout release/1.5.3b2
git pull origin release/1.5.3b2
```

Add a temporary debug print right after the `include_dirs = [...]` list in `setup.py` (around line 239-245, confirmed in Task 1's log context):
```python
    include_dirs = [
        osp.join(current_dir, "gsplat", "cuda", "include"),
        osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm"),
        f"{os.environ['HOME']}/.local/include",
        f"/opt/conda/include",
        f"/opt/conda/envs/py_3.12/lib/python3.12/site-packages/",
        f"/opt/rocm/include",
    ]
    print(f"DEBUG include_dirs = {include_dirs}", flush=True)
    import os as _os
    print(f"DEBUG glm exists = {_os.path.isdir(include_dirs[1])}", flush=True)
```

- [ ] **Step 2: Rebuild inside the container and capture the debug output**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/git/gsplat:/tmp/gsplat-debug:ro \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cp -r /tmp/gsplat-debug /tmp/gsplat
cd /tmp/gsplat
git submodule update --init --recursive
CUDA_HOME=/opt/rocm python -m pip install -v --no-build-isolation . 2>&1 | grep -A2 "DEBUG\|hipcc.*IntersectTile"
'
```

Compare: does `DEBUG include_dirs` list the glm path, does `DEBUG glm exists` say `True`, and does the *actual hipcc command line* later in the log contain `-I` followed by that same path? If include_dirs is correct and glm exists but the hipcc command doesn't have the `-I` flag, the bug is in how `torch.utils.cpp_extension`'s HIP compilation path drops custom include_dirs for hipified `.hip` sources — this is a known category of issue with older `hipify`/`cpp_extension` versions not forwarding `include_dirs` to the per-source hipcc invocation for `.hip`-suffixed files (as opposed to `.cu`).

- [ ] **Step 3: Apply the fix based on Step 2's finding**

If the include_dirs value is correct but not reaching hipcc, the fix is to also pass the glm path via `extra_compile_args["nvcc"]` (which becomes the HIP compile flags per the surrounding code) as an explicit `-I` flag, since that list *does* reach the hipcc invocation (confirmed in Task 1's log — the `hipcc_flags` list assembled earlier in `setup.py` is what appeared in the failing command). Find the `hipcc_flags` list construction (near `extra_compile_args["nvcc"] = hipcc_flags`) and add the include path directly there:

```python
        glm_include_path = str(osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm"))
        hipcc_flags = [
            "-D__HIP_PLATFORM_AMD__",
            "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
            "-DUSE_ROCM",
            f"-I{glm_include_path}",
            f"--offload-arch={gpu_arch}",
        ]
```

This snippet is illustrative of *what* to add (the `glm_include_path` line and the `f"-I{glm_include_path}"` entry) — merge those two additions into whatever the existing `hipcc_flags` list already contains at that point in the file (found via Task 1's log / Step 1's debug output), rather than replacing the whole list wholesale. Inspect the real surrounding code in `setup.py` before editing since its exact current contents must be preserved.

- [ ] **Step 4: Remove the debug prints, rebuild, verify success**

Remove the two `print(...)` lines and the `import os as _os` line added in Step 1. Rebuild using the same command as Task 1 Step 2. Expected: `CUDA_HOME=/opt/rocm python -m pip install --no-build-isolation .` completes with no errors, and:
```bash
python -c "from gsplat.cuda._backend import _C; print(_C)"
```
prints a module path.

- [ ] **Step 5: Commit and push the fix to bjoernellens1/gsplat**

```bash
cd /home/bjoern/git/gsplat
git add setup.py
git commit -m "fix: pass glm include path directly via hipcc flags (fixes #3)

CUDAExtension's include_dirs wasn't reaching the per-source hipcc
invocation for hipified .hip sources, so the compiler fell back to
whatever glm was findable on the default include path instead of the
vendored, HIP-compatible submodule. Passing the include path via the
hipcc_flags list (which does reach the actual compile command) fixes it."
git push origin release/1.5.3b2
```

- [ ] **Step 6: Close the GitHub issue**

```bash
gh issue close 3 --repo bjoernellens1/gsplat --comment "Fixed in $(git rev-parse --short HEAD): the glm include path wasn't reaching hipcc's per-source invocation for hipified .hip sources via CUDAExtension's include_dirs. Passing it directly through the hipcc_flags list resolves it — verified building and importing cleanly on gfx1151."
```

---

### Task 2b: Port the wave32 fix onto release/1.5.3b2 (inserted mid-execution)

> **CORRECTION — what was actually applied (post-execution note).** The rest of
> this section describes the plan *as written before execution*; the source it
> names was rejected during execution. Read this note first.
>
> - **Rejected source:** `origin/fix/gfx1151-wave32-review-cleanup` (the 4-file
>   diff described below) — **it does not compile**: it deletes the definition
>   of `dpp_warpMax` while leaving its call site intact.
> - **Actual source:** [AMD-Ecosystem/gsplat PR #17](https://github.com/AMD-Ecosystem/gsplat/pull/17),
>   commit `20f38d5`, authored by Warren Ross — ported onto
>   `bjoernellens1/gsplat` `release/1.5.3b2` as commit `9b6d7db`
>   ("use a single WARP_SIZE constant (arch-gated) instead of hardcoded wave64
>   in reduction/tiling kernels", fixes gsplat#4).
> - **Actual scope: 9 files**, not 4 — the port also covers the projection
>   kernels and adds a new shared header:
>   1. `gsplat/cuda/include/Common.cuh` *(new file — the arch-gated
>      `WARP_SIZE` constant)*
>   2. `gsplat/cuda/csrc/Projection2DGSFused.cu`
>   3. `gsplat/cuda/csrc/Projection2DGSPacked.cu`
>   4. `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`
>   5. `gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu`
>   6. `gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu` *(planned)*
>   7. `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` *(planned)*
>   8. `gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu` *(planned)*
>   9. `gsplat/cuda/include/Utils.cuh` *(planned)*
>
> The "Do NOT modify `setup.py` / `_backend.py`" constraint below still held —
> Task 2's glm fix (`42d17c9`) was preserved untouched.

**Why this task exists:** Task 2 fixed gsplat#3 (glm include path) but, in
verifying it, discovered gsplat still fails to build at all on gfx1151: a
new bug, filed as
[gsplat#4](https://github.com/bjoernellens1/gsplat/issues/4) — the backward
rasterization kernels hardcode a rocprim logical warp size of 64
(`rocprim_warpSum<CDIM, 64>`, `rocprim::warp_reduce<float,64>`,
`cg::tiled_partition<64>`), but gfx1151 (RDNA3.5) has a native wavefront
size of 32, so `rocprim::check_virtual_wave_size`'s `static_assert` fails at
compile time. This isn't confined to `RasterizeToPixels2DGSBwd.cu` (the file
Task 2's diagnostic build happened to reach first) — the same `<64>`
hardcoding exists in `RasterizeToPixels3DGSBwd.cu` (the kernel Splatfacto's
default 3DGS mode actually needs for Task 4) and
`RasterizeToPixelsFromWorld3DGSBwd.cu`. Confirmed via:
```bash
grep -rn "rocprim_warpSum<.*64\|warp_reduce<float,64\|tiled_partition<64" \
  gsplat/cuda/csrc/*.cu gsplat/cuda/include/*.cuh
```

A candidate fix already exists on an unmerged branch,
`origin/fix/gfx1151-wave32-review-cleanup`, which touches exactly these
files with a wave32-aware version. It also bundles ~700 unrelated lines
(CI workflow, `docker/Dockerfile.rocm-gfx11`, `docker/patch_glm_platform_h.py`,
docs, examples, and its own different — and now superseded by Task 2's
verified fix — approach to the glm problem in `setup.py`/`gsplat/cuda/_backend.py`).
Confirmed via:
```bash
git -C /home/bjoern/git/gsplat diff --stat \
  b01acd43e3c7fa942f95fda0974e9125e4de7395..origin/fix/gfx1151-wave32-review-cleanup
```
which lists 14 changed files; only 4 are the actual kernel/wave-size fix:
`gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu`,
`gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`,
`gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu`,
`gsplat/cuda/include/Utils.cuh`.

**Files:**
- Modify: `/home/bjoern/git/gsplat/gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu`
- Modify: `/home/bjoern/git/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`
- Modify: `/home/bjoern/git/gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu`
- Modify: `/home/bjoern/git/gsplat/gsplat/cuda/include/Utils.cuh`
- Do NOT modify `setup.py` or `gsplat/cuda/_backend.py` in this task — Task 2's
  glm fix in `setup.py` is already verified and must not be overwritten or
  reintroduced-and-conflicted-with by the branch's different approach there.

**Interfaces:**
- Consumes: Task 2's commit `42d17c9` on `release/1.5.3b2` (glm fix, already
  verified working for both compile paths).
- Produces: a gsplat commit on `release/1.5.3b2` where `CUDA_HOME=/opt/rocm
  python -m pip install --no-build-isolation .` succeeds all the way through
  (not just past the glm errors) and
  `python -c "from gsplat.cuda._backend import _C; print(_C)"` prints a
  module path — needed before Task 3 (test suite) or Task 4 (Splatfacto
  training) can proceed at all, since neither can run against a package that
  doesn't build.

- [ ] **Step 1: Extract only the wave32-relevant hunks from the candidate branch for the 4 files above**

```bash
cd /home/bjoern/git/gsplat
git diff b01acd43e3c7fa942f95fda0974e9125e4de7395..origin/fix/gfx1151-wave32-review-cleanup -- \
  gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu \
  gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu \
  gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu \
  gsplat/cuda/include/Utils.cuh \
  > /tmp/gsplat-wave32-fix.diff
```

Read the resulting diff in full before applying anything — confirm every
hunk is genuinely about wave32/warp-size (the `<64>` → `<32>`-style changes
already sampled during planning) and not something else incidentally
touching the same files. If any hunk looks unrelated, exclude it and note
why in your report.

- [ ] **Step 2: Apply the fix on top of Task 2's commit**

```bash
git checkout release/1.5.3b2
git pull origin release/1.5.3b2   # picks up Task 2's 42d17c9
git apply /tmp/gsplat-wave32-fix.diff
```

If `git apply` fails (context mismatch against Task 2's changes, since both
touch code near each other in spirit even if not the same lines), fall back
to manually applying each hunk's intent using the file structure at HEAD —
the underlying change is mechanical (every hardcoded `64` warp/wave-size
literal relevant to RDNA becomes `32`, or becomes conditional on
`__AMDGCN_WAVEFRONT_SIZE`/`warpSize` per the sampled hunks in this task's
"Why this task exists" section above), not something requiring new design.

- [ ] **Step 3: Build and verify with real GPU devices attached**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/git/gsplat:/tmp/gsplat-debug:ro \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cp -r /tmp/gsplat-debug /tmp/gsplat
cd /tmp/gsplat
git submodule update --init --recursive
CUDA_HOME=/opt/rocm python -m pip install -v --no-build-isolation . 2>&1 | tail -60
python -c "from gsplat.cuda._backend import _C; print(_C)"
'
```

Expected: build completes with no errors, import prints a module path (not
`None`, not "No CUDA toolkit found").

- [ ] **Step 4: Commit and push**

```bash
cd /home/bjoern/git/gsplat
git add gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu \
        gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu \
        gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu \
        gsplat/cuda/include/Utils.cuh
git commit -m "fix: use native wave32 warp/wavefront size on RDNA (gfx1151) in backward rasterization kernels (fixes #4)

Ports the kernel-only portion of the wave32 fix from
origin/fix/gfx1151-wave32-review-cleanup — rocprim_warpSum, warp_reduce,
and cg::tiled_partition were hardcoded to a logical warp size of 64
(matching CDNA/Instinct wavefronts), which fails rocprim's
check_virtual_wave_size static_assert on RDNA3.5's native 32-lane
wavefronts. Does not port that branch's setup.py/docker/CI changes —
this repo's glm fix (42d17c9) already covers the include-path problem
via a different, already-verified approach."
git push origin release/1.5.3b2
```

- [ ] **Step 5: Close the GitHub issue**

```bash
gh issue close 4 --repo bjoernellens1/gsplat --comment "Fixed in $(git -C /home/bjoern/git/gsplat rev-parse --short HEAD): ported the kernel-only wave32 fix (RasterizeToPixels2DGSBwd/3DGSBwd/FromWorld3DGSBwd.cu, Utils.cuh) from origin/fix/gfx1151-wave32-review-cleanup, isolated from that branch's unrelated CI/Docker/setup.py changes. Verified building and importing cleanly on gfx1151 with real GPU devices attached."
```

---

### Task 3: Run gsplat's test suite on gfx1151

**Files:**
- None modified (unless Task 3 uncovers a real bug requiring a fix — see Step 3's branch).

**Interfaces:**
- Consumes: working gsplat install from Task 1 or Task 2 (whichever concluded successfully) inside `localhost/nerfstudio-rocm:phase1-pillowfix`.
- Produces: a pass/fail report per test file, referenced by Task 5's `ROCM.md` update.

- [ ] **Step 1: Install gsplat's test dependencies and run the suite**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
git clone --branch release/1.5.3b2 https://github.com/bjoernellens1/gsplat.git /tmp/gsplat
cd /tmp/gsplat
git submodule update --init --recursive
CUDA_HOME=/opt/rocm python -m pip install --no-build-isolation .
python -m pip install pytest
pytest tests/ --ignore=tests/_test_distributed.py -v 2>&1 | tee /tmp/gsplat-test-results.log
' 2>&1 | tee /home/bjoern/git/gsplat-test-results-gfx1151.log
```

Expected: pytest runs and reports pass/fail counts per test file. Do not expect 100% pass on the first run — this is genuinely unverified territory per the design spec.

- [ ] **Step 2: Triage any failures**

For each failing test, read the failure output and classify it as one of:
- A genuine RDNA/gfx1151-specific numerical or kernel bug (matches the risk profile described in the design spec — wave32/wave64, atomics, cooperative groups).
- A test environment issue (missing test data, wrong tolerance for this hardware, etc.) unrelated to correctness.
- An already-known issue (cross-reference gsplat#2/#3 and anything found in Task 1/2).

Record findings by appending a summary section to `/home/bjoern/git/gsplat-test-results-gfx1151.log`.

- [ ] **Step 3: Fix or document each failure**

For test-environment issues: fix directly (e.g. adjust a fixture, skip with a documented reason). For genuine kernel bugs: this is real gsplat HIP kernel debugging — use **superpowers:systematic-debugging** for each one. If a fix isn't reasonably scoped to finish within this task (e.g. requires deep kernel rework), document it as a known limitation with a filed GitHub issue on `bjoernellens1/gsplat` instead of blocking the whole plan on it — Splatfacto training in Task 4 doesn't require every gsplat test to pass, only that the specific kernels Splatfacto actually exercises (projection, rasterization forward/backward, Adam) work correctly.

- [ ] **Step 4: Re-run the full suite and record final pass/fail counts**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/git/gsplat:/tmp/gsplat-src:ro \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cp -r /tmp/gsplat-src /tmp/gsplat && cd /tmp/gsplat
git submodule update --init --recursive
CUDA_HOME=/opt/rocm python -m pip install --no-build-isolation .
python -m pip install pytest
pytest tests/ --ignore=tests/_test_distributed.py -v
'
```

Save the final summary line (e.g. "142 passed, 3 skipped, 0 failed") for Task 5's documentation update.

---

### Task 4: Train Splatfacto on the bonsai scene to 7000 iterations

**Files:**
- Create: `tests/rocm/logs/splatfacto.log` (gitignored, produced by the training run — matches Phase 1's `tests/rocm/logs/` pattern)

**Interfaces:**
- Consumes: working gsplat (from Task 1/2/3), existing `nerfstudio-rocm` install inside `localhost/nerfstudio-rocm:phase1-pillowfix`, and the `bonsai` dataset at `/home/bjoern/Downloads/mipnerf360_v2_dataset/bonsai`.
- Produces: a completed Splatfacto training run with a saved checkpoint under `outputs/`, referenced by Task 5.

- [ ] **Step 1: Confirm the dataset structure matches what Nerfstudio's ColmapDataParser expects**

```bash
ls /home/bjoern/Downloads/mipnerf360_v2_dataset/bonsai/sparse/0/
ls /home/bjoern/Downloads/mipnerf360_v2_dataset/bonsai/images_4 | wc -l
```
Expected: `cameras.bin`, `images.bin`, `points3D.bin` present; `images_4` contains the same count as `images_2`/`images` (292, per the design spec's earlier finding).

- [ ] **Step 2: Do a very short trial run (100 iterations) first to catch config/path errors cheaply**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360:ro \
  -v /home/bjoern/git/nerfstudio-rocm:/workspace/nerfstudio-rocm \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cd /workspace/nerfstudio-rocm
pip install --no-build-isolation -e . -q
git clone --branch release/1.5.3b2 https://github.com/bjoernellens1/gsplat.git /tmp/gsplat 2>&1 | tail -2
cd /tmp/gsplat && git submodule update --init --recursive && CUDA_HOME=/opt/rocm python -m pip install --no-build-isolation . -q
cd /workspace/nerfstudio-rocm
ns-train splatfacto \
  --data /data/mipnerf360/bonsai \
  --max-num-iterations 100 \
  --viewer.quit-on-train-completion True --vis tensorboard \
  colmap --images-path images_4 --colmap-path sparse/0
' 2>&1 | tee /tmp/splatfacto-trial.log
```

Note the flag ordering: `colmap` (the dataparser subcommand) must be the last positional token, and *its own* flags (`--images-path`, `--colmap-path`) come after it, not before — this is the same subcommand-ordering rule established in Phase 1 (see `nerfstudio/data/dataparsers/colmap_dataparser.py:93-99` for the `ColmapDataParserConfig` fields: `images_path: Path = Path("images")` and `colmap_path: Path = Path("colmap/sparse/0")` — `bonsai`'s reconstruction is at `sparse/0`, not the default `colmap/sparse/0`, so `--colmap-path sparse/0` is required, not optional).

Expected: training starts, runs 100 iterations, ends with "Training Finished." If it errors on data loading or config, run `ns-train splatfacto colmap --help` inside the container to double check current flag names before retrying — do not spend the full 7000-iteration run debugging config issues.

- [ ] **Step 3: Run the full 7000-iteration training**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360:ro \
  -v /home/bjoern/git/nerfstudio-rocm:/workspace/nerfstudio-rocm \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cd /workspace/nerfstudio-rocm
pip install --no-build-isolation -e . -q
ns-train splatfacto \
  --data /data/mipnerf360/bonsai \
  --max-num-iterations 7000 \
  --viewer.quit-on-train-completion True --vis tensorboard \
  colmap --images-path images_4 --colmap-path sparse/0
' 2>&1 | tee /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/splatfacto.log
```

This will take a while — monitor progress rather than blocking synchronously (matches how Phase 1's long-running training was monitored). Check periodically:
```bash
grep -a "Step\|error\|Error\|nan\|NaN\|Training Finished" /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/splatfacto.log | tail -20
```

- [ ] **Step 4: Verify success criteria**

```bash
grep -a "Training Finished" /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/splatfacto.log
grep -a -i "nan\|inf" /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/splatfacto.log
```

Expected: "Training Finished" banner present, no NaN/Inf matches. Additionally confirm densification happened by checking the Gaussian count grew — Nerfstudio's Splatfacto logs a `Gaussian Count` metric during training; grep for it:
```bash
grep -a "Gaussian" /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/splatfacto.log | head -5
grep -a "Gaussian" /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/splatfacto.log | tail -5
```
The tail value should be larger than the head value (the model started from the sparse COLMAP point cloud and added Gaussians during training).

---

### Task 5: Make gsplat a hard build step and update documentation

**Files:**
- Modify: `docker/Dockerfile.rocm`
- Modify: `dependencies/rocm-lock.toml`
- Modify: `ROCM.md`

**Interfaces:**
- Consumes: the working gsplat commit (from Task 1 or Task 2) and gsplat test results (from Task 3).
- Produces: an updated `nerfstudio-rocm` where gsplat installs unconditionally at Docker build time.

- [ ] **Step 1: Change the gsplat Dockerfile step from best-effort to hard**

In `docker/Dockerfile.rocm`, replace:
```dockerfile
RUN git clone --branch release/1.5.3b2 https://github.com/bjoernellens1/gsplat.git && \
    cd gsplat && git submodule update --init --recursive && \
    (CUDA_HOME=/opt/rocm python -m pip install --no-cache-dir --no-build-isolation . || \
     echo "gsplat: build-time install failed (expected, see comment above) — run scripts/install-rocm-deps.sh at container runtime instead")
```
with:
```dockerfile
RUN git clone --branch release/1.5.3b2 https://github.com/bjoernellens1/gsplat.git && \
    cd gsplat && git submodule update --init --recursive && \
    CUDA_HOME=/opt/rocm python -m pip install --no-cache-dir --no-build-isolation .
```

Also remove or rewrite the comment block above it (the one explaining the gsplat#2/#3 workaround) to reflect the actual current state — if Task 2 ran, reference the fix commit; if Task 1 found it already resolved, say so plainly:
```dockerfile
# --- gsplat: verified ROCm fork, pinned commit in dependencies/rocm-lock.toml ---
# Builds cleanly with real GPU devices attached (docker build has none, which
# is why this must run at image-build time on a host that already has ROCm
# device access via BuildKit/podman's device passthrough, or be rebuilt from
# a running container — see scripts/install-rocm-deps.sh for the equivalent
# runtime install path used during development/debugging).
```

- [ ] **Step 2: Verify the image still builds successfully with the hard dependency**

```bash
cd /home/bjoern/git/nerfstudio-rocm
podman build -f docker/Dockerfile.rocm -t nerfstudio-rocm:phase2-gsplat --build-arg PYTORCH_ROCM_ARCH=gfx1151 .
```
Expected: build succeeds through the gsplat step without falling into the `||` fallback (there is no fallback anymore — a failure here should hard-fail the build, which is correct).

If this fails because `docker build` genuinely lacks GPU device access (the gsplat#2 root cause), this confirms gsplat's `setup.py` still needs the `PYTORCH_ROCM_ARCH` env var respected even without live device access (i.e. gsplat#2 needs an actual fix in `setup.py`, not just a runtime workaround) — apply the same style of fix as Task 2 but for `get_rocm_arch()`: make it check `os.environ.get("PYTORCH_ROCM_ARCH")` first, matching the pattern already present on gsplat's default branch (`docs/ecosystem-contrib-link`), before falling back to `rocminfo`. Push that fix to `bjoernellens1/gsplat` the same way as Task 2 Step 5, and close gsplat#2 the same way as Task 2 Step 6.

- [ ] **Step 3: Update dependencies/rocm-lock.toml**

Update the `[gsplat]` section's `commit` field to the new HEAD of `release/1.5.3b2` (after Task 2/5's fixes, if any were pushed) and rewrite `status`:
```bash
git -C /home/bjoern/git/gsplat rev-parse origin/release/1.5.3b2
```
Update `dependencies/rocm-lock.toml`:
```toml
[gsplat]
repo = "https://github.com/bjoernellens1/gsplat"
branch = "release/1.5.3b2"
commit = "<paste the commit hash from above>"
status = "verified on gfx1151: full pytest suite <N passed, M skipped, 0 failed>; Splatfacto trained 7000 iterations on bonsai (mip-nerf360) with no NaN/Inf and confirmed densification"
```

- [ ] **Step 4: Update ROCM.md**

In the fork-status table, change the gsplat row's status column from the placeholder text to reflect Task 3's actual test results and Task 4's training result. In the "Known build issues" section, remove the gsplat#2/#3 bullet entirely if both were fixed, or update it to reflect only what remains unresolved.

- [ ] **Step 5: Commit and push**

```bash
cd /home/bjoern/git/nerfstudio-rocm
git add docker/Dockerfile.rocm dependencies/rocm-lock.toml ROCM.md
git commit -m "$(cat <<'EOF'
Make gsplat a hard build dependency: verified on gfx1151

gsplat's test suite and Splatfacto training (7000 iterations on the
bonsai mip-nerf360 scene) both verified working on real gfx1151
hardware. docker/Dockerfile.rocm no longer falls back silently on
gsplat build failure — it's a real, required dependency now.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
git push origin main
```

---

### Task 6: Add a splatfacto smoke test entry

**Files:**
- Modify: `tests/rocm/README.md`
- Modify: `.github/workflows/tests-rocm.yml`

**Interfaces:**
- Consumes: the working training command validated in Task 4.
- Produces: a documented, dormant (manual-trigger-only, matching Phase 1's existing methods) CI step for future use once a self-hosted gfx1151 runner exists.

- [ ] **Step 1: Update tests/rocm/README.md**

Read the current file:
```bash
cat /home/bjoern/git/nerfstudio-rocm/tests/rocm/README.md
```

Add a line documenting the splatfacto log:
```markdown
`splatfacto.log` — Splatfacto training on the bonsai mip-nerf360 scene
(7000 iterations, `images_4` downsample). See `ROCM.md` for the gsplat
verification this depends on.
```

- [ ] **Step 2: Add a splatfacto job step to tests-rocm.yml**

Read the current workflow:
```bash
cat /home/bjoern/git/nerfstudio-rocm/.github/workflows/tests-rocm.yml
```

Add a step after the existing Phase 1 smoke-test step (same `phase1-baseline` job, or a new job — match whatever structure is already there), using the exact `ns-train splatfacto` command validated in Task 4 Step 3, with the dataset path adjusted to wherever a CI runner would have it mounted (document this as a TODO comment in the workflow itself, since no self-hosted runner exists yet to test the actual mount path):
```yaml
      - name: Run Splatfacto smoke test (bonsai, mip-nerf360)
        # NOTE: dataset mount path below is a placeholder — no self-hosted
        # gfx1151 runner exists yet to validate the real path. Update when
        # one is provisioned.
        run: |
          docker run --rm \
            --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
            -v /data/mipnerf360:/data/mipnerf360:ro \
            -v "$PWD/tests/rocm/logs:/workspace/nerfstudio-rocm/tests/rocm/logs" \
            nerfstudio-rocm:ci bash -c "ns-train splatfacto --data /data/mipnerf360/bonsai --max-num-iterations 7000 --viewer.quit-on-train-completion True --vis tensorboard colmap --images-path images_4 --colmap-path sparse/0"
```

- [ ] **Step 3: Commit and push**

```bash
cd /home/bjoern/git/nerfstudio-rocm
git add tests/rocm/README.md .github/workflows/tests-rocm.yml
git commit -m "$(cat <<'EOF'
Document splatfacto smoke test for future self-hosted CI runner

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
git push origin main
```
