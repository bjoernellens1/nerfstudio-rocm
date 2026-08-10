# nerfacc validation + Instant-NGP training on gfx1151 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-fork nerfacc-rocm from AMD-Ecosystem/nerfacc's working ROCm port, fix its two known bug patterns, verify on real gfx1151 hardware via its own test suite, prove it end-to-end by training Instant-NGP, and make it a hard dependency in nerfstudio-rocm.

**Architecture:** Mirrors the gsplat sub-project exactly. All build/test/train work happens inside the `nerfstudio-rocm` ROCm Docker image (`localhost/nerfstudio-rocm:phase1-pillowfix` or rebuilt from `docker/Dockerfile.rocm`), run with real GPU devices attached (`--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host`). `bjoernellens1/nerfacc-rocm` (new fork, based on `AMD-Ecosystem/nerfacc`'s `release/0.5.3`) is installed from source inside the running container during validation, then baked into the image as a hard build step once proven reliable (Task 5).

**Tech Stack:** Podman, PyTorch 2.10.0+rocm7.2.2, nerfacc (HIP/hipcc), Nerfstudio, pytest.

## Global Constraints

- Target architecture: gfx1151 (Radeon 8060S / Strix Halo).
- New nerfacc fork: `bjoernellens1/nerfacc-rocm`, forked from `AMD-Ecosystem/nerfacc`, based on branch `release/0.5.3` (not `main` — closest to nerfstudio's pinned `nerfacc==0.5.2`).
- No AI/Claude/agent attribution anywhere in any commit message or GitHub comment, in either repo (`nerfstudio-rocm` or `nerfacc-rocm`) — established and enforced throughout the prior gsplat sub-project, same rule applies here without exception.
- Instant-NGP's default dataparser is `NerfstudioDataParserConfig` (colmap-format), confirmed via `nerfstudio/configs/method_configs.py:251-271` — NOT `InstantNGPDataParserConfig` (that's only `instant-ngp-bounded`'s default). Use the same `bonsai` dataset and the same verified dataparser command shape established in the gsplat plan's Task 4: `colmap --images-path images --colmap-path sparse/0 --downscale-factor 4 --downscale-rounding-mode round` as the LAST tokens, after all other `ns-train` flags.
- `instant-ngp`'s method config defaults to `vis="viewer"` and `mixed_precision=True` (`nerfstudio/configs/method_configs.py:270-271`) — override `vis` to `tensorboard` for non-interactive training (same as every other method in this project), keep `mixed_precision` as-is (it's the method's real default, not something to second-guess without evidence it's a problem).
- Do not modify `dependencies/rocm-lock.toml`'s `tiny_rocm_nn` section — stays unported, out of scope.

---

### Task 1: Re-fork nerfacc-rocm and set up local checkout

**Files:**
- Modify: `dependencies/rocm-lock.toml` (this repo, `[nerfacc]` section's `repo`/`branch`/`commit` fields)

**Interfaces:**
- Consumes: nothing from earlier tasks (first task).
- Produces: a public GitHub fork `bjoernellens1/nerfacc-rocm` (of `AMD-Ecosystem/nerfacc`), a local checkout at `/home/bjoern/git/nerfacc-rocm` (replacing the current unported checkout) on branch `release/0.5.3`, with `origin` (own fork) and `upstream` (`nerfstudio-project/nerfacc`, reference only) remotes. `dependencies/rocm-lock.toml` updated to point at the new fork.

- [ ] **Step 1: Confirm current nerfacc-rocm has nothing to preserve, then delete it**

```bash
gh issue list --repo bjoernellens1/nerfacc-rocm --state all
gh pr list --repo bjoernellens1/nerfacc-rocm --state all
git -C /home/bjoern/git/nerfacc-rocm log --oneline origin/master..HEAD 2>&1
```

Expected: no issues, no PRs, and the local branch has at most the one "Add nerfstudio-rocm ecosystem banner" commit beyond upstream — confirming there's nothing repo-specific to preserve. If this finds something unexpected (an issue, a PR, or unexpected local commits), STOP and report to the controller before deleting — don't delete on autopilot if the assumption from planning doesn't hold.

```bash
gh repo delete bjoernellens1/nerfacc-rocm --yes
```

- [ ] **Step 2: Fork AMD-Ecosystem/nerfacc**

```bash
gh repo fork AMD-Ecosystem/nerfacc --fork-name nerfacc-rocm --clone=false
gh api repos/bjoernellens1/nerfacc-rocm --jq '{full_name, fork, parent: .parent.full_name, visibility, default_branch}'
```

Expected: `fork: true`, `parent: "AMD-Ecosystem/nerfacc"`, `visibility: "public"` (forks inherit parent visibility).

- [ ] **Step 3: Set up local checkout on release/0.5.3**

```bash
rm -rf /home/bjoern/git/nerfacc-rocm
git clone https://github.com/bjoernellens1/nerfacc-rocm.git /home/bjoern/git/nerfacc-rocm
cd /home/bjoern/git/nerfacc-rocm
git checkout release/0.5.3
git remote add upstream https://github.com/nerfstudio-project/nerfacc.git
git remote -v
```

Expected: `origin` → `bjoernellens1/nerfacc-rocm`, `upstream` → `nerfstudio-project/nerfacc`, currently on branch `release/0.5.3`.

- [ ] **Step 4: Update dependencies/rocm-lock.toml**

```bash
cd /home/bjoern/git/nerfacc-rocm
git rev-parse HEAD
```

In `/home/bjoern/git/nerfstudio-rocm/dependencies/rocm-lock.toml`, update the `[nerfacc]` section:
```toml
[nerfacc]
repo = "https://github.com/bjoernellens1/nerfacc-rocm"
branch = "release/0.5.3"
commit = "<paste the commit hash from above>"
status = "re-forked from AMD-Ecosystem/nerfacc (working ROCm port); IS_ROCM/arch-hardcoding fixes not yet applied"
```

- [ ] **Step 5: Commit and push**

```bash
cd /home/bjoern/git/nerfstudio-rocm
git add dependencies/rocm-lock.toml
git commit -m "$(cat <<'EOF'
Point rocm-lock.toml at the re-forked nerfacc-rocm

bjoernellens1/nerfacc-rocm is now a real fork of AMD-Ecosystem/nerfacc
(which has a working ROCm port on release/0.5.3) instead of a
from-scratch unported fork of nerfstudio-project/nerfacc.
EOF
)"
git push origin main
```

---

### Task 2: Fix IS_ROCM and offload-arch hardcoding

**Files:**
- Modify: `/home/bjoern/git/nerfacc-rocm/setup.py`

**Interfaces:**
- Consumes: the working checkout from Task 1 (branch `release/0.5.3`).
- Produces: a nerfacc-rocm commit where `pip install --no-build-isolation .` builds correctly for the actual target architecture (not hardcoded gfx942) and only compiles with ROCm/HIP flags when actually running under a ROCm PyTorch build (not unconditionally).

- [ ] **Step 1: Read the current setup.py and locate both hardcoded values**

```bash
cd /home/bjoern/git/nerfacc-rocm
grep -n "IS_ROCM\|offload-arch\|gfx942" setup.py
```

Expected to find: `IS_ROCM = True` near the top of the file (module-level, before any conditional), and `--offload-arch=gfx942` inside the `hipcc_flags` list construction, inside the `if IS_ROCM:` branch of `get_extensions()`.

- [ ] **Step 2: Fix IS_ROCM to check torch.version.hip**

Replace the hardcoded `IS_ROCM = True` with a conditional check. Since `setup.py`'s top-level code runs before `get_extensions()` (which does its own local `import torch`), the cleanest fix is to move the check into `get_extensions()` itself where `torch` is already imported, or add a top-level `import torch` guarded by a try/except (matching how other ROCm forks in this project — gsplat, tiny-rocm-nn — structure this same check). Inspect the actual surrounding code first (imports at top of file, where `IS_ROCM` is read elsewhere in the file besides `get_extensions()`) before choosing the exact mechanism — `grep -n "IS_ROCM" setup.py` to find every use site, since the fix must work at every one of them, not just the one inside `get_extensions()`.

The resulting behavior: `IS_ROCM` should be `True` when the installed PyTorch is a ROCm build (`torch.version.hip is not None`) and `False` otherwise — matching gsplat's own final, verified pattern of deriving ROCm-ness from the actual torch build rather than a hardcoded flag.

- [ ] **Step 3: Fix the offload-arch hardcoding to respect PYTORCH_ROCM_ARCH**

Find the `hipcc_flags` list construction (contains `"--offload-arch=gfx942"`). Apply the same pattern as `bjoernellens1/gsplat`'s verified fix (commit `4515618291906856d8471c95b65c060d0cec4e43` on `release/1.5.3b2` — you can read it directly: `git -C /home/bjoern/git/gsplat show 4515618 -- setup.py` for the exact reference implementation): check `os.environ.get("PYTORCH_ROCM_ARCH")` first, split on `,`/`;` if it lists multiple archs and take the first one, use that to build the `--offload-arch=<arch>` flag; fall back to the existing hardcoded `gfx942` only if the env var is unset.

- [ ] **Step 4: Build and verify with real GPU devices attached, targeting gfx1151**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/git/nerfacc-rocm:/tmp/nerfacc-debug:ro \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cp -r /tmp/nerfacc-debug /tmp/nerfacc
cd /tmp/nerfacc
PYTORCH_ROCM_ARCH=gfx1151 CUDA_HOME=/opt/rocm python -m pip install -v --no-build-isolation . 2>&1 | tail -60
python -c "import nerfacc; print(nerfacc.__file__); print(nerfacc.__version__)"
'
```

Expected: build succeeds, `import nerfacc` prints a version consistent with `0.5.3`, no errors. If it fails, debug using the same systematic-debugging approach as the gsplat sub-project — reproduce, isolate, hypothesize, test. Don't assume this build is trivially clean just because the diff was small; verify empirically.

Also verify the fix actually targets gfx1151, not silently falling back to gfx942 — check the build log for the actual `--offload-arch=` flag used:
```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/git/nerfacc-rocm:/tmp/nerfacc-debug:ro \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cp -r /tmp/nerfacc-debug /tmp/nerfacc && cd /tmp/nerfacc
PYTORCH_ROCM_ARCH=gfx1151 CUDA_HOME=/opt/rocm python -m pip install -v --no-build-isolation . 2>&1 | grep -o "\-\-offload-arch=[a-z0-9]*" | sort -u
'
```
Expected: only `--offload-arch=gfx1151`, not `gfx942`.

- [ ] **Step 5: Commit and push**

```bash
cd /home/bjoern/git/nerfacc-rocm
git add setup.py
git commit -m "$(cat <<'EOF'
fix: derive IS_ROCM from torch.version.hip, respect PYTORCH_ROCM_ARCH

Two bug-pattern fixes, same shape as bjoernellens1/gsplat's earlier
fixes for the identical problems:
- IS_ROCM was a hardcoded module-level True, meaning this always
  compiled with ROCm/HIP flags regardless of the actual installed
  PyTorch build. Now derived from torch.version.hip.
- hipcc_flags hardcoded --offload-arch=gfx942 (CDNA/Instinct only).
  Now checks PYTORCH_ROCM_ARCH first, falls back to gfx942 only if
  unset -- verified building correctly for gfx1151 (RDNA3.5) with the
  env var set, matching gsplat's release/1.5.3b2 commit 4515618's
  fix for the same issue.
EOF
)"
git push origin release/0.5.3
```

---

### Task 3: Run nerfacc's test suite on gfx1151

**Files:**
- None modified (unless a fixable test-environment issue is found — see Step 3's branch).

**Interfaces:**
- Consumes: the working nerfacc build from Task 2.
- Produces: a pass/fail report per test file, informing Task 5's documentation.

- [ ] **Step 1: Install nerfacc's test dependencies and run the suite**

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
git clone --branch release/0.5.3 https://github.com/bjoernellens1/nerfacc-rocm.git /tmp/nerfacc
cd /tmp/nerfacc
PYTORCH_ROCM_ARCH=gfx1151 CUDA_HOME=/opt/rocm python -m pip install --no-build-isolation .
python -m pip install pytest
pytest tests/ -v 2>&1 | tee /tmp/nerfacc-test-results.log
' 2>&1 | tee /home/bjoern/git/nerfacc-test-results-gfx1151.log
```

Expected: pytest runs and reports pass/fail counts per test file (`test_camera.py`, `test_grid.py`, `test_pack.py`, `test_pdf.py`, `test_rendering.py`, `test_scan.py`, `test_vdb.py`). Do not assume 100% pass — this is genuinely unverified on RDNA hardware, and the design spec's hypothesis that no wave32-class bug exists needs this run to confirm it, not just the absence of `rocprim` usage in a grep.

- [ ] **Step 2: Triage any failures**

For each failing test, classify as: a genuine gfx1151/RDNA-specific numerical or kernel bug (would confirm or refute the design spec's "probably no wave32 issue" hypothesis), a test-environment issue unrelated to correctness, or an already-known limitation. Record findings.

- [ ] **Step 3: Fix or document each failure**

Test-environment issues: fix directly if scoped and clear. Genuine kernel bugs: use **systematic-debugging** — reproduce, isolate, hypothesize, test. If a fix isn't reasonably scoped to finish within this task, document as a known limitation with a filed GitHub issue on `bjoernellens1/nerfacc-rocm` (issues are likely disabled by default on a fresh fork — check with `gh api repos/bjoernellens1/nerfacc-rocm --jq '.has_issues'` and enable via `gh repo edit bjoernellens1/nerfacc-rocm --enable-issues` if needed, same as was done for gsplat). Task 4 (Instant-NGP training) doesn't require every nerfacc test to pass — only the code paths Instant-NGP's `OccGridEstimator` actually exercises (occupancy grid updates, ray marching/sampling, rendering weight computation) need to be solid.

- [ ] **Step 4: Re-run the full suite and record final pass/fail counts**

Same command as Step 1, from the final committed tree. Save the summary line for Task 5's documentation update.

---

### Task 4: Train Instant-NGP on the bonsai scene

**Files:**
- Create: `tests/rocm/logs/instant-ngp.log` (gitignored, matches the existing pattern for `splatfacto.log` etc.)

**Interfaces:**
- Consumes: working nerfacc from Task 2/3, existing `nerfstudio-rocm` install, the `bonsai` dataset at `/home/bjoern/Downloads/mipnerf360_v2_dataset/bonsai` (already has COLMAP `sparse/0/{cameras,images,points3D}.bin`, same dataset the gsplat plan's Splatfacto training used).
- Produces: a completed Instant-NGP training run with a saved checkpoint, verifying nerfacc's `OccGridEstimator` works correctly end-to-end.

- [ ] **Step 1: Do a short trial run first (500 iterations) to catch config/path errors cheaply**

Instant-NGP is known to converge fast (the whole point of the method), but still do a short trial before committing to a longer run — same lesson as the gsplat plan's Task 4, which found the naively-assumed dataparser command didn't actually work.

First, check whether Task 2 or Task 3 already produced a reusable nerfacc wheel to avoid rebuilding from source (same shortcut the gsplat plan's Task 4 took):
```bash
find /home/bjoern/git/nerfstudio-rocm /tmp -maxdepth 3 -iname "*nerfacc*.whl" 2>/dev/null
```

If a wheel exists, install it directly (`python -m pip install <path-to-wheel> -q`). If not, build from source inline:
```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360:ro \
  -v /home/bjoern/git/nerfstudio-rocm:/workspace/nerfstudio-rocm \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cd /workspace/nerfstudio-rocm
pip install --no-build-isolation -e . -q
git clone --branch release/0.5.3 https://github.com/bjoernellens1/nerfacc-rocm.git /tmp/nerfacc
cd /tmp/nerfacc
PYTORCH_ROCM_ARCH=gfx1151 CUDA_HOME=/opt/rocm python -m pip install --no-build-isolation . -q
cd /workspace/nerfstudio-rocm
ns-train instant-ngp \
  --data /data/mipnerf360/bonsai \
  --max-num-iterations 500 \
  --viewer.quit-on-train-completion True --vis tensorboard \
  colmap --images-path images --colmap-path sparse/0 --downscale-factor 4 --downscale-rounding-mode round
' 2>&1 | tee /tmp/instant-ngp-trial.log
```

If a wheel from Task 2/3 was found instead, build a wheel once (`python -m pip wheel . --no-deps --no-build-isolation -w /workspace/nerfstudio-rocm/tests/rocm/wheels` from the nerfacc-rocm checkout, mirroring the gsplat plan's approach) and `pip install` it in both this trial run and Task 4 Step 2's full run, rather than rebuilding from source each time.

Expected: training starts, runs 500 iterations, ends with "Training Finished." If it errors on data loading or config (verify with `ns-train instant-ngp colmap --help` inside the container if something doesn't match), fix the command and retry before proceeding — do not spend a long run debugging config issues, same lesson from the gsplat plan.

- [ ] **Step 2: Run the full training (pick an iteration count during this step based on Step 1's observed iteration speed)**

Instant-NGP trains much faster per-iteration than Splatfacto typically, and its default `max_num_iterations` is 30000 — but this task doesn't need full convergence, just a real, meaningful validation run. Use Step 1's observed iteration time to pick a target that's practical (a few thousand iterations, enough for the occupancy grid to update multiple times and loss to visibly trend down) — document the actual number chosen and why in the task report, don't blindly copy a number from this plan without checking it's reasonable given the observed speed.

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360:ro \
  -v /home/bjoern/git/nerfstudio-rocm:/workspace/nerfstudio-rocm \
  --entrypoint bash localhost/nerfstudio-rocm:phase1-pillowfix -c '
cd /workspace/nerfstudio-rocm
pip install --no-build-isolation -e . -q
ns-train instant-ngp \
  --data /data/mipnerf360/bonsai \
  --max-num-iterations <N> \
  --viewer.quit-on-train-completion True --vis tensorboard \
  colmap --images-path images --colmap-path sparse/0 --downscale-factor 4 --downscale-rounding-mode round
' 2>&1 | tee /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/instant-ngp.log
```

- [ ] **Step 3: Verify success criteria**

```bash
grep -a "Training Finished" /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/instant-ngp.log
grep -a -i "nan\|inf" /home/bjoern/git/nerfstudio-rocm/tests/rocm/logs/instant-ngp.log
```

Expected: "Training Finished" banner present, no NaN/Inf. Additionally check the loss/PSNR trend from the TensorBoard event file (same technique as the gsplat plan's Task 4 — `tensorboard.backend.event_processing.event_file_loader.LegacyEventFileLoader`) to confirm training is actually converging in a healthy way, not just completing without crashing. Instant-NGP's occupancy grid should also show evidence of updating (check for any logged occupancy/grid-related metric, or at minimum confirm the loss trend and rendered eval images look reasonable) — this is the direct evidence that nerfacc's `OccGridEstimator` is working correctly on gfx1151, the actual point of this task.

---

### Task 5: Integrate nerfacc into nerfstudio-rocm as a hard dependency

**Files:**
- Modify: `docker/Dockerfile.rocm`
- Modify: `dependencies/rocm-lock.toml`
- Modify: `ROCM.md`

**Interfaces:**
- Consumes: the working nerfacc commit (from Task 2) and test/training results (from Task 3/4).
- Produces: an updated `nerfstudio-rocm` where nerfacc installs unconditionally at Docker build time, same as gsplat.

- [ ] **Step 1: Change the nerfacc Dockerfile step from best-effort to hard**

In `docker/Dockerfile.rocm`, find the current nerfacc step (best-effort, with a `||` fallback echo) and replace it with a hard `RUN` step, same pattern as gsplat's Task 5 fix. Point the `git clone` at `bjoernellens1/nerfacc-rocm`'s `release/0.5.3` branch. Rewrite the surrounding comment to reflect current state instead of "not yet ROCm-buildable."

- [ ] **Step 2: Verify the image builds successfully with the hard dependency**

```bash
cd /home/bjoern/git/nerfstudio-rocm
podman build -f docker/Dockerfile.rocm -t nerfstudio-rocm:phase3-nerfacc --build-arg PYTORCH_ROCM_ARCH=gfx1151 .
```

Expected: build succeeds through the nerfacc step without falling into a fallback (there shouldn't be one anymore). If `docker build`'s lack of live GPU device access causes a problem here (same class of issue as gsplat#2's original build-time-vs-runtime gap, already fixed via `PYTORCH_ROCM_ARCH` in Task 2), verify the env var is actually set in the Dockerfile's `ENV` block before this step runs (it should already be, from the existing `PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}` line near the top of `docker/Dockerfile.rocm`).

Also verify no other package silently overwrites nerfacc later in the build, the way `gsplat==1.4.0`'s pin did for gsplat in the earlier sub-project — check `pyproject.toml`'s dependency list for a `nerfacc==X` pin:
```bash
grep -n "nerfacc" pyproject.toml
```
nerfstudio's `pyproject.toml` currently pins `"nerfacc==0.5.2"` — since the ROCm fork is versioned `0.5.3` (matching its own `nerfacc/version.py`) but likely still installs under the same `nerfacc` distribution/import name (unlike gsplat's fork, which deliberately renamed to `amd_gsplat` to coexist with PyPI — check whether `bjoernellens1/nerfacc-rocm`'s `setup.py`/`nerfacc/version.py` did anything similar), this pin may or may not cause the same silent-overwrite problem gsplat had. Actually verify with a real build + `import nerfacc; print(nerfacc.__version__)` check inside the built image, don't assume either way.

- [ ] **Step 3: Fix the pin conflict if it reproduces**

If `import nerfacc` inside the built image reports `0.5.2` (stock PyPI) instead of `0.5.3` (the ROCm fork), apply the same fix as gsplat's Task 5: remove or adjust the `nerfacc==0.5.2` pin in `pyproject.toml`'s `dependencies` list, with an explanatory comment (following the existing `gsplat`-related comment already in that file as a template for tone/detail level).

- [ ] **Step 4: Update dependencies/rocm-lock.toml**

```toml
[nerfacc]
repo = "https://github.com/bjoernellens1/nerfacc-rocm"
branch = "release/0.5.3"
commit = "<paste final HEAD commit hash>"
status = "verified on gfx1151: pytest suite <N passed, M failed/skipped — see task 3 findings>; Instant-NGP trained <N> iterations on bonsai (mip-nerf360) with no NaN/Inf and healthy loss/PSNR trend, confirming nerfacc's OccGridEstimator works correctly; builds as a hard docker build-time dependency (docker/Dockerfile.rocm)"
```

- [ ] **Step 5: Update ROCM.md**

Update the nerfacc row in the fork-status table (currently says "unported, fresh fork") to reflect the new fork lineage (`AMD-Ecosystem/nerfacc` → `bjoernellens1/nerfacc-rocm`) and verified status. Update the "nerfacc-rocm: not started" section (currently describes a porting plan that's now been executed) to reflect what was actually done — similar correction to what the gsplat plan's final review required for Task 2b's documentation (state the real fork source and what was fixed, not the old aspirational plan text).

- [ ] **Step 6: Commit and push**

```bash
cd /home/bjoern/git/nerfstudio-rocm
git add docker/Dockerfile.rocm dependencies/rocm-lock.toml ROCM.md pyproject.toml
git commit -m "$(cat <<'EOF'
Make nerfacc a hard build dependency: verified on gfx1151

nerfacc (re-forked from AMD-Ecosystem/nerfacc's working ROCm port,
release/0.5.3) verified via its own test suite and an end-to-end
Instant-NGP training run confirming OccGridEstimator works correctly
on gfx1151. docker/Dockerfile.rocm no longer falls back silently on
nerfacc build failure.
EOF
)"
git push origin main
```

(Adjust the file list in `git add` if Step 3 didn't need to touch `pyproject.toml`.)
