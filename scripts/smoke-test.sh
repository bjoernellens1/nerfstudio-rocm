#!/usr/bin/env bash
# Phase-1 baseline smoke test: pure-Torch Nerfstudio methods, no accelerated
# deps required. Run inside the ROCm container. Logs go to tests/rocm/logs/.
set -euo pipefail

# nerfstudio's built-in `ns-download-data` sources (nerfstudio/blender captures)
# are hosted on Google Drive and frequently rate-limit/block gdown — don't
# depend on that for a smoke test. Use the tiny blender-format fixture
# nerfstudio ships for its own tests instead (1 train + 1 val image).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/tests/data/lego_test}"
LOG_DIR="${REPO_ROOT}/tests/rocm/logs"
mkdir -p "${LOG_DIR}"

python -m rocm.diagnostics | tee "${LOG_DIR}/rocm-info.log"

run() {
    local name="$1"; shift
    echo "==> ${name}"
    ns-train "$@" --data "${DATA_DIR}" --max-num-iterations 20 \
        --viewer.quit-on-train-completion True --vis tensorboard \
        2>&1 | tee "${LOG_DIR}/${name}.log"
}

# vanilla-nerf/mipnerf default to BlenderDataParserConfig already; nerfacto
# defaults to NerfstudioDataParserConfig (COLMAP-style) so it needs an
# explicit dataparser override to work with this fixture.
run vanilla-nerf vanilla-nerf
run mipnerf mipnerf
run nerfacto-torch nerfacto --pipeline.model.implementation torch blender-data

echo "==> smoke test complete, logs in ${LOG_DIR}"
