#!/usr/bin/env bash
# Phase-1 baseline smoke test: pure-Torch Nerfstudio methods, no accelerated
# deps required. Run inside the ROCm container. Logs go to tests/rocm/logs/.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/data/nerfstudio/poster}"
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/rocm/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "==> downloading nerfstudio 'poster' test dataset to ${DATA_DIR%/poster}"
    ns-download-data nerfstudio --capture-name poster
fi

python -m rocm.diagnostics | tee "${LOG_DIR}/rocm-info.log"

run() {
    local name="$1"; shift
    echo "==> ${name}"
    ns-train "$@" --data "${DATA_DIR}" --max-num-iterations 20 \
        --viewer.quit-on-train-completion True \
        2>&1 | tee "${LOG_DIR}/${name}.log"
}

run vanilla-nerf vanilla-nerf
run mipnerf mipnerf
run nerfacto-torch nerfacto --pipeline.model.implementation torch

echo "==> smoke test complete, logs in ${LOG_DIR}"
