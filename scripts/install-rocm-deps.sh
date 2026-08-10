#!/usr/bin/env bash
# Install the pinned ROCm forks (gsplat, nerfacc, tiny-rocm-nn) into the current
# Python environment. Intended to run inside the ROCm container built from
# docker/Dockerfile.rocm; see dependencies/rocm-lock.toml for pinned commits.
set -euo pipefail

WORKDIR="${WORKDIR:-/tmp/nerfstudio-rocm-deps}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1151}"
export PYTORCH_ROCM_ARCH CUDA_HOME=/opt/rocm

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# gsplat#2 (PYTORCH_ROCM_ARCH ignored) and gsplat#3 (glm include path) are both
# fixed, so gsplat now builds as a hard dependency in docker/Dockerfile.rocm with
# no GPU device access needed. This script remains an alternative manual install
# path (e.g. inside an already-running container) and no longer requires the old
# "attach real GPU devices" workaround.
echo "==> gsplat (verified ROCm fork — builds from PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}, no GPU device access required)"
rm -rf gsplat
git clone --branch release/1.5.3b2 https://github.com/bjoernellens1/gsplat.git
cd gsplat
git submodule update --init --recursive
python -m pip install --no-build-isolation .
cd ..

echo "==> nerfacc (unported fork — best effort)"
rm -rf nerfacc
git clone https://github.com/bjoernellens1/nerfacc-rocm.git nerfacc
python -m pip install --no-build-isolation --no-deps ./nerfacc || \
    echo "nerfacc-rocm build failed — expected until ported, see ROCM.md"

echo "==> tiny-rocm-nn (partially ported — best effort on this arch)"
rm -rf tiny-rocm-nn
git clone --recursive https://github.com/bjoernellens1/tiny-rocm-nn.git
(cd tiny-rocm-nn/bindings/torch && python -m pip install --no-build-isolation .) || \
    echo "tiny-rocm-nn build failed on ${PYTORCH_ROCM_ARCH} — falls back to implementation=torch"

echo "==> done"
