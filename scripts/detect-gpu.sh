#!/usr/bin/env bash
# Print the detected AMD GPU gfx arch, or exit 1 if none found / rocminfo missing.
set -euo pipefail

if ! command -v rocminfo >/dev/null 2>&1; then
    echo "rocminfo not found — not a ROCm host" >&2
    exit 1
fi

arch=$(rocminfo | grep -m1 -oE 'gfx[0-9a-fA-F]+' || true)
if [[ -z "${arch}" ]]; then
    echo "no AMD GPU detected by rocminfo" >&2
    exit 1
fi

echo "${arch}"
