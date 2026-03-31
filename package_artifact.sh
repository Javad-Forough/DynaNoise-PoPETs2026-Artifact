#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "$ROOT_DIR")"
OUT_PATH="${1:-$ROOT_DIR/../${REPO_NAME}-bundle.tar.gz}"

REQUIRED_FILES=(
  "saved_models/alexnet_cifar10_target.pt"
  "saved_models/hamp_alexnet_cifar10.pt"
  "saved_models_shadow/shadow_model.pt"
  "saved_models_shadow/lira_shadow0.pt"
  "saved_models_shadow/lira_shadow1.pt"
  "saved_models_shadow/lira_shadow2.pt"
)

cd "$ROOT_DIR"

bad_files=()
missing_files=()

for rel_path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$rel_path" ]]; then
    missing_files+=("$rel_path")
    continue
  fi
  if head -n 1 "$rel_path" 2>/dev/null | grep -q "version https://git-lfs.github.com/spec/v1"; then
    bad_files+=("$rel_path")
  fi
done

if (( ${#missing_files[@]} > 0 )); then
  echo "Cannot package artifact: required checkpoints are missing." >&2
  printf '  %s\n' "${missing_files[@]}" >&2
  exit 1
fi

if (( ${#bad_files[@]} > 0 )); then
  echo "Cannot package artifact: the following files are still Git LFS pointer stubs." >&2
  printf '  %s\n' "${bad_files[@]}" >&2
  echo "Run 'bash fetch_checkpoints.sh' from a real Git checkout first." >&2
  exit 1
fi

echo "[PACKAGE] Creating bundled artifact at $OUT_PATH"

tar \
  --exclude="./.git" \
  --exclude="./pets_env" \
  --exclude="./data" \
  --exclude="./results_artifact" \
  --exclude="./__pycache__" \
  --exclude="./*.zip" \
  -czf "$OUT_PATH" \
  -C "$(dirname "$ROOT_DIR")" \
  "$REPO_NAME"

echo "[PACKAGE] Done."
