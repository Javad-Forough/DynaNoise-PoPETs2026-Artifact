#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REQUIRED_FILES=(
  "saved_models/alexnet_cifar10_target.pt"
  "saved_models/hamp_alexnet_cifar10.pt"
  "saved_models_shadow/shadow_model.pt"
  "saved_models_shadow/lira_shadow0.pt"
  "saved_models_shadow/lira_shadow1.pt"
  "saved_models_shadow/lira_shadow2.pt"
)

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  echo "This directory is not a Git checkout." >&2
  echo "GitHub source ZIP archives and copied folders do not include Git LFS metadata or blobs." >&2
  echo "Please obtain the artifact with a normal Git clone instead:" >&2
  echo "  git lfs install" >&2
  echo "  git clone https://github.com/Javad-Forough/DynaNoise-PoPETs2026-Artifact.git" >&2
  echo "  cd DynaNoise-PoPETs2026-Artifact" >&2
  echo "  bash fetch_checkpoints.sh" >&2
  exit 1
fi

if ! git lfs version >/dev/null 2>&1; then
  echo "git-lfs is required but is not installed." >&2
  echo "Install it from https://git-lfs.com/, then run:" >&2
  echo "  git lfs install" >&2
  echo "  bash fetch_checkpoints.sh" >&2
  exit 1
fi

echo "[LFS] Pulling checkpoint blobs..."
git lfs pull --include="saved_models/*.pt,saved_models_shadow/*.pt"

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
  echo "Missing required checkpoint files after git lfs pull:" >&2
  printf '  %s\n' "${missing_files[@]}" >&2
  exit 1
fi

if (( ${#bad_files[@]} > 0 )); then
  echo "The following files are still Git LFS pointer stubs after git lfs pull:" >&2
  printf '  %s\n' "${bad_files[@]}" >&2
  echo "Please verify that Git LFS is installed and that the repository was cloned normally." >&2
  exit 1
fi

echo "[LFS] Checkpoints are present and ready."
