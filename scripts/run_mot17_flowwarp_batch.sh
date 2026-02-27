#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"
PY_SCRIPT="${REPO_ROOT}/scripts/mot17_bbox_flow_warp.py"

# Dataset root (symlinked as you did: w2-net/data/MOT17 -> /Users/akshay/Downloads/MOT17)
MOT17_ROOT="${REPO_ROOT}/data/MOT17"

# Choose device: mps (Apple) or cpu
DEVICE="${DEVICE:-mps}"

# GMFlow checkpoint preset for PTLFlow: chairs|things|sintel|kitti
CKPT="${CKPT:-things}"

# Number of frame pairs per sequence
MAX_PAIRS="${MAX_PAIRS:-200}"

# Output dir
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results}"

# Sequences to run (edit this list)
SEQS=(
  "train/MOT17-02-FRCNN"
  "train/MOT17-04-FRCNN"
  "train/MOT17-09-FRCNN"
)

# --- Run ---
echo "Repo: ${REPO_ROOT}"
echo "MOT17_ROOT: ${MOT17_ROOT}"
echo "DEVICE: ${DEVICE}  CKPT: ${CKPT}  MAX_PAIRS: ${MAX_PAIRS}"
echo "OUT_DIR: ${OUT_DIR}"
echo

mkdir -p "${OUT_DIR}"

# Activate venv
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "ERROR: venv not found at ${VENV_PATH}. Create it with: python3 -m venv .venv"
  exit 1
fi

# Sanity checks
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: script not found: ${PY_SCRIPT}"
  exit 1
fi

if [[ ! -d "${MOT17_ROOT}" ]]; then
  echo "ERROR: MOT17 root not found: ${MOT17_ROOT}"
  echo "Tip: ensure symlink exists: ln -s /Users/akshay/Downloads/MOT17 data/MOT17"
  exit 1
fi

for rel in "${SEQS[@]}"; do
  SEQ_PATH="${MOT17_ROOT}/${rel}"
  echo "=============================="
  echo "Running: ${SEQ_PATH}"
  echo "=============================="

  if [[ ! -d "${SEQ_PATH}/img1" ]]; then
    echo "WARNING: img1 not found, skipping: ${SEQ_PATH}"
    continue
  fi

  python "${PY_SCRIPT}" \
    --seq "${SEQ_PATH}" \
    --device "${DEVICE}" \
    --max_pairs "${MAX_PAIRS}" \
    --ckpt "${CKPT}" \
    --out_dir "${OUT_DIR}" \
    --out_csv auto

  echo
done

echo "All done. CSVs saved in: ${OUT_DIR}"