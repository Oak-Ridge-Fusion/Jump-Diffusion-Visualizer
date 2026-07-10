#!/usr/bin/env bash
# Train the Conditional RealNVP flow. Any config.py flag can be overridden, e.g.:
#   ./train_nf.sh --nf_epochs 300 --batch_size 2048 --lr 1e-4
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python train_nf.py "$@"
