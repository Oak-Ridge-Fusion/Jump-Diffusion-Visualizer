#!/usr/bin/env bash
# Run the full benchmark: evaluate both trained models against Ground Truth,
# write results/results.{csv,json}, and generate all figures.
#   ./compare.sh --eval_n_anchors 7
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python compare.py "$@"
