#!/usr/bin/env bash
# Train the Conditional DDPM. Any config.py flag can be overridden, e.g.:
#   ./train_diffusion.sh --diff_epochs 300 --diff_n_timesteps 1000
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python train_diffusion.py "$@"
