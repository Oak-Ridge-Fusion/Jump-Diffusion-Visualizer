#!/usr/bin/env bash
# run_all.sh -- Problem 3 (2D runaway electron): end-to-end ML pipeline.
#
# Single entry point meant to be run on the GPU box: generates the
# ground-truth data, trains the flow map + 3-class exit classifier, then
# runs inference (B0/B1/B2 vs the reference integrator). Nothing else needs
# to be set up by hand -- Config already defaults device="cuda" and falls
# back to CPU automatically if CUDA isn't available (see common_2.get_device).
#
# Safe to re-run: each stage is SKIPPED if its output already exists, so a
# crash partway through (e.g. a GPU OOM during flow-map training) doesn't
# force you to regenerate the ~1M-particle dataset again. Pass --force to
# rerun every stage from scratch regardless.
#
# Usage:
#   bash run_all.sh            # skip stages whose output already exists
#   bash run_all.sh --force    # rerun every stage
#
# Output (all in <repo>/Diffusion_CPP/code_2d/artifacts_re2d/):
#   data_pairs.npz, data_exit.npz, data_exit_extra.npz, config.json  (stage 1)
#   ckpt_flow.pt                                                     (stage 2)
#   ckpt_exit.pt                                                     (stage 3)
#   B0_onestep.png, B1_exit_maps.png, B2_rollout.png, metrics.json   (stage 4)

set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../problem3/ml
CODE_2D_DIR="$(cd "$SCRIPT_DIR/../../code_2d" && pwd)"
DATA_DIR="$CODE_2D_DIR/artifacts_re2d"

echo "=================================================================="
echo "Problem 3 (2D runaway electron) -- ML pipeline"
echo "code_2d dir : $CODE_2D_DIR"
echo "data dir    : $DATA_DIR"
echo "force rerun : $FORCE"
echo "=================================================================="
echo

python3 -c "
import torch
print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
else:
    print('WARNING: no CUDA device visible -- will fall back to CPU '
          '(common_2.get_device), which will be much slower for '
          'flow_training.py\'s ODE label generation.')
"
echo

need() {  # need <filename-under-DATA_DIR>  -- true if --force or file missing
    [[ "$FORCE" == "1" || ! -f "$DATA_DIR/$1" ]]
}

echo "---- stage 1/4: data generation (common_2.py ground truth) ----"
if need "data_pairs.npz" || need "data_exit.npz" || need "data_exit_extra.npz"; then
    cd "$CODE_2D_DIR"
    time python3 data_generation_2.py
else
    echo "  skip: data_pairs.npz / data_exit.npz / data_exit_extra.npz already exist"
fi
echo

echo "---- stage 2/4: flow-map training ----"
cd "$SCRIPT_DIR"
if need "ckpt_flow.pt"; then
    time python3 flow_training.py
else
    echo "  skip: ckpt_flow.pt already exists"
fi
echo

echo "---- stage 3/4: exit-classifier training ----"
cd "$SCRIPT_DIR"
if need "ckpt_exit.pt"; then
    time python3 exit_training.py
else
    echo "  skip: ckpt_exit.pt already exists"
fi
echo

echo "---- stage 4/4: inference (B0 / B1 / B2) ----"
cd "$SCRIPT_DIR"
time python3 inference.py
echo

echo "=================================================================="
echo "DONE. Outputs in $DATA_DIR:"
echo "  data_pairs.npz, data_exit.npz, data_exit_extra.npz, config.json"
echo "  ckpt_flow.pt, ckpt_exit.pt"
echo "  B0_onestep.png, B1_exit_maps.png, B2_rollout.png, metrics.json"
echo "=================================================================="
