#!/bin/bash
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
# smallgpu nodes are 128 cores / 2 GPUs and allocate memory proportional to CPUs, so it
# requires 64 CPUs per GPU. (The `ai` partition wants exactly 14 per GPU instead.)
#SBATCH -N 1 -n 64 -t 11:30:00 -A lilly-rob1
#SBATCH -o /scratch/gautschi/manors/merlin_drone/moeridge_%j.log
# One cell of the AAT-site capacity test. Usage: sbatch sbatch_moe_ridge.sh <arm> <k> <rank>
#
# Why the AAT and not the old site: the earlier screen wrapped dpt_regressor_head.conv2.0, a
# 147K-param conv, so even doubling that site moved the 230M model by 0.018%. The 8 AAT FFN
# blocks are 37.8M params, and at K=8/r=128 the experts add 62.9M -- capacity that can actually
# register. Depth only, no metric scale: run_cell already trains under si_loss.
set -eu
ARM=${1:?arm}; K=${2:?k}; RANK=${3:?rank}; WIDEN=${4:-0}
STEPS=${STEPS:-3000}; SEED=${SEED:-0}; METRIC=${METRIC:-0}
[ "$METRIC" = "1" ] && MFLAG="--metric" || MFLAG=""
# WIDEN>0 widens the dense FFN instead of adding experts: the matched-parameter control.
# cap_widen puts w5200 at +63.94M against MoE K=8/r=128's +62.91M, a 1.6% match.

WS=${MERLIN_WS:-/scratch/gautschi/manors/merlin_drone}
export HF_HOME=$WS/hf_home TORCH_HOME=$WS/torch_home \
       HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# 8 AAT blocks x {fc1, fc2}; info_sharing.self_attention_blocks is the AAT (matches aat-depth 8)
SITES=$(python3 - <<'EOF'
p = "info_sharing.self_attention_blocks.%d.mlp.%s"
print(",".join(p % (b, f) for b in range(8) for f in ("fc1", "fc2")))
EOF
)

TAG=${ARM}_k${K}_r${RANK}$([ "$WIDEN" -gt 0 ] && echo "_w${WIDEN}" || true)
TAG=$TAG$([ "$STEPS" != "3000" ] && echo "_s${STEPS}" || true)
TAG=$TAG$([ "$METRIC" = "1" ] && echo "_metric" || true)
TAG=$TAG$([ "$SEED" != "0" ] && echo "_seed${SEED}" || true)
OUT=$WS/out/moeridge/$TAG.json
mkdir -p "$(dirname "$OUT")"

echo "[ridge] arm=$ARM k=$K rank=$RANK widen=$WIDEN steps=$STEPS seed=$SEED metric=$METRIC"
cd "$WS"
exec "$WS/envs/moeenv/bin/python" model/moe/run_cell.py \
    --arm "$ARM" --k "$K" --rank "$RANK" --seed "$SEED" --steps "$STEPS" $MFLAG \
    --ffn-widen "$WIDEN" \
    --sites "$SITES" \
    --init "$WS/ckpt/student_hab_nav.pt" \
    --train-root "$WS/data/nearfield/train" \
    --val-root "$WS/data/nearfield/val" \
    --out "$OUT"
