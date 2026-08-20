#!/bin/bash
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 64 -t 06:00:00 -A lilly-rob1
#SBATCH -o /scratch/gautschi/manors/merlin_drone/drone_%j.log
# Reusable driver wrapper: launch PX4 SIH, then run eval/drone/run_experiments.py with the
# forwarded args. The renderer is spawned by the driver itself (at the model resolution).
set -u
# Overridable so the same driver runs off-cluster (e.g. a single cloud GPU box), where the
# SBATCH headers above are inert and there is no SLURM_JOB_ID.
WS=${MERLIN_WS:-/scratch/gautschi/manors/merlin_drone}
JOB=${SLURM_JOB_ID:-$$}
# Offline by default because Gautschi compute nodes have no internet; a host that does have
# it (and so has not pre-warmed the caches) exports 0.
export HF_HOME=$WS/hf_home TORCH_HOME=$WS/torch_home \
       HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
cd $WS/repo
# MERLIN_PX4_INSTANCE selects the PX4 instance (MAVLink 14540+i) and renderer port (5601+i),
# so co-located jobs do not collide.
I=${MERLIN_PX4_INSTANCE:-0}
bash eval/drone/run_px4.sh $I > $WS/drone_px4_${JOB}.log 2>&1 &
sleep 15
PYTHONPATH="$WS/repo:$WS/repo/model/distill:$WS/repo/model/backbones" \
  $WS/envs/modelenv/bin/python eval/drone/run_experiments.py "$@"
RC=$?
# Match only this job's instance: a bare binary-path match kills co-located jobs' PX4 too,
# which silently freezes their drone (position stops updating) rather than erroring.
pkill -TERM -f "bin/px4 -i $I -d" 2>/dev/null
echo "DRIVER_RC=$RC"
