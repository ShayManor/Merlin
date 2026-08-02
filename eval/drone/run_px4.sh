#!/bin/bash
# Launch PX4 SITL SIH quad headless inside the Apptainer container. No root.
# Uses PX4's own canonical SITL invocation (Tools/simulation/sitl_multiple_run.sh):
#   px4 -i 0 -d <build>/etc     run from a writable working dir.
# -d = daemon mode (no interactive pxh> shell, so headless/sbatch stdin is a non-issue).
# Do NOT pass -t <test_data>: that directory is not produced by the build and a missing
# -t path makes PX4 abort 255 before printing anything. PX4_SIM_MODEL selects the SIH
# airframe (internal dynamics + IMU, no external simulator). MAVLink OFFBOARD -> UDP 14540.
# Instance id (arg 1, default 0) offsets the MAVLink port to 14540+i and gives the instance
# its own rootfs, so concurrent PX4 jobs on one node do not fight over UDP 14540.
set -e
WS=/scratch/gautschi/manors/merlin_drone
I=${1:-0}
B=$WS/px4/build/px4_sitl_default
RD=$B/rootfs_$I; mkdir -p "$RD"
LD_PRELOAD= BASH_ENV= apptainer exec --cleanenv --bind /scratch/gautschi/manors \
  "$WS/images/px4sim.sif" bash -c \
  "unset LD_PRELOAD; cd $RD && PX4_SIM_MODEL=sihsim_quadx exec $B/bin/px4 -i $I -d $B/etc"
