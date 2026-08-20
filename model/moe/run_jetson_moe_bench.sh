#!/usr/bin/env bash
# Drive the FFNMoE deployment measurements on the Orin Nano, ONE CELL PER PROCESS.
#
# One process per cell is not tidiness. The first attempt ran four block_m values and three K
# cells in a single process; triton recompiles its whole 18-config autotune space for every
# distinct BLOCK_M, and the board went off the network and needed a power cycle. Per-process
# isolation bounds both the JIT arena and the resident expert weights, and a cell that does kill
# the board loses one json instead of the run.
#
# Resumable: a cell whose json already exists is skipped, so re-running after a reboot continues.
# Ordered by value, because the box may not survive to the end.
set -u
PY=${PY:-$HOME/merlin/venv/bin/python}
REPO=${REPO:-$HOME/merlin/Merlin}
OUT=${OUT:-$HOME/merlin/moebench}
mkdir -p "$OUT"
cd "$REPO" || exit 1

swapon --show=NAME --noheadings | grep -q . \
  || echo "WARNING: no swap. An 8 GB shared-memory OOM hard-reboots this board." >&2

# Blunt the CPU+GPU current step. A steady 4096^3 GEMM loop survives 23.1 W on this board with
# all six cores idle; the bench dies, and the difference is triton autotune -- 18 kernel variants
# JIT-compiled across every core WHILE the GPU bursts. Persisting the cache means that only
# happens on the first run, and capping threads keeps the CPU side of the peak down.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/merlin/.triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
echo "triton cache $TRITON_CACHE_DIR | OMP threads $OMP_NUM_THREADS | mode $(nvpmodel -q | tail -1)"

run () {
  local tag=$1; shift
  if [ -s "$OUT/$tag.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag  $(date +%H:%M:%S)"
  export MOE_HEARTBEAT="$OUT/$tag.beat"
  if "$PY" "$@" --out "$OUT/$tag.json" >"$OUT/$tag.log" 2>&1; then
    echo "[ok  ] $tag"
  else
    echo "[FAIL] $tag -- see $OUT/$tag.log"
  fi
  # Flush THIS FILE, not the whole system. A bare `sync` is a global writeback barrier, and on a
  # 114 GB eMMC at 86% full it can block for a long time. systemd holds /dev/watchdog with
  # RuntimeWatchdogUSec=2min, so anything that stalls PID 1 for two minutes hard-resets the
  # board with no log -- which is a live suspect for the hangs, and would mean the crash
  # hardening was causing the crashes.
  sync -d "$OUT/$tag.json" 2>/dev/null || true
}

B=model/moe/bench_ffn_moe_latency.py
T=model/moe/trt_ffn_moe.py

# 1. is there an INT8 engine at all. Done and stable; skipped on re-run via the json check.
run trt_k8 "$T" --k 8
run trt_k2 "$T" --k 2

# 2. site level WITHOUT CUDA graphs. Eager is enough to find the padding floor, since block_m
#    changes GEMM work rather than launch overhead, and every board hang so far has landed in a
#    cell that was capturing graphs. Cheapest useful data first.
for bm in 64 32 16 128; do
  run "site_eager_bm$bm" "$B" --level site --block-ms "$bm" --no-graph
done

# 3. then the captured numbers, which are the deployable ones.
for bm in 64 32; do
  run "site_bm$bm" "$B" --level site --block-ms "$bm"
done

# NOTE: the whole-model cells are OUT of this queue. They fail in the geometric encoders'
# einops einsum ("expected scalar type Half but found Float"), which needs the teacher's
# multi-patch fp16 recipe and is not what this measurement is about. The frame numbers come
# from the site delta on the measured 169.49 ms fp16 base instead (matched_latency.py).

echo "ALL CELLS ATTEMPTED $(date +%H:%M:%S)"
ls -la "$OUT"/*.json 2>/dev/null
