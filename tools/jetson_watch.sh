#!/usr/bin/env bash
# Record what the board was doing at the moment it died.
#
# Three site-bench runs have taken the Orin Nano off the network hard: it never came back on its
# own, it needed a physical power cycle every time, and it left NO kernel evidence because the
# journal is not persistent. That combination is what this script fixes. It is diagnostic only
# and starts no workload of its own.
#
#   ./jetson_watch.sh start [logdir]   begin sampling
#   ./jetson_watch.sh stop
#   ./jetson_watch.sh persist-journal  one-time, needs sudo: keep logs across a reboot
#
# tegrastats carries RAM, per-core load, GPU frequency, the thermal zones and, on this board,
# the VDD power rails. Sampling is fsynced every few lines, because a crash that eats the last
# ten seconds of log eats exactly the ten seconds worth having.
set -u
CMD=${1:-start}
LOG=${2:-$HOME/merlin/watch}
PIDF=$LOG/tegrastats.pid

case "$CMD" in
  persist-journal)
    # Left in place but NOT the mechanism relied on: journald on this image stays on the runtime
    # journal (/run/log/journal) whatever Storage= says, so nothing survives a hang. `start`
    # follows the kernel ring buffer into a plain rootfs file instead, which does survive.
    echo 123456 | sudo -S sed -i 's/^#\?Storage=.*/Storage=persistent/' /etc/systemd/journald.conf
    echo 123456 | sudo -S systemctl restart systemd-journald
    journalctl --disk-usage
    ;;
  start)
    mkdir -p "$LOG"
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null
    setsid nohup sh -c '
      n=0
      tegrastats --interval 1000 | while IFS= read -r line; do
        printf "%s %s\n" "$(date +%s)" "$line" >> '"$LOG"'/tegrastats.log
        n=$((n+1)); [ $((n % 30)) -eq 0 ] && sync -d '"$LOG"'/tegrastats.log
      done' >/dev/null 2>&1 < /dev/null &
    echo $! > "$PIDF"
    # Kernel ring buffer -> rootfs file. This is where an Xid, an OOM kill, a thermal trip or a
    # regulator fault will show up, and it is the evidence three hangs so far have not left.
    setsid nohup sh -c '
      m=0
      echo 123456 | sudo -S dmesg --follow --time-format iso 2>/dev/null | while IFS= read -r l; do
        printf "%s\n" "$l" >> '"$LOG"'/kmsg.log
        m=$((m+1)); [ $((m % 30)) -eq 0 ] && sync -d '"$LOG"'/kmsg.log
      done' >/dev/null 2>&1 < /dev/null &
    echo $! > "$LOG/kmsg.pid"
    nvpmodel -q | tr '\n' ' ' > "$LOG/powermode.txt"; sync
    echo "watching -> $LOG/tegrastats.log (pid $(cat "$PIDF"), mode: $(cat "$LOG/powermode.txt"))"
    ;;
  stop)
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null && rm -f "$PIDF"
    [ -f "$LOG/kmsg.pid" ] && kill "$(cat "$LOG/kmsg.pid")" 2>/dev/null && rm -f "$LOG/kmsg.pid"
    pkill -f "tegrastats --interval" 2>/dev/null
    pkill -f "dmesg --follow" 2>/dev/null
    echo stopped
    ;;
  *) echo "usage: $0 {start|stop|persist-journal} [logdir]" >&2; exit 2 ;;
esac
