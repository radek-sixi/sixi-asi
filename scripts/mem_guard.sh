#!/usr/bin/env bash
# Memory watchdog for model servers on unified-memory devices (Jetson): GPU allocations are not charged to the
# process, so the kernel OOM killer does not pick the server and the box stalls instead (2026-11-11: a 32-stream
# run hung Thor until a power cycle). Kills PID when MemAvailable drops below MIN_GB; logs the low-water mark.
#   mem_guard.sh PID [MIN_GB=10] [LOG]
set -u
PID=$1; MIN_KB=$(( ${2:-10} * 1024 * 1024 )); LOG=${3:-/dev/stderr}
low=999999999
while kill -0 "$PID" 2>/dev/null; do
  a=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)
  [ "$a" -lt "$low" ] && low=$a
  if [ "$a" -lt "$MIN_KB" ]; then
    kill -9 "$PID" 2>/dev/null
    echo "mem_guard: MemAvailable $((a / 1024)) MiB < $((MIN_KB / 1024)) MiB, killed $PID  $(date +%T)" >> "$LOG"
    break
  fi
  sleep 0.2
done
echo "mem_guard: lowest MemAvailable $((low / 1024)) MiB ($(awk -v k=$low 'BEGIN{printf "%.1f GiB", k/1048576}'))" >> "$LOG"
