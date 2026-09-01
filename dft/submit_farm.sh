#!/usr/bin/env bash
# Submit one package as a farm of whole nodes.
#
# These partitions are OverSubscribe=EXCLUSIVE, so a job holds a whole 64-core node no
# matter what it asks for, and the account allows only 64 jobs in flight. One array element
# per task would be rejected outright. Instead each array element takes a node and works
# through the package's tasklist one task at a time, giving every task all 64 cores.
#
#   ./submit_farm.sh <package> [jobs] [workers] [cores] [partition] [time]
#
# Defaults: 8 jobs, one task at a time per node, 64 cores per task. Re-running is how a
# partly finished package is resumed: a finished task carries a marker and is skipped, and
# run.sh skips any stage that already exited zero.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PKG=${1:?usage: submit_farm.sh <package> [jobs] [workers] [cores] [partition] [time]}
JOBS=${2:-8}
WORKERS=${3:-1}
CORES=${4:-64}
PART=${5:-deimos}
TIME=${6:-24:00:00}
MAX_JOBS=20

[ -d "$HERE/$PKG" ] || { echo "no package $PKG"; exit 1; }
[ "$JOBS" -le "$MAX_JOBS" ] || { echo "refusing $JOBS jobs; the working limit is $MAX_JOBS"; exit 1; }
# The limit is on jobs in flight, not on one submission: without this check several
# submissions each under the cap add up to more than the cap.
INFLIGHT=$(squeue -h -u "$USER" -r 2>/dev/null | wc -l)
ROOM=$(( MAX_JOBS - INFLIGHT ))
if [ "$ROOM" -le 0 ]; then
  echo "$INFLIGHT job(s) already in flight against a limit of $MAX_JOBS; submitting nothing"
  exit 0
fi
if [ "$JOBS" -gt "$ROOM" ]; then
  echo "$INFLIGHT in flight; trimming $JOBS to $ROOM to stay under $MAX_JOBS"
  JOBS=$ROOM
fi
[ $((WORKERS * CORES)) -le 64 ] || { echo "workers x cores exceeds the 64 cores on a node"; exit 1; }

# stage_b lives inside E4_design, so the name has to distinguish the two
NAME="pris-$(echo "$PKG" | sed 's|_[a-z_]*||g; s|/|-|g')"
mkdir -p "$HERE/$PKG/.farm/done" "$HERE/$PKG/.farm/claim" "$HERE/$PKG/logs"

# A claim only means something while a job holds it. If nothing of ours is queued for this
# package, any claim left behind is from a job that was killed, and clearing it is what
# lets the work resume instead of being skipped forever.
running=$(squeue -h -u "$USER" -n "$NAME" 2>/dev/null | wc -l)
if [ "$running" -eq 0 ]; then
  stale=0
  for c in "$HERE/$PKG/.farm/claim"/*; do
    [ -e "$c" ] || continue
    [ -e "$HERE/$PKG/.farm/done/$(basename "$c")" ] && continue
    rmdir "$c" 2>/dev/null && stale=$((stale + 1))
  done
  [ "$stale" -gt 0 ] && echo "cleared $stale stale claim(s) from an interrupted run"
else
  echo "note: $running job(s) named $NAME are already queued; leaving claims alone"
fi

total=$(grep -c . "$HERE/$PKG/tasklist.txt")
done_n=$(find "$HERE/$PKG/.farm/done" -maxdepth 1 -type f | wc -l)
# A task that has been given up on is settled, not pending. Counting it as remaining makes
# a finished package look like it still has work, and the queue then refills with jobs that
# sweep the list, find nothing they may claim, and exit -- burning a node each cycle.
dead_n=$(find "$HERE/$PKG/.farm/dead" -maxdepth 1 -type f 2>/dev/null | wc -l)
left=$(( total - done_n - dead_n ))
echo "$PKG: $total tasks, $done_n done, $dead_n dead, $left left; $JOBS nodes x $WORKERS task(s) x $CORES cores"
[ "$left" -gt 0 ] || { echo "nothing left to run"; exit 0; }
[ "$JOBS" -le "$left" ] || { JOBS=$left; echo "trimmed to $JOBS jobs, one per remaining task"; }

cd "$HERE/$PKG"
sbatch --parsable \
  --job-name="$NAME" \
  --partition="$PART" --time="$TIME" \
  --array="1-${JOBS}" \
  --export=ALL,FARM_PKG="$HERE/$PKG",FARM_ROOT="$HERE",WORKERS="$WORKERS",CORES_PER_WORKER="$CORES" \
  "$HERE/farm.slurm"
