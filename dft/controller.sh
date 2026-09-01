#!/usr/bin/env bash
# Keep the campaign moving without a person in the loop.
#
# Every cycle it looks at each package in order, tops the queue up to the working limit of
# 20 jobs, releases claims left by killed jobs, and moves on to the next package when one
# finishes. E4 is two stages: when stage A is complete it builds stage B and starts that.
#
# It only ever submits work that is already frozen in the packages; it never edits an INCAR
# or retries a failed stage with different settings. Failures are left for a human to read
# in .farm/failed, which is exactly where the farm writes them.
#
#   nohup setsid ./controller.sh > controller.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE=$(pwd)
MAX_INFLIGHT=20
PER_PACKAGE=8
MIN_PER_PACKAGE=3
CYCLE=${CYCLE:-300}

log() { echo "[$(date -Is)] $*"; }

pkg_total()   { grep -c . "$1/tasklist.txt" 2>/dev/null || echo 0; }
pkg_done()    { find "$1/.farm/done" -maxdepth 1 -type f 2>/dev/null | wc -l; }
# A task given up on will never be done, so a package holding one would otherwise never
# count as finished and the stage that depends on it would never be built.
pkg_dead()    { find "$1/.farm/dead" -maxdepth 1 -type f 2>/dev/null | wc -l; }
pkg_left()    { echo $(( $(pkg_total "$1") - $(pkg_done "$1") - $(pkg_dead "$1") )); }
pkg_jobname() { echo "pris-$(echo "$1" | sed 's|_[a-z_]*||g; s|/|-|g')"; }
inflight()    { squeue -h -u "$USER" -r 2>/dev/null | wc -l; }
pkg_running() { squeue -h -u "$USER" -n "$(pkg_jobname "$1")" 2>/dev/null | wc -l; }

release_stale() {
  local p=$1 n=0
  [ "$(pkg_running "$p")" -eq 0 ] || return 0
  for c in "$p"/.farm/claim/*; do
    [ -e "$c" ] || continue
    [ -e "$p/.farm/done/$(basename "$c")" ] && continue
    rmdir "$c" 2>/dev/null && n=$((n+1))
  done
  [ "$n" -gt 0 ] && log "$p: released $n claim(s) from an interrupted job"
  return 0
}

advance() {                       # advance <package> <walltime> <target_jobs>
  local p=$1 t=$2 target=$3 tot dn left running free want
  [ -f "$p/tasklist.txt" ] || return 1
  tot=$(pkg_total "$p"); dn=$(pkg_done "$p"); left=$(pkg_left "$p")
  if [ "$left" -le 0 ]; then return 2; fi          # package finished
  release_stale "$p"
  running=$(pkg_running "$p")
  # Top the package up to its target rather than only starting it when it has none. Jobs
  # end as they exhaust the list, so waiting for zero lets the queue drain while nodes sit
  # idle and the largest package crawls.
  want=$(( target - running ))
  [ "$want" -le 0 ] && return 0
  free=$(( MAX_INFLIGHT - $(inflight) ))
  [ "$want" -gt "$free" ] && want=$free
  [ "$want" -gt "$left" ] && want=$left
  [ "$want" -le 0 ] && return 0
  log "$p: $dn/$tot done, $running running, adding $want job(s)"
  ./submit_farm.sh "$p" "$want" 1 64 deimos "$t" 2>&1 | sed 's/^/    /'
  return 0
}

# Slots are shared out in proportion to what each package has left, so the package with the
# most work gets the most nodes instead of every package getting the same handful.
allocate() {
  local names=("$@") rem=() total=0 i
  for p in "${names[@]}"; do
    if [ -f "$p/tasklist.txt" ]; then
      r=$(pkg_left "$p"); [ "$r" -lt 0 ] && r=0
    else r=0; fi
    rem+=("$r"); total=$(( total + r ))
  done
  for i in "${!names[@]}"; do
    if [ "$total" -le 0 ] || [ "${rem[$i]}" -le 0 ]; then echo 0; continue; fi
    share=$(( MAX_INFLIGHT * rem[i] / total ))
    # A floor, not a bare minimum of one: a package with few tasks left is often the one
    # holding up an analysis, and strict proportionality starves it behind the largest
    # package. Three nodes finish a short tail quickly without denting the big package.
    [ "$share" -lt "$MIN_PER_PACKAGE" ] && share=$MIN_PER_PACKAGE
    [ "$share" -gt "${rem[$i]}" ] && share=${rem[$i]}
    echo "$share"
  done
}

# A heartbeat on the shared filesystem, because the controller runs on whichever login
# node the connection landed on and cannot be found with pgrep from another one.
HEARTBEAT="$HERE/.farm_controller.heartbeat"
PIDFILE="$HERE/.farm_controller.pid"
# A pid file, not pkill: a pattern like "controller.sh" also matches the shell that was
# asked to restart it, so pkill -f kills the very session issuing the command.
if [ -f "$PIDFILE" ]; then
  old_pid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$old_pid" ] && kill "$old_pid" 2>/dev/null; then
    log "stopped the previous controller (pid $old_pid)"
    # wait for it to go, or its exit trap deletes the pid file this one is about to write
    for _ in 1 2 3 4 5; do kill -0 "$old_pid" 2>/dev/null || break; sleep 1; done
  fi
fi
echo $$ > "$PIDFILE"
# only clean up files that still belong to this process, so a restart does not have its
# own pid file removed by the predecessor it just replaced
trap '[ "$(cat "$PIDFILE" 2>/dev/null)" = "$$" ] && rm -f "$HEARTBEAT" "$PIDFILE"' EXIT
log "controller started on $(hostname); limit $MAX_INFLIGHT jobs, cycle ${CYCLE}s"
while true; do
  # A superseded instance cannot be killed from another login node, and it would go on
  # looping and logging. The pid file on the shared filesystem is the authority: if it no
  # longer names this process, a newer controller has taken over and this one steps aside.
  if [ "$(cat "$PIDFILE" 2>/dev/null)" != "$$" ]; then
    log "pid file now names $(cat "$PIDFILE" 2>/dev/null); this instance is superseded, exiting"
    exit 0
  fi
  date -Is > "$HEARTBEAT"

  PKGS=(E2_ordering E3_crosscheck E4_design)
  [ -f E4_design/stage_b/tasklist.txt ] && PKGS+=(E4_design/stage_b)
  mapfile -t SHARES < <(allocate "${PKGS[@]}")
  for i in "${!PKGS[@]}"; do
    advance "${PKGS[$i]}" 48:00:00 "${SHARES[$i]:-1}"
  done

  # E4 stage B only exists once stage A has produced relaxed cells
  if [ ! -f E4_design/stage_b/tasklist.txt ] \
     && [ "$(pkg_left E4_design)" -le 0 ]; then
    log "E4 stage A complete; building stage B"
    ( cd E4_design && python3 make_stage_b.py ) 2>&1 | sed 's/^/    /'
  fi

  # stop when nothing is left anywhere
  allrc=0
  for p in E2_ordering E3_crosscheck E4_design E4_design/stage_b; do
    [ -f "$p/tasklist.txt" ] || continue
    [ "$(pkg_left "$p")" -gt 0 ] && allrc=1
  done
  if [ "$allrc" = "0" ] && [ "$(inflight)" -eq 0 ]; then
    log "every package is complete; controller exiting"
    break
  fi
  sleep "$CYCLE"
done
