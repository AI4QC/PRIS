#!/usr/bin/env bash
# Emit one machine-readable snapshot of the campaign. Runs on the login node, kept to a
# single pass over the package directories so it can be called every few seconds.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
now=$(date +%s)

for p in E1_rho_curve E1b_paw_control E2_ordering E3_crosscheck E4_design E4_design/stage_b; do
  [ -f "$p/tasklist.txt" ] || continue
  tot=$(grep -c . "$p/tasklist.txt")
  dn=$(find "$p/.farm/done" -maxdepth 1 -type f 2>/dev/null | wc -l)
  # count claims that have not yet produced a done marker, rather than subtracting totals:
  # claims are released as tasks finish, so the difference can go negative
  cl=0
  for c in "$p"/.farm/claim/*; do
    [ -e "$c" ] || continue
    [ -e "$p/.farm/done/$(basename "$c")" ] || cl=$((cl + 1))
  done
  fb=$(find "$p/tasks" -name FALLBACK_APPLIED 2>/dev/null | wc -l)
  # tasks finished in the last hour, from the done markers' timestamps
  recent=$(find "$p/.farm/done" -maxdepth 1 -type f -newermt "-1 hour" 2>/dev/null | wc -l)
  echo "PKG $p $tot $dn $cl $fb $recent"
done

echo "QUEUE $(squeue -h -u "$USER" -r -t R 2>/dev/null | wc -l) $(squeue -h -u "$USER" -r -t PD 2>/dev/null | wc -l)"

hb=$(cat .farm_controller.heartbeat 2>/dev/null)
if [ -n "$hb" ]; then
  echo "HB $(( now - $(date -d "$hb" +%s 2>/dev/null || echo "$now") ))"
else
  echo "HB -1"
fi

unk=0; known=0
for f in E*/.farm/failed/*.txt; do
  [ -e "$f" ] || continue
  # classes already diagnosed and decided on: handled automatically, or reported and
  # excluded. Only a signature not on this list is worth interrupting for.
  if grep -qE "ZBRENT|Inconsistent Bravais|RHOSYG|INVGRP|IBZKPT|FEXCF|not hermitian" "$f"
  then known=$((known+1)); else unk=$((unk+1)); fi
done
dead=0
for p in E1_rho_curve E1b_paw_control E2_ordering E3_crosscheck E4_design E4_design/stage_b; do
  dead=$(( dead + $(find "$p/.farm/dead" -maxdepth 1 -type f 2>/dev/null | wc -l) ))
done
echo "FAIL $unk $known $dead"

# node-hours of VASP consumed, as a running cost figure
# Cost is the one expensive line here: it reads every OUTCAR, and there are thousands of
# them on a shared filesystem. A snapshot must never wait for that scan - a poll that times
# out reports nothing at all, which is worse than a figure a cycle old. So the cached value
# is printed immediately and a stale cache is refreshed in the background.
CACHE=.farm_cost_cache
STALE=1800
cached=$(cat "$CACHE" 2>/dev/null)
age=$(( now - $(stat -c %Y "$CACHE" 2>/dev/null || echo 0) ))
if [ "$age" -gt "$STALE" ] && [ ! -f "$CACHE.lock" ]; then
  ( touch "$CACHE.lock"
    find E1_rho_curve E1b_paw_control E2_ordering E3_crosscheck E4_design \
      -name OUTCAR -path "*/tasks/*" 2>/dev/null -exec grep -h "Elapsed time" {} + 2>/dev/null \
      | awk '{s+=$NF} END{printf "%.1f", s/3600}' > "$CACHE.new" 2>/dev/null
    [ -s "$CACHE.new" ] && mv "$CACHE.new" "$CACHE"
    rm -f "$CACHE.lock" ) >/dev/null 2>&1 &
fi
echo "COST ${cached:-0}"
