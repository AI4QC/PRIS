#!/usr/bin/env bash
# Live view of the campaign. Redraws every REFRESH seconds until interrupted.
#
#   dft/monitor.sh              # refresh every 30 s
#   REFRESH=10 dft/monitor.sh
#
# Reads through the same connection wrapper the rest of the work uses, so it inherits the
# LAN binding that gets past the local proxy.
SSH=${HPC_SSH:-<path>
REFRESH=${REFRESH:-30}
bar() {  # bar <done> <total> <width>
  local d=$1 t=$2 w=$3 f
  [ "$t" -eq 0 ] && { printf '%*s' "$w" ""; return; }
  f=$(( d * w / t ))
  printf '%s%s' "$(printf '█%.0s' $(seq 1 $f 2>/dev/null))" "$(printf '·%.0s' $(seq 1 $((w-f)) 2>/dev/null))"
}
while true; do
  snap=$(timeout 120 "$SSH" 'bash ~/dft/status.sh' 2>/dev/null)
  clear
  printf '\033[1mPRIS first-principles campaign\033[0m    %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  if [ -z "$snap" ]; then
    printf '  no answer from the cluster; retrying in %ss\n' "$REFRESH"
  else
    printf '  %-20s %-24s %8s %7s %8s %8s\n' package progress done running "last h" fallback
    tot_all=0; done_all=0
    while read -r kind a b c d e f; do
      case $kind in
        PKG) tot=$b; dn=$c; run=$d; fb=$e; rec=$f
             tot_all=$((tot_all+tot)); done_all=$((done_all+dn))
             pct=0; [ "$tot" -gt 0 ] && pct=$(( dn * 100 / tot ))
             printf '  %-20s [%s] %3s%% %4s/%-4s %6s %8s %8s\n' \
               "$(basename "$a")" "$(bar "$dn" "$tot" 20)" "$pct" "$dn" "$tot" "$run" "$rec" "$fb" ;;
        QUEUE) qr=$a; qp=$b ;;
        HB)    hb=$a ;;
        FAIL)  unk=$a; kn=$b ;;
        COST)  cost=$a ;;
      esac
    done <<< "$snap"
    pct_all=0; [ "$tot_all" -gt 0 ] && pct_all=$(( done_all * 100 / tot_all ))
    printf '\n  %-20s [%s] %3s%% %4s/%-4s\n' TOTAL "$(bar "$done_all" "$tot_all" 20)" "$pct_all" "$done_all" "$tot_all"
    printf '\n  jobs: %s running, %s pending' "${qr:-?}" "${qp:-?}"
    if [ "${hb:--1}" -lt 0 ]; then printf '   controller: \033[31mnot running\033[0m'
    elif [ "${hb:-0}" -gt 1200 ]; then printf '   controller: \033[33mheartbeat %ss stale\033[0m' "$hb"
    else printf '   controller: alive (%ss ago)' "$hb"; fi
    printf '\n  failures: %s handled by the driver' "${kn:-0}"
    [ "${unk:-0}" -gt 0 ] && printf ', \033[31m%s needing attention\033[0m' "$unk"
    printf '\n  VASP consumed: %s node-hours\n' "${cost:-?}"
  fi
  printf '\n  refreshing every %ss — Ctrl-C to stop\n' "$REFRESH"
  sleep "$REFRESH"
done
