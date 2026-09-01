#!/bin/bash
# Kill controller.sh instances that are not the one named by the pid file.
# Matches the FULL command line exactly, so this script and the shell running it
# (whose command lines differ) can never match themselves -- the mistake that once
# killed the invoking ssh session.
cd "$(dirname "$0")" || exit 1
live=$(cat .farm_controller.pid 2>/dev/null)
ps -u "$USER" -o pid=,args= | while read -r pid args; do
  case "$args" in
    "bash ./controller.sh"|"/bin/bash ./controller.sh"|"./controller.sh") ;;
    *) continue ;;
  esac
  [ "$pid" = "$live" ] && { echo "$(hostname): keeping live $pid"; continue; }
  kill "$pid" 2>/dev/null && echo "$(hostname): killed zombie $pid"
done
