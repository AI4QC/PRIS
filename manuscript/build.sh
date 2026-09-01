#!/usr/bin/env bash
# Build SI first, export its labels, then build the main text twice.
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-tectonic}"
echo "==> si.tex"          ; "$TECTONIC" --keep-intermediates --keep-logs si.tex
echo "==> si-xr.tex"       ; python3 gen-si-xr.py
echo "==> main.tex pass 1" ; "$TECTONIC" --keep-intermediates --keep-logs main.tex
echo "==> main.tex pass 2" ; "$TECTONIC" --keep-intermediates --keep-logs main.tex
echo "==> done: main.pdf si.pdf"
