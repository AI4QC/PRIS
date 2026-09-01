#!/usr/bin/env bash
# Per-task driver. Stages run in order; each writes into its own subdirectory so a failed
# stage never overwrites a good one. Pure shell, so a compute node needs nothing but VASP.
# VASP_BIN and VASP_PP_PATH come from the submit script.
set -uo pipefail
cd "$(dirname "$0")"
: "${VASP_BIN:?VASP_BIN not set}"
: "${VASP_PP_PATH:?VASP_PP_PATH not set}"

build_potcar() {
  : > POTCAR
  while read -r rel; do
    [ -z "$rel" ] && continue
    cat "$VASP_PP_PATH/$rel" >> POTCAR || return 1
  done < POTCAR.list
  [ -s POTCAR ]
}

run_stage() {
  stage="$1"; incar="$2"; poscar="$3"
  mkdir -p "$stage"
  cp "$poscar" "$stage/POSCAR"
  cp "INCAR.$incar" "$stage/INCAR"
  cp KPOINTS "$stage/KPOINTS"
  cp POTCAR.list POTCAR.spec.json "$stage/"
  ( cd "$stage" && build_potcar && $VASP_BIN > vasp.out 2> vasp.err ; echo $? > exit_code )
  # the POTCAR is licensed and rebuildable from the list, so it is never left behind
  rm -f "$stage/POTCAR" "$stage/CHG" "$stage/CHGCAR" "$stage/WAVECAR" "$stage/PROCAR"
  return 0
}

while IFS="$(printf '\t')" read -r stage incar poscar from_contcar; do
  [ -z "$stage" ] && continue
  if [ -f "$stage/exit_code" ] && [ "$(cat "$stage/exit_code")" = "0" ]; then
    echo "skip $stage (already converged)"; continue
  fi
  if [ "$from_contcar" != "-" ]; then
    if [ ! -s "$from_contcar/CONTCAR" ]; then
      echo "missing $from_contcar/CONTCAR, stopping" >&2; break
    fi
    cp "$from_contcar/CONTCAR" "$poscar"
  fi
  echo "==> $stage"
  run_stage "$stage" "$incar" "$poscar"
  if [ "$(cat "$stage/exit_code")" != "0" ]; then
    echo "stage $stage failed" >&2; break
  fi
done < stages.tsv
