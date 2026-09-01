#!/usr/bin/env bash
# Does LSCAAWARE=.FALSE. change any answer?
#
# It was added after 1,266 stages had already run, to get past a ScaLAPACK failure in the
# distributed subspace diagonalisation. The switch selects a different linear-algebra path
# and should leave the converged energy alone, but "should" is not evidence: this recomputes
# finished stages with the new setting and prints the difference against what is on disk.
#
#SBATCH --job-name=pris-lsca
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=lsca-%j.out
#SBATCH --error=lsca-%j.err

set -uo pipefail
cd ~/dft || exit 1
export VASP_PP_PATH=${VASP_PP_PATH:-$HOME/POTCAR/PBE}
VASP_EXE=${VASP_EXE:-$HOME/vasp.6.3.0/bin/vasp_std}
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module purge >/dev/null 2>&1; module load intel/oneapi2023.2_impi || exit 1
export OMP_NUM_THREADS=1 I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm

W=~/dft/.lsca_check; rm -rf $W; mkdir -p $W
printf "%-42s %18s %18s %14s\n" "stage" "on disk (eV)" "recomputed (eV)" "difference"
n=0
for src in $(ls -d E1_rho_curve/tasks/*/v08 E2_ordering/tasks/*/static E3_crosscheck/tasks/*/static 2>/dev/null | head -60); do
  [ "$(cat "$src/exit_code" 2>/dev/null)" = "0" ] || continue
  ref=$(grep "energy(sigma->0)" "$src/OUTCAR" 2>/dev/null | tail -1 | awk '{print $NF}')
  [ -z "$ref" ] && continue
  t=$(dirname "$src"); s=$(basename "$src")
  inc=$(awk -F'\t' -v s="$s" '$1==s{print $2}' "$t/stages.tsv")
  d="$W/$(basename "$t")_$s"; mkdir -p "$d"
  cp "$src/POSCAR" "$d/"; cp "$t/KPOINTS" "$d/"
  # the INCAR on disk already carries the switch; strip it to be sure of what is compared
  grep -v "^LSCAAWARE" "$t/INCAR.$inc" > "$d/INCAR"; echo "LSCAAWARE = .FALSE." >> "$d/INCAR"
  : > "$d/POTCAR"
  while read -r rel; do [ -n "$rel" ] && cat "$VASP_PP_PATH/$rel" >> "$d/POTCAR"; done < "$t/POTCAR.list"
  ( cd "$d" && mpirun -n 64 "$VASP_EXE" > vasp.out 2> vasp.err < /dev/null )
  new=$(grep "energy(sigma->0)" "$d/OUTCAR" 2>/dev/null | tail -1 | awk '{print $NF}')
  if [ -n "$new" ]; then
    printf "%-42s %18s %18s %14s\n" "$(basename "$t")/$s" "$ref" "$new" "$(echo "$new - $ref" | bc)"
  else
    printf "%-42s %18s %18s %14s\n" "$(basename "$t")/$s" "$ref" "FAILED" "-"
  fi
  rm -f "$d/POTCAR" "$d/CHG"* "$d/WAVECAR"
  n=$((n+1)); [ "$n" -ge 6 ] && break
done
echo "done"
