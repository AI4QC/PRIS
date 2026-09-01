#!/usr/bin/env bash
# Measure VASP wall time against KPAR, NCORE and rank count on this cluster's nodes.
#
# The cells here are small and carry many k-points, so k-point parallelism is expected to
# matter far more than band parallelism, and VASP itself warns that the default NCORE=1 is
# inefficient. Rather than guess, this runs the same cell under a matrix of settings and
# reports seconds. Nothing here touches the physics: KPAR, NCORE and rank count change only
# how the work is divided.
#
#SBATCH --job-name=pris-bench
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=bench-%j.out
#SBATCH --error=bench-%j.err

set -uo pipefail
cd "${BENCH_DIR:-$SLURM_SUBMIT_DIR}" || exit 1

VASP_EXE=${VASP_EXE:-$HOME/vasp.6.3.0/bin/vasp_std}
export VASP_PP_PATH=${VASP_PP_PATH:-$HOME/POTCAR/PBE}

# `module` is a shell function from /etc/profile.d, and a batch script is not a login
# shell, so it has to be brought in explicitly. Silently missing it leaves mpirun off PATH
# and every task fails instantly, which is exactly what happened the first time.
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
    [ -r "$f" ] && . "$f" && break
  done
fi
command -v module >/dev/null 2>&1 || { echo "no module command; cannot set up the toolchain" >&2; exit 1; }
module purge >/dev/null 2>&1
module load "${ONEAPI_MODULE:-intel/oneapi2023.2_impi}" || { echo "cannot load the oneAPI module" >&2; exit 1; }
command -v mpirun >/dev/null 2>&1 || { echo "mpirun still not on PATH after module load" >&2; exit 1; }
export OMP_NUM_THREADS=1 I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm

run_case() {
  local label=$1 src=$2 incar=$3 ranks=$4 kpar=$5 ncore=$6 extra=$7
  local d="run/${label}_r${ranks}_k${kpar}_n${ncore}"
  rm -rf "$d"; mkdir -p "$d"
  cp "$src/POSCAR.bench" "$d/POSCAR"
  cp "$src/KPOINTS" "$d/"
  { grep -viE "^(KPAR|NCORE|NPAR|NSW)" "$src/INCAR.$incar"; echo "KPAR = $kpar"; echo "NCORE = $ncore"; echo "$extra"; } > "$d/INCAR"
  : > "$d/POTCAR"
  while read -r rel; do [ -n "$rel" ] && cat "$VASP_PP_PATH/$rel" >> "$d/POTCAR"; done < "$src/POTCAR.list"
  local t0 t1
  t0=$(date +%s.%N)
  ( cd "$d" && mpirun -n "$ranks" "$VASP_EXE" > vasp.out 2> vasp.err )
  t1=$(date +%s.%N)
  local ok="ok"
  grep -q "General timing" "$d/OUTCAR" 2>/dev/null || ok="FAILED"
  local kpts irr
  irr=$(grep -m1 "irreducible k-points" "$d/OUTCAR" 2>/dev/null | awk '{print $2}')
  printf "%-10s ranks=%-3s KPAR=%-2s NCORE=%-2s  %8.1f s  irrk=%-5s %s\n" \
    "$label" "$ranks" "$kpar" "$ncore" "$(echo "$t1 - $t0" | bc)" "${irr:-?}" "$ok"
  rm -f "$d/POTCAR" "$d/CHG" "$d/CHGCAR" "$d/WAVECAR"
}

echo "node $(hostname), $(nproc) cores"
echo "=== case A: E1 static single point (small cell, symmetry-folded k-mesh) ==="
for combo in "8 1 1" "8 1 2" "8 1 4" "8 2 1" "8 2 2" "8 4 1" "8 4 2" "8 8 1" "4 4 1" "16 8 1"; do
  set -- $combo; run_case A "$BENCH_A" static_soft "$1" "$2" "$3" ""
done
echo
echo "=== case B: E4 relaxation, two ionic steps (metal, unfolded k-mesh) ==="
for combo in "8 1 1" "8 2 2" "8 4 1" "8 4 2" "8 8 1" "16 8 2"; do
  set -- $combo; run_case B "$BENCH_B" relax_cell "$1" "$2" "$3" "NSW = 2"
done
echo "done"
