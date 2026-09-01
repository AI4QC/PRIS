#!/usr/bin/env bash
# How many tasks should share one node?
#
# A sweep measured with the node to itself says 8 ranks is the most efficient use of
# cores, but that is not the question. In production several VASP runs share one node's
# memory bandwidth, and each is slower than it was alone. This measures each packing the
# way it would actually run: N copies started together, timed until the last one finishes.
# The number that decides the layout is tasks per node-hour under contention.
#
#SBATCH --job-name=pris-pack
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=pack-%j.out
#SBATCH --error=pack-%j.err

set -uo pipefail
cd "${BENCH_DIR:?}" || exit 1
VASP_EXE=${VASP_EXE:-$HOME/vasp.6.3.0/bin/vasp_std}
export VASP_PP_PATH=${VASP_PP_PATH:-$HOME/POTCAR/PBE}

if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    [ -r "$f" ] && . "$f" && break
  done
fi
module purge >/dev/null 2>&1
module load "${ONEAPI_MODULE:-intel/oneapi2023.2_impi}" || exit 1
command -v mpirun >/dev/null 2>&1 || { echo "no mpirun" >&2; exit 1; }
export OMP_NUM_THREADS=1 I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm

prep() {
  local d=$1 kpar=$2 ncore=$3
  rm -rf "$d"; mkdir -p "$d"
  cp "$BENCH_A/POSCAR.bench" "$d/POSCAR"; cp "$BENCH_A/KPOINTS" "$d/"
  { grep -viE "^(KPAR|NCORE|NPAR)" "$BENCH_A/INCAR.static_soft"; echo "KPAR = $kpar"; echo "NCORE = $ncore"; } > "$d/INCAR"
  : > "$d/POTCAR"
  while read -r rel; do [ -n "$rel" ] && cat "$VASP_PP_PATH/$rel" >> "$d/POTCAR"; done < "$BENCH_A/POTCAR.list"
}

echo "node $(hostname), cores $(grep -c ^processor /proc/cpuinfo)"
printf "%-28s %10s %10s %14s\n" "packing" "wall s" "per task" "tasks/node-h"
for spec in "1 64 2 8" "2 32 1 8" "4 16 1 8" "8 8 1 8" "16 4 1 4"; do
  set -- $spec; n=$1; c=$2; kpar=$3; ncore=$4
  for w in $(seq 0 $((n - 1))); do prep "run/P_${n}x${c}_w${w}" "$kpar" "$ncore"; done
  t0=$(date +%s.%N)
  for w in $(seq 0 $((n - 1))); do
    ( cd "run/P_${n}x${c}_w${w}" \
      && I_MPI_PIN_PROCESSOR_LIST="$((w * c))-$((w * c + c - 1))" \
         mpirun -n "$c" "$VASP_EXE" > vasp.out 2> vasp.err ) &
  done
  wait
  t1=$(date +%s.%N)
  s=$(echo "$t1 - $t0" | bc)
  bad=0
  for w in $(seq 0 $((n - 1))); do
    grep -q "General timing" "run/P_${n}x${c}_w${w}/OUTCAR" 2>/dev/null || bad=$((bad + 1))
  done
  printf "%-28s %10.1f %10.1f %14s  %s\n" \
    "${n} task(s) x ${c} cores" "$s" "$(echo "scale=1; $s / 1" | bc)" \
    "$(echo "scale=1; 3600 / $s * $n" | bc)" \
    "$([ $bad -eq 0 ] && echo ok || echo "$bad FAILED")"
  rm -f run/P_${n}x${c}_w*/POTCAR run/P_${n}x${c}_w*/CHG* run/P_${n}x${c}_w*/WAVECAR
done
echo done
