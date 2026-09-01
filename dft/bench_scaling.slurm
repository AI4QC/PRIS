#!/usr/bin/env bash
# How does one VASP run scale from 8 to 64 ranks on one node, and what does running
# several at once cost each of them?
#
# The first sweep gives the best KPAR/NCORE at each rank count, measured with the node to
# itself. The second measures the thing an isolated sweep cannot see: eight concurrent
# runs share one node's memory bandwidth, so each is slower than when measured alone.
# Throughput per node-hour, not the single-run time, is what decides how to fill a node.
#
#SBATCH --job-name=pris-scale
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=scale-%j.out
#SBATCH --error=scale-%j.err

set -uo pipefail
cd "${BENCH_DIR:?}" || exit 1
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

prep() {   # prep <dir> <src> <incar> <kpar> <ncore> <extra>
  local d=$1 src=$2 incar=$3 kpar=$4 ncore=$5 extra=$6
  rm -rf "$d"; mkdir -p "$d"
  cp "$src/POSCAR.bench" "$d/POSCAR"; cp "$src/KPOINTS" "$d/"
  { grep -viE "^(KPAR|NCORE|NPAR|NSW)" "$src/INCAR.$incar"; echo "KPAR = $kpar"; echo "NCORE = $ncore"; echo "$extra"; } > "$d/INCAR"
  : > "$d/POTCAR"
  while read -r rel; do [ -n "$rel" ] && cat "$VASP_PP_PATH/$rel" >> "$d/POTCAR"; done < "$src/POTCAR.list"
}

echo "node $(hostname), cores $(grep -c ^processor /proc/cpuinfo), mask $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
echo
echo "=== sweep: one run alone on the node ==="
printf "%-6s %-6s %-7s %10s %12s\n" ranks KPAR NCORE "seconds" "tasks/node-h"
for combo in "8 1 4" "8 1 8" "16 1 4" "16 1 8" "16 2 4" "16 4 4" \
             "32 1 8" "32 2 8" "32 4 8" "32 8 4" \
             "64 1 8" "64 2 8" "64 4 8" "64 8 8" "64 4 16"; do
  set -- $combo; r=$1; k=$2; n=$3
  d="run/S_r${r}_k${k}_n${n}"
  prep "$d" "$BENCH_A" static_soft "$k" "$n" ""
  t0=$(date +%s.%N); ( cd "$d" && mpirun -n "$r" "$VASP_EXE" > vasp.out 2> vasp.err ); t1=$(date +%s.%N)
  s=$(echo "$t1 - $t0" | bc)
  ok=$(grep -q "General timing" "$d/OUTCAR" 2>/dev/null && echo ok || echo FAILED)
  # a node runs 64/r of these at once, so throughput is (64/r) runs per s seconds
  thr=$(echo "scale=1; 3600 / $s * 64 / $r" | bc)
  printf "%-6s %-6s %-7s %10.1f %12s  %s\n" "$r" "$k" "$n" "$s" "$thr" "$ok"
  rm -f "$d/POTCAR" "$d/CHG" "$d/CHGCAR" "$d/WAVECAR"
done

echo
echo "=== contention: eight 8-rank runs at once, versus the same run alone ==="
for w in $(seq 0 7); do
  d="run/C_w${w}"; prep "$d" "$BENCH_A" static_soft 1 4 ""
done
t0=$(date +%s.%N)
for w in $(seq 0 7); do
  ( cd "run/C_w${w}" && I_MPI_PIN_PROCESSOR_LIST="$((w*8))-$((w*8+7))" mpirun -n 8 "$VASP_EXE" > vasp.out 2> vasp.err ) &
done
wait
t1=$(date +%s.%N)
s=$(echo "$t1 - $t0" | bc)
thr=$(echo "scale=1; 3600 / $s * 8" | bc)
printf "8 concurrent 8-rank runs: wall %.1f s for 8 tasks -> %s tasks/node-h\n" "$s" "$thr"
for w in 0 7; do printf "   worker %s elapsed: %s s\n" "$w" "$(grep 'Elapsed time' run/C_w${w}/OUTCAR 2>/dev/null | awk '{print $NF}')"; done
rm -f run/C_w*/POTCAR
echo "done"
