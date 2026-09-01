#!/usr/bin/env bash
# One whole node runs many small VASP tasks side by side.
#
# The partitions here are OverSubscribe=EXCLUSIVE, so a job holds a whole 64-core node
# whatever it asks for, and the account allows only 64 jobs in flight. One array element
# per task would therefore be rejected outright and would waste most of every node. This
# script instead treats each array element as a worker pool that pulls tasks from the
# package's tasklist until it is empty.
#
# Claiming is a mkdir, which is atomic on Lustre, so several jobs can share one list
# without stepping on each other. A finished task drops a marker in .farm/done and is
# never recomputed; an interrupted one releases its claim so a later job resumes it, and
# run.sh itself skips any stage that already exited zero.
#
# Submit through submit_farm.sh, which sizes the array and sets the variables below.
#
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -uo pipefail
cd "${FARM_PKG:-$SLURM_SUBMIT_DIR}" || exit 1

: "${WORKERS:=8}"
: "${CORES_PER_WORKER:=8}"
: "${VASP_EXE:=$HOME/vasp.6.3.0/bin/vasp_std}"
: "${VASP_PP_PATH:=$HOME/POTCAR/PBE}"
: "${ONEAPI_MODULE:=intel/oneapi2023.2_impi}"


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
[ -x "$VASP_EXE" ] || { echo "no VASP at $VASP_EXE" >&2; exit 1; }

export VASP_PP_PATH
export OMP_NUM_THREADS=1
# every rank stays on this node: the farm runs many independent single-node jobs, so the
# launcher must not try to spread across the allocation
export I_MPI_HYDRA_BOOTSTRAP=fork
export I_MPI_FABRICS=shm

CLAIM=.farm/claim
DONE=.farm/done
DEAD=.farm/dead
ATTEMPTS=.farm/attempts
mkdir -p "$CLAIM" "$DONE" "$DEAD" "$ATTEMPTS" logs
# A task that keeps failing must stop being retried. Workers now re-sweep the list until
# it is dry, so without this a permanently broken cell would be picked up again and again
# and burn nodes indefinitely.
: "${FARM_MAX_ATTEMPTS:=3}"

# a task counts as finished only when every stage it declares exited zero
task_complete() {
  local t=$1 stage code
  [ -f "$t/stages.tsv" ] || return 1
  while IFS=$'\t' read -r stage _ _ _; do
    [ -z "$stage" ] && continue
    code=$(cat "$t/$stage/exit_code" 2>/dev/null)
    [ "$code" = "0" ] || return 1
  done < "$t/stages.tsv"
  return 0
}

worker() {
  local w=$1 first last name code
  first=$(( w * CORES_PER_WORKER ))
  last=$(( first + CORES_PER_WORKER - 1 ))
  export I_MPI_PIN_PROCESSOR_LIST="${first}-${last}"
  export VASP_BIN="mpirun -n ${CORES_PER_WORKER} ${VASP_EXE}"
  # Sweep the list repeatedly, not once. A task can become available after a worker has
  # already walked past it — a claim released by a killed job, or a stage cleared for a
  # retry — and a single pass would leave it for nobody. The sweep ends when a whole pass
  # claims nothing.
  while : ; do
   claimed_any=0
   while read -r t <&4; do
    [ -z "$t" ] && continue
    name=$(basename "$t")
    [ -e "$DONE/$name" ] && continue
    [ -e "$DEAD/$name" ] && continue                # given up on; see .farm/dead
    mkdir "$CLAIM/$name" 2>/dev/null || continue   # another worker already has it
    claimed_any=1
    if task_complete "$t"; then
      touch "$DONE/$name"; continue                # finished in an earlier submission
    fi
    echo "[w$w] $(date +%T) start $name"
    bash "$t/run.sh" > "logs/${name}.log" 2>&1 < /dev/null
    if task_complete "$t"; then
      touch "$DONE/$name"
      echo "[w$w] $(date +%T) done  $name"
    else
      # Keep the evidence. A failed stage is deleted before it is retried, so whatever
      # VASP said has to be copied out first or the reason is lost for good.
      mkdir -p .farm/failed
      while IFS=$'\t' read -r st _ _ _ <&5; do
        [ -z "$st" ] && continue
        [ "$(cat "$t/$st/exit_code" 2>/dev/null)" = "0" ] && continue
        { echo "### $name stage $st  $(date -Is)";
          echo "--- vasp.out tail:"; tail -40 "$t/$st/vasp.out" 2>/dev/null;
          echo "--- vasp.err tail:"; tail -20 "$t/$st/vasp.err" 2>/dev/null;
        } >> ".farm/failed/${name}.txt"
      done 5< "$t/stages.tsv"
      n_att=$(( $(cat "$ATTEMPTS/$name" 2>/dev/null || echo 0) + 1 ))
      echo "$n_att" > "$ATTEMPTS/$name"
      if [ "$n_att" -ge "$FARM_MAX_ATTEMPTS" ]; then
        echo "$name failed $n_att times; not retried again" > "$DEAD/$name"
        echo "[w$w] $(date +%T) GIVING UP on $name after $n_att attempts"
      else
        echo "[w$w] $(date +%T) INCOMPLETE $name, attempt $n_att (see .farm/failed/${name}.txt)"
      fi
      # release the claim so a later job can resume; run.sh skips whatever already passed
      rmdir "$CLAIM/$name" 2>/dev/null
    fi
   done 4< "$FARM_TASKLIST"
   [ "$claimed_any" -eq 0 ] && break
  done
}

echo "node $(hostname)  array ${SLURM_ARRAY_TASK_ID:-0}  workers $WORKERS x $CORES_PER_WORKER cores"
# Workers claim in list order, so a short critical task queued behind slow ones waits for
# them. FARM_TASKLIST points a run at a subset without editing the package's own list;
# claims and done markers are shared, so subset and full runs cannot collide.
: "${FARM_TASKLIST:=tasklist.txt}"
[ -f "$FARM_TASKLIST" ] || { echo "no task list at $FARM_TASKLIST" >&2; exit 1; }
export FARM_DONE_AT_START=$(find "$DONE" -maxdepth 1 -type f 2>/dev/null | wc -l)
echo "package $(pwd)  list $FARM_TASKLIST  tasks $(grep -c . "$FARM_TASKLIST")  already done $FARM_DONE_AT_START"
for w in $(seq 0 $((WORKERS - 1))); do worker "$w" & done
wait
done_now=$(find "$DONE" -maxdepth 1 -type f 2>/dev/null | wc -l)
total=$(grep -c . "$FARM_TASKLIST")
echo "array element ${SLURM_ARRAY_TASK_ID:-0} finished; $done_now/$total done, $(find "$DEAD" -maxdepth 1 -type f 2>/dev/null | wc -l) given up"

# Keep the package moving without a daemon. A login-node loop is killed sooner or later and
# cannot even be found reliably from another login node, so a job that still sees work left
# submits its own replacement before exiting. The in-flight ceiling is what stops this from
# running away, and a job that completed nothing does not requeue, so a package that cannot
# progress stops instead of spinning.
dead_now=$(find "$DEAD" -maxdepth 1 -type f 2>/dev/null | wc -l)
if [ "${FARM_REQUEUE:-1}" = "1" ] && [ "$(( done_now + dead_now ))" -lt "$total" ]; then
  claimed=$(find "$CLAIM" -maxdepth 1 -mindepth 1 2>/dev/null | wc -l)
  inflight=$(squeue -h -u "$USER" -r 2>/dev/null | wc -l)
  if [ "$claimed" -lt "$total" ] && [ "$inflight" -lt "${FARM_MAX_INFLIGHT:-20}" ]; then
    if [ "$done_now" -gt "${FARM_DONE_AT_START:-0}" ]; then
      echo "work remains and the queue has room; submitting a replacement"
      # The submitter lives at the root of the package tree, not beside the package: for
      # a nested package such as E4_design/stage_b, dirname points one level too deep.
      root=${FARM_ROOT:-$(dirname "$FARM_PKG")}
      ( cd "$root" && ./submit_farm.sh "${FARM_PKG#$root/}" 1 \
          "$WORKERS" "$CORES_PER_WORKER" "${SLURM_JOB_PARTITION:-deimos}" \
          "${FARM_TIME:-48:00:00}" ) 2>&1 | sed 's/^/    /'
    else
      echo "this element completed nothing; not requeueing"
    fi
  fi
fi
