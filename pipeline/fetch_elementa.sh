#!/usr/bin/env bash
# Downloading ELEMENTA: the resumable curl version.
#
# Why not hf download: it defaults to the xet backend, which was measured degrading to
# 30 KB/s at 8.68/16.5 GiB (the process is alive with a socket and CPU, but transfers almost
# nothing); at that rate the remainder would take 74 hours.
# At the same moment the plain HTTP CDN path measured 4.4 MB/s, 147x faster.
#
# The resume point comes from what hf already downloaded, in
# .cache/huggingface/download/*.incomplete. Both files were verified to have apparent size ==
# blocks actually occupied (dense, no holes), i.e. xet writes sequentially, so the partial
# file is a valid prefix and can be continued directly.
#
# Usage: nohup ./fetch_elementa.sh > fetch.log 2>&1 &

set -u

DIR=<path>
BASE=https://huggingface.co/datasets/kairosmaterial/ELEMENTA/resolve/main

# filename -> expected byte count (from the HTTP x-linked-size header, checked)
declare -A WANT=(
    [ELEMENTA_open.extxyz.tar.zst]=14613451503
    [ELEMENTA_spin.extxyz.tar.zst]=3097446799
)
# existing partial -> target file (determined uniquely by size: 6.18 GiB can only be open,
# 2.50 GiB can only be spin)
declare -A SEED=(
    [ELEMENTA_open.extxyz.tar.zst]=mqE4JW6JpfGqu7wKUUBxureqnQ0=.ebb6898f3c322e5c9ae507c043827afe57a8c90219a873e5ff59a16a5a4f2309.incomplete
    [ELEMENTA_spin.extxyz.tar.zst]=k8jmaLpJTp7MwA_94ca-mkw7Z9s=.53f6e7563bae7ce7ba0e437058ef5eca88dc817d645ba180773b53d54bc81cbb.incomplete
)

MAX_TRY=60
cd "$DIR" || { echo "directory does not exist: $DIR"; exit 1; }

for f in "${!WANT[@]}"; do
    want=${WANT[$f]}

    # first time: move hf's partial file over as the seed
    if [ ! -f "$f" ]; then
        seed=".cache/huggingface/download/${SEED[$f]}"
        if [ -f "$seed" ]; then
            echo "[$(date +%T)] $f <- taking over a partial file of $(stat -c %s "$seed") bytes"
            mv "$seed" "$f"
        else
            echo "[$(date +%T)] $f <- no partial file, downloading from the start"
            : > "$f"
        fi
    fi

    for try in $(seq 1 $MAX_TRY); do
        have=$(stat -c %s "$f" 2>/dev/null || echo 0)
        if [ "$have" -ge "$want" ]; then break; fi
        pct=$(python3 -c "print(f'{100*$have/$want:.1f}')")
        echo "[$(date +%T)] $f attempt ${try}: have ${have}/${want} (${pct}%)"

        # --speed-limit/--speed-time: drop and retry after 60 continuous seconds below
        # 100 KB/s, which is the cure for a silent stall
        curl -sL -C - "$BASE/$f" -o "$f" \
             --speed-limit 102400 --speed-time 60 \
             --retry 5 --retry-delay 5 --retry-all-errors
        rc=$?
        [ $rc -ne 0 ] && echo "[$(date +%T)]   curl exited $rc, retrying"
        sleep 2
    done

    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$have" -eq "$want" ]; then
        echo "[$(date +%T)] OK $f complete, byte count matches exactly ($want)"
    else
        echo "[$(date +%T)] FAIL $f incomplete: ${have}/${want}"
        exit 1
    fi
done

echo "[$(date +%T)] both files downloaded; starting the integrity check (full zstd decompression, nothing written to disk)"
for f in "${!WANT[@]}"; do
    if zstd -t "$f" 2>/dev/null; then
        echo "[$(date +%T)] OK $f passed the zstd check"
    else
        echo "[$(date +%T)] FAIL $f did not pass the zstd check -- the resumed prefix may be bad; download it again"
        exit 1
    fi
done
echo "[$(date +%T)] all done"
