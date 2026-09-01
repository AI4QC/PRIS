#!/usr/bin/env bash
# ELEMENTA 下载:curl 断点续传版。
#
# 为什么不用 hf download:它默认走 xet 后端,实测在 8.68/16.5 GiB 处退化到 30 KB/s
# (进程活着、有 socket、有 CPU,但基本不传数据),按那速度剩余部分要 74 小时。
# 同一时刻普通 HTTP CDN 通道实测 4.4 MB/s,快 147 倍。
#
# 断点来源:hf 已下的 .cache/huggingface/download/*.incomplete。已验证两个文件
# 表观大小 == 实占块(dense,无空洞),即 xet 是顺序写的,可以当作合法前缀直接接。
#
# 用法:nohup ./fetch_elementa.sh > fetch.log 2>&1 &

set -u

DIR=<path>
BASE=https://huggingface.co/datasets/kairosmaterial/ELEMENTA/resolve/main

# 文件名 → 期望字节数(来自 HTTP x-linked-size,已核对)
declare -A WANT=(
    [ELEMENTA_open.extxyz.tar.zst]=14613451503
    [ELEMENTA_spin.extxyz.tar.zst]=3097446799
)
# 现有断点 → 目标文件(按大小唯一确定:6.18 GiB 只能是 open,2.50 GiB 只能是 spin)
declare -A SEED=(
    [ELEMENTA_open.extxyz.tar.zst]=mqE4JW6JpfGqu7wKUUBxureqnQ0=.ebb6898f3c322e5c9ae507c043827afe57a8c90219a873e5ff59a16a5a4f2309.incomplete
    [ELEMENTA_spin.extxyz.tar.zst]=k8jmaLpJTp7MwA_94ca-mkw7Z9s=.53f6e7563bae7ce7ba0e437058ef5eca88dc817d645ba180773b53d54bc81cbb.incomplete
)

MAX_TRY=60
cd "$DIR" || { echo "目录不存在: $DIR"; exit 1; }

for f in "${!WANT[@]}"; do
    want=${WANT[$f]}

    # 首次:把 hf 的断点搬过来当种子
    if [ ! -f "$f" ]; then
        seed=".cache/huggingface/download/${SEED[$f]}"
        if [ -f "$seed" ]; then
            echo "[$(date +%T)] $f ← 接管断点 $(stat -c %s "$seed") 字节"
            mv "$seed" "$f"
        else
            echo "[$(date +%T)] $f ← 无断点,从头下"
            : > "$f"
        fi
    fi

    for try in $(seq 1 $MAX_TRY); do
        have=$(stat -c %s "$f" 2>/dev/null || echo 0)
        if [ "$have" -ge "$want" ]; then break; fi
        pct=$(python3 -c "print(f'{100*$have/$want:.1f}')")
        echo "[$(date +%T)] $f 第 ${try} 次:已有 ${have}/${want} (${pct}%)"

        # --speed-limit/--speed-time:连续 60 秒低于 100 KB/s 就断开重来,专治静默卡死
        curl -sL -C - "$BASE/$f" -o "$f" \
             --speed-limit 102400 --speed-time 60 \
             --retry 5 --retry-delay 5 --retry-all-errors
        rc=$?
        [ $rc -ne 0 ] && echo "[$(date +%T)]   curl 退出码 $rc,重试"
        sleep 2
    done

    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$have" -eq "$want" ]; then
        echo "[$(date +%T)] ✓ $f 完成,字节数精确匹配 ($want)"
    else
        echo "[$(date +%T)] ✗ $f 未完成:${have}/${want}"
        exit 1
    fi
done

echo "[$(date +%T)] 两个文件都已下完,开始完整性校验(zstd 全流解压,不落盘)"
for f in "${!WANT[@]}"; do
    if zstd -t "$f" 2>/dev/null; then
        echo "[$(date +%T)] ✓ $f zstd 校验通过"
    else
        echo "[$(date +%T)] ✗ $f zstd 校验失败 —— 断点前缀可能有问题,需要重下"
        exit 1
    fi
done
echo "[$(date +%T)] 全部完成"
