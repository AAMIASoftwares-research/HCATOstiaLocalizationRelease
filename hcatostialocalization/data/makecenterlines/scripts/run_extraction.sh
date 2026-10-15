#!/usr/bin/env bash
# Complete v2 centerline extraction on ASOCA, ImageCAS and ImageCAS-X: builds the input lists,
# then runs the three datasets one after the other, each split over the requested number of
# parallel jobs (shards). Can be launched from any folder.
#
#   bash run_extraction.sh -o <run folder> [--use-shards [K]] [--resume] [--data <data root>]
#                          [--datasets "ASOCA ImageCAS ImageCAS-X"] [-- <extra v2 options>]
#
#   -o, --output      run folder (one subfolder per dataset, plus lists/ and logs/);
#                     default results/makecenterlines_v2_final; a relative path is taken from the folder you launch from
#   --use-shards [K]  K parallel jobs per dataset (K = available cores); 2 if K is omitted;
#                     without this option, one job
#   --resume          continue an interrupted run in the same folder: same lists, same shard
#                     count and datasets as the first launch (read from <run folder>/run_config.env);
#                     every job restarts after its last processed case, redoing the last 2
#   --data            data root, default /path/to/data
#   --datasets        subset of the three datasets, quoted
#   -- ...            anything after -- is passed to every v2 job (e.g. -- --trim-endpoints-mm 3)
#
# Logs: <run folder>/logs/<dataset>.shard<i>of<K>.log. Exit status 1 if any job failed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../../../.." && pwd)"          # repository root: python -m must run from here
PYTHON="${PYTHON:-python}"
OUT=""; SHARDS=1; RESUME=0
DATA="/path/to/data"; DATASETS="ASOCA ImageCAS ImageCAS-X"; EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output) OUT="$2"; shift 2 ;;
        --use-shards)
            if [[ $# -gt 1 && "$2" =~ ^[0-9]+$ ]]; then SHARDS="$2"; shift 2; else SHARDS=2; shift; fi ;;
        --resume) RESUME=1; shift ;;
        --data) DATA="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done
[[ "$SHARDS" -ge 1 ]] || { echo "--use-shards needs K >= 1" >&2; exit 2; }

OUT="$(realpath -m "${OUT:-$CODE/results/makecenterlines_v2_final}")"   # relative to where you launch from
cd "$CODE"
CONFIG="$OUT/run_config.env"
if [[ $RESUME -eq 1 ]]; then
    [[ -f "$CONFIG" ]] || { echo "--resume: no $CONFIG (was the run started with this script?)" >&2; exit 2; }
    # shard count, datasets, data root and extra options of the first launch
    source "$CONFIG"
    echo "Resuming $OUT: $SHARDS shard(s) per dataset, datasets: $DATASETS"
else
    mkdir -p "$OUT"
    {
        printf 'SHARDS=%q\n' "$SHARDS"; printf 'DATASETS=%q\n' "$DATASETS"; printf 'DATA=%q\n' "$DATA"
        printf 'EXTRA=('; for x in "${EXTRA[@]+"${EXTRA[@]}"}"; do printf '%q ' "$x"; done; printf ')\n'
    } > "$CONFIG"
    bash "$HERE/../collections/make_lists.sh" "$DATA" "$OUT" "$OUT/lists" || exit 1
fi
L="$OUT/lists"; mkdir -p "$OUT/logs"

status=0
for d in $DATASETS; do
    artifacts=(graph qc debug_figures markup-3d-slicer)   # Slicer curves for every dataset, to inspect critical cases
    if [[ ! -s "$L/${d}_all_masks.txt" ]]; then echo "WARNING: no cases in $L/${d}_all_masks.txt, $d skipped" >&2; continue; fi
    echo "== $d: $(wc -l < "$L/${d}_all_masks.txt") cases, $SHARDS job(s)"
    pids=()
    for ((i = 0; i < SHARDS; i++)); do
        args=(-m hcatostialocalization.data.makecenterlines.v2
              --masks "$L/${d}_all_masks.txt" --ostia "$L/${d}_all_ostia.txt"
              -o "$OUT/$d" -a "${artifacts[@]}" --smooth-graph)
        [[ $SHARDS -gt 1 ]] && args+=(--shard "$i/$SHARDS")
        [[ $RESUME -eq 1 ]] && args+=(--resume)
        args+=("${EXTRA[@]+"${EXTRA[@]}"}")
        log="$OUT/logs/$d.shard${i}of${SHARDS}.log"
        [[ $RESUME -eq 1 ]] && echo "---- resumed $(date -Is)" >> "$log"
        "$PYTHON" "${args[@]}" >> "$log" 2>&1 &
        pids+=($!)
    done
    for k in "${!pids[@]}"; do
        if ! wait "${pids[$k]}"; then
            echo "   shard $k/$SHARDS of $d ended with failures or errors: see $OUT/logs/$d.shard${k}of${SHARDS}.log"
            status=1
        fi
    done
    echo "   $d done: $(ls "$OUT/$d/qc" 2>/dev/null | wc -l) ok, $(ls "$OUT/$d/errors" 2>/dev/null | wc -l) failed"
done
exit $status
