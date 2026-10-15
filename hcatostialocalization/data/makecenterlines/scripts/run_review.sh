#!/usr/bin/env bash
# Review of a finished (or partial) run_extraction.sh run: v2.review on every dataset, which
# writes review.csv and review.html into <run folder>/<dataset>/, then the inspection report of the whole run,
# <run folder>/inspection.html (makecenterlines/inspection.py; run_compare.sh refreshes it with the ASOCA
# comparison). Can be launched from any folder.
#
#   bash run_review.sh -o <run folder> [--use-shards [K]] [--resume] [--audit N]
#                      [--datasets "ASOCA ImageCAS ImageCAS-X"] [-- <extra review options>]
#
#   -o, --output      the run folder of run_extraction.sh; default results/makecenterlines_v2_final
#   --use-shards [K]  review up to K datasets at the same time; 2 if K is omitted
#   --resume          skip the datasets that already have review.csv
#   --audit N         random unflagged cases added to the gallery (default 50)
#   --datasets        subset, quoted; default: the datasets of the run (run_config.env), else all three
#
# Logs: <run folder>/logs/review.<dataset>.log. Exit status 1 if any review failed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../../../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT=""; JOBS=1; RESUME=0; AUDIT=50; DATASETS_ARG=""; EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output) OUT="$2"; shift 2 ;;
        --use-shards)
            if [[ $# -gt 1 && "$2" =~ ^[0-9]+$ ]]; then JOBS="$2"; shift 2; else JOBS=2; shift; fi ;;
        --resume) RESUME=1; shift ;;
        --audit) AUDIT="$2"; shift 2 ;;
        --datasets) DATASETS_ARG="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

OUT="$(realpath -m "${OUT:-$CODE/results/makecenterlines_v2_final}")"   # relative to where you launch from
cd "$CODE"
DATASETS="ASOCA ImageCAS ImageCAS-X"
[[ -f "$OUT/run_config.env" ]] && DATASETS="$(source "$OUT/run_config.env"; echo "$DATASETS")"
[[ -n "$DATASETS_ARG" ]] && DATASETS="$DATASETS_ARG"
mkdir -p "$OUT/logs" "$OUT/logs/.status"

for d in $DATASETS; do
    if [[ ! -d "$OUT/$d/qc" ]]; then echo "WARNING: no $OUT/$d/qc, $d skipped" >&2; continue; fi
    if [[ $RESUME -eq 1 && -f "$OUT/$d/review.csv" ]]; then echo "== $d: already reviewed, skipped (--resume)"; continue; fi
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
    echo "== $d: reviewing"
    ( "$PYTHON" -m hcatostialocalization.data.makecenterlines.v2.review "$OUT/$d" --audit "$AUDIT" "${EXTRA[@]+"${EXTRA[@]}"}" \
          > "$OUT/logs/review.$d.log" 2>&1
      echo $? > "$OUT/logs/.status/review.$d" ) &
done
wait

status=0
for d in $DATASETS; do
    f="$OUT/logs/.status/review.$d"
    [[ -f "$f" ]] || continue
    if [[ "$(cat "$f")" != "0" ]]; then echo "   $d: review failed, see $OUT/logs/review.$d.log"; status=1
    else echo "   $d: $OUT/$d/review.html"; fi
    rm -f "$f"
done

# The inspection report gathers every case worth a visual check before the release (all datasets, plus the
# ASOCA comparison when it exists): <run folder>/inspection.html.
if "$PYTHON" -m hcatostialocalization.data.makecenterlines.inspection --run "$OUT" > "$OUT/logs/inspection.log" 2>&1; then
    tail -n 1 "$OUT/logs/inspection.log"
else
    echo "   inspection report failed, see $OUT/logs/inspection.log"; status=1
fi
exit $status
