#!/usr/bin/env bash
# Comparison of a run_extraction.sh run with the reference centerlines (compare.py):
#   ASOCA                                ours (ASOCA masks)      vs ASOCA centerlines
#   ImageCAS-X_vs_ours_from_ImageCAS     ours (ImageCAS masks)   vs ImageCAS-X centerlines
#   ImageCAS-X_vs_ours_from_ImageCAS-X   ours (ImageCAS-X masks) vs ImageCAS-X centerlines
# Results in <run folder>/compare/<name>/. Uses the lists written by run_extraction.sh
# (<run folder>/lists). Can be launched from any folder.
#
#   bash run_compare.sh -o <run folder> [--use-shards [K]] [--resume] [--comparisons "..."]
#                       [-- <extra compare.py options>]
#
#   -o, --output      the run folder of run_extraction.sh; default results/makecenterlines_v2_final
#   --use-shards [K]  run up to K comparisons at the same time; 2 if K is omitted
#   --resume          skip the comparisons that already have summary.json
#   --comparisons     subset of the three names above, quoted
#   -- ...            passed to every compare.py (e.g. -- --no-figures)
#
# Logs: <run folder>/logs/compare.<name>.log. Exit status 1 if any comparison failed
# Refreshes the run's inspection report at the end (<run folder>/inspection.html, see run_review.sh).
# (compare.py also exits 1 when a case could not be compared: see its flags.csv).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../../../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT=""; JOBS=1; RESUME=0; EXTRA=()
COMPARISONS="ASOCA ImageCAS-X_vs_ours_from_ImageCAS ImageCAS-X_vs_ours_from_ImageCAS-X"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output) OUT="$2"; shift 2 ;;
        --use-shards)
            if [[ $# -gt 1 && "$2" =~ ^[0-9]+$ ]]; then JOBS="$2"; shift 2; else JOBS=2; shift; fi ;;
        --resume) RESUME=1; shift ;;
        --comparisons) COMPARISONS="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

OUT="$(realpath -m "${OUT:-$CODE/results/makecenterlines_v2_final}")"   # relative to where you launch from
cd "$CODE"
L="$OUT/lists"
[[ -d "$L" ]] || { echo "no $L: run run_extraction.sh first (it writes the lists)" >&2; exit 2; }
mkdir -p "$OUT/logs/.status" "$OUT/compare"

for name in $COMPARISONS; do
    case "$name" in
        ASOCA)
            args=(--pred "$L/compare_ASOCA_pred.txt" --ref "$L/compare_ASOCA_ref.txt" --ostia "$L/compare_ASOCA_ostia.txt"
                  --images "$L/compare_ASOCA_images.txt" --case-ids "$L/compare_ASOCA_ids.txt") ;;
        ImageCAS-X_vs_ours_from_ImageCAS|ImageCAS-X_vs_ours_from_ImageCAS-X)
            source_dataset="${name#ImageCAS-X_vs_ours_from_}"
            args=(--pred "$L/compare_ImageCAS-X_pred_from_${source_dataset}.txt" --ref "$L/compare_ImageCAS-X_ref.txt"
                  --ostia "$L/compare_ImageCAS-X_ostia.txt" --case-ids "$L/compare_ImageCAS-X_ids.txt") ;;
        *) echo "unknown comparison: $name" >&2; exit 2 ;;
    esac
    if [[ ! -s "${args[1]}" ]]; then echo "WARNING: ${args[1]} is empty, $name skipped" >&2; continue; fi
    if [[ $RESUME -eq 1 && -f "$OUT/compare/$name/summary.json" ]]; then echo "== $name: already compared, skipped (--resume)"; continue; fi
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
    echo "== $name: comparing $(wc -l < "${args[1]}") cases"
    ( "$PYTHON" -m hcatostialocalization.data.makecenterlines.compare "${args[@]}" -o "$OUT/compare/$name" "${EXTRA[@]+"${EXTRA[@]}"}" \
          > "$OUT/logs/compare.$name.log" 2>&1
      echo $? > "$OUT/logs/.status/compare.$name" ) &
done
wait

status=0
for name in $COMPARISONS; do
    f="$OUT/logs/.status/compare.$name"
    [[ -f "$f" ]] || continue
    if [[ "$(cat "$f")" != "0" ]]; then echo "   $name: exit $(cat "$f"), see $OUT/logs/compare.$name.log and $OUT/compare/$name/flags.csv"; status=1
    else echo "   $name: $OUT/compare/$name/report.html"; fi
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
