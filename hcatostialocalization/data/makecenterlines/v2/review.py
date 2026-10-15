"""
Batch review of a v2 run: ranks every case by how likely it is to miss a
branch, so only the flagged cases (plus a random audit of the rest) need
opening by hand.

    python -m hcatostialocalization.data.makecenterlines.v2.review <output-dir> [--audit 50]

Reads <output-dir>/qc/*.qc.json, errors/ and batch_summary*.json, writes
<output-dir>/review.csv (every case with its category and reasons, most severe
first) and review.html (summary, then one gallery of targets_mip.png per
category). Every case falls in the most severe category that applies:

  1. failed: the extraction raised an error; or a mask piece at least
     --failed-extent-mm long could not be attached to the tree (no ostium reaches
     it, and it is farther than the gap-bridging limit): a big branch is missing;
     or a side's tree is shorter than --min-side-length-mm;
  2. likely missed branch: a mask region the final tree does not explain at least
     the QC's suspect extent long (10 mm by default), or an unattached mask piece
     shorter than --failed-extent-mm but at least that long;
  3. other checks (worth a look, not misses): a gap bridged, the left and right
     trees touching, an ostium more than 3 mm from the mask, a side whose length or
     endpoint count is below the batch's 5th percentile;
  4. clean: none of the above. --audit of them are drawn at random: if k of n
     audited clean cases have a miss, that is the miss rate to quote.

Separately (whatever the category), suspiciously short branches that were accepted:
every branch whose run outside the already-covered tube was shorter than
--short-branch-mm (the extraction keeps runs >= --min-branch-length-mm, 5 mm),
with its side, length and RAS position, so possible spurious stubs can be checked.
"""
import os
import csv
import sys
import glob
import json
import html
import random
import argparse

import numpy

_LOW_PERCENTILE = 5.0
_MIN_CASES_FOR_PERCENTILES = 20
_MAX_OSTIUM_TO_MASK_MM = 3.0


def _load(output_dir: str):
    cases = {}
    for path in sorted(glob.glob(os.path.join(output_dir, "qc", "*.qc.json"))):
        with open(path) as f:
            qc = json.load(f)
        cases[qc["case_id"]] = {"status": "ok", "qc": qc}
    # failures: per-case error files (written as they happen) and every batch summary
    # (plain, per shard, per resume); a case with a qc file succeeded in a later run
    failures = []
    for path in sorted(glob.glob(os.path.join(output_dir, "errors", "*.error.json"))):
        with open(path) as f:
            failures.append(json.load(f))
    for path in sorted(glob.glob(os.path.join(output_dir, "batch_summary*.json"))):
        with open(path) as f:
            failures.extend(r for r in json.load(f) if r["status"] != "ok")
    for result in failures:
        if result["case_id"] not in cases:
            cases[result["case_id"]] = {"status": "error", "error": result.get("error", "")}
    return cases


CATEGORIES = ("failed", "likely_missed", "other_checks", "clean")
TITLES = {"failed": "Failed", "likely_missed": "Likely missed branches", "other_checks": "Other checks", "clean": "Clean"}


def _row(case_id: str, case: dict, failed_extent_mm: float, min_side_length_mm: float, short_branch_mm: float) -> dict:
    row = {"case_id": case_id, "status": case["status"], "reasons": [], "failed": [], "likely_missed": [], "other_checks": [],
           "short_accepted": []}
    if case["status"] != "ok":
        row["failed"].append(f"extraction error: {case['error']}")
        return row
    qc = case["qc"]
    suspect_mm = qc.get("suspect_missed_branch", {}).get("extent_threshold_mm", 10.0)
    for region in qc["unreachable"].get("largest", []):
        extent = region.get("extent_mm", 0.0)
        gap = region.get("distance_to_tree_mm", {}).get("min")
        where = f" ({gap:.0f} mm from the tree)" if gap is not None else ""
        if extent >= failed_extent_mm:
            row["failed"].append(f"mask piece {extent:.0f} mm long not attached{where}")
        elif extent >= suspect_mm:
            row["likely_missed"].append(f"mask piece {extent:.0f} mm long not attached{where}")
    affine = numpy.asarray(qc["working_grid"]["affine_ijk_to_ras"], dtype=float)
    for side in ("L", "R"):
        for branch in qc["sides"][side].get("accepted", []):
            if branch["length_mm"] < short_branch_mm:
                ras = affine[:3, :3] @ numpy.asarray(branch["target_ijk"], dtype=float) + affine[:3, 3]
                row["short_accepted"].append(f"{side}: {branch['length_mm']:.1f} mm, tip at RAS ({ras[0]:.1f}, {ras[1]:.1f}, {ras[2]:.1f})")
    for side in ("L", "R"):
        s = qc["sides"][side]
        if s["length_mm"] < min_side_length_mm:
            row["failed"].append(f"{side}: tree only {s['length_mm']:.0f} mm long")
        u = s["unexplained"]
        if u.get("n_suspect"):
            row["likely_missed"].append(f"{side}: {u['n_suspect']} mask region(s) not explained by the tree, up to {u['max_extent_mm']:.0f} mm")
    for side in ("L", "R"):
        s = qc["sides"][side]
        row[f"length_{side}_mm"] = s["length_mm"]
        row[f"endpoints_{side}"] = s["n_endpoints"]
        row[f"unexplained_max_extent_{side}_mm"] = s["unexplained"]["max_extent_mm"]
        row[f"ostium_to_mask_{side}_mm"] = s["ostium_to_mask_mm"]
        row[f"shell_voxels_{side}"] = s["n_shell_voxels_used"]
        if s["ostium_to_mask_mm"] > _MAX_OSTIUM_TO_MASK_MM:
            row["other_checks"].append(f"{side}: mask {s['ostium_to_mask_mm']:.1f} mm from the ostium")
    row["unreachable_max_extent_mm"] = qc["unreachable"]["max_extent_mm"]
    row["unreachable_volume_mm3"] = qc["unreachable"]["volume_mm3"]
    bridges = qc.get("gap_bridges", {}).get("bridges", [])
    row["gap_bridges"] = len(bridges)
    row["gap_bridge_max_mm"] = max((b["gap_mm"] for b in bridges), default=0.0)
    if bridges:  # straight links across mask interruptions: worth a look in Slicer
        row["other_checks"].append(f"{len(bridges)} mask gap(s) bridged, up to {row['gap_bridge_max_mm']:.1f} mm")
    row["merged_component"] = qc["merged_component"]
    if qc["merged_component"]:
        row["other_checks"].append("left and right trees touch in the mask")
    return row


def _categorise(row: dict) -> None:
    """Sets row['category'] (the most severe that applies) and row['reasons'] (all of them, prefixed). In place."""
    row["category"] = next((c for c in CATEGORIES[:3] if row[c]), "clean")
    row["reasons"] = [f"{TITLES[c].upper()}: {x}" for c in CATEGORIES[:3] for x in row[c]]


def _batch_outliers(rows: list) -> None:
    """Adds a reason to every side whose length or endpoint count is below the
    batch's low percentile. In place."""
    ok = [r for r in rows if r["status"] == "ok"]
    if len(ok) < _MIN_CASES_FOR_PERCENTILES:
        return
    for side in ("L", "R"):
        for key, label in ((f"length_{side}_mm", "total length"), (f"endpoints_{side}", "endpoint count")):
            threshold = float(numpy.percentile([r[key] for r in ok], _LOW_PERCENTILE))
            for r in ok:
                if r[key] < threshold:
                    r["other_checks"].append(f"{side}: {label} {r[key]:.0f} below the batch p{_LOW_PERCENTILE:.0f} ({threshold:.0f})")


def _sort_key(row: dict):
    extent = max(row.get("unexplained_max_extent_L_mm", 0.0), row.get("unexplained_max_extent_R_mm", 0.0),
                 row.get("unreachable_max_extent_mm", 0.0))
    return (CATEGORIES.index(row["category"]), row["status"] == "ok", -extent, -len(row["reasons"]))


def _write_csv(rows: list, path: str) -> None:
    columns = ["case_id", "category", "status", "n_reasons", "reasons"] + sorted(
        {k for r in rows for k in r} - {"case_id", "category", "status", "reasons", "failed", "likely_missed", "other_checks", "short_accepted"}
    ) + ["n_short_accepted", "short_accepted"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "n_reasons": len(r["reasons"]), "reasons": " | ".join(r["reasons"]),
                             "n_short_accepted": len(r["short_accepted"]), "short_accepted": " | ".join(r["short_accepted"])})


def _write_html(rows: list, audit: list, path: str, short_branch_mm: float) -> None:
    def cards(selection, field="reasons"):
        out = []
        for r in selection:
            image = f"debug_figures/{r['case_id']}/targets_mip.png"
            curves = f"markup-3d-slicer/{r['case_id']}/"
            reasons = "".join(f"<li>{html.escape(x)}</li>" for x in r[field]) or "<li>no flag</li>"
            out.append(
                f'<div class="card {r["category"]}"><h3>{html.escape(r["case_id"])}</h3><ul>{reasons}</ul>'
                f'<a href="{image}"><img src="{image}" loading="lazy" alt="targets_mip {html.escape(r["case_id"])}"></a>'
                f'<p class="small">Slicer curves: <code>{html.escape(curves)}</code></p></div>'
            )
        return "\n".join(out) or "<p>none</p>"

    counts = {c: sum(r["category"] == c for r in rows) for c in CATEGORIES}
    n = len(rows)
    short = sorted((r for r in rows if r["short_accepted"]), key=lambda r: (-len(r["short_accepted"]), r["case_id"]))
    summary = "".join(f"<tr><td>{TITLES[c]}</td><td>{counts[c]}</td><td>{100 * counts[c] / n:.1f} %</td></tr>" for c in CATEGORIES)
    sections = "".join(
        f'<h2 id="{c}">{TITLES[c]} ({counts[c]})</h2><div class="grid">{cards([r for r in rows if r["category"] == c])}</div>'
        for c in CATEGORIES[:3]
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Centerline review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
table {{ border-collapse: collapse; margin-bottom: 12px; }} td, th {{ border-bottom: 1px solid #ddd; padding: 4px 10px; text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #ccc; border-left-width: 5px; border-radius: 6px; padding: 8px; }}
.card.failed {{ border-left-color: #c62828; }} .card.likely_missed {{ border-left-color: #ef6c00; }}
.card.other_checks {{ border-left-color: #1565c0; }} .card.clean {{ border-left-color: #9e9e9e; }}
.card h3 {{ margin: 0 0 4px; }} .card ul {{ margin: 0 0 6px; padding-left: 18px; font-size: 13px; }}
.card img {{ width: 100%; }} .small {{ font-size: 12px; color: #555; margin: 4px 0 0; }}
</style></head><body>
<h1>Centerline review</h1>
<table><tr><th>Category</th><th>Cases</th><th>Share</th></tr>{summary}</table>
<p>Every case is in the most severe category that applies (see review.csv for all reasons). Magenta crosses in the
images mark the suspect regions; the Slicer curves of each case are in the folder shown under its image.</p>
{sections}
<h2 id="short">Suspiciously short branches accepted (&lt; {short_branch_mm:g} mm outside the covered tube): {len(short)} cases, {sum(len(r["short_accepted"]) for r in short)} branches</h2>
<p>Most cases first. Each line: side, length, and the tip position in RAS (type it in Slicer's markup or crosshair to jump there).</p>
<div class="grid">{cards(short, "short_accepted")}</div>
<h2 id="audit">Audit: {len(audit)} clean cases drawn at random</h2>
<p>Check each for a missed branch; k misses out of {len(audit)} is the miss rate among clean cases.</p>
<div class="grid">{cards(audit)}</div>
</body></html>"""
    with open(path, "w") as f:
        f.write(page)


def main():
    parser = argparse.ArgumentParser(description="Rank the cases of a v2 run by how likely they are to miss a branch.")
    parser.add_argument("output_dir", help="The --output-dir of the v2 run (needs qc/; debug_figures/ for the gallery).")
    parser.add_argument("--audit", type=int, default=50, help="Number of unflagged cases drawn at random for the audit.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the audit draw.")
    parser.add_argument("--failed-extent-mm", type=float, default=20.0, help="An unattached mask piece at least this long makes the case 'failed' (a big branch is missing); shorter ones (down to the QC suspect extent) are 'likely missed'.")
    parser.add_argument("--short-branch-mm", type=float, default=10.0, help="Accepted branches whose run outside the covered tube is shorter than this are listed as suspiciously short.")
    parser.add_argument("--min-side-length-mm", type=float, default=20.0, help="A side whose tree is shorter than this makes the case 'failed'.")
    args = parser.parse_args()

    cases = _load(args.output_dir)
    if not cases:
        parser.error(f"no qc/*.qc.json in {args.output_dir}")
    rows = [_row(case_id, case, args.failed_extent_mm, args.min_side_length_mm, args.short_branch_mm) for case_id, case in cases.items()]
    _batch_outliers(rows)
    for r in rows:
        _categorise(r)
    rows.sort(key=_sort_key)

    clean = sorted((r for r in rows if r["category"] == "clean"), key=lambda r: r["case_id"])
    audit = random.Random(args.seed).sample(clean, min(args.audit, len(clean)))
    for r in rows:
        r["in_audit"] = r in audit

    _write_csv(rows, os.path.join(args.output_dir, "review.csv"))
    _write_html(rows, audit, os.path.join(args.output_dir, "review.html"), args.short_branch_mm)
    counts = {c: sum(r["category"] == c for r in rows) for c in CATEGORIES}
    print(f"{len(rows)} cases: " + ", ".join(f"{counts[c]} {TITLES[c].lower()}" for c in CATEGORIES)
          + f"; {len(audit)} clean cases in the audit sample; "
          + f"{sum(len(r['short_accepted']) for r in rows)} short accepted branches in {sum(bool(r['short_accepted']) for r in rows)} cases.")
    print(f"Wrote {os.path.join(args.output_dir, 'review.csv')} and review.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
