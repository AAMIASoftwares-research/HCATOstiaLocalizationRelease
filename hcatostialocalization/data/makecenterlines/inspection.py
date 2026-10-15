"""
Inspection report of a complete centerline run: every case worth a visual check before the release, grouped by the
kind of possible trouble, with the debug figures inline, so the inspection can start in the browser.

    python -m hcatostialocalization.data.makecenterlines.inspection --run <run folder>
        [--notes release_case_notes.csv] [--suspect-mm 10]

Called at the end of scripts/run_review.sh and scripts/run_compare.sh (the ASOCA group needs the comparison), so it
is always up to date. Reads the run's qc/, review.csv, debug_figures/, graph/ and compare/; writes
<run>/inspection.html (open it from the run folder: images and links are relative) and <run>/inspection.csv (one
row per issue). Cases explained in the notes file are listed apart, as known.

Groups (ImageCAS masks, the official set, unless stated):
  0  failed in the review, all datasets
  1  left and right trees touch in the mask
  2  possibly missed branch: a mask region the tree does not explain, or unreachable, >= --suspect-mm
  3  ostium more than 3 mm from the mask
  4  mask gap bridged
  5  radius cap moved the path by more than 1 mm
  6  ASOCA: extra branch absent from the reference (needs run_compare.sh)
  7  ImageCAS-X graph that the release would recommend (release.recommend) but whose own review flagged it
"""
import argparse
import csv
import glob
import html
import json
import os
import sys

import numpy

from hcatostialocalization.data.makecenterlines.release import qc_label, recommend

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ("ASOCA", "ImageCAS", "ImageCAS-X")

GROUPS = {
    0: ("Failed in the review",
        "The extraction failed, a large mask piece could not be attached, or a tree is shorter than 20 mm. Confirm that "
        "the cause lies in the mask (or the ostium annotation), then note the explanation."),
    1: ("Left and right trees touch in the mask",
        "The two coronary trees never connect anatomically, but their masks touch here, and every shared voxel was given "
        "to the nearer ostium. Check where the two masks meet and that every vessel belongs to the right tree. The "
        "position is the midpoint of the closest pair of left and right nodes: a starting point, not necessarily the "
        "contact."),
    2: ("Possibly missed branch",
        "A mask region at least 10 mm long is not explained by the tree, or cannot be reached from either ostium (big "
        "magenta crosses in the figure). Decide whether it is a vessel the tree misses, or a blob or artefact of the "
        "mask."),
    3: ("Ostium more than 3 mm from the mask",
        "The ostium was connected to the mask through a straight bridge. Check whether the ostium annotation is right and "
        "whether the mask stops short of the aorta, and look at the first millimetres of the tree."),
    4: ("Mask gap bridged",
        "A disconnected mask piece was attached across a gap of up to 10 mm. Is it the continuation of a vessel "
        "interrupted by the segmentation (correct), or a fragment that does not belong to the tree (wrong)? The position "
        "is a voxel of the attached piece."),
    5: ("Radius cap moved the path by more than 1 mm",
        "The second pass changed the path at a bulky part of the mask, usually a bifurcation. Check that the final path "
        "is centred in the vessels there."),
    6: ("ASOCA: extra branch absent from the reference",
        "A branch of at least 10 mm traced in the ASOCA mask but not in the ASOCA reference centerlines. Usually a real "
        "side branch the reference does not trace; check that it is not spurious. The position is its start."),
    7: ("ImageCAS-X graph recommended, but flagged by its own review",
        "For these cases the release recommends the ImageCAS-X graph, because the ImageCAS one is worse. Check that the "
        "ImageCAS-X graph is really the better of the two (both figures are shown)."),
}
FIGURES = (("targets_mip.png", "trees and targets"), ("graph_3d_smoothed.png", "3D"),
           ("graph_2d_smoothed.png", "2D"), ("distance_map.png", "distance map"))


def _ras(qc, ijk):
    affine = numpy.asarray(qc["working_grid"]["affine_ijk_to_ras"], float)
    return [float(v) for v in (affine @ numpy.append(numpy.asarray(ijk, float), 1.0))[:3]]


def _fmt(p):
    return "" if p is None else "(" + ", ".join(f"{v:.1f}" for v in p) + ")"


def _case_key(case):
    return (0, int(case), "") if case.isdigit() else (1, 0, case)


def _closest_left_right(gml_path):
    """(distance mm, midpoint RAS) of the closest pair of left and right nodes of a graph, or (None, None)."""
    if not os.path.exists(gml_path):
        return None, None
    import networkx
    g = networkx.read_gml(gml_path, label="id")
    pos = {side: numpy.array([[d["x"], d["y"], d["z"]] for _, d in g.nodes(data=True) if d.get("side") == side])
           for side in ("LEFT", "RIGHT")}
    if not len(pos["LEFT"]) or not len(pos["RIGHT"]):
        return None, None
    dist = numpy.linalg.norm(pos["LEFT"][:, None, :] - pos["RIGHT"][None, :, :], axis=2)
    i, j = numpy.unravel_index(numpy.argmin(dist), dist.shape)
    return float(dist[i, j]), list((pos["LEFT"][i] + pos["RIGHT"][j]) / 2)


def _read_csv(path):
    return list(csv.DictReader(open(path, newline=""))) if os.path.exists(path) else []


def collect(run, notes_path, suspect_mm):
    """(issues, reviews, known): one dict per issue; review rows per dataset and case; notes per (dataset, case)."""
    known = {}
    for r in _read_csv(notes_path):
        known.setdefault((r["mask_source"], r["case_id"]), []).append(r["note"])
    reviews = {d: {r["case_id"]: r for r in _read_csv(f"{run}/{d}/review.csv")} for d in DATASETS}
    issues = []

    def add(group, dataset, case, issue, side="", size="", where=None):
        issues.append({"group": group, "dataset": dataset, "case_id": case, "issue": issue, "side": side,
                       "size": size, "location_ras": _fmt(where), "known": "yes" if (dataset, case) in known else ""})

    for dataset in DATASETS:
        for r in reviews[dataset].values():
            if r["category"] == "failed":
                add(0, dataset, r["case_id"], "failed", "", r["reasons"])

    for f in glob.glob(f"{run}/ImageCAS/qc/*.qc.json"):
        qc = json.load(open(f))
        case = qc["case_id"]
        if qc.get("merged_component"):
            gap, where = _closest_left_right(f"{run}/ImageCAS/graph/{case}.centerlines.gml")
            add(1, "ImageCAS", case, "left and right trees touch", "L+R",
                "" if gap is None else f"closest left/right nodes {gap:.1f} mm apart", where)
        for side in ("L", "R"):
            s = qc["sides"][side]
            for region in s["unexplained"].get("largest", []):
                if region["extent_mm"] >= suspect_mm:
                    add(2, "ImageCAS", case, "mask region not explained by the tree", side,
                        f"{region['extent_mm']:.0f} mm long, {region['volume_mm3']:.0f} mm3, "
                        f"{region['distance_to_tree_mm']['min']:.1f} mm from the tree", region.get("centroid_ras"))
            if s["ostium_to_mask_mm"] > 3.0:
                add(3, "ImageCAS", case, "ostium far from the mask", side, f"{s['ostium_to_mask_mm']:.1f} mm")
        for region in qc["unreachable"].get("largest", []):
            if region["extent_mm"] >= suspect_mm:
                add(2, "ImageCAS", case, "mask piece unreachable from both ostia", "",
                    f"{region['extent_mm']:.0f} mm long, {region['volume_mm3']:.0f} mm3, "
                    f"{region['distance_to_tree_mm']['min']:.1f} mm from the tree", region.get("centroid_ras"))
        for bridge in qc["gap_bridges"]["bridges"]:
            add(4, "ImageCAS", case, "gap bridged", "",
                f"gap {bridge['gap_mm']:.1f} mm, piece {bridge['piece_volume_mm3']:.0f} mm3", _ras(qc, bridge["piece_ijk"]))
        if qc.get("radius_cap", {}).get("tree_changed"):
            for side, change in qc["radius_cap"]["changes"].items():
                if change["max_distance_from_pass1_mm"] > 1.0:
                    add(5, "ImageCAS", case, "radius cap moved the path", side,
                        f"up to {change['max_distance_from_pass1_mm']:.1f} mm, {change['n_voxels_moved']} voxels moved")

    for r in _read_csv(f"{run}/compare/ASOCA/discrepancies.csv"):
        if r["kind"] == "extra":
            add(6, "ASOCA", r["case_id"], "extra branch, absent from the reference", r["side"],
                f"{r['length_mm']} mm long, starting {r['distance_from_ostium_mm']} mm from the ostium",
                json.loads(r["start_ras"]) if r["start_ras"] else None)

    for case, x in reviews["ImageCAS-X"].items():
        has = {d: os.path.exists(f"{run}/{d}/graph/{case}.centerlines.gml") for d in ("ImageCAS", "ImageCAS-X")}
        labels = [qc_label(reviews[d].get(case)) if has[d] else None for d in ("ImageCAS", "ImageCAS-X")]
        if recommend(*labels)[0] == "ImageCAS-X" and x["category"] != "clean":
            add(7, "ImageCAS-X", case, f"ImageCAS-X review: {x['category'].replace('_', ' ')}", "", x["reasons"])

    issues.sort(key=lambda r: (r["group"], r["dataset"], _case_key(r["case_id"])))
    return issues, reviews, known


def _figures(run, dataset, case):
    """(inline image, links to the other figures) of a case, relative to the run folder."""
    folder = f"{dataset}/debug_figures/{case}"
    main = f"{folder}/{FIGURES[0][0]}"
    img = (f'<a href="{main}"><img src="{main}" loading="lazy" alt="{html.escape(dataset)} {html.escape(case)}"></a>'
           if os.path.exists(f"{run}/{main}") else '<p class="small">no debug figure</p>')
    links = " · ".join(f'<a href="{folder}/{name}">{label}</a>' for name, label in FIGURES[1:]
                       if os.path.exists(f"{run}/{folder}/{name}"))
    return img, links


def _card(run, group, dataset, case, rows, reviews, known, groups_of_case):
    review = reviews.get(dataset, {}).get(case, {})
    items = "".join(
        f"<li><b>{html.escape(r['issue'])}</b>{' (' + html.escape(r['side']) + ')' if r['side'] else ''}"
        f"{': ' + html.escape(r['size']) if r['size'] else ''}"
        f"{' <span class=ras>RAS ' + r['location_ras'] + '</span>' if r['location_ras'] else ''}</li>"
        for r in rows)
    others = sorted(groups_of_case - {group})
    also = (f'<p class="small">Also in: ' + ", ".join(f'<a href="#g{g}">{g}. {GROUPS[g][0]}</a>' for g in others) + "</p>"
            if others else "")
    note = "".join(f'<p class="note">{html.escape(n)}</p>' for n in known.get((dataset, case), []))
    img, links = _figures(run, dataset, case)
    extra = ""
    if dataset == "ASOCA" and os.path.exists(f"{run}/compare/ASOCA/figures/{case}.png"):
        extra = f'<p class="small">Comparison with the reference: <a href="compare/ASOCA/figures/{case}.png">figure</a></p>'
    if group == 7:
        ic_img, ic_links = _figures(run, "ImageCAS", case)
        ic_review = reviews["ImageCAS"].get(case, {})
        extra = (f'<p class="small">ImageCAS-mask graph ({html.escape(ic_review.get("category", "no graph"))}): '
                 f'{html.escape(ic_review.get("reasons", ""))}</p>{ic_img}<p class="small">{ic_links}</p>')
    return (f'<div class="card g{group}" id="g{group}-{html.escape(dataset)}-{html.escape(case)}">'
            f'<h3>{html.escape(dataset)} {html.escape(case)} <span class="cat">{html.escape(review.get("category", "").replace("_", " "))}</span></h3>'
            f'<ul>{items}</ul>{note}{also}{img}<p class="small">{links}</p>{extra}'
            f'<p class="small">Slicer: <code>{html.escape(dataset)}/markup-3d-slicer/{html.escape(case)}/</code> · graph: '
            f'<code>{html.escape(dataset)}/graph/{html.escape(case)}.centerlines.gml</code></p></div>')


def write_html(run, issues, reviews, known):
    name = os.path.basename(os.path.normpath(run))
    todo = [r for r in issues if not r["known"]]
    known_rows = [r for r in issues if r["known"]]
    groups_of = {}
    for r in issues:
        groups_of.setdefault((r["dataset"], r["case_id"]), set()).add(r["group"])
    cases_todo = sorted({(r["dataset"], r["case_id"]) for r in todo}, key=lambda k: (k[0], _case_key(k[1])))

    summary = []
    for g, (title, _) in GROUPS.items():
        sel = [r for r in todo if r["group"] == g]
        summary.append(f'<tr><td>{g}</td><td><a href="#g{g}">{title}</a></td>'
                       f'<td>{len({(r["dataset"], r["case_id"]) for r in sel})}</td><td>{len(sel)}</td></tr>')
    links = []
    for d in DATASETS:
        if os.path.exists(f"{run}/{d}/review.html"):
            links.append(f'<li><a href="{d}/review.html">{d}: review</a> (all flagged cases, short accepted branches, '
                         f'random audit of clean cases)</li>')
    for f in sorted(glob.glob(f"{run}/compare/*/report.html")):
        c = os.path.basename(os.path.dirname(f))
        links.append(f'<li><a href="compare/{c}/report.html">comparison {c}</a> · '
                     f'<a href="compare/{c}/discrepancies.html">discrepancies</a></li>')
    index = " ".join(
        f'<a class="chip" href="#g{min(groups_of[k])}-{html.escape(k[0])}-{html.escape(k[1])}">'
        f'{html.escape(k[0])} {html.escape(k[1])}</a>' for k in cases_todo)

    sections = []
    for g, (title, text) in GROUPS.items():
        sel = [r for r in todo if r["group"] == g]
        by_case = {}
        for r in sel:
            by_case.setdefault((r["dataset"], r["case_id"]), []).append(r)
        cards = "".join(_card(run, g, d, c, rows, reviews, known, groups_of[(d, c)]) for (d, c), rows in by_case.items())
        sections.append(f'<h2 id="g{g}">{g}. {title} ({len(by_case)} cases)</h2><p class="what">{text}</p>'
                        f'<div class="grid">{cards or "<p>none</p>"}</div>')
    by_known = {}
    for r in known_rows:
        by_known.setdefault((r["dataset"], r["case_id"]), []).append(r)
    known_cards = "".join(_card(run, min(groups_of[k]), k[0], k[1], rows, reviews, known, groups_of[k])
                          for k, rows in by_known.items())

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Centerline inspection</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fff; color: #111; max-width: 1800px; }}
table {{ border-collapse: collapse; margin-bottom: 12px; }} td, th {{ border-bottom: 1px solid #ddd; padding: 4px 10px; text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #ccc; border-left-width: 6px; border-radius: 6px; padding: 8px; }}
.card.g0 {{ border-left-color: #c62828; }} .card.g1 {{ border-left-color: #6a1b9a; }} .card.g2 {{ border-left-color: #ef6c00; }}
.card.g3 {{ border-left-color: #2e7d32; }} .card.g4 {{ border-left-color: #1565c0; }} .card.g5 {{ border-left-color: #00838f; }}
.card.g6 {{ border-left-color: #6d4c41; }} .card.g7 {{ border-left-color: #ad1457; }}
.card h3 {{ margin: 0 0 4px; }} .cat {{ font-size: 12px; font-weight: normal; color: #555; }}
.card ul {{ margin: 0 0 6px; padding-left: 18px; font-size: 13px; }} .card img {{ width: 100%; }}
.ras {{ font-family: monospace; color: #444; }} .small {{ font-size: 12px; color: #555; margin: 4px 0; }}
.note {{ font-size: 13px; background: #fff8e1; padding: 4px 6px; margin: 4px 0; }}
.what {{ max-width: 1100px; }} .chip {{ display: inline-block; font-size: 12px; padding: 1px 6px; margin: 2px;
border: 1px solid #ccc; border-radius: 10px; text-decoration: none; color: #111; }}
code {{ font-size: 11px; }}
</style></head><body>
<h1>Centerline inspection: <code>{html.escape(name)}</code></h1>
<p class="what">These are the cases that may be in trouble, found by the quality checks of the extraction and by the
comparison with the ASOCA reference, so that the release can be inspected without opening every patient. Each card shows
what was flagged, its size and, where available, a RAS position: type it in 3D Slicer (crosshair or a markup point) to
jump there. The first figure of each card is the top and side view of the trees, with the accepted (green) and rejected
(orange) targets and the possibly missed regions (big magenta crosses); the other figures are linked under it. The Slicer
curves and the graph of each case are in the folders shown, relative to the run folder (this page's folder). A case can
appear in more than one group; its card in each group shows only that group's issues.</p>
<table><tr><th>#</th><th>Group</th><th>Cases</th><th>Issues</th></tr>{"".join(summary)}
<tr><td></td><td>Total to inspect</td><td>{len(cases_todo)}</td><td>{len(todo)}</td></tr>
<tr><td></td><td><a href="#known">Known, already explained</a></td><td>{len(by_known)}</td><td>{len(known_rows)}</td></tr></table>
<p>Other reports of this run:</p><ul>{"".join(links)}</ul>
<p>The miss rate among unflagged cases comes from the random audit in each dataset's review, not from this page.</p>
<h2>All cases to inspect ({len(cases_todo)})</h2><p>{index}</p>
{"".join(sections)}
<h2 id="known">Known, already explained in the notes file ({len(by_known)} cases)</h2>
<div class="grid">{known_cards or "<p>none</p>"}</div>
</body></html>"""
    open(f"{run}/inspection.html", "w").write(page)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--notes", default=os.path.join(HERE, "release_case_notes.csv"))
    p.add_argument("--suspect-mm", type=float, default=10.0)
    a = p.parse_args()
    issues, reviews, known = collect(a.run, a.notes, a.suspect_mm)
    with open(f"{a.run}/inspection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "dataset", "case_id", "issue", "side", "size", "location_ras", "known"])
        w.writeheader()
        w.writerows(issues)
    write_html(a.run, issues, reviews, known)
    todo = [r for r in issues if not r["known"]]
    print(f"inspection: {len(todo)} issues in {len({(r['dataset'], r['case_id']) for r in todo})} cases "
          f"(+ {len(issues) - len(todo)} known); {a.run}/inspection.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
