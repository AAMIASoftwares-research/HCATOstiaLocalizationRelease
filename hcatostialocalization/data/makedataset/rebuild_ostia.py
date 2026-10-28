"""
Rebuilds the released ostium files of every dataset from their sources, in one go:

    <data>/Annotations/OstiumAnnotations_<DS>/OSTIAv2/JSON/<DS>__<id>.ostium.json   (blueprint schema, io.io)
    <data>/Annotations/OstiumAnnotations_<DS>/OSTIAv2/Slicer/<DS>__<id>.mrk.json    (3D Slicer fiducials R/L, RAS)

    python -m hcatostialocalization.data.makedataset.rebuild_ostia [--data <root>]
        [--datasets ASOCA CAT08 ImageCAS] [--cases 338 501] [--fix-swapped]

OSTIAv2 is a new folder next to the previous release, OstiumAnnotations_<DS>/Ostia/<DS>__<id>.ostium.json,
which is never modified: trainings and evaluations that read it are unaffected.

Sources (all under <data>/Annotations/):
  ASOCA     OstiumAnnotations_ASOCA/_legacy/{Normal,Diseased}_ijk_ras_json/<id>.json, with the image header
            (ASOCA/<Group>/CTCA/<id>.nrrd) for its raw frame
  CAT08     OstiumAnnotations_CAT08/_legacy/JSON_Completi/datasetNN_labels.json, image CAT08/datasetNN/imageNN.mhd
  ImageCAS  cases 1-500: consensus of the annotators' Slicer files,
            OstiumAnnotations_ImageCAS/Manual_Annotations/SlicerObjects_1_500/<id>_*.mrk.json: per side,
            the mean of the annotators' points (the same as the k-means with one cluster used originally);
            cases 501-1000: the corrected model-assisted markups,
            OstiumAnnotations_ImageCAS/Manual_Annotations/SlicerObjects_501_1000/markups_501_1000_final/
            case_*_<id>.img.mrk.json; image ImageCAS/Data/<id>.img.nii.gz

Checks, written to OSTIAv2/rebuild_report.csv per dataset and summarised at the end:
  - ImageCAS consensus: number of annotators and of points per side; an annotator whose R and L
    labels look swapped (its R point nearer the other annotators' L, and vice versa); any annotator
    point more than --outlier-mm from the mean of the others. --fix-swapped swaps a suspected
    annotator's labels in the consensus (its file is not modified) and records it.
  - ImageCAS 1-500: distance from the consensus stored in
    _legacy/Annotations ImageCAS_RAS_IJK/<id>_final_ostia.json (reproducibility; 0 unless a source
    file changed).
  - Every case: distance from the previous released file (Ostia/<DS>__<id>.ostium.json), so the
    cases that change are listed; both ostia inside the image; L-R distance.
Existing files in OSTIAv2/JSON and OSTIAv2/Slicer are overwritten; Ostia/ is only read.

Run from the repository root, on the cluster (it reads every image once).
"""
import os
import re
import csv
import sys
import glob
import json
import shutil
import argparse

import numpy

from hcatostialocalization.io.io import (
    extract_simplified_ras, compute_image_metadata, save_blueprint_json, save_slicer_json, read_image_header_lps,
)

DEFAULT_DATA = "/path/to/data"
SIDES = ("R", "L")


# ----------------------------------------------------------------------------- sources

def _id_key(value: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", value)]


def _source_glob(root, pattern):
    """Source annotations live in <root>/_legacy/ (moved there with OSTIAv2); <root>/ is the older location."""
    return glob.glob(f"{root}/_legacy/{pattern}") or glob.glob(f"{root}/{pattern}")


def asoca_cases(data):
    out = []
    for group in ("Normal", "Diseased"):
        for src in sorted(_source_glob(f"{data}/Annotations/OstiumAnnotations_ASOCA", f"{group}_ijk_ras_json/{group}_*.json"), key=_id_key):
            case = os.path.basename(src)[:-len(".json")]
            images = sorted(glob.glob(f"{data}/ASOCA/{group}/CTCA/{case}.*"))
            out.append({"id": case, "image": images[0] if images else None, "source": src, "kind": "file"})
    return out


def cat08_cases(data):
    out = []
    for src in sorted(_source_glob(f"{data}/Annotations/OstiumAnnotations_CAT08", "JSON_Completi/dataset*_labels.json"), key=_id_key):
        number = re.search(r"dataset(\d+)_labels", os.path.basename(src)).group(1)
        image = f"{data}/CAT08/dataset{number}/image{number}.mhd"
        out.append({"id": f"image{number}", "image": image if os.path.exists(image) else None, "source": src, "kind": "file"})
    return out


def imagecas_cases(data):
    root = f"{data}/Annotations/OstiumAnnotations_ImageCAS"
    annotators = {}
    for f in glob.glob(f"{root}/Manual_Annotations/SlicerObjects_1_500/*.mrk.json"):
        head = os.path.basename(f).split("_")[0]
        if head.isdigit():
            annotators.setdefault(str(int(head)), []).append(f)
    corrected = {}
    for f in glob.glob(f"{root}/Manual_Annotations/SlicerObjects_501_1000/markups_501_1000_final/*.mrk.json"):
        m = re.search(r"_(\d+)\.img\.mrk\.json$", os.path.basename(f))
        if m:
            corrected[m.group(1)] = f
    out = []
    for n in range(1, 1001):
        case = str(n)
        image = f"{data}/ImageCAS/Data/{case}.img.nii.gz"
        legacy = f"{root}/_legacy/Annotations ImageCAS_RAS_IJK/{case}_final_ostia.json"
        if n <= 500:
            entry = {"id": case, "image": image, "kind": "consensus", "annotator_files": sorted(annotators.get(case, [])),
                     "legacy": legacy if os.path.exists(legacy) else None}
        else:
            entry = {"id": case, "image": image, "kind": "file", "source": corrected.get(case)}
        if not os.path.exists(image):
            entry["image"] = None
        out.append(entry)
    return out


DATASETS = {"ASOCA": asoca_cases, "CAT08": cat08_cases, "ImageCAS": imagecas_cases}


# ImageCAS consensus

def _markup_points(path):
    """{'R': [ras, ...], 'L': [...]} from one annotator's Slicer file. A label counts for a
    side if it contains that letter (R first), as in the original consensus; positions are
    converted to RAS when the file is in LPS (Slicer's default for saved markups)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    points = {"R": [], "L": []}
    for markup in data.get("markups", []):
        flip = markup.get("coordinateSystem", "LPS").upper() == "LPS"
        for cp in markup.get("controlPoints", []):
            label, pos = cp.get("label", "").upper(), cp.get("position")
            if pos is None or len(pos) != 3:
                continue
            ras = numpy.asarray(pos, float) * (numpy.array([-1.0, -1.0, 1.0]) if flip else 1.0)
            if "R" in label:
                points["R"].append(ras)
            elif "L" in label:
                points["L"].append(ras)
    return points


def consensus(entry, fix_swapped: bool, outlier_mm: float):
    """Mean of the annotators' points per side, with checks. Returns (simple_pos or None, qc)."""
    qc = {"n_annotators": len(entry["annotator_files"]), "swapped": [], "ambiguous_swap": [], "outliers": [], "point_counts": {}}
    per, single = {}, {s: {} for s in SIDES}
    for f in entry["annotator_files"]:
        name = os.path.basename(f)[len(entry["id"]) + 1:].split(".")[0]
        pts = _markup_points(f)
        qc["point_counts"][name] = {s: len(pts[s]) for s in SIDES}
        for s in SIDES:  # an annotator who placed exactly one point on a side counts for that side
            if len(pts[s]) == 1:
                single[s][name] = pts[s][0]
        if all(len(pts[s]) == 1 for s in SIDES):
            per[name] = {s: pts[s][0] for s in SIDES}
    if min(len(single[s]) for s in SIDES) < 2:
        return None, qc
    # swapped labels: among the annotators with one point on each side, find which ones to swap so
    # that each side's points (partial annotators included) cluster most tightly; ties -> ambiguous
    import itertools
    complete = sorted(per)

    def spread(flips):
        total = 0.0
        for s in SIDES:
            points = [(single["L" if s == "R" else "R"][n] if n in flips else single[s][n]) for n in single[s]]
            centre = numpy.mean(points, axis=0)
            total += float(sum(numpy.sum((p - centre) ** 2) for p in points))
        return total
    # at most half of the annotators can be the swapped ones
    candidates = [set(c) for k in range(len(complete) // 2 + 1) for c in itertools.combinations(complete, k)]
    scored = sorted(((spread(c), len(c), sorted(c)) for c in candidates), key=lambda x: (x[0], x[1]))
    best = scored[0]
    ties = [x for x in scored[1:] if abs(x[0] - best[0]) <= 1e-6 * max(1.0, best[0]) and x[2] != best[2]]
    if ties and best[2]:
        qc["ambiguous_swap"] = [best[2]] + [x[2] for x in ties]
    elif best[2]:
        qc["swapped"] = best[2]
    if fix_swapped:
        for name in qc["swapped"]:
            single["R"][name], single["L"][name] = single["L"][name], single["R"][name]
    for s in SIDES:
        for name, point in single[s].items():
            others = [q for k, q in single[s].items() if k != name]
            d = float(numpy.linalg.norm(point - numpy.mean(others, axis=0)))
            if d > outlier_mm:
                qc["outliers"].append(f"{name}:{s}:{d:.1f}mm")
    # per side, the mean of every annotator's single point on that side (as the original consensus)
    pos = {s: {"ras": [float(c) for c in numpy.mean(list(single[s].values()), axis=0)]} for s in SIDES}
    qc["n_used"] = min(len(single[s]) for s in SIDES)
    return pos, qc


def public_notes(qc, fix_swapped: bool):
    """The consensus checks of qc without annotator names, for the released JSON ('other.checks'); the names stay
    only in rebuild_report.csv. Annotators are numbered 1, 2, 3 in the sorted order of their files' names."""
    number = {name: f"annotator {k}" for k, name in enumerate(sorted(qc["point_counts"]), start=1)}
    notes = []
    if qc["swapped"]:
        notes.append(("swapped R/L FIXED: " if fix_swapped else "swapped R/L suspected: ")
                     + ", ".join(number.get(n, "annotator") for n in qc["swapped"]))
    if qc.get("ambiguous_swap"):
        sets = [[number.get(n, "annotator") for n in group] for group in qc["ambiguous_swap"]]
        notes.append(f"R/L swap suspected but ambiguous, NOT fixed (equally good swap sets: {sets})")
    if qc["outliers"]:
        notes.append("annotator far from the others: " + ", ".join(
            f"{number.get(o.split(':')[0], 'annotator')}:{':'.join(o.split(':')[1:])}" for o in qc["outliers"]))
    bad = {number[k]: v for k, v in qc["point_counts"].items() if v != {"R": 1, "L": 1}}
    if bad:
        notes.append(f"annotators without exactly one R and one L (their single points still used): {bad}")
    return notes


# Main

def _ras(path):
    if not path or not os.path.exists(path):
        return None
    pos = extract_simplified_ras(path)
    if not pos or not all(len(pos.get(s, {}).get("ras") or []) == 3 for s in SIDES):
        return None
    return {s: numpy.asarray(pos[s]["ras"], float) for s in SIDES}


def rebuild(dataset, data, cases_filter, fix_swapped, outlier_mm, min_lr_mm=18.0, max_lr_mm=65.0):
    previous_dir = f"{data}/Annotations/OstiumAnnotations_{dataset}/Ostia"
    root = f"{data}/Annotations/OstiumAnnotations_{dataset}/OSTIAv2"
    json_dir, slicer_dir = f"{root}/JSON", f"{root}/Slicer"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(slicer_dir, exist_ok=True)
    cases = DATASETS[dataset](data)
    if not cases:
        print(f"== {dataset}: ERROR, no source annotations found under {data}/Annotations/OstiumAnnotations_{dataset}/ "
              f"(see the source layout in this file's docstring); nothing written", file=sys.stderr)
        return False
    if cases_filter:
        cases = [c for c in cases if c["id"] in cases_filter]
    rows, n_ok = [], 0
    for c in cases:
        image_id = f"{dataset}__{c['id']}"
        row = {"case": c["id"], "status": "", "source": "", "notes": []}
        rows.append(row)
        if c["image"] is None:
            row["status"] = "error: image not found"
            continue
        qc = {}
        if c["kind"] == "consensus":
            row["source"] = f"consensus of {len(c['annotator_files'])} annotator files"
            simple_pos, qc = consensus(c, fix_swapped, outlier_mm)
            if simple_pos is None and c.get("legacy"):
                simple_pos = extract_simplified_ras(c["legacy"])
                row["source"] = "legacy consensus (annotator files unusable)"
                row["notes"].append(f"point counts {qc['point_counts']}")
        else:
            row["source"] = os.path.basename(c["source"]) if c.get("source") else ""
            simple_pos = extract_simplified_ras(c["source"], image_path=c["image"]) if c.get("source") else None
        if not simple_pos or not all(len(simple_pos.get(s, {}).get("ras") or []) == 3 for s in SIDES):
            row["status"] = "error: no R and L ostium in the source"
            continue
        try:
            full_pos, affine_ijk_to_ras, affine_ras_to_ijk = compute_image_metadata(c["image"], simple_pos)
        except Exception as e:
            row["status"] = f"error: {type(e).__name__}: {e}"
            continue
        size = read_image_header_lps(c["image"])[3]
        for s in SIDES:
            ijk = full_pos[s]["ijk"]["sitk"]
            if not all(-0.5 <= v <= n - 0.5 for v, n in zip(ijk, size)):
                row["notes"].append(f"{s} outside the image")
        new = {s: numpy.asarray(full_pos[s]["ras"], float) for s in SIDES}
        row["L_R_distance_mm"] = round(float(numpy.linalg.norm(new["L"] - new["R"])), 2)
        if not (min_lr_mm <= row["L_R_distance_mm"] <= max_lr_mm):  # 99 % of ImageCAS cases lie in 22-57 mm
            row["notes"].append(f"left-right distance {row['L_R_distance_mm']} mm outside [{min_lr_mm:g}, {max_lr_mm:g}]")
        named = (len(row["notes"]),) * 2  # notes[named[0]:named[1]] carry annotator names
        if qc:
            row["annotators_used"] = qc.get("n_used")
            if qc["swapped"]:
                row["notes"].append(("swapped R/L FIXED: " if fix_swapped else "swapped R/L suspected: ") + ", ".join(qc["swapped"]))
            if qc.get("ambiguous_swap"):
                row["notes"].append(f"R/L swap suspected but ambiguous, NOT fixed (equally good swap sets: {qc['ambiguous_swap']})")
            if qc["outliers"]:
                row["notes"].append("annotator far from the others: " + ", ".join(qc["outliers"]))
            bad = {k: v for k, v in qc["point_counts"].items() if v != {"R": 1, "L": 1}}
            if bad:
                row["notes"].append(f"annotators without exactly one R and one L (their single points still used): {bad}")
            named = (named[0], len(row["notes"]))
            legacy = _ras(c.get("legacy"))
            if legacy is not None:
                row["vs_legacy_consensus_mm"] = round(max(float(numpy.linalg.norm(new[s] - legacy[s])) for s in SIDES), 3)
        previous = _ras(f"{previous_dir}/{image_id}.ostium.json")
        if previous is not None:
            row["vs_previous_file_mm"] = round(max(float(numpy.linalg.norm(new[s] - previous[s])) for s in SIDES), 3)
        comment = f"Ostium annotation of {dataset} case {c['id']}, rebuilt from: {row['source']}"
        # the released JSON names no annotator: their checks are numbered (the report keeps the names)
        checks = row["notes"][:named[0]] + row["notes"][named[1]:] + (public_notes(qc, fix_swapped) if qc else [])
        if save_blueprint_json(f"{json_dir}/{image_id}.ostium.json", image_id, os.path.basename(c["image"]), full_pos,
                               affine_ijk_to_ras, affine_ras_to_ijk, comment=comment,
                               other={"source": row["source"], "checks": checks}) is None:
            row["status"] = "error: could not write the JSON"
            continue
        save_slicer_json(f"{slicer_dir}/{image_id}.mrk.json", {s: {"ras": full_pos[s]["ras"]} for s in SIDES})
        row["status"] = "ok"
        n_ok += 1
        print(f"  {image_id}: ok" + (f"  [{'; '.join(row['notes'])}]" if row["notes"] else ""))

    report = f"{root}/rebuild_report.csv" if not cases_filter else f"{root}/rebuild_report_subset.csv"
    fields = ["case", "status", "source", "annotators_used", "L_R_distance_mm", "vs_previous_file_mm", "vs_legacy_consensus_mm", "notes"]
    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "notes": " | ".join(r["notes"])})
    changed = [(r["case"], r["vs_previous_file_mm"]) for r in rows if r.get("vs_previous_file_mm", 0) > 0.01]
    flagged = [(r["case"], " | ".join(r["notes"])) for r in rows if r["notes"]]
    errors = [(r["case"], r["status"]) for r in rows if r["status"] != "ok"]
    print(f"== {dataset}: {n_ok}/{len(rows)} written to {json_dir} and {slicer_dir}; report {report}")
    print(f"   changed vs the previous files (> 0.01 mm): {len(changed)} {changed[:20]}")
    print(f"   flagged: {len(flagged)}")
    for case, note in flagged[:40]:
        print(f"     {case}: {note}")
    if errors:
        print(f"   ERRORS: {errors}")
    return not errors


def main():
    p = argparse.ArgumentParser(description="Rebuild OSTIAv2/JSON (.ostium.json) and OSTIAv2/Slicer (.mrk.json) for ASOCA, CAT08 and ImageCAS from their sources (the previous Ostia/ folder is only read).")
    p.add_argument("--data", default=DEFAULT_DATA, help=f"data root (default {DEFAULT_DATA})")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--cases", nargs="+", default=None, help="only these case ids (e.g. 338 501, Normal_1, image01); the report goes to rebuild_report_subset.csv")
    p.add_argument("--fix-swapped", action="store_true", help="ImageCAS consensus: swap the R/L labels of an annotator whose labels look swapped (the file itself is not changed)")
    p.add_argument("--outlier-mm", type=float, default=10.0, help="ImageCAS consensus: flag an annotator point farther than this from the mean of the others")
    a = p.parse_args()
    ok = True
    for ds in a.datasets:
        print(f"\n######## {ds}")
        ok &= rebuild(ds, a.data, set(a.cases) if a.cases else None, a.fix_swapped, a.outlier_mm)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
