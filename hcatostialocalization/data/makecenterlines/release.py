"""
Builds the public centerline release from a run_extraction.sh run (+ run_review.sh and run_compare.sh).

    python -m hcatostialocalization.data.makecenterlines.release --run <run folder> --out <release folder>
        [--notes release_case_notes.csv] [--no-copy]

Release layout:

    <out>/README.md                                     release_README.md, copied
    <out>/cases.csv                                     one row per ImageCAS case (see below)
    <out>/centerlines_from_ImageCAS_masks/gml/<id>.centerlines.gml
    <out>/centerlines_from_ImageCAS_masks/vtk/<id>.graph.vtk, <id>.paths.vtk   (undirected graph; ostium-to-endpoint polylines)
    <out>/centerlines_from_ImageCAS_masks/slicer/<id>/*.mkr.json
    <out>/centerlines_from_ImageCAS-X_masks/gml/...     same, traced on the ImageCAS-X masks (800 cases)
    <out>/centerlines_from_ImageCAS-X_masks/slicer/...
    <out>/validation.json                               every released GML re-read with plain networkx

The ImageCAS-mask centerlines are the official set. The recommended graph of a case is the ImageCAS one, unless
it is missing, or the review found it failed, at risk of missed branches, or with the left and right trees touching
in the mask, while the ImageCAS-X one exists and is free of all three; the notes file can override this. The Slicer folder of each case is the one written by the
extraction (HCATNetwork), plus <id>_notable_landmarks.mrk.json: the ostia, bifurcations and endpoints, each named
with side, kind, rank from the ostium and GML node id, e.g. "Left bifurcation 2 (412)".

cases.csv columns: case_id, recommended (ImageCAS / ImageCAS-X / none), why_recommended, in_imagecas_x,
gml_<set>, vtk_<set> (prefix of the two VTK files) and slicer_<set> (paths relative to <out>, empty when there is no graph), and per set (imagecas,
imagecas_x): status (ok / extraction failed / not in ImageCAS-X), qc (clean / check / left and right trees touch /
possible missed branch / failed, from the review), qc_notes (the review reasons), short_branches (accepted branches shorter than 10 mm,
with their tip in RAS), length_L_mm, length_R_mm; then the agreement of the
ImageCAS-mask graph with the ImageCAS-X reference centerlines (recall, precision, missing length; empty outside
ImageCAS-X), and comment (from the notes file).

The notes file (default: release_case_notes.csv next to this script) has columns case_id, mask_source,
recommended (optional override), note.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {"ImageCAS": "centerlines_from_ImageCAS_masks", "ImageCAS-X": "centerlines_from_ImageCAS-X_masks"}
QC_NAMES = {"clean": "clean", "other_checks": "check", "likely_missed": "possible missed branch", "failed": "failed"}
RISKY = ("failed", "possible missed branch", "left and right trees touch")  # review results that make the ImageCAS-X graph preferable


def qc_label(review_row):
    """The release's quality label of a case from its review.csv row ('' without a review)."""
    label = QC_NAMES.get((review_row or {}).get("category", ""), "")
    if label and review_row.get("merged_component") == "True" and label not in RISKY:
        label = "left and right trees touch"  # should never happen anatomically: substantial
    return label


def recommend(imagecas_label, imagecas_x_label):
    """(recommended set, why) from the quality labels of the two graphs of a case, None where there is no graph.
    The ImageCAS-mask graph is the official one; ImageCAS-X only as a fallback: no ImageCAS-mask graph, or one the
    review found failed, at risk of missed branches or with touching trees, while the ImageCAS-X graph is free of all three.
    Shared with inspection.py."""
    ok = lambda label: label is not None and label not in RISKY
    if ok(imagecas_label):
        return "ImageCAS", "official set"
    if ok(imagecas_x_label):
        return "ImageCAS-X", ("no graph on the ImageCAS mask" if imagecas_label is None
                              else f"quality checks of the ImageCAS-mask graph: {imagecas_label}")
    if imagecas_label is not None:
        return "ImageCAS", f"official set (quality checks: {imagecas_label}; no better alternative)"
    if imagecas_x_label is not None:
        return "ImageCAS-X", f"no graph on the ImageCAS mask (ImageCAS-X quality checks: {imagecas_x_label})"
    return "none", "no graph"


def _ids(list_file):
    """Case ids from an ostium list (…/ImageCAS__<id>.ostium.json)."""
    with open(list_file) as f:
        return [os.path.basename(l.strip())[len("ImageCAS__"):-len(".ostium.json")] for l in f if l.strip()]


def _read_csv(path, key="case_id", where=None):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found", file=sys.stderr)
        return {}
    with open(path, newline="") as f:
        return {r[key]: r for r in csv.DictReader(f) if where is None or where(r)}


def _num(value, digits=1):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


def _validate(gml_path):
    """Problems found re-reading a released GML with plain networkx (empty list: fine)."""
    import networkx
    try:
        g = networkx.read_gml(gml_path, label="id")
    except Exception as e:
        return [f"networkx cannot read it: {type(e).__name__}: {e}"]
    problems = []
    ostia = sorted(d.get("side") for _, d in g.nodes(data=True) if d.get("topology") == "OSTIUM")
    if ostia != ["LEFT", "RIGHT"]:
        problems.append(f"ostia {ostia}, expected one LEFT and one RIGHT")
    if not networkx.is_forest(g):
        problems.append("not a forest")
    if networkx.number_connected_components(g) != 2:
        problems.append(f"{networkx.number_connected_components(g)} connected components, expected 2")
    for _, d in g.nodes(data=True):
        if not all(isinstance(d.get(k), (int, float)) for k in ("x", "y", "z", "r")):
            problems.append("a node without numeric x, y, z, r")
            break
    return problems


def _write_notable_landmarks(gml_path, out_path):
    """3D Slicer point list of the ostia, bifurcations and endpoints of a released GML (RAS, mm), each named
    with its side, its kind, its rank along the tree (by path distance from the ostium) and its GML node id
    in parentheses, e.g. "Left bifurcation 2 (412)"."""
    import networkx
    g = networkx.read_gml(gml_path, label="id")
    kinds = {"OSTIUM": "ostium", "INTERSECTION": "bifurcation", "ENDPOINT": "endpoint"}
    points = []
    for ostium, d in sorted(((n, d) for n, d in g.nodes(data=True) if d.get("topology") == "OSTIUM"),
                            key=lambda item: item[1]["side"]):
        side = d["side"].capitalize()
        distance = networkx.single_source_dijkstra_path_length(g, ostium, weight="euclidean_distance")
        rank = {}
        for n in sorted(distance, key=distance.get):
            kind = kinds.get(g.nodes[n].get("topology"))
            if kind is None:
                continue
            rank[kind] = rank.get(kind, 0) + 1
            name = f"{side} ostium ({n})" if kind == "ostium" else f"{side} {kind} {rank[kind]} ({n})"
            x = g.nodes[n]
            points.append({"id": str(len(points) + 1), "label": name, "position": [x["x"], x["y"], x["z"]],
                           "positionStatus": "defined"})
    markup = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/"
                         "Schema/markups-schema-v1.0.3.json#",
              "markups": [{"type": "Fiducial", "coordinateSystem": "RAS", "coordinateUnits": "mm", "locked": True,
                           "labelFormat": "%N-%d", "controlPoints": points}]}
    with open(out_path, "w") as f:
        json.dump(markup, f, indent=1)


TOPOLOGY_CODES = {"OSTIUM": 1, "SEGMENT": 2, "INTERSECTION": 3, "ENDPOINT": 4}
SIDE_CODES = {"LEFT": 1, "RIGHT": 2}


def _vtk_polydata(out_path, title, points, lines, point_data, cell_data):
    """Legacy ASCII VTK polydata. points: [(x, y, z)]; lines: [[point index, ...]]; point_data / cell_data:
    {name: (vtk type, values)}. 'SPACE=RAS' in the title line tells 3D Slicer the coordinates are RAS."""
    with open(out_path, "w") as f:
        f.write(f"# vtk DataFile Version 3.0\n{title} SPACE=RAS\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} double\n")
        f.writelines(f"{x!r} {y!r} {z!r}\n" for x, y, z in points)
        f.write(f"LINES {len(lines)} {sum(len(l) + 1 for l in lines)}\n")
        f.writelines(f"{len(l)} " + " ".join(map(str, l)) + "\n" for l in lines)
        for header, n, data in (("POINT_DATA", len(points), point_data), ("CELL_DATA", len(lines), cell_data)):
            if not data:
                continue
            f.write(f"{header} {n}\n")
            for name, (vtk_type, values) in data.items():
                f.write(f"SCALARS {name} {vtk_type} 1\nLOOKUP_TABLE default\n")
                f.writelines(f"{v!r}\n" for v in values)


def _write_vtk(gml_path, graph_path, paths_path, case):
    """Two VTK files from a released GML (RAS, mm):
    graph_path: the undirected graph, one point per node and one 2-point line per edge; point data radius_mm,
        topology (1 ostium, 2 segment, 3 bifurcation, 4 endpoint), side (1 left, 2 right), node_id (GML id).
    paths_path: one polyline per ostium-to-endpoint path (as the Slicer open curves: shared proximal parts are
        repeated); point data radius_mm, topology, node_id, distance_from_ostium_mm; cell data side,
        endpoint_node_id, length_mm.
    Returns the problems found (empty list: fine)."""
    import networkx
    g = networkx.read_gml(gml_path, label="id")
    nodes = list(g.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    attr = lambda n, k: g.nodes[n][k]
    _vtk_polydata(
        graph_path, f"ImageCAS {case} coronary centerline graph",
        [(attr(n, "x"), attr(n, "y"), attr(n, "z")) for n in nodes],
        [[index[u], index[v]] for u, v in g.edges],
        {"radius_mm": ("double", [float(attr(n, "r")) for n in nodes]),
         "topology": ("int", [TOPOLOGY_CODES[attr(n, "topology")] for n in nodes]),
         "side": ("int", [SIDE_CODES[attr(n, "side")] for n in nodes]),
         "node_id": ("int", nodes)},
        {},
    )
    points, lines, pdata, cdata = [], [], {k: [] for k in ("radius_mm", "topology", "node_id", "distance_from_ostium_mm")}, \
        {k: [] for k in ("side", "endpoint_node_id", "length_mm")}
    for ostium in sorted((n for n in nodes if attr(n, "topology") == "OSTIUM"), key=lambda n: attr(n, "side")):
        distance, route = networkx.single_source_dijkstra(g, ostium, weight="euclidean_distance")
        for end in sorted((n for n in route if attr(n, "topology") == "ENDPOINT"), key=distance.get):
            line = []
            for n in route[end]:
                line.append(len(points))
                points.append((attr(n, "x"), attr(n, "y"), attr(n, "z")))
                pdata["radius_mm"].append(float(attr(n, "r")))
                pdata["topology"].append(TOPOLOGY_CODES[attr(n, "topology")])
                pdata["node_id"].append(n)
                pdata["distance_from_ostium_mm"].append(float(distance[n]))
            lines.append(line)
            cdata["side"].append(SIDE_CODES[attr(ostium, "side")])
            cdata["endpoint_node_id"].append(end)
            cdata["length_mm"].append(float(distance[end]))
    _vtk_polydata(paths_path, f"ImageCAS {case} coronary centerline paths (ostium to endpoint)", points, lines,
                  {"radius_mm": ("double", pdata["radius_mm"]), "topology": ("int", pdata["topology"]),
                   "node_id": ("int", pdata["node_id"]),
                   "distance_from_ostium_mm": ("double", pdata["distance_from_ostium_mm"])},
                  {"side": ("int", cdata["side"]), "endpoint_node_id": ("int", cdata["endpoint_node_id"]),
                   "length_mm": ("double", cdata["length_mm"])})
    problems = []
    n_ends = sum(attr(n, "topology") == "ENDPOINT" for n in nodes)
    if len(lines) != n_ends:
        problems.append(f"VTK paths: {len(lines)} polylines for {n_ends} endpoints")
    for path in (graph_path, paths_path):
        tokens = open(path).read().split()
        n_points = int(tokens[tokens.index("POINTS") + 1])
        if n_points == 0:
            problems.append(f"{os.path.basename(path)}: no points")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run_extraction.sh output folder")
    p.add_argument("--out", required=True, help="release folder (created)")
    p.add_argument("--notes", default=os.path.join(HERE, "release_case_notes.csv"))
    p.add_argument("--no-copy", action="store_true", help="only write cases.csv and the README (no graph copies)")
    a = p.parse_args()

    ids = {"ImageCAS": _ids(f"{a.run}/lists/ImageCAS_all_ostia.txt"),
           "ImageCAS-X": _ids(f"{a.run}/lists/ImageCAS-X_all_ostia.txt")}
    in_x = set(ids["ImageCAS-X"])
    review = {s: _read_csv(f"{a.run}/{s}/review.csv") for s in SETS}
    qc_lengths = {}
    for s in SETS:
        for f in glob.glob(f"{a.run}/{s}/qc/*.qc.json"):
            d = json.load(open(f))
            qc_lengths[(s, d["case_id"])] = {side: d["sides"][side]["length_mm"] for side in ("L", "R")}
    agreement = _read_csv(f"{a.run}/compare/ImageCAS-X_vs_ours_from_ImageCAS/per_case.csv",
                          where=lambda r: r.get("scope") == "ALL")
    notes = {}
    if os.path.exists(a.notes):
        for r in csv.DictReader(open(a.notes, newline="")):
            notes.setdefault(r["case_id"], []).append(r)

    os.makedirs(a.out, exist_ok=True)
    rows, validation = [], {}
    for case in ids["ImageCAS"]:
        row = {"case_id": case, "in_imagecas_x": "yes" if case in in_x else "no"}
        labels = {}
        for s, folder in SETS.items():
            prefix = "imagecas" if s == "ImageCAS" else "imagecas_x"
            gml = f"{a.run}/{s}/graph/{case}.centerlines.gml"
            slicer = f"{a.run}/{s}/markup-3d-slicer/{case}"
            rv = review[s].get(case, {})
            if s == "ImageCAS-X" and case not in in_x:
                status = "not in ImageCAS-X"
            elif os.path.exists(gml):
                status = "ok"
            else:
                status = "extraction failed"
            qc = qc_label(rv)
            row[f"{prefix}_status"] = status
            row[f"{prefix}_qc"] = qc if status == "ok" else ""
            row[f"{prefix}_qc_notes"] = rv.get("reasons", "") if status == "ok" else ""
            row[f"{prefix}_short_branches"] = rv.get("short_accepted", "") if status == "ok" else ""
            lengths = qc_lengths.get((s, case), {})
            row[f"{prefix}_length_L_mm"] = _num(lengths.get("L"))
            row[f"{prefix}_length_R_mm"] = _num(lengths.get("R"))
            row[f"gml_{prefix}"] = row[f"slicer_{prefix}"] = row[f"vtk_{prefix}"] = ""
            if status != "ok":
                continue
            labels[s] = qc
            row[f"gml_{prefix}"] = f"{folder}/gml/{case}.centerlines.gml"
            row[f"slicer_{prefix}"] = f"{folder}/slicer/{case}" if os.path.isdir(slicer) else ""
            if not a.no_copy:
                os.makedirs(f"{a.out}/{folder}/gml", exist_ok=True)
                shutil.copy2(gml, f"{a.out}/{row[f'gml_{prefix}']}")
                if row[f"slicer_{prefix}"]:
                    shutil.copytree(slicer, f"{a.out}/{row[f'slicer_{prefix}']}", dirs_exist_ok=True)
                    _write_notable_landmarks(f"{a.out}/{row[f'gml_{prefix}']}",
                                             f"{a.out}/{row[f'slicer_{prefix}']}/{case}_notable_landmarks.mrk.json")
                problems = _validate(f"{a.out}/{row[f'gml_{prefix}']}")
                if not problems:
                    os.makedirs(f"{a.out}/{folder}/vtk", exist_ok=True)
                    row[f"vtk_{prefix}"] = f"{folder}/vtk/{case}"
                    problems = _write_vtk(f"{a.out}/{row[f'gml_{prefix}']}", f"{a.out}/{folder}/vtk/{case}.graph.vtk",
                                          f"{a.out}/{folder}/vtk/{case}.paths.vtk", case)
                if problems:
                    validation[row[f"gml_{prefix}"]] = problems

        rec, why = recommend(labels.get("ImageCAS"), labels.get("ImageCAS-X"))
        for n in notes.get(case, []):
            wanted = n.get("recommended")
            if wanted and wanted != rec and wanted in labels:
                rec, why = wanted, "expert review"
        row["recommended"], row["why_recommended"] = rec, why
        ag = agreement.get(case, {})
        row["recall_vs_imagecas_x_reference"] = _num(ag.get("recall"), 3)
        row["precision_vs_imagecas_x_reference"] = _num(ag.get("precision"), 3)
        row["missing_length_vs_imagecas_x_reference_mm"] = _num(ag.get("missing_length_mm"))
        row["comment"] = " ".join(f"[{n['mask_source']} mask] {n['note']}" for n in notes.get(case, []))
        rows.append(row)

    fields = ["case_id", "recommended", "why_recommended", "in_imagecas_x",
              "gml_imagecas", "vtk_imagecas", "slicer_imagecas", "gml_imagecas_x", "vtk_imagecas_x", "slicer_imagecas_x"]
    for prefix in ("imagecas", "imagecas_x"):
        fields += [f"{prefix}_{k}" for k in ("status", "qc", "qc_notes", "short_branches", "length_L_mm", "length_R_mm")]
    fields += ["recall_vs_imagecas_x_reference", "precision_vs_imagecas_x_reference",
               "missing_length_vs_imagecas_x_reference_mm", "comment"]
    with open(f"{a.out}/cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    shutil.copy2(os.path.join(HERE, "release_README.md"), f"{a.out}/README.md")
    if not a.no_copy:
        json.dump(validation, open(f"{a.out}/validation.json", "w"), indent=1)

    count = lambda key, value: sum(r[key] == value for r in rows)
    print(f"{len(rows)} cases: recommended ImageCAS {count('recommended', 'ImageCAS')}, "
          f"ImageCAS-X {count('recommended', 'ImageCAS-X')}, none {count('recommended', 'none')}")
    for s, prefix in (("ImageCAS", "imagecas"), ("ImageCAS-X", "imagecas_x")):
        print(f"  {s}: {count(f'{prefix}_status', 'ok')} graphs, "
              + ", ".join(f"{q} {count(f'{prefix}_qc', q)}" for q in list(QC_NAMES.values()) + ["left and right trees touch"]))
    if not a.no_copy:
        print(f"validation: {len(validation)} released GML file(s) with problems" + (f" (see {a.out}/validation.json)" if validation else ""))
    return 1 if validation else 0


if __name__ == "__main__":
    sys.exit(main())
