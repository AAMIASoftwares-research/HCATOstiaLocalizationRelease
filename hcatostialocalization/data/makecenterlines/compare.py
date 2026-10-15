"""
Compares centerline graphs (ours, v1 or v2) against reference centerlines (ASOCA,
ImageCAS-X), case by case, and summarises the batch.

    python -m hcatostialocalization.data.makecenterlines.compare \\
        --pred  <list.txt of our graphs> \\
        --ref   <list.txt of reference centerlines> \\
        [--ostia <list.txt of ostium JSONs>] [--images <list.txt of CCTA volumes>] \\
        -o <outdir>

Every list is a .txt/.list/.lst file (one path per line) or paths given inline, in the
same case order. A --pred entry that does not exist (e.g. '-') marks the case as
missing: it is flagged, not skipped silently. Supported formats, for both sides
(io.io.load_centerline_graph): .gml, .vtk/.vtp PolyData, and a folder of 3D Slicer
*.SlicerOpenCurve.mkr.json (or a single markup file).

What is measured (README.md "compare.py" has the definitions and the reasoning):
  - spatial agreement: distance from every reference point to our centerline and back,
    on both graphs resampled every --step-mm (p25, mean, median, p75, p95, max), the
    symmetric mean distance and the (95%) Hausdorff distance, and the same statistics
    restricted to the matched part (localisation accuracy where both have a vessel);
  - overlap: recall (= coverage of the reference, % of its length), precision, overlap
    OV and F1, with a point matched when it lies within the reference lumen radius
    (floored at --min-tolerance-mm; --tolerance-mm when the reference has no radius),
    as in the CAT08 evaluation framework (Schaap et al. 2009, .bib key 'cat08'); OT
    (overlap on vessels >= 1.5 mm wide, with radii); OF (overlap until the first error)
    per reference path, ignoring the first --of-skip-mm (40 mm); recall/precision at
    fixed tolerances; recall per anatomical segment when the reference has segment
    labels (ImageCAS-X);
  - missing centerlines: every connected stretch of the reference that we do not cover
    and that is at least --missing-min-length-mm long is reported and classified as a
    missing tree, a missing branch, a truncated branch or an interior gap; stretches of
    ours with no reference counterpart are reported as 'extra';
  - topology: endpoint detection (one-to-one matching within a tolerance), bifurcation
    distances binned 0-5 / 5-10 / 10-15 / >15 mm, ostium (root) distance, total length
    ratio, L/R side agreement;
  - radius error where both graphs carry radii.

Coordinate frames: our graphs are RAS mm. A VTK file has no frame, so for the reference
--ref-frame auto tries RAS, LPS and (with --images) ASOCA's raw frame
(io.io.asoca_raw_point_to_ras) and keeps the one closest to our graph; the choice and
every candidate's median distance are written to the output. Run from the repository root.
"""
import os
import sys
import csv
import json
import html
import argparse

import numpy
import networkx
from termcolor import colored

from hcatostialocalization.io import io as hio
from .utils.cli import expand_paths


SCOPES = ("ALL", "L", "R")
DISTANCE_STATS = ("p25", "mean", "median", "p75", "p95", "max")
AGGREGATE_STATS = ("n", "p25", "mean", "median", "p75", "max")
COLORS = {"ref": "#9a9993", "missed": "#e34948", "pred": "#2a78d6", "extra": "#eb6834", "L": "#2a78d6", "R": "#eb6834"}
_SUFFIXES = (".centerlines.gml", ".centerline.gml", ".gml", ".centerlines.vtk", ".centerlines.vtp", ".vtk", ".vtp",
             ".mkr.json", ".mrk.json", ".nii.gz", ".nrrd")


# ----------------------------------------------------------------------------- helpers

def _stats(values, keys=DISTANCE_STATS) -> dict:
    values = numpy.asarray(values, dtype=float)
    values = values[numpy.isfinite(values)]
    out = {}
    for key in keys:
        if key == "n":
            out[key] = int(len(values))
        elif len(values) == 0:
            out[key] = None
        elif key == "mean":
            out[key] = float(values.mean())
        elif key == "max":
            out[key] = float(values.max())
        elif key == "median":
            out[key] = float(numpy.median(values))
        else:
            out[key] = float(numpy.percentile(values, float(key[1:])))
    return out


def _case_id(path: str) -> str:
    name = os.path.basename(os.path.normpath(path))
    for suffix in _SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    for suffix in (".centerlines", "_centerlines", ".centerline", "_centerline", ".label", ".img"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _num(value):
    """None -> nan, for printing (a plain 'or' would turn 0.0 into nan too)."""
    return float("nan") if value is None else value


def _ratio(a, b):
    return float(a / b) if b else None


class _UnionFind:
    def __init__(self, n):
        self.parent = numpy.arange(n)

    def find(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def _runs(selected: numpy.ndarray, pairs: numpy.ndarray) -> list:
    """Connected groups of selected samples, adjacency given by sample pairs."""
    index = numpy.flatnonzero(selected)
    if len(index) == 0:
        return []
    uf = _UnionFind(len(selected))
    for i, j in pairs:
        if selected[i] and selected[j]:
            uf.union(i, j)
    groups = {}
    for i in index:
        groups.setdefault(uf.find(i), []).append(i)
    return [numpy.asarray(g) for g in groups.values()]


# ----------------------------------------------------------------------------- frames

def _frame_transforms(header) -> dict:
    """Candidate maps from a file's own coordinates to RAS mm."""
    transforms = {
        "ras": lambda p: p,
        "lps": lambda p: p * numpy.array([-1.0, -1.0, 1.0]),
    }
    if header is not None:
        origin, spacing, direction = header
        transforms["asoca-raw"] = lambda p: hio.asoca_raw_points_to_ras(p, origin, spacing, direction)
    return transforms


def _apply_frame(graph, transform):
    positions = numpy.array([graph.nodes[n]["pos"] for n in graph.nodes]).reshape(-1, 3)
    converted = transform(positions)
    for node, position in zip(graph.nodes, converted):
        graph.nodes[node]["pos"] = numpy.asarray(position, dtype=float)
    for u, v, data in graph.edges(data=True):
        data["length"] = float(numpy.linalg.norm(graph.nodes[u]["pos"] - graph.nodes[v]["pos"]))


def _resolve_frame(graph, requested, header, target_points, warn_mm, from_targets=False):
    """Puts graph in RAS. requested='auto' keeps the candidate frame whose points are
    closest (median nearest distance) to target_points (the other graph, already RAS).
    from_targets=True measures from the targets to the graph instead (used with the two
    annotated ostia when there is no graph of ours to compare against)."""

    def median_distance(points):
        if from_targets:
            return float(numpy.median(hio._nearest_neighbours(target_points, points)[0]))
        return float(numpy.median(hio._nearest_neighbours(points, target_points)[0]))

    transforms = _frame_transforms(header)
    info = {"requested": requested, "candidates_median_mm": {}}
    if requested != "auto":
        if requested not in transforms:
            raise ValueError(f"frame '{requested}' needs --images (image header) for this case")
        chosen = requested
    elif graph.graph.get("frame_hint") == "ras":
        chosen = "ras"
    elif target_points is None or len(target_points) == 0:
        chosen = "ras"
        info["note"] = "nothing to compare against: RAS assumed"
    else:
        positions = numpy.array([graph.nodes[n]["pos"] for n in graph.nodes]).reshape(-1, 3)
        for name, transform in transforms.items():
            info["candidates_median_mm"][name] = median_distance(transform(positions))
        chosen = min(info["candidates_median_mm"], key=info["candidates_median_mm"].get)
    _apply_frame(graph, transforms[chosen])
    info["chosen"] = chosen
    if target_points is not None and len(target_points):
        positions = numpy.array([graph.nodes[n]["pos"] for n in graph.nodes]).reshape(-1, 3)
        median = median_distance(positions)
        info["median_mm_after"] = median
        info["warning"] = median > warn_mm
    return info


# ----------------------------------------------------------------------------- sides and roots

def _label_sides_and_roots(graph, ostia) -> dict:
    """Gives every node a side and finds one root per side. In order of preference:
    ostium nodes and side labels stored in the file; nodes the file flags as tree
    starts (ImageCAS-X start_points); the annotated ostia (--ostia):
    the component endpoint nearest an ostium is its root and the component takes that
    side; ASOCA-style files: the node most polylines start from is the root; with none
    of these, the two largest components are told apart by the root's R coordinate
    (right ostium = larger R). Small unlabelled pieces take the side of the nearest
    labelled node. Returns {'roots': {side: node}, 'side_source': str}."""
    components = sorted(networkx.connected_components(graph), key=len, reverse=True)
    sources = set()
    component_root, component_side = {}, {}
    for c, nodes in enumerate(components):
        nodes = list(nodes)
        ostium_nodes = [n for n in nodes if graph.nodes[n]["is_ostium"]]
        start_nodes = [n for n in nodes if graph.nodes[n].get("is_start")]
        endpoints = [n for n in nodes if graph.degree(n) <= 1] or nodes
        sides = [graph.nodes[n]["side"] for n in nodes if graph.nodes[n]["side"] in ("L", "R")]
        root, side = None, None
        if ostium_nodes:
            root = ostium_nodes[0]
            side = graph.nodes[root]["side"]
            sources.add("file")
        elif start_nodes:
            # several start flags (e.g. ImageCAS-X roots placed near the aorta): with the
            # annotated ostia, the one nearest an ostium; else the first
            root = start_nodes[0]
            if ostia and len(start_nodes) > 1:
                points = numpy.array(list(ostia.values()))
                root = min(start_nodes, key=lambda n: numpy.linalg.norm(points - graph.nodes[n]["pos"], axis=1).min())
            sources.add("start-flag")
        elif ostia:
            best = None
            for s, point in ostia.items():
                positions = numpy.array([graph.nodes[n]["pos"] for n in endpoints])
                distance = numpy.linalg.norm(positions - point, axis=1)
                k = int(distance.argmin())
                if best is None or distance[k] < best[0]:
                    best = (float(distance[k]), s, endpoints[k])
            _, side, root = best
            sources.add("ostia")
        elif any(graph.nodes[n]["line_start"] for n in nodes):
            root = max(nodes, key=lambda n: (graph.nodes[n]["line_start"], -graph.degree(n)))
            sources.add("line-start")
        if sides:
            side = max(("L", "R"), key=sides.count)
            sources.add("file")
        component_root[c], component_side[c] = root, side

    unlabelled = [c for c in range(len(components)) if component_side[c] is None and component_root[c] is not None]
    labelled_sides = {component_side[c] for c in component_side if component_side[c]}
    if unlabelled and not labelled_sides and len(unlabelled) >= 2:
        a, b = unlabelled[:2]
        xa, xb = graph.nodes[component_root[a]]["pos"][0], graph.nodes[component_root[b]]["pos"][0]
        component_side[a], component_side[b] = ("R", "L") if xa > xb else ("L", "R")
        sources.add("heuristic-x")
    elif len(unlabelled) == 1 and len(labelled_sides) == 1:
        component_side[unlabelled[0]] = ({"L", "R"} - labelled_sides).pop()
        sources.add("complement")

    for c, nodes in enumerate(components):
        for n in nodes:
            if graph.nodes[n]["side"] not in ("L", "R", "RL") and component_side[c]:
                graph.nodes[n]["side"] = component_side[c]
    labelled = [n for n in graph.nodes if graph.nodes[n]["side"] in ("L", "R", "RL")]
    missing = [n for n in graph.nodes if graph.nodes[n]["side"] not in ("L", "R", "RL")]
    if labelled and missing:
        _, index = hio._nearest_neighbours(numpy.array([graph.nodes[n]["pos"] for n in missing]),
                                           numpy.array([graph.nodes[n]["pos"] for n in labelled]))
        for n, k in zip(missing, index):
            graph.nodes[n]["side"] = graph.nodes[labelled[k]]["side"]
        sources.add("nearest")

    # one root per side: with the annotated ostia, the candidate nearest that side's
    # ostium (a fragmented reference, e.g. 30 % of ImageCAS-X cases, has several pieces
    # per side, and its largest piece need not start at the ostium); else the root of
    # the largest piece
    roots = {}
    for c, nodes in enumerate(components):
        side = component_side[c]
        if component_root[c] is None or side not in ("L", "R"):
            continue
        if side not in roots:
            roots[side] = component_root[c]
        elif ostia and side in ostia:
            nearer = lambda n: float(numpy.linalg.norm(graph.nodes[n]["pos"] - ostia[side]))
            if nearer(component_root[c]) < nearer(roots[side]):
                roots[side] = component_root[c]
    for n in graph.nodes:  # ostium nodes always win
        if graph.nodes[n]["is_ostium"] and graph.nodes[n]["side"] in ("L", "R"):
            roots[graph.nodes[n]["side"]] = n
    return {"roots": roots, "side_source": "+".join(sorted(sources)) or "none",
            "n_components": len(components)}


# ----------------------------------------------------------------------------- sampling

def _sample(graph, roots: dict, step: float) -> dict:
    """Resamples every edge into points spaced at most `step` apart (midpoints of equal
    sub-segments), so each point stands for `weight` mm of centerline and sums of
    weights are exact lengths. Also returns sample adjacency (along edges and across
    nodes), the distance from the side root along the graph, and which samples touch
    an endpoint or a bifurcation."""
    root_nodes = set(roots.values())
    geodesic = {}
    for root in root_nodes:
        for node, d in networkx.single_source_dijkstra_path_length(graph, root, weight="length").items():
            geodesic[node] = min(d, geodesic.get(node, numpy.inf))
    pos, weight, radius, side, geo, pairs, edge_samples, sample_node, segment = [], [], [], [], [], [], {}, [], []
    node_ends = {}
    count = 0
    for u, v, data in graph.edges(data=True):
        length = data["length"]
        if length <= 0:
            continue
        n = max(1, int(numpy.ceil(length / step)))
        t = (numpy.arange(n) + 0.5) / n
        pu, pv = graph.nodes[u]["pos"], graph.nodes[v]["pos"]
        pos.append(pu + t[:, None] * (pv - pu))
        weight.append(numpy.full(n, length / n))
        radius.append(graph.nodes[u]["r"] + t * (graph.nodes[v]["r"] - graph.nodes[u]["r"]))
        su, sv = graph.nodes[u]["side"], graph.nodes[v]["side"]
        side.append(numpy.array([su if su in ("L", "R") else sv] * n, dtype=object))
        gu, gv = graph.nodes[u].get("segment", numpy.nan), graph.nodes[v].get("segment", numpy.nan)
        segment.append(numpy.full(n, gv if (numpy.isnan(gu) or (not numpy.isnan(gv) and graph.degree(u) >= 3)) else gu))
        du, dv = geodesic.get(u, numpy.nan), geodesic.get(v, numpy.nan)
        geo.append(du + t * length if not (dv < du) else dv + (1 - t) * length)
        sample_node.extend([u if not (dv < du) else v] * n)  # proximal node of the sample's edge
        index = numpy.arange(count, count + n)
        pairs.extend(zip(index[:-1], index[1:]))
        edge_samples[(u, v)] = index
        node_ends.setdefault(u, []).append(index[0])
        node_ends.setdefault(v, []).append(index[-1])
        count += n
    near_endpoint = numpy.zeros(count, dtype=bool)
    near_bifurcation = numpy.zeros(count, dtype=bool)
    for node, ends in node_ends.items():
        pairs.extend((ends[0], e) for e in ends[1:])
        degree = graph.degree(node)
        if degree == 1 and node not in root_nodes:
            near_endpoint[ends] = True
        elif degree >= 3:
            near_bifurcation[ends] = True
    cat = (lambda parts, empty: numpy.concatenate(parts) if parts else empty)
    return {
        "pos": cat(pos, numpy.zeros((0, 3))), "weight": cat(weight, numpy.zeros(0)),
        "radius": cat(radius, numpy.zeros(0)), "side": cat(side, numpy.zeros(0, dtype=object)),
        "segment": cat(segment, numpy.zeros(0)),
        "geo": cat(geo, numpy.zeros(0)), "pairs": numpy.asarray(pairs, dtype=int).reshape(-1, 2),
        "edge_samples": edge_samples, "near_endpoint": near_endpoint, "near_bifurcation": near_bifurcation,
        "proximal_node": sample_node, "geodesic": geodesic,
    }


def _path_samples(path_nodes, edge_samples) -> numpy.ndarray:
    parts = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        if (a, b) in edge_samples:
            parts.append(edge_samples[(a, b)])
        elif (b, a) in edge_samples:
            parts.append(edge_samples[(b, a)][::-1])
    return numpy.concatenate(parts) if parts else numpy.zeros(0, dtype=int)


# ----------------------------------------------------------------------------- metrics

def _match_landmarks(ref_points, pred_points, tolerance) -> dict:
    """Greedy one-to-one matching by increasing distance, within tolerance."""
    out = {"n_ref": len(ref_points), "n_pred": len(pred_points), "n_matched": 0,
           "recall": None, "precision": None, "error_mean_mm": None}
    if len(ref_points) == 0 or len(pred_points) == 0:
        out["recall"] = 0.0 if len(ref_points) else None
        out["precision"] = 0.0 if len(pred_points) else None
        return out
    distance = numpy.linalg.norm(numpy.asarray(ref_points)[:, None] - numpy.asarray(pred_points)[None], axis=2)
    used_r, used_p, errors = set(), set(), []
    for flat in numpy.argsort(distance, axis=None):
        i, j = numpy.unravel_index(flat, distance.shape)
        if distance[i, j] > tolerance:
            break
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        errors.append(distance[i, j])
    out.update(n_matched=len(errors), recall=len(errors) / len(ref_points), precision=len(errors) / len(pred_points),
               error_mean_mm=float(numpy.mean(errors)) if errors else None)
    return out


def _bucket_landmarks(ref_points, pred_points, edges) -> dict:
    """Every reference landmark is paired one-to-one (greedy, by increasing distance, no
    distance cap) with one of ours; the pairing distances are binned at `edges` (mm).
    Reference landmarks left unpaired (we have fewer) go in the last, open bin."""
    labels = [f"{lo:g}-{hi:g}mm" for lo, hi in zip([0] + list(edges[:-1]), edges)] + [f">{edges[-1]:g}mm"]
    out = {f"pct_{label}": None for label in labels}
    out.update(n_ref=len(ref_points), n_pred=len(pred_points), error_median_mm=None)
    if len(ref_points) == 0:
        return out
    distances = numpy.full(len(ref_points), numpy.inf)
    if len(pred_points):
        distance = numpy.linalg.norm(numpy.asarray(ref_points)[:, None] - numpy.asarray(pred_points)[None], axis=2)
        used_r, used_p = set(), set()
        for flat in numpy.argsort(distance, axis=None):
            i, j = numpy.unravel_index(flat, distance.shape)
            if i in used_r or j in used_p:
                continue
            used_r.add(i)
            used_p.add(j)
            distances[i] = distance[i, j]
    bins = numpy.digitize(distances, edges, right=True)  # 0: <= edges[0], ..., len(edges): beyond
    for k, label in enumerate(labels):
        out[f"pct_{label}"] = float((bins == k).mean())
    finite = distances[numpy.isfinite(distances)]
    out["error_median_mm"] = float(numpy.median(finite)) if len(finite) else None
    return out


def _landmarks(graph, roots, side):
    nodes = [n for n in graph.nodes if side == "ALL" or graph.nodes[n]["side"] == side]
    root_nodes = set(roots.values())
    endpoints = [graph.nodes[n]["pos"] for n in nodes if graph.degree(n) == 1 and n not in root_nodes]
    bifurcations = [graph.nodes[n]["pos"] for n in nodes if graph.degree(n) >= 3]
    return endpoints, bifurcations


def _upstream_bifurcation_distance(graph, roots, node, geodesic):
    """Distance along the tree from node back to the nearest bifurcation (degree >= 3)
    between it and its root, or None."""
    for root in roots.values():
        if node in geodesic and networkx.has_path(graph, root, node):
            for upstream in reversed(networkx.shortest_path(graph, root, node, weight="length")):
                if graph.degree(upstream) >= 3:
                    return float(geodesic[node] - geodesic[upstream])
            return None
    return None


def _classify_runs(samples, covered, min_length, graph=None, roots=None, branch_origin_mm=5.0, segment_names=None) -> list:
    """Uncovered stretches of a sampled graph at least min_length long, proximal end
    first. kind: 'missing_branch' (reaches an endpoint and starts within branch_origin_mm
    of a bifurcation: the first mm of a missed side branch lie inside the parent lumen,
    so they count as matched), 'truncated' (reaches an endpoint, starts mid-vessel),
    'gap' (interior)."""
    out = []
    for run in _runs(~covered, samples["pairs"]):
        length = float(samples["weight"][run].sum())
        if length < min_length:
            continue
        geo = samples["geo"][run]
        order = numpy.argsort(numpy.where(numpy.isfinite(geo), geo, numpy.inf))
        proximal, distal = run[order[0]], run[order[-1]]
        reaches_endpoint = bool(samples["near_endpoint"][run].any())
        if reaches_endpoint:
            at_branch = bool(samples["near_bifurcation"][proximal])
            if not at_branch and graph is not None and numpy.isfinite(samples["geo"][proximal]):
                node = samples["proximal_node"][proximal]
                upstream = _upstream_bifurcation_distance(graph, roots, node, samples["geodesic"])
                if upstream is not None:
                    at_branch = upstream + (samples["geo"][proximal] - samples["geodesic"][node]) <= branch_origin_mm
            kind = "missing_branch" if at_branch else "truncated"
        else:
            kind = "gap"
        sides = list(samples["side"][run])
        labels = [int(x) for x in samples.get("segment", numpy.zeros(0))[run] if numpy.isfinite(x)] if len(samples.get("segment", [])) else []
        segment = None
        if labels:  # the anatomical segment of the stretch (references with labels, e.g. ImageCAS-X)
            label = max(set(labels), key=labels.count)
            segment = (segment_names or {}).get(str(label), str(label))
        out.append({
            "kind": kind,
            "side": max(set(sides), key=sides.count) if sides else None,
            "segment": segment,
            "length_mm": length,
            "distance_from_ostium_mm": float(geo[order[0]]) if numpy.isfinite(geo[order[0]]) else None,
            "start_ras": [float(c) for c in samples["pos"][proximal]],
            "end_ras": [float(c) for c in samples["pos"][distal]],
            "mid_ras": [float(c) for c in samples["pos"][run[order[len(order) // 2]]]],
        })
    return sorted(out, key=lambda r: -r["length_mm"])


def compare_graphs(ref, pred, ref_roots, pred_roots, args) -> dict:
    """All metrics for one case (both graphs in RAS, sides and roots labelled)."""
    R, P = _sample(ref, ref_roots, args.step_mm), _sample(pred, pred_roots, args.step_mm)
    Rf, Pf = _sample(ref, ref_roots, args.fine_step_mm), _sample(pred, pred_roots, args.fine_step_mm)
    d_rp, i_rp = hio._nearest_neighbours(R["pos"], Pf["pos"])
    d_pr, i_pr = hio._nearest_neighbours(P["pos"], Rf["pos"])

    has_radius = len(R["radius"]) > 0 and numpy.isfinite(R["radius"]).mean() > 0.9 and numpy.nanmax(R["radius"]) > 0
    mode = "radius" if (args.tolerance == "radius" and has_radius) else "fixed"

    def tolerance(radius):
        if mode == "radius":
            return numpy.where(numpy.isfinite(radius), numpy.maximum(radius, args.min_tolerance_mm), args.tolerance_mm)
        return numpy.full(len(radius), args.tolerance_mm)

    cov_r = d_rp <= tolerance(R["radius"])
    cov_p = d_pr <= tolerance(Rf["radius"][i_pr]) if len(i_pr) and len(Rf["radius"]) else numpy.zeros(len(d_pr), bool)

    result = {"tolerance_mode": mode, "scopes": {}}
    for scope in SCOPES:
        rm = numpy.ones(len(d_rp), bool) if scope == "ALL" else (R["side"] == scope)
        pm = numpy.ones(len(d_pr), bool) if scope == "ALL" else (P["side"] == scope)
        ref_len, pred_len = float(R["weight"][rm].sum()), float(P["weight"][pm].sum())
        tpr, tpm = float(R["weight"][rm & cov_r].sum()), float(P["weight"][pm & cov_p].sum())
        recall, precision = _ratio(tpr, ref_len), _ratio(tpm, pred_len)
        m = {
            "ref_length_mm": ref_len, "pred_length_mm": pred_len, "length_ratio": _ratio(pred_len, ref_len),
            "recall": recall, "precision": precision,
            "overlap": _ratio(tpr + tpm, ref_len + pred_len),
            "f1": (2 * recall * precision / (recall + precision)) if recall and precision else 0.0,
        }
        if has_radius:
            # OT: overlap on the clinically relevant vessels (reference diameter >= --clinical-diameter-mm)
            rc = rm & (R["radius"] >= args.clinical_diameter_mm / 2)
            pc = pm & (Rf["radius"][i_pr] >= args.clinical_diameter_mm / 2) if len(i_pr) else pm
            m["overlap_clinical"] = _ratio(float(R["weight"][rc & cov_r].sum() + P["weight"][pc & cov_p].sum()),
                                           float(R["weight"][rc].sum() + P["weight"][pc].sum()))
        m.update({f"ref_to_pred_{k}": v for k, v in _stats(d_rp[rm]).items()})
        m.update({f"pred_to_ref_{k}": v for k, v in _stats(d_pr[pm]).items()})
        m.update({f"matched_{k}": v for k, v in _stats(d_rp[rm & cov_r]).items()})
        both = ref_len + pred_len
        m["assd_mm"] = float((d_rp[rm] @ R["weight"][rm] + d_pr[pm] @ P["weight"][pm]) / both) if both and rm.any() and pm.any() else None
        m["hd_mm"] = max(m["ref_to_pred_max"] or 0, m["pred_to_ref_max"] or 0) if rm.any() and pm.any() else None
        m["hd95_mm"] = max(m["ref_to_pred_p95"] or 0, m["pred_to_ref_p95"] or 0) if rm.any() and pm.any() else None
        for t in args.tolerance_curve:
            m[f"recall_at_{t:g}mm"] = _ratio(float(R["weight"][rm & (d_rp <= t)].sum()), ref_len)
            m[f"precision_at_{t:g}mm"] = _ratio(float(P["weight"][pm & (d_pr <= t)].sum()), pred_len)
        # radius error where both have a radius
        rad_ok = rm & cov_r & numpy.isfinite(R["radius"])
        if rad_ok.any() and len(Pf["radius"]):
            diff = Pf["radius"][i_rp[rad_ok]] - R["radius"][rad_ok]
            diff = diff[numpy.isfinite(diff)]
            m["radius_error_mean_mm"] = float(diff.mean()) if len(diff) else None
            m["radius_error_median_mm"] = float(numpy.median(diff)) if len(diff) else None
            m["radius_error_mae_mm"] = float(numpy.abs(diff).mean()) if len(diff) else None
        # side agreement on the matched part
        sm = rm & cov_r
        if scope == "ALL" and sm.any() and len(Pf["side"]):
            m["side_agreement"] = float(R["weight"][sm][Pf["side"][i_rp[sm]] == R["side"][sm]].sum() / R["weight"][sm].sum())
        # landmarks
        ref_end, ref_bif = _landmarks(ref, ref_roots, scope)
        pred_end, pred_bif = _landmarks(pred, pred_roots, scope)
        m.update({f"endpoint_{k}": v for k, v in _match_landmarks(ref_end, pred_end, args.endpoint_tol_mm).items()})
        m.update({f"bifurcation_{k}": v for k, v in _bucket_landmarks(ref_bif, pred_bif, args.bifurcation_buckets_mm).items()})
        # root (ostium) distance
        if scope in ("L", "R") and scope in ref_roots and scope in pred_roots:
            m["root_distance_mm"] = float(numpy.linalg.norm(ref.nodes[ref_roots[scope]]["pos"] - pred.nodes[pred_roots[scope]]["pos"]))
        # OF: overlap until the first error, per reference path, ignoring the first
        # --of-skip-mm from the root (ostium definitions differ between datasets)
        of_values = []
        for side in (("L", "R") if scope == "ALL" else (scope,)):
            if side not in ref_roots:
                continue
            root = ref_roots[side]
            component = networkx.node_connected_component(ref, root)
            paths = networkx.single_source_dijkstra_path(ref.subgraph(component), root, weight="length")
            for node in component:
                if node == root or ref.degree(node) != 1:
                    continue
                index = _path_samples(paths[node], R["edge_samples"])
                arc = numpy.cumsum(R["weight"][index]) if len(index) else numpy.zeros(0)
                index = index[arc > args.of_skip_mm]
                if len(index) == 0:
                    continue  # path not longer than the skipped part
                misses = numpy.flatnonzero(~cov_r[index])
                upto = index[: misses[0]] if len(misses) else index
                of_values.append(float(R["weight"][upto].sum() / R["weight"][index].sum()))
        m["of_mean"] = float(numpy.mean(of_values)) if of_values else None
        m["n_ref_paths"] = len(of_values)
        result["scopes"][scope] = m

    missing = _classify_runs(R, cov_r, args.missing_min_length_mm, ref, ref_roots, args.branch_origin_mm,
                             segment_names=ref.graph.get("segment_names", {}))
    extra = [dict(r, kind="extra") for r in _classify_runs(P, cov_p, args.missing_min_length_mm)]
    for side in ("L", "R"):
        s = result["scopes"][side]
        if s["ref_length_mm"] > 0 and (s["recall"] or 0) < args.missing_tree_recall:
            missing = [r for r in missing if r["side"] != side]
            missing.insert(0, {"kind": "missing_tree", "side": side, "segment": None, "length_mm": s["ref_length_mm"],
                               "distance_from_ostium_mm": 0.0, "start_ras": None, "end_ras": None, "mid_ras": None})
    result["missing"] = missing
    result["extra"] = extra
    # recall per anatomical segment, when the reference carries labels (ImageCAS-X: segment_label
    # ids, with their names -- LM, LAD, LCX, D1, OM1, ... -- read from its segment_name array)
    segment_names = ref.graph.get("segment_names", {})
    result["segments"] = {}
    labelled = numpy.isfinite(R["segment"])
    for label in sorted(set(R["segment"][labelled].astype(int).tolist())):
        sel = labelled & (R["segment"] == label)
        sides = list(R["side"][sel])
        result["segments"][str(label)] = {
            "segment_name": segment_names.get(str(label)),
            "side": max(set(sides), key=sides.count) if sides else None,
            "ref_length_mm": float(R["weight"][sel].sum()),
            "recall": _ratio(float(R["weight"][sel & cov_r].sum()), float(R["weight"][sel].sum())),
            "ref_to_pred_median": _stats(d_rp[sel])["median"],
        }
    for scope in SCOPES:
        sel = [r for r in missing if scope == "ALL" or r["side"] == scope]
        for kind in ("missing_tree", "missing_branch", "truncated", "gap"):
            result["scopes"][scope][f"n_{kind}"] = sum(1 for r in sel if r["kind"] == kind)
        result["scopes"][scope]["missing_length_mm"] = float(sum(r["length_mm"] for r in sel if r["kind"] != "missing_tree"))
        result["scopes"][scope]["n_extra"] = sum(1 for r in extra if scope == "ALL" or r["side"] == scope)
    result["_plot"] = {"ref": R["pos"], "ref_covered": cov_r, "pred": P["pos"], "pred_covered": cov_p,
                       "d_rp": d_rp, "d_pr": d_pr}
    return result


# ----------------------------------------------------------------------------- figures

def _plot_case(case_id, result, ref, pred, ref_roots, pred_roots, ostia, file_path):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    data = result["_plot"]
    s = result["scopes"]["ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, (y, ylabel, title) in zip(axes, ((1, "A (mm)", "Top view (R-A)"), (2, "S (mm)", "Front view (R-S)"))):
        ref_pos, cov = data["ref"], data["ref_covered"]
        ax.scatter(ref_pos[cov, 0], ref_pos[cov, y], s=6, color=COLORS["ref"], linewidths=0, label="reference, matched")
        ax.scatter(ref_pos[~cov, 0], ref_pos[~cov, y], s=9, color=COLORS["missed"], linewidths=0, label="reference, not matched")
        pred_pos, pcov = data["pred"], data["pred_covered"]
        ax.scatter(pred_pos[pcov, 0], pred_pos[pcov, y], s=1.2, color=COLORS["pred"], linewidths=0, label="ours, matched")
        ax.scatter(pred_pos[~pcov, 0], pred_pos[~pcov, y], s=2.5, color=COLORS["extra"], linewidths=0, label="ours, extra")
        for roots, graph, marker, label in ((ref_roots, ref, "o", "reference root"), (pred_roots, pred, "^", "our root")):
            points = numpy.array([graph.nodes[n]["pos"] for n in roots.values()]).reshape(-1, 3)
            if len(points):
                ax.scatter(points[:, 0], points[:, y], s=70, marker=marker, facecolors="none", edgecolors="black",
                           linewidths=1.2, label=label)
        if ostia:
            points = numpy.array(list(ostia.values()))
            ax.scatter(points[:, 0], points[:, y], s=25, marker="x", color="black", linewidths=1.2, label="annotated ostia")
        for run in result["missing"]:
            if run["mid_ras"] is not None:
                ax.annotate(f"{run['kind'].replace('_', ' ')} {run['length_mm']:.0f} mm", (run["mid_ras"][0], run["mid_ras"][y]),
                            fontsize=7, color="#52514e", xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("R (mm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#e6e5e0", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.suptitle(f"{case_id}: recall {100 * (s['recall'] or 0):.1f}%, precision {100 * (s['precision'] or 0):.1f}%, "
                 f"median distance {_num(s['ref_to_pred_median']):.2f} mm "
                 f"(tolerance: {result['tolerance_mode']})", fontsize=11)
    fig.savefig(file_path, dpi=150)
    plt.close(fig)


def _plot_batch(cases, pooled, tolerance_curve, out_dir):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    ok = [c for c in cases if c["status"] == "ok"]
    if not ok:
        return []
    written = []

    # recall / precision vs tolerance: one thin line per case, median bold
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, sharey=True)
    for ax, key, title in ((axes[0], "recall", "Recall (coverage of the reference)"), (axes[1], "precision", "Precision")):
        curves = numpy.array([[c["metrics"]["scopes"]["ALL"][f"{key}_at_{t:g}mm"] or 0 for t in tolerance_curve] for c in ok]) * 100
        for curve in curves:
            ax.plot(tolerance_curve, curve, color=COLORS["ref"], linewidth=0.6, alpha=0.6)
        ax.plot(tolerance_curve, numpy.median(curves, axis=0), color=COLORS["pred"], linewidth=2, marker="o", markersize=4)
        ax.annotate("median", (tolerance_curve[-1], numpy.median(curves, axis=0)[-1]), xytext=(-40, -14),
                    textcoords="offset points", fontsize=8, color="#52514e")
        ax.set_xlabel("Distance tolerance (mm)")
        ax.set_title(title, fontsize=10)
        ax.grid(color="#e6e5e0", linewidth=0.5)
    axes[0].set_ylabel("% of length (one line per case)")
    axes[0].set_ylim(0, 101)
    path = os.path.join(out_dir, "coverage_vs_tolerance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # pooled distance distributions (ECDF)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for key, label, color in (("ref_to_pred", "reference to ours", COLORS["pred"]), ("pred_to_ref", "ours to reference", COLORS["extra"])):
        values = numpy.sort(pooled[key])
        if len(values):
            ax.plot(values, numpy.arange(1, len(values) + 1) / len(values) * 100, color=color, linewidth=2, label=label)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("Distance to the other centerline (mm)")
    ax.set_ylabel("% of centerline length")
    ax.set_title("Pooled distances, all cases", fontsize=10)
    ax.grid(color="#e6e5e0", linewidth=0.5)
    ax.legend(fontsize=8)
    path = os.path.join(out_dir, "distance_ecdf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # per-case metrics by side
    panels = (("recall", "Recall (%)", 100), ("precision", "Precision (%)", 100),
              ("ref_to_pred_median", "Median distance (mm)", 1), ("of_mean", "OF (%)", 100))
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.8), constrained_layout=True)
    for ax, (key, label, scale) in zip(axes, panels):
        series = [[c["metrics"]["scopes"][side][key] * scale for c in ok if c["metrics"]["scopes"][side].get(key) is not None]
                  for side in ("L", "R")]
        series = [values if values else [numpy.nan] for values in series]
        boxes = ax.boxplot(series, patch_artist=True, widths=0.5, medianprops={"color": "black"}, flierprops={"markersize": 3})
        ax.set_xticks([1, 2], ["Left", "Right"])
        for patch, side in zip(boxes["boxes"], ("L", "R")):
            patch.set_facecolor(COLORS[side])
            patch.set_alpha(0.6)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.5)
    path = os.path.join(out_dir, "per_case_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def _write_html(cases, out_dir, file_path):
    rows = []
    order = sorted(cases, key=lambda c: (c["status"] == "ok", (c.get("metrics") or {}).get("scopes", {}).get("ALL", {}).get("recall") or 0))
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}"
    mm = lambda v: "-" if v is None else f"{v:.2f}"
    for c in order:
        if c["status"] != "ok":
            rows.append(f"<tr class='bad'><td>{html.escape(c['case_id'])}</td><td colspan='9'>{html.escape(c['status'])}: "
                        f"{html.escape(str(c.get('error', '')))}</td></tr>")
            continue
        s = c["metrics"]["scopes"]
        flags = ", ".join(f"{r['kind'].replace('_', ' ')} {r['side'] or ''} {r['length_mm']:.0f} mm" for r in c["metrics"]["missing"]) or "-"
        figure = os.path.relpath(c["figure"], out_dir) if c.get("figure") else None
        image = f"<a href='{html.escape(figure)}'><img src='{html.escape(figure)}' loading='lazy'></a>" if figure else ""
        rows.append(
            f"<tr class='{'flag' if c['metrics']['missing'] else ''}'><td>{html.escape(c['case_id'])}</td>"
            f"<td>{pct(s['ALL']['recall'])}</td><td>{pct(s['L']['recall'])}</td><td>{pct(s['R']['recall'])}</td>"
            f"<td>{pct(s['ALL']['precision'])}</td><td>{mm(s['ALL']['ref_to_pred_median'])}</td>"
            f"<td>{mm(s['ALL']['hd95_mm'])}</td><td>{html.escape(c['frame']['chosen'])}</td>"
            f"<td>{html.escape(flags)}</td><td>{image}</td></tr>"
        )
    batch = "".join(f"<img class='wide' src='{os.path.basename(p)}'>" for p in ("coverage_vs_tolerance.png", "distance_ecdf.png", "per_case_metrics.png")
                    if os.path.exists(os.path.join(out_dir, p)))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Centerline comparison</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;color:#0b0b0b;background:#fcfcfb}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #e6e5e0;padding:4px 8px;text-align:left;vertical-align:top}}
th{{position:sticky;top:0;background:#fcfcfb}}tr.flag td:first-child{{border-left:4px solid #e34948}}tr.bad{{background:#fbe9e9}}
img{{width:360px}}img.wide{{width:min(100%,900px);display:block;margin-bottom:12px}}
</style></head><body><h1>Centerline comparison</h1><p>Cases sorted by recall, worst first. Recall = % of the reference
length matched by our centerline; distances in mm; red bar = at least one missing stretch flagged.</p>{batch}
<table><tr><th>case</th><th>recall %</th><th>L</th><th>R</th><th>precision %</th><th>median d</th><th>HD95</th><th>frame</th>
<th>flags</th><th>figure</th></tr>{''.join(rows)}</table></body></html>"""
    with open(file_path, "w") as f:
        f.write(page)


# ----------------------------------------------------------------------------- batch

def _ostia_for_case(path, image_path):
    if not path:
        return None
    simple = hio.extract_simplified_ras(path, image_path)
    if simple is None:
        raise ValueError(f"cannot read ostia from {path}")
    return {s: numpy.asarray(simple[s]["ras"], dtype=float) for s in ("L", "R") if len(simple.get(s, {}).get("ras") or []) == 3}


def _load(path_field, fmt, overlap_mm):
    """One graph from a list entry; 'a.vtk;b.vtk' (e.g. one file per side) is read as the
    union of the files, each keeping the side its own name suggests."""
    graph = None
    for path in path_field.split(";"):
        part = hio.load_centerline_graph(path.strip(), fmt, overlap_tol_mm=overlap_mm)
        if graph is None:
            graph = part
        else:
            attrs = dict(graph.graph)
            names = {**attrs.get("segment_names", {}), **part.graph.get("segment_names", {})}
            graph = networkx.disjoint_union(graph, part)
            graph.graph.update(attrs, source=attrs["source"] + ";" + part.graph["source"], segment_names=names)
    return graph


SEVERITIES = ("failed_comparison", "major", "minor", "agreement")
SEVERITY_TITLES = {"failed_comparison": "Not compared", "major": "Major discrepancies", "minor": "Minor discrepancies",
                   "agreement": "Agreement"}
KIND_TITLES = {"missing_tree": "missing tree", "missing_branch": "missing branch", "truncated": "truncated (ours stops early)",
               "gap": "gap (ours leaves the reference and comes back)", "extra": "ours only (not in the reference)"}


def _case_severity(case, major_mm: float, major_recall: float) -> str:
    """failed_comparison: no graph of ours or an error; major: a missing tree, any discrepancy
    at least major_mm long, recall below major_recall, or a frame warning; minor: any other
    discrepancy; agreement: none."""
    if case["status"] != "ok":
        return "failed_comparison"
    runs = case["metrics"]["missing"] + case["metrics"]["extra"]
    recall = case["metrics"]["scopes"]["ALL"]["recall"] or 0.0
    if (case["frame"].get("warning") or recall < major_recall or any(r["kind"] == "missing_tree" for r in runs)
            or any(r["length_mm"] >= major_mm for r in runs)):
        return "major"
    return "minor" if runs else "agreement"


def _write_discrepancies(cases, out_dir, major_mm: float, major_recall: float) -> dict:
    """discrepancies.csv (one row per discrepancy, with the case severity) and
    discrepancies.html (cases grouped by severity, worst first, each with what is
    different, where and how much). Returns the counts per severity."""
    for c in cases:
        c["severity"] = _case_severity(c, major_mm, major_recall)
    order = {s: k for k, s in enumerate(SEVERITIES)}

    def worst(c):
        runs = (c.get("metrics") or {}).get("missing", []) + (c.get("metrics") or {}).get("extra", [])
        return (order[c["severity"]], -max((r["length_mm"] for r in runs), default=0.0), c["case_id"])
    ranked = sorted(cases, key=worst)

    rows = []
    for c in ranked:
        if c["status"] != "ok":
            rows.append({"case_id": c["case_id"], "severity": c["severity"], "kind": c["status"], "note": c.get("error", "")})
            continue
        for r in sorted(c["metrics"]["missing"] + c["metrics"]["extra"], key=lambda r: -r["length_mm"]):
            rows.append({"case_id": c["case_id"], "severity": c["severity"], "kind": r["kind"], "side": r["side"] or "",
                         "segment": r.get("segment") or "", "length_mm": round(r["length_mm"], 1),
                         "distance_from_ostium_mm": None if r["distance_from_ostium_mm"] is None else round(r["distance_from_ostium_mm"], 1),
                         "start_ras": r["start_ras"], "end_ras": r["end_ras"],
                         "recall": round(c["metrics"]["scopes"]["ALL"]["recall"] or 0, 4)})
    with open(os.path.join(out_dir, "discrepancies.csv"), "w", newline="") as f:
        fields = ["case_id", "severity", "kind", "side", "segment", "length_mm", "distance_from_ostium_mm", "start_ras", "end_ras", "recall", "note"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(c["severity"] == s for c in cases) for s in SEVERITIES}
    fmt_ras = lambda p: "–" if p is None else "(" + ", ".join(f"{v:.1f}" for v in p) + ")"

    def case_block(c):
        figure = os.path.relpath(c["figure"], out_dir) if c.get("figure") else None
        image = f'<a href="{html.escape(figure)}"><img src="{html.escape(figure)}" loading="lazy"></a>' if figure else ""
        if c["status"] != "ok":
            return f'<div class="case failed_comparison"><h3>{html.escape(c["case_id"])}</h3><p>{html.escape(c["status"])}: {html.escape(str(c.get("error", "")))}</p></div>'
        s = c["metrics"]["scopes"]["ALL"]
        lines = []
        for r in sorted(c["metrics"]["missing"] + c["metrics"]["extra"], key=lambda r: -r["length_mm"]):
            where = ", ".join(x for x in (f"side {r['side']}" if r["side"] else "", f"segment {r['segment']}" if r.get("segment") else "",
                                          f"{r['distance_from_ostium_mm']:.0f} mm from the ostium" if r["distance_from_ostium_mm"] is not None else "") if x)
            lines.append(f"<tr><td>{html.escape(KIND_TITLES.get(r['kind'], r['kind']))}</td><td>{r['length_mm']:.1f} mm</td>"
                         f"<td>{html.escape(where)}</td><td>{fmt_ras(r['start_ras'])}</td></tr>")
        frame = f"<p class='warn'>frame warning: median distance {c['frame'].get('median_mm_after', 0):.1f} mm</p>" if c["frame"].get("warning") else ""
        table = ("<table><tr><th>what</th><th>how much</th><th>where</th><th>starts at (RAS)</th></tr>" + "".join(lines) + "</table>") if lines else "<p>no discrepancy</p>"
        return (f'<div class="case {c["severity"]}"><h3>{html.escape(c["case_id"])}</h3>'
                f'<p class="small">recall {100 * (s["recall"] or 0):.1f} %, precision {100 * (s["precision"] or 0):.1f} %, '
                f'median distance {_num(s["ref_to_pred_median"]):.2f} mm</p>{frame}{table}{image}</div>')

    sections = "".join(
        f'<h2 id="{sev}">{SEVERITY_TITLES[sev]} ({counts[sev]})</h2>' + "".join(case_block(c) for c in ranked if c["severity"] == sev)
        for sev in SEVERITIES[:3])
    agreeing = ", ".join(html.escape(c["case_id"]) for c in ranked if c["severity"] == "agreement")
    summary = "".join(f"<tr><td>{SEVERITY_TITLES[s]}</td><td>{counts[s]}</td></tr>" for s in SEVERITIES)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Discrepancies with the reference</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;color:#111;background:#fff}} table{{border-collapse:collapse;font-size:13px;margin:4px 0}}
td,th{{border-bottom:1px solid #ddd;padding:3px 8px;text-align:left;vertical-align:top}}
.case{{border:1px solid #ccc;border-left-width:5px;border-radius:6px;padding:8px;margin:10px 0}} .case h3{{margin:0 0 4px}}
.case.failed_comparison{{border-left-color:#c62828}} .case.major{{border-left-color:#ef6c00}} .case.minor{{border-left-color:#1565c0}}
.case img{{max-width:640px;width:100%;display:block;margin-top:6px}} .small{{font-size:12px;color:#555;margin:0}} .warn{{color:#c62828}}
</style></head><body><h1>Discrepancies with the reference</h1>
<table><tr><th>Severity</th><th>Cases</th></tr>{summary}</table>
<p>Major: a missing tree, a discrepancy of at least {major_mm:g} mm, recall below {100 * major_recall:g} %, or a frame warning.
Minor: shorter discrepancies (unmatched stretches from the flag threshold up). Within each group, the case with the
longest discrepancy comes first. Segment names appear when the reference carries them (ImageCAS-X). Every discrepancy is
also in discrepancies.csv.</p>
{sections}
<h2 id="agreement">Agreement ({counts["agreement"]})</h2><p>{agreeing or "none"}</p></body></html>"""
    with open(os.path.join(out_dir, "discrepancies.html"), "w") as f:
        f.write(page)
    return counts


def compare_case(case_id, ref_path, pred_path, ostium_path, image_path, args, figure_dir):
    result = {"case_id": case_id, "ref": ref_path, "pred": pred_path}
    header = None
    if image_path:
        origin, spacing, direction, _ = hio.read_image_header_lps(image_path)
        header = (origin, spacing, direction)
    ostia = _ostia_for_case(ostium_path, image_path)
    if pred_path is None or not all(os.path.exists(p) for p in pred_path.split(";")):
        ref = _load(ref_path, args.ref_format, args.overlap_mm)
        result["frame"] = _resolve_frame(ref, args.ref_frame, header, numpy.array(list(ostia.values())) if ostia else None,
                                         args.frame_warn_mm, from_targets=True)
        labels = _label_sides_and_roots(ref, ostia)
        result.update(status="missing_prediction", error=f"prediction not found: {pred_path}",
                      ref_length_mm=float(sum(d["length"] for _, _, d in ref.edges(data=True))),
                      ref_sides=sorted({ref.nodes[n]["side"] for n in ref.nodes if ref.nodes[n]["side"]}),
                      ref_side_source=labels["side_source"])
        return result

    pred = _load(pred_path, args.pred_format, args.overlap_mm)
    if args.pred_frame != "ras":
        transforms = _frame_transforms(header)
        if args.pred_frame not in transforms:
            raise ValueError(f"--pred-frame {args.pred_frame} needs --images")
        _apply_frame(pred, transforms[args.pred_frame])
    pred_points = numpy.array([pred.nodes[n]["pos"] for n in pred.nodes]).reshape(-1, 3)
    ref = _load(ref_path, args.ref_format, args.overlap_mm)
    result["frame"] = _resolve_frame(ref, args.ref_frame, header, pred_points, args.frame_warn_mm)
    ref_labels = _label_sides_and_roots(ref, ostia)
    pred_labels = _label_sides_and_roots(pred, ostia)
    result["ref_side_source"], result["pred_side_source"] = ref_labels["side_source"], pred_labels["side_source"]
    result["ref_components"], result["pred_components"] = ref_labels["n_components"], pred_labels["n_components"]
    metrics = compare_graphs(ref, pred, ref_labels["roots"], pred_labels["roots"], args)
    if ostia:
        for side, point in ostia.items():
            for name, graph, roots in (("ref", ref, ref_labels["roots"]), ("pred", pred, pred_labels["roots"])):
                if side in roots:
                    metrics["scopes"][side][f"{name}_root_to_annotated_ostium_mm"] = float(numpy.linalg.norm(graph.nodes[roots[side]]["pos"] - point))
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
        result["figure"] = os.path.join(figure_dir, f"{case_id}.png")
        _plot_case(case_id, metrics, ref, pred, ref_labels["roots"], pred_labels["roots"], ostia, result["figure"])
    result["_plot"] = metrics.pop("_plot")
    result.update(status="ok", metrics=metrics)
    return result


def _flag_rows(case):
    rows = []
    base = {"case_id": case["case_id"]}
    if case["status"] == "missing_prediction":
        rows.append(dict(base, kind="missing_case", side="", length_mm=case.get("ref_length_mm"), note=case.get("error", "")))
        return rows
    if case["status"] != "ok":
        rows.append(dict(base, kind="error", side="", note=case.get("error", "")))
        return rows
    if case["frame"].get("warning"):
        rows.append(dict(base, kind="frame_warning", side="",
                         note=f"median distance {case['frame'].get('median_mm_after', 0):.1f} mm after frame '{case['frame']['chosen']}'"))
    for run in case["metrics"]["missing"] + case["metrics"]["extra"]:
        rows.append(dict(base, kind=run["kind"], side=run["side"] or "", segment=run.get("segment") or "", length_mm=round(run["length_mm"], 2),
                         distance_from_ostium_mm=None if run["distance_from_ostium_mm"] is None else round(run["distance_from_ostium_mm"], 1),
                         start_ras=run["start_ras"], end_ras=run["end_ras"]))
    return rows


_SEVERITY = {"missing_case": 0, "error": 0, "missing_tree": 1, "frame_warning": 2, "missing_branch": 3,
             "truncated": 4, "gap": 5, "extra": 6}


def main():
    parser = argparse.ArgumentParser(description="Compare our centerline graphs against reference centerlines (ASOCA, ImageCAS-X, ...).")
    parser.add_argument("--pred", nargs="+", required=True, help="Our centerlines (.gml, .vtk/.vtp, or folder of Slicer open curves), or one .txt list. A path that does not exist marks the case as missing.")
    parser.add_argument("--ref", nargs="+", required=True, help="Reference centerlines, same order/count as --pred, or one .txt list.")
    parser.add_argument("--ostia", nargs="+", default=None, help="Optional ostium JSONs (any format io.io.extract_simplified_ras reads), same order. Used to assign L/R and roots to graphs that carry no labels, and to report root-to-ostium distances.")
    parser.add_argument("--images", nargs="+", default=None, help="Optional CCTA volumes, same order. Only their headers are read: needed for the ASOCA raw frame (--ref-frame asoca-raw, or as an 'auto' candidate) and for raw ASOCA ostium files.")
    parser.add_argument("--case-ids", nargs="+", default=None, help="Optional case ids, same order (default: from the reference file names).")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--ref-format", default="auto", choices=["auto", "gml", "vtk", "slicer"])
    parser.add_argument("--pred-format", default="auto", choices=["auto", "gml", "vtk", "slicer"])
    parser.add_argument("--ref-frame", default="auto", choices=["auto", "ras", "lps", "asoca-raw"], help="Frame of the reference coordinates. 'auto' keeps, among ras/lps(/asoca-raw with --images), the one closest to our graph. GML and Slicer files are taken as RAS.")
    parser.add_argument("--pred-frame", default="ras", choices=["ras", "lps", "asoca-raw"], help="Frame of our coordinates (our GML files are RAS).")
    parser.add_argument("--frame-warn-mm", type=float, default=5.0, help="Flag the case if the median reference-to-ours distance is above this after the frame choice (likely a frame problem, not a centerline one).")
    parser.add_argument("--overlap-mm", type=float, default=1.0, help="Polyline formats only: geometry repeated within this distance (one polyline per ostium-to-endpoint path, as in ASOCA) is merged into one tree. 0 = keep every polyline as is.")
    parser.add_argument("--step-mm", type=float, default=0.25, help="Sampling step along both graphs for the measured points.")
    parser.add_argument("--fine-step-mm", type=float, default=0.05, help="Sampling step of the curve each point is measured against (distance error <= half of it).")
    parser.add_argument("--tolerance", default="radius", choices=["radius", "fixed"], help="'radius': a point is matched within the reference lumen radius (floored at --min-tolerance-mm), as in the CAT08 framework; falls back to 'fixed' when the reference has no radius. 'fixed': within --tolerance-mm.")
    parser.add_argument("--tolerance-mm", type=float, default=2.0)
    parser.add_argument("--min-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--tolerance-curve", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0], help="Fixed tolerances (mm) for the recall/precision curves.")
    parser.add_argument("--major-mm", type=float, default=20.0, help="discrepancies.html: a discrepancy at least this long makes the case 'major'.")
    parser.add_argument("--major-recall", type=float, default=0.9, help="discrepancies.html: a recall below this makes the case 'major'.")
    parser.add_argument("--missing-min-length-mm", type=float, default=10.0, help="Unmatched stretches shorter than this are not flagged (v2 trims 2 mm from every endpoint; tips are matched within --endpoint-tol-mm).")
    parser.add_argument("--branch-origin-mm", type=float, default=5.0, help="An unmatched stretch reaching an endpoint is a 'missing_branch' if it starts within this distance (along the reference) of a bifurcation, else 'truncated'.")
    parser.add_argument("--missing-tree-recall", type=float, default=0.05, help="A side whose recall is below this is flagged as a missing tree.")
    parser.add_argument("--endpoint-tol-mm", type=float, default=10.0)
    parser.add_argument("--clinical-diameter-mm", type=float, default=1.5, help="OT (references with radii only): overlap restricted to reference vessels at least this wide, as in the CAT08 framework.")
    parser.add_argument("--of-skip-mm", type=float, default=40.0, help="OF ignores the first this-many mm of every reference path from the root (paths not longer are skipped).")
    parser.add_argument("--bifurcation-buckets-mm", type=float, nargs="+", default=[5.0, 10.0, 15.0], help="Bin edges (mm) for the distance between each reference bifurcation and its one-to-one paired bifurcation of ours: 0-5, 5-10, 10-15, >15 by default (unpaired ones fall in the last bin).")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    refs = expand_paths(args.ref)
    preds = expand_paths(args.pred)
    ostia = expand_paths(args.ostia) if args.ostia else [None] * len(refs)
    images = expand_paths(args.images) if args.images else [None] * len(refs)
    case_ids = [os.path.basename(c) for c in expand_paths(args.case_ids)] if args.case_ids else [_case_id(p.split(";")[0]) for p in refs]
    if not (len(refs) == len(preds) == len(ostia) == len(images) == len(case_ids)):
        parser.error(f"--ref ({len(refs)}), --pred ({len(preds)}), --ostia ({len(ostia)}), --images ({len(images)}) "
                     f"and --case-ids ({len(case_ids)}) must have the same count.")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    figure_dir = None if args.no_figures else os.path.join(out_dir, "figures")
    cases = []
    pooled = {"ref_to_pred": [], "pred_to_ref": []}
    for k, (case_id, ref_path, pred_path, ostium_path, image_path) in enumerate(zip(case_ids, refs, preds, ostia, images), start=1):
        print(colored(f"[{k}/{len(refs)}] {case_id}", "cyan"))
        try:
            case = compare_case(case_id, ref_path, pred_path, ostium_path, image_path, args, figure_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            case = {"case_id": case_id, "ref": ref_path, "pred": pred_path, "status": "error", "error": f"{type(e).__name__}: {e}"}
        if case["status"] == "ok":
            pooled["ref_to_pred"].append(case["_plot"]["d_rp"])
            pooled["pred_to_ref"].append(case["_plot"]["d_pr"])
            s = case["metrics"]["scopes"]["ALL"]
            flags = [f"{r['kind']} {r['side'] or ''} {r['length_mm']:.0f} mm" for r in case["metrics"]["missing"]]
            print(colored(f"    recall {100 * (s['recall'] or 0):.1f}%  precision {100 * (s['precision'] or 0):.1f}%  "
                          f"median {_num(s['ref_to_pred_median']):.2f} mm  frame {case['frame']['chosen']}", "white"))
            if flags or case["frame"].get("warning"):
                print(colored("    FLAGS: " + "; ".join(flags + (["frame warning"] if case["frame"].get("warning") else [])), "yellow"))
        else:
            print(colored(f"    {case['status'].upper()}: {case.get('error', '')}", "red"))
        case.pop("_plot", None)
        cases.append(case)
    pooled = {k: numpy.concatenate(v) if v else numpy.zeros(0) for k, v in pooled.items()}

    # per-case CSV (one row per case and scope)
    ok = [c for c in cases if c["status"] == "ok"]
    metric_keys = []
    for c in ok:
        for scope in SCOPES:
            for key in c["metrics"]["scopes"][scope]:
                if key not in metric_keys:
                    metric_keys.append(key)
    with open(os.path.join(out_dir, "per_case.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "scope", "status", "frame", "tolerance_mode"] + metric_keys)
        for c in cases:
            if c["status"] != "ok":
                writer.writerow([c["case_id"], "ALL", c["status"], "", ""] + [""] * len(metric_keys))
                continue
            for scope in SCOPES:
                m = c["metrics"]["scopes"][scope]
                writer.writerow([c["case_id"], scope, "ok", c["frame"]["chosen"], c["metrics"]["tolerance_mode"]]
                                + ["" if m.get(k) is None else m.get(k) for k in metric_keys])

    # per-segment recall (references with segment labels only)
    segment_rows = [dict(case_id=c["case_id"], segment=label, **values)
                    for c in ok for label, values in c["metrics"].get("segments", {}).items()]
    if segment_rows:
        with open(os.path.join(out_dir, "segments.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "segment", "segment_name", "side", "ref_length_mm", "recall", "ref_to_pred_median"])
            writer.writeheader()
            writer.writerows(segment_rows)
    # per segment id, and per segment name (ids can differ between files, names are the anatomy)
    segment_aggregate = {"by_id": {}, "by_name": {}}
    for group, key_of in (("by_id", lambda r: r["segment"]), ("by_name", lambda r: r["segment_name"])):
        keys = sorted({key_of(r) for r in segment_rows if key_of(r) is not None}, key=str)
        for key in keys:
            rows = [r for r in segment_rows if key_of(r) == key]
            entry = {m: _stats([r[m] for r in rows if r[m] is not None], AGGREGATE_STATS)
                     for m in ("recall", "ref_length_mm", "ref_to_pred_median")}
            if group == "by_id":
                names = [r["segment_name"] for r in rows if r["segment_name"]]
                entry["segment_name"] = max(set(names), key=names.count) if names else None
            segment_aggregate[group][str(key)] = entry

    # flags CSV, most severe first
    flag_rows = sorted((row for c in cases for row in _flag_rows(c)),
                       key=lambda r: (_SEVERITY.get(r["kind"], 9), -(r.get("length_mm") or 0)))
    with open(os.path.join(out_dir, "flags.csv"), "w", newline="") as f:
        fields = ["case_id", "kind", "side", "segment", "length_mm", "distance_from_ostium_mm", "start_ras", "end_ras", "note"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flag_rows)

    # summary JSON: statistics over cases of every per-case metric, pooled distances, every case
    aggregate = {scope: {key: _stats([c["metrics"]["scopes"][scope].get(key) for c in ok
                                      if isinstance(c["metrics"]["scopes"][scope].get(key), (int, float))], AGGREGATE_STATS)
                         for key in metric_keys} for scope in SCOPES}
    summary = {
        "options": vars(args),
        "n_cases": len(cases),
        "n_ok": len(ok),
        "n_missing_prediction": sum(1 for c in cases if c["status"] == "missing_prediction"),
        "n_error": sum(1 for c in cases if c["status"] == "error"),
        "n_cases_flagged": sum(1 for c in ok if c["metrics"]["missing"]),
        "flag_counts": {kind: sum(1 for r in flag_rows if r["kind"] == kind) for kind in _SEVERITY},
        "pooled_distances_mm": {k: _stats(v) for k, v in pooled.items()},
        "aggregate_over_cases": aggregate,
        "segments_over_cases": segment_aggregate,
        "cases": cases,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    if not args.no_figures:
        _plot_batch(cases, pooled, args.tolerance_curve, out_dir)
    _write_html(cases, out_dir, os.path.join(out_dir, "report.html"))
    severity_counts = _write_discrepancies(cases, out_dir, args.major_mm, args.major_recall)
    print(colored("Discrepancies: " + ", ".join(f"{n} {SEVERITY_TITLES[s].lower()}" for s, n in severity_counts.items())
                  + f" (discrepancies.html / .csv)", "cyan"))

    a = aggregate["ALL"]
    print(colored(f"Done: {len(ok)} compared, {summary['n_missing_prediction']} missing, {summary['n_error']} failed, "
                  f"{summary['n_cases_flagged']} flagged.", "cyan"))
    if ok:
        print(colored(f"Recall median {100 * (a['recall']['median'] or 0):.1f}% (p25 {100 * (a['recall']['p25'] or 0):.1f}%), "
                      f"median distance median {_num(a['ref_to_pred_median']['median']):.2f} mm. "
                      f"Outputs in {out_dir}", "cyan"))
    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
