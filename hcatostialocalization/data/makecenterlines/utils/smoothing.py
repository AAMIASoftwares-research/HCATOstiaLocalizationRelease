"""
Branch-wise geometric smoothing of a rooted centerline tree, and the direction
roughness measure that motivates it.

Why: a minimal path on a voxel grid is a staircase of 26-connected steps. Its
position is within a fraction of a voxel of the vessel axis, but its direction
jumps by up to ~45 degrees from one step to the next. Anything that uses the
local direction -- a curved planar reformat (each slice is a plane normal to the
curve), cross-sections, curvature or tortuosity measures -- sees that jitter. A
light smoothing (sigma = 0.5 mm on 1 mm-spaced nodes, v1's setting) leaves a
median turn of ~5 degrees per 0.5 mm, i.e. a radius of curvature of ~6 mm almost
everywhere, and adjacent reformat planes then cross each other a few mm from the
curve (checked on the first 5 ImageCAS cases, 2026-10-01).

How:
  1. The tree is split into branches by longest-path decomposition: from each
     ostium, the main branch follows, at every bifurcation, the child with the
     longest path to a tip; every other child starts a side branch at that
     bifurcation, decomposed the same way. Branches are processed parent first.
  2. Each branch is smoothed as one curve: positions (and radii) are replaced by
     a Gaussian-weighted average over arc length (sigma_mm). Padding by point
     reflection at both ends (p[-k] = 2 p[0] - p[k]) keeps a straight branch
     straight and leaves the two end nodes exactly in place. So the ostium, the
     tips and the attachment of every side branch to its (already smoothed)
     parent do not move, and the main vessel passes smoothly through its
     bifurcations instead of kinking at every landmark.
  3. A node never moves farther than max_shift_fraction x its radius (and
     max_shift_mm), so the smoothed curve stays well inside the lumen; where the
     limit bites (tight bends in thin vessels), the shift is shortened, not
     dropped.
Topology is untouched: no node or edge is added or removed.
"""
import numpy
import networkx


def _children(tree, parent_of, node):
    return [n for n in tree.neighbors(node) if parent_of.get(n) == node]


def branch_decomposition(tree: networkx.Graph, root, length: str = "length") -> list:
    """Longest-path decomposition of a tree from root: a list of node paths, the
    main branch first, every side branch starting at its bifurcation node, each
    listed after its parent branch."""
    parent_of = dict(networkx.bfs_predecessors(tree, root))
    order = [root] + [n for _, n in networkx.bfs_edges(tree, root)]
    downstream = {}
    for node in reversed(order):
        downstream[node] = max((tree.edges[node, c][length] + downstream[c] for c in _children(tree, parent_of, node)),
                               default=0.0)
    branches, queue = [], [[root]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        while True:
            children = _children(tree, parent_of, node)
            if not children:
                break
            children.sort(key=lambda c: tree.edges[node, c][length] + downstream[c], reverse=True)
            for side in children[1:]:
                queue.append([node, side])
            node = children[0]
            path.append(node)
        branches.append(path)
    return branches


def _smooth_curve(points: numpy.ndarray, values: numpy.ndarray, sigma: float):
    """Gaussian smoothing over arc length of an (n, 3) polyline and of (n,) per-node
    values, ends fixed (point reflection for positions, mirror for values)."""
    n = len(points)
    if n < 3 or sigma <= 0:
        return points.copy(), values.copy()
    s = numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(points, axis=0), axis=1))])
    reach = 3.0 * sigma
    k_left = int(numpy.searchsorted(s, reach, side="right"))
    k_right = int(numpy.searchsorted(s[-1] - s[::-1], reach, side="right"))
    idx_l = numpy.arange(1, min(k_left, n - 1) + 1)
    idx_r = numpy.arange(1, min(k_right, n - 1) + 1)
    pad_points = numpy.vstack([2 * points[0] - points[idx_l][::-1], points, 2 * points[-1] - points[n - 1 - idx_r]])
    pad_values = numpy.concatenate([values[idx_l][::-1], values, values[n - 1 - idx_r]])
    pad_s = numpy.concatenate([-s[idx_l][::-1], s, 2 * s[-1] - s[n - 1 - idx_r]])
    weights = numpy.exp(-((s[:, None] - pad_s[None, :]) ** 2) / (2.0 * sigma ** 2))
    weights[numpy.abs(s[:, None] - pad_s[None, :]) > reach] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    smoothed_points = weights @ pad_points
    finite = numpy.isfinite(pad_values)
    smoothed_values = (weights * numpy.where(finite, pad_values, 0.0)) @ numpy.ones(len(pad_values))
    norm = weights @ finite.astype(float)
    smoothed_values = numpy.where(norm > 0, smoothed_values / numpy.where(norm > 0, norm, 1.0), values)
    smoothed_points[0], smoothed_points[-1] = points[0], points[-1]
    smoothed_values[0], smoothed_values[-1] = values[0], values[-1]
    return smoothed_points, smoothed_values


def smooth_tree_branches(graph: networkx.Graph, roots, sigma_mm: float = 2.0, max_shift_fraction: float = 0.4,
                         max_shift_mm: float = 1.0, min_shift_mm: float = 0.0, pos: str = "pos_ras", radius: str = "r",
                         smooth_radius: bool = True, inside=None) -> dict:
    """In place: smooths every tree of graph that contains one of roots (see the
    module docstring). Returns QC: number of branches, nodes moved, the largest
    shift (mm) and how many shifts were shortened by the limits.

    inside: optional callable, position -> bool (is it in the segmentation mask?).
    With it, the shift limit is the mask itself rather than the radius: a node that is
    inside the mask may move up to max_shift_mm as long as it stays inside; if its
    smoothed position would fall outside, the shift is halved until it is inside, and
    if that never happens (down to the radius limit) the radius limit applies. Thin
    vessels are then smoothed as much as wide ones, and no node inside the mask can be
    moved out of it. Nodes outside the mask (dilation shell, gap bridges, ostium side)
    keep the radius limit. Without inside: radius limit only (max_shift_fraction x
    radius, floored at min_shift_mm, at most max_shift_mm)."""
    for u, v in graph.edges:
        graph.edges[u, v]["_length"] = float(numpy.linalg.norm(numpy.asarray(graph.nodes[u][pos]) - numpy.asarray(graph.nodes[v][pos])))
    original = {n: numpy.asarray(graph.nodes[n][pos], dtype=float).copy() for n in graph.nodes}
    qc = {"sigma_mm": sigma_mm, "max_shift_fraction": max_shift_fraction, "max_shift_mm": max_shift_mm, "min_shift_mm": min_shift_mm,
          "shift_limit": "mask" if inside is not None else "radius",
          "n_branches": 0, "n_nodes_moved": 0, "n_shifts_limited": 0, "max_shift_mm_applied": 0.0}
    for root in roots:
        if root not in graph:
            continue
        component = graph.subgraph(networkx.node_connected_component(graph, root))
        for branch in branch_decomposition(component, root, length="_length"):
            qc["n_branches"] += 1
            # parent first: branch[0] already holds its final (smoothed) position
            points = numpy.array([graph.nodes[n][pos] for n in branch], dtype=float)
            radii = numpy.array([graph.nodes[n].get(radius, numpy.nan) for n in branch], dtype=float)
            new_points, new_radii = _smooth_curve(points, radii, sigma_mm)
            for k, node in enumerate(branch[1:-1], start=1):
                shift = new_points[k] - original[node]
                distance = float(numpy.linalg.norm(shift))
                radius_limit = max_shift_mm
                if numpy.isfinite(radii[k]) and radii[k] > 0:
                    # never below min_shift_mm: in thin vessels a fraction of the radius is
                    # smaller than the voxel staircase the smoothing has to remove
                    radius_limit = min(radius_limit, max(max_shift_fraction * radii[k], min_shift_mm))
                limited = False
                if inside is not None and inside(original[node]):
                    if distance > max_shift_mm:
                        shift *= max_shift_mm / distance
                        distance, limited = max_shift_mm, True
                    while not inside(original[node] + shift) and distance > radius_limit:
                        shift *= 0.5
                        distance, limited = distance * 0.5, True
                    if not inside(original[node] + shift) and distance > 0:
                        shift *= min(1.0, radius_limit / distance)
                        distance, limited = min(distance, radius_limit), True
                elif distance > radius_limit:
                    shift *= radius_limit / distance
                    distance, limited = radius_limit, True
                qc["n_shifts_limited"] += int(limited)
                graph.nodes[node][pos] = original[node] + shift
                if smooth_radius and numpy.isfinite(new_radii[k]):
                    graph.nodes[node][radius] = float(new_radii[k])
                qc["n_nodes_moved"] += 1
                qc["max_shift_mm_applied"] = max(qc["max_shift_mm_applied"], distance)
    for u, v in graph.edges:
        graph.edges[u, v].pop("_length", None)
    return qc


def direction_roughness(graph: networkx.Graph, roots, pos: str = "pos_ras", step_mm: float = 0.5) -> dict:
    """Turn angle (degrees) between consecutive steps along every branch of the
    tree, after resampling each branch to step_mm, so the numbers do not depend on
    the node spacing. A real coronary artery turns by ~1 degree per 0.5 mm or less
    away from tight bends (radius of curvature >= ~30 mm); a voxel staircase by
    10-45 degrees. Also the radius of curvature this implies (mm)."""
    angles = []
    for root in roots:
        if root not in graph:
            continue
        component = graph.subgraph(networkx.node_connected_component(graph, root)).copy()
        for u, v in component.edges:
            component.edges[u, v]["_length"] = float(numpy.linalg.norm(numpy.asarray(component.nodes[u][pos]) - numpy.asarray(component.nodes[v][pos])))
        for branch in branch_decomposition(component, root, length="_length"):
            points = numpy.array([component.nodes[n][pos] for n in branch], dtype=float)
            s = numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(points, axis=0), axis=1))])
            if s[-1] < 3 * step_mm:
                continue
            t = numpy.arange(0.0, s[-1], step_mm)
            resampled = numpy.column_stack([numpy.interp(t, s, points[:, k]) for k in range(3)])
            d = numpy.diff(resampled, axis=0)
            d /= numpy.linalg.norm(d, axis=1, keepdims=True)
            angles.extend(numpy.degrees(numpy.arccos(numpy.clip((d[1:] * d[:-1]).sum(1), -1.0, 1.0))).tolist())
    angles = numpy.asarray(angles)
    if len(angles) == 0:
        return {"n": 0}
    radius = step_mm / numpy.maximum(numpy.radians(angles), 1e-9)
    return {
        "n": int(len(angles)), "step_mm": step_mm,
        "turn_deg_median": float(numpy.median(angles)), "turn_deg_p95": float(numpy.percentile(angles, 95)),
        "turn_deg_max": float(angles.max()), "fraction_turn_above_5deg": float((angles > 5).mean()),
        "curvature_radius_mm_median": float(numpy.median(radius)), "curvature_radius_mm_p5": float(numpy.percentile(radius, 5)),
    }
