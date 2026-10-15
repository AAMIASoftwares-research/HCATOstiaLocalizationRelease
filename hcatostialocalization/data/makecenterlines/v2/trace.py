"""
Root-driven centerline tracing: grows each coronary tree from its annotated
ostium through the mask, with no skeletonisation.

Per side, everything follows from two single-source minimal-cost fronts started
at the ostium (skimage MCP_Geometric, 26-connected):
  - G, the geodesic distance through the mask with unit cost (the flood fill).
    It only orders the targets: the farthest voxel along the mask is the tip of
    the longest vessel not yet traced.
  - C, the cumulative centred cost, with per-voxel cost (DT + 0.5)^-p, DT being
    the distance (voxels) to the nearest background voxel. A minimal path then
    minimises  integral ds / DT^p : short AND far from the wall in one cost, p
    setting how strongly centring wins over length. Its traceback array is a
    single predecessor map rooted at the ostium.

Every path is read from that one predecessor map, so a voxel's path to the
ostium is always the same: the union of all paths is a tree by construction,
and a new path joins the tree at its first voxel already in it and never
leaves it again. No loop can form, so none has to be removed.

Tree growth is the TEASAR farthest-point loop: take the uncovered mask voxel
with the largest G, trace its path, measure how far the path runs outside the
tube already covered by the tree, cover a radius-aware tube around it (only
voxels in line of sight of the path through the mask, so a parallel vessel
next to it is not swallowed), and keep
the path only if that length is at least min_branch_length_mm. Bumps and wall
pockets are selected too, but their paths leave the covered tube for only a few
mm, so they are rejected; this is what makes endpoint detection reliable
without looking for local maxima.

Domain and costs (working grid, cropped to the arteries):
  mask      the resampled mask, the source of truth for what is a vessel;
  domain    mask dilated by dilation_voxels, to repair connections that
            nearest-neighbour resampling broke. Its extra voxels (the "shell")
            cost shell_penalty times more, so paths use them only where the
            mask itself is broken;
  bridge    a ball around each ostium, just large enough to reach the domain:
            the ostium sits on the aortic wall, mostly outside the mask, and
            enters it by the shortest route (cost shell_penalty per voxel,
            more than anywhere in the domain).
Mask pieces reachable from neither ostium are not bridged, only reported.

If the two ostia reach the same domain (trees touching in the mask), each
voxel goes to the ostium with the smaller G (geodesic Voronoi split).

Bulky junctions: (DT + 0.5)^-p rewards depth without limit, so where the mask
bulges (an over-segmented bifurcation) the deep centre of the bulge attracts the
path, off the real lumen. By default the trees are therefore grown twice, the
second time with DT capped at about the vessel's own radius, measured along the
first tree (capped_distance). The cap only bites where DT exceeds the local
vessel radius, i.e. in bulges; elsewhere the cost is unchanged.

Endpoints: the tips run to the very end of the mask; trim_endpoints() shortens
each by a fixed length on the finished tree.

This module only needs numpy/scipy/scikit-image/networkx, so it can be tested
on small synthetic volumes without the rest of the package.
"""
import networkx
import numpy
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as _ndimage_label
from scipy.spatial import cKDTree

from ..utils.grid import longest_uncovered_stretch

_CONNECTIVITY_26 = numpy.ones((3, 3, 3), dtype=int)
_BRIDGE_MARGIN_MM = 1.0
_MAX_TARGETS_RECORDED = 50
_MAX_PIECES_RECORDED = 20


def connectivity_domain(mask: numpy.ndarray, dilation_voxels: int) -> numpy.ndarray:
    """mask dilated by a ball of radius dilation_voxels (0 = mask itself)."""
    if dilation_voxels <= 0:
        return mask.copy()
    from skimage.morphology import ball
    return binary_dilation(mask, structure=ball(dilation_voxels))


def bridge_mask_gaps(domain, ostium_ijk: dict, max_gap_mm: float, spacing_mm: float,
                     min_piece_mm3: float = 10.0, max_bridges: int = 20):
    """Joins mask pieces disconnected from the tree domain by short gaps. A coronary
    mask should be one tree per side, but segmentations can be interrupted (stenosis,
    calcium, motion, partial volume) by a few millimetres; everything beyond such a gap
    would otherwise be unreachable from the ostium and lost.

    The pieces reached from the ostia (the components holding the domain voxels nearest
    to each ostium) form the connected set. Repeatedly, the remaining piece nearest to
    the connected set (at least min_piece_mm3) is joined to it by a straight one-voxel
    line between their closest voxels, if that gap is at most max_gap_mm; the piece then
    joins the connected set, so chains of pieces are joined in order. The line voxels
    become part of the domain but not of the mask: they cost like the dilation shell
    (shell penalty), so paths use them only to cross the gap. max_gap_mm <= 0: off.

    Returns (domain with the bridges, bridge voxels bool array, list of bridges with
    their length, end points (ijk) and the piece volume)."""
    bridges_mask = numpy.zeros(domain.shape, dtype=bool)
    if max_gap_mm <= 0:
        return domain, bridges_mask, []
    labels, n_labels = _ndimage_label(domain, structure=_CONNECTIVITY_26)
    if n_labels < 2:
        return domain, bridges_mask, []
    voxel_mm3 = spacing_mm ** 3
    sizes = numpy.bincount(labels.ravel())
    voxels = numpy.argwhere(domain)
    voxel_labels = labels[tuple(voxels.T)]
    _, nearest = cKDTree(voxels).query(numpy.array([ostium_ijk["L"], ostium_ijk["R"]]))
    connected = {int(voxel_labels[k]) for k in nearest}
    candidates = {k for k in range(1, n_labels + 1) if k not in connected and sizes[k] * voxel_mm3 >= min_piece_mm3}
    bridges = []
    while candidates and len(bridges) < max_bridges:
        connected_voxels = voxels[numpy.isin(voxel_labels, list(connected))]
        tree = cKDTree(connected_voxels)
        best = None
        for k in candidates:
            piece = voxels[voxel_labels == k]
            distance, index = tree.query(piece)
            j = int(numpy.argmin(distance))
            if best is None or distance[j] < best[0]:
                best = (float(distance[j]), k, piece[j], connected_voxels[index[j]])
        gap_voxels, k, a, b = best
        if gap_voxels * spacing_mm > max_gap_mm:
            break
        n_steps = max(2, int(numpy.ceil(gap_voxels * 4)))
        line = numpy.rint(a[None] + (b - a)[None] * numpy.linspace(0.0, 1.0, n_steps)[:, None]).astype(int)
        bridges_mask[tuple(line.T)] = True
        bridges.append({"gap_mm": gap_voxels * spacing_mm, "piece_ijk": [int(c) for c in a],
                        "tree_side_ijk": [int(c) for c in b], "piece_volume_mm3": float(sizes[k] * voxel_mm3)})
        connected.add(k)
        candidates.discard(k)
    bridges_mask &= ~domain
    return domain | bridges_mask, bridges_mask, bridges


def centred_cost(mask, domain, dt_voxels, centering_power: float, shell_penalty: float) -> numpy.ndarray:
    """(DT + 0.5)^-p in the mask, shell_penalty times that in the dilation shell,
    infinite (impassable) elsewhere."""
    base = (dt_voxels + 0.5) ** (-float(centering_power))
    cost = numpy.full(mask.shape, numpy.inf)
    cost[domain] = shell_penalty * base[domain]
    cost[mask] = base[mask]
    return cost


def ostium_bridge(domain, ostium_ijk, domain_tree: cKDTree, max_bridge_mm: float, spacing_mm: float):
    """Voxels outside the domain within (entry distance + _BRIDGE_MARGIN_MM) of
    the ostium, plus the ostium's own voxel. Raises if the domain is farther
    than max_bridge_mm from the ostium.

    Returns (bridge bool array, start voxel tuple, entry distance in mm).
    """
    entry_voxels, _ = domain_tree.query(ostium_ijk)
    entry_mm = float(entry_voxels) * spacing_mm
    if entry_mm > max_bridge_mm:
        raise ValueError(
            f"the mask is {entry_mm:.1f} mm from the ostium, more than the "
            f"{max_bridge_mm:.1f} mm allowed (--max-ostium-bridge-mm)"
        )
    shape = numpy.asarray(domain.shape)
    start = tuple(int(c) for c in numpy.clip(numpy.rint(ostium_ijk), 0, shape - 1))
    radius = entry_voxels + _BRIDGE_MARGIN_MM / spacing_mm
    lo = numpy.maximum(numpy.floor(ostium_ijk - radius), 0).astype(int)
    hi = numpy.minimum(numpy.ceil(ostium_ijk + radius) + 1, shape).astype(int)
    window = tuple(slice(a, b) for a, b in zip(lo, hi))
    grid = numpy.stack(numpy.meshgrid(*[numpy.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), axis=-1)
    within = numpy.linalg.norm(grid - ostium_ijk, axis=-1) <= radius
    bridge = numpy.zeros(domain.shape, dtype=bool)
    bridge[window] = within & ~domain[window]
    bridge[start] = not domain[start]
    return bridge, start, entry_mm


def single_source_front(cost: numpy.ndarray, start):
    """Minimal-cost front from one voxel. Returns (cumulative cost, MCP object,
    whose traceback() gives the minimal path from start to any reached voxel)."""
    from skimage.graph import MCP_Geometric
    mcp = MCP_Geometric(cost, fully_connected=True)
    cumulative, _ = mcp.find_costs(starts=[start])
    return cumulative, mcp


def split_left_right(geodesic_l: numpy.ndarray, geodesic_r: numpy.ndarray):
    """Domains reached by each ostium; where both reach, the voxel goes to the
    geodesically closer ostium. Returns (domain_l, domain_r, merged)."""
    reach_l = numpy.isfinite(geodesic_l)
    reach_r = numpy.isfinite(geodesic_r)
    overlap = reach_l & reach_r
    domain_l = reach_l & ~(overlap & (geodesic_r < geodesic_l))
    domain_r = reach_r & ~(overlap & (geodesic_l <= geodesic_r))
    return domain_l, domain_r, bool(overlap.any())


def _line_of_sight(starts: numpy.ndarray, ends: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    """True where the segment start -> end runs inside the mask, sampled every
    half voxel. Samples within one voxel of the start are not tested, since a
    path node may sit in the dilation shell."""
    delta = ends - starts
    length = numpy.linalg.norm(delta, axis=1)
    t = numpy.linspace(0.0, 1.0, max(2, int(numpy.ceil(length.max() * 2.0)) + 1))
    samples = numpy.rint(starts[:, None, :] + t[None, :, None] * delta[:, None, :]).astype(int)
    inside = mask[tuple(numpy.moveaxis(samples, -1, 0))]
    near_start = t[None, :] * length[:, None] < 1.0
    return numpy.all(inside | near_start, axis=1)


def grow_tree(mask, side_domain, geodesic, mcp, start, dt_voxels, spacing_mm: float,
              min_branch_length_mm: float, cover_radius_scale: float, cover_tol_mm: float):
    """Farthest-point loop over the side's mask voxels (see module docstring).

    A mask voxel inside the tube around a new path is covered only if the
    segment from its nearest path node to it stays inside the mask. A vessel
    running parallel to the path, separated by background (or only joined
    through the dilation shell), is therefore not swallowed by the tube and
    stays available as a target.

    Returns (edges, tree_volume, accepted, rejected): edges as (voxel, voxel)
    pairs, tree_volume a bool array of tree voxels, accepted/rejected lists of
    (target voxel, length outside the covered tube in mm).
    """
    candidates = numpy.argwhere(side_domain & mask)
    candidates = candidates[numpy.argsort(-geodesic[tuple(candidates.T)], kind="stable")]
    cover_ijk = numpy.argwhere(side_domain & mask)
    cover_tree = cKDTree(cover_ijk)
    covered = numpy.zeros(mask.shape, dtype=bool)
    tree_volume = numpy.zeros(mask.shape, dtype=bool)
    tree_volume[start] = True
    min_length_voxels = min_branch_length_mm / spacing_mm
    tol_voxels = cover_tol_mm / spacing_mm

    edges, accepted, rejected = [], [], []
    for target in (tuple(int(c) for c in v) for v in candidates):
        if covered[target]:
            continue
        path = [tuple(int(c) for c in p) for p in mcp.traceback(target)]
        # The tree is closed under the predecessor map, so its voxels form a
        # prefix of the path: keep from the last tree voxel (the junction) on.
        junction = 0
        while junction + 1 < len(path) and tree_volume[path[junction + 1]]:
            junction += 1
        new_part = path[junction:]
        length_voxels = longest_uncovered_stretch(new_part, covered)

        points = numpy.asarray(new_part, dtype=float)
        radii = cover_radius_scale * dt_voxels[tuple(numpy.asarray(new_part).T)] + tol_voxels
        hits = [h for h in cover_tree.query_ball_point(points, radii) if h]
        if hits:
            in_tube = cover_ijk[numpy.unique(numpy.concatenate(hits))]
            _, nearest = cKDTree(points).query(in_tube)
            visible = _line_of_sight(points[nearest], in_tube.astype(float), mask)
            covered[tuple(in_tube[visible].T)] = True
        covered[target] = True  # guarantees progress even for a zero-length path

        record = (target, float(length_voxels * spacing_mm))
        if length_voxels >= min_length_voxels:
            for a, b in zip(new_part[:-1], new_part[1:]):
                edges.append((a, b))
                tree_volume[b] = True
            accepted.append(record)
        else:
            rejected.append(record)
    return edges, tree_volume, accepted, rejected


def vessel_radius_along_tree(edges, dt_voxels, window_voxels: float):
    """Per tree voxel, the median DT of the tree voxels within window_voxels of
    it along the tree: the vessel's own radius, robust to the short DT spike of
    a bulky junction. Voxels outside the domain (the ostium bridge, DT = 0) are
    left out. Returns (voxels (N, 3), radius in voxels (N,))."""
    tree = networkx.Graph()
    for a, b in edges:
        tree.add_edge(a, b, distance=float(numpy.linalg.norm(numpy.subtract(a, b))))
    voxels, radius = [], []
    for voxel in tree.nodes:
        near = networkx.single_source_dijkstra_path_length(tree, voxel, cutoff=window_voxels, weight="distance")
        values = [dt_voxels[v] for v in near if dt_voxels[v] > 0]
        if values:
            voxels.append(voxel)
            radius.append(float(numpy.median(values)))
    return numpy.asarray(voxels, dtype=float).reshape(-1, 3), numpy.asarray(radius)


def capped_distance(dt_voxels, side_domains: dict, side_radii: dict, cap_scale: float):
    """DT with every domain voxel capped at cap_scale x the vessel radius of its
    nearest tree voxel (same side), never below 1 voxel. Inside a bulky junction
    the cost then stops rewarding depth beyond the vessel's own radius: the path
    takes the shortest route that is still a full radius deep, instead of being
    pulled to the centre of the bulge. Elsewhere DT stays at most ~1 radius, so
    the cap leaves it unchanged. Returns (capped DT, fraction of domain voxels
    capped)."""
    capped = dt_voxels.copy()
    for side, domain in side_domains.items():
        voxels, radius = side_radii[side]
        inside = numpy.argwhere(domain & (dt_voxels > 0))
        if len(voxels) == 0 or len(inside) == 0:
            continue
        _, nearest = cKDTree(voxels).query(inside)
        cap = numpy.maximum(cap_scale * radius[nearest], 1.0)
        index = tuple(inside.T)
        capped[index] = numpy.minimum(dt_voxels[index], cap)
    n_domain = int(sum((d & (dt_voxels > 0)).sum() for d in side_domains.values()))
    fraction = float((capped < dt_voxels).sum() / max(n_domain, 1))
    return capped, fraction


def trim_endpoints(graph: networkx.Graph, trim_mm: float, spacing_mm: float) -> int:
    """In place: removes trim_mm of centerline from every endpoint (the tips run
    to the very end of the mask), never past a bifurcation or an ostium. Uses
    the edges' 'distance' (voxels). Returns the number of endpoints trimmed."""
    if trim_mm <= 0:
        return 0
    trim_voxels = trim_mm / spacing_mm
    endpoints = [n for n, d in graph.degree() if d == 1 and not graph.nodes[n].get("is_ostium", False)]
    for current in endpoints:
        removed = 0.0
        while removed < trim_voxels and graph.degree(current) == 1:
            (following,) = graph.neighbors(current)
            if graph.degree(following) != 2 or graph.nodes[following].get("is_ostium", False):
                break
            removed += graph.edges[current, following]["distance"]
            graph.remove_node(current)
            current = following
    return len(endpoints)


def _centredness(tree_volume, dt_voxels) -> numpy.ndarray:
    """Per tree voxel, DT / (max DT within a ball of radius DT around it): 1 on
    the medial axis, lower where the path cuts a corner toward the wall. The
    ball reaches the axis whenever the voxel is at least half a radius deep."""
    out = []
    shape = numpy.asarray(dt_voxels.shape)
    for voxel in numpy.argwhere(tree_volume):
        dt = float(dt_voxels[tuple(voxel)])
        if dt <= 0:
            continue
        r = int(numpy.ceil(dt))
        lo = numpy.maximum(voxel - r, 0)
        hi = numpy.minimum(voxel + r + 1, shape)
        grid = numpy.stack(numpy.meshgrid(*[numpy.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), axis=-1)
        within = numpy.linalg.norm(grid - voxel, axis=-1) <= dt
        local = dt_voxels[tuple(slice(a, b) for a, b in zip(lo, hi))][within]
        out.append(dt / float(local.max()))
    return numpy.asarray(out)


def _extent_mm(voxels: numpy.ndarray, spacing_mm: float) -> float:
    """Span of a voxel set along its first principal axis: about the length of
    an elongated (vessel-like) region, small for a blob or a sliver of wall."""
    if len(voxels) < 2:
        return 0.0
    centred = voxels - voxels.mean(axis=0)
    _, _, axes = numpy.linalg.svd(centred, full_matrices=False)
    projection = centred @ axes[0]
    return float((projection.max() - projection.min()) * spacing_mm)


def _describe_regions(labels, n, tree_ijk, ijk_to_ras, spacing_mm: float, suspect_extent_mm: float) -> dict:
    """Volume, extent, distance to the tree and centroid of each labelled
    region; regions with extent >= suspect_extent_mm are counted as suspect.
    Regions too small to reach that extent are only counted in the volume, not
    described."""
    from scipy.ndimage import find_objects

    voxel_mm3 = spacing_mm ** 3
    out = {"n_regions": int(n), "volume_mm3": float((labels > 0).sum() * voxel_mm3),
           "n_suspect": 0, "max_extent_mm": 0.0, "largest": []}
    if n == 0:
        return out
    counts = numpy.bincount(labels.ravel())[1:]
    # A region of k voxels cannot span more than k voxel diagonals.
    min_voxels = suspect_extent_mm / (spacing_mm * numpy.sqrt(3.0))
    tree = cKDTree(tree_ijk) if len(tree_ijk) else None
    described = []
    for index, box in enumerate(find_objects(labels)):
        if box is None or counts[index] < min_voxels:
            continue
        voxels = numpy.argwhere(labels[box] == index + 1) + numpy.array([s.start for s in box])
        region = {
            "volume_mm3": float(counts[index] * voxel_mm3),
            "extent_mm": _extent_mm(voxels, spacing_mm),
            "centroid_ras": [float(v) for v in ijk_to_ras(voxels.mean(axis=0))],
            "centroid_ijk": [float(v) for v in voxels.mean(axis=0)],
        }
        if tree is not None:
            distance = tree.query(voxels)[0] * spacing_mm
            region["distance_to_tree_mm"] = {"min": float(distance.min()), "max": float(distance.max())}
        region["suspect"] = region["extent_mm"] >= suspect_extent_mm
        described.append(region)
    described.sort(key=lambda r: -r["extent_mm"])
    out["n_suspect"] = sum(r["suspect"] for r in described)
    out["max_extent_mm"] = described[0]["extent_mm"] if described else 0.0
    out["largest"] = described[:_MAX_PIECES_RECORDED]
    return out


def unexplained_regions(mask, side_domain, tree_volume, dt_voxels, ijk_to_ras, spacing_mm: float,
                        cover_radius_scale: float, cover_tol_mm: float, suspect_extent_mm: float) -> dict:
    """Mask voxels of one side that the FINAL tree does not explain, i.e. outside
    the same tube (scale x r + tol, line of sight through the mask) around the
    accepted tree only. During growth every voxel ends up covered, partly by the
    tubes of rejected stubs; measured against the accepted tree alone, a missed
    branch is left as a long uncovered region, while wall pockets and rejected
    stubs leave short ones. Call it on the tree before endpoint trimming."""
    voxels = numpy.argwhere(side_domain & mask)
    tree_ijk = numpy.argwhere(tree_volume)
    if len(voxels) == 0 or len(tree_ijk) == 0:
        return _describe_regions(numpy.zeros(mask.shape, dtype=int), 0, tree_ijk, ijk_to_ras, spacing_mm, suspect_extent_mm)
    distance, nearest = cKDTree(tree_ijk).query(voxels)
    radius = cover_radius_scale * dt_voxels[tuple(tree_ijk[nearest].T)] + cover_tol_mm / spacing_mm
    explained = distance <= radius
    in_tube = numpy.flatnonzero(explained)
    for chunk in numpy.array_split(in_tube, len(in_tube) // 50000 + 1):
        if len(chunk):
            explained[chunk] = _line_of_sight(
                tree_ijk[nearest[chunk]].astype(float), voxels[chunk].astype(float), mask
            )
    unexplained = numpy.zeros(mask.shape, dtype=bool)
    unexplained[tuple(voxels[~explained].T)] = True
    labels, n = _ndimage_label(unexplained, structure=_CONNECTIVITY_26)
    out = _describe_regions(labels, n, tree_ijk, ijk_to_ras, spacing_mm, suspect_extent_mm)
    out["fraction_of_side_mask"] = float((~explained).sum() / len(voxels))
    return out


def _unreachable_pieces(mask, reached, tree_ijk, ijk_to_ras, spacing_mm: float, suspect_extent_mm: float) -> dict:
    """Connected pieces of the mask that neither ostium reaches: reported, never
    bridged. Long ones are suspect too: a vessel the segmentation cut off."""
    labels, n = _ndimage_label(mask & ~reached, structure=_CONNECTIVITY_26)
    out = _describe_regions(labels, n, tree_ijk, ijk_to_ras, spacing_mm, suspect_extent_mm)
    out["fraction_of_mask"] = float((labels > 0).sum() / max(int(mask.sum()), 1))
    return out


def trace_centerlines(
    mask: numpy.ndarray,
    ostium_ijk: dict,
    ijk_to_ras,
    spacing_mm: float,
    dilation_voxels: int = 1,
    centering_power: float = 4.0,
    shell_penalty: float = 1000.0,
    max_bridge_mm: float = 25.0,
    min_branch_length_mm: float = 5.0,
    cover_radius_scale: float = 1.5,
    cover_tol_mm: float = 1.0,
    radius_cap_scale: float = 1.2,
    radius_cap_window_mm: float = 5.0,
    suspect_extent_mm: float = 10.0,
    max_gap_bridge_mm: float = 10.0,
):
    """Traces both trees. ostium_ijk = {"L": (3,), "R": (3,)}, continuous
    working-grid indices; ijk_to_ras maps (N, 3) or (3,) indices to RAS (QC only).

    With radius_cap_scale > 0 the trees are grown twice: pass 1 with the plain
    centred cost, then pass 2 with DT capped at radius_cap_scale x the vessel
    radius measured along the pass-1 tree (capped_distance), which keeps paths
    from being pulled into the centre of bulky junctions. Targets and domains
    are the same in both passes; only the paths change.

    Returns (graph, qc, debug). graph is a plain networkx.Graph whose nodes carry
    'ijk' and 'side' ("L"/"R"), the ostium nodes ("ostium_L"/"ostium_R") also
    'is_ostium' and their exact continuous 'ijk'; edges carry 'distance'
    (voxels). Each side is a tree by construction, and the two never share a node.
    """
    domain = connectivity_domain(mask, dilation_voxels)
    domain, gap_bridges, gap_bridge_list = bridge_mask_gaps(domain, ostium_ijk, max_gap_bridge_mm, spacing_mm)
    dt_voxels = distance_transform_edt(domain)
    domain_tree = cKDTree(numpy.argwhere(domain))

    sides = {}
    for side in ("L", "R"):
        bridge, start, entry_mm = ostium_bridge(domain, ostium_ijk[side], domain_tree, max_bridge_mm, spacing_mm)
        reachable = domain | bridge
        geodesic, _ = single_source_front(numpy.where(reachable, 1.0, numpy.inf), start)
        sides[side] = {"bridge": bridge, "start": start, "entry_mm": entry_mm, "geodesic": geodesic}
    domain_l, domain_r, merged = split_left_right(sides["L"]["geodesic"], sides["R"]["geodesic"])
    sides["L"]["domain"], sides["R"]["domain"] = domain_l, domain_r

    def grow_both(dt_for_cost):
        cost = centred_cost(mask, domain, dt_for_cost, centering_power, shell_penalty)
        grown = {}
        for side in ("L", "R"):
            s = sides[side]
            side_cost = numpy.where(s["domain"], cost, numpy.inf)
            side_cost[s["bridge"]] = shell_penalty
            _, mcp = single_source_front(side_cost, s["start"])
            grown[side] = grow_tree(
                mask, s["domain"], s["geodesic"], mcp, s["start"], dt_voxels, spacing_mm,
                min_branch_length_mm, cover_radius_scale, cover_tol_mm,
            )
        return grown

    qc = {"merged_component": merged, "sides": {},
          "gap_bridges": {"max_gap_mm": max_gap_bridge_mm, "n": len(gap_bridge_list), "bridges": gap_bridge_list}}
    grown = grow_both(dt_voxels)
    if radius_cap_scale > 0:
        window_voxels = radius_cap_window_mm / spacing_mm
        side_radii = {side: vessel_radius_along_tree(grown[side][0], dt_voxels, window_voxels) for side in ("L", "R")}
        dt_capped, fraction_capped = capped_distance(
            dt_voxels, {"L": domain_l, "R": domain_r}, side_radii, radius_cap_scale
        )
        qc["radius_cap"] = {
            "scale": radius_cap_scale,
            "window_mm": radius_cap_window_mm,
            "fraction_domain_voxels_capped": fraction_capped,
            "pass1_n_tree_voxels": {side: int(grown[side][1].sum()) for side in ("L", "R")},
        }
        pass1 = grown
        grown = grow_both(dt_capped)
        # How much the cap changed each tree: pass-2 tree voxels not on the pass-1 tree,
        # and how far the pass-2 tree strays from the pass-1 one (mm).
        changes = {}
        for side in ("L", "R"):
            before, after = numpy.argwhere(pass1[side][1]), numpy.argwhere(grown[side][1])
            far = cKDTree(before).query(after)[0] * spacing_mm if len(before) and len(after) else numpy.zeros(0)
            changes[side] = {
                "n_voxels_moved": int((grown[side][1] & ~pass1[side][1]).sum()),
                "max_distance_from_pass1_mm": float(far.max()) if len(far) else 0.0,
            }
        qc["radius_cap"]["changes"] = changes
        # more than a one-voxel (diagonal) move, which the cap can cause anywhere
        qc["radius_cap"]["tree_changed"] = any(c["max_distance_from_pass1_mm"] > 1.0 for c in changes.values())

    graph = networkx.Graph()
    debug = {"mask": mask, "dt_mm": dt_voxels * spacing_mm, "targets": {}}
    for side in ("L", "R"):
        s = sides[side]
        edges, tree_volume, accepted, rejected = grown[side]
        s["tree_volume"] = tree_volume

        ostium_node = f"ostium_{side}"
        def node_id(voxel):
            return ostium_node if voxel == s["start"] else "{}_{}_{}".format(*voxel)
        graph.add_node(ostium_node, ijk=numpy.asarray(ostium_ijk[side], dtype=float), side=side, is_ostium=True)
        for a, b in edges:
            for v in (a, b):
                if node_id(v) not in graph:
                    graph.add_node(node_id(v), ijk=numpy.asarray(v, dtype=float), side=side)
            graph.add_edge(node_id(a), node_id(b), distance=float(numpy.linalg.norm(numpy.subtract(a, b))))

        side_nodes = [n for n, d in graph.nodes(data=True) if d["side"] == side]
        sub = graph.subgraph(side_nodes)
        centredness = _centredness(tree_volume & mask, dt_voxels)
        qc["sides"][side] = {
            "ostium_to_mask_mm": s["entry_mm"],
            "n_nodes": sub.number_of_nodes(),
            "n_endpoints": sum(1 for n, d in sub.degree() if d == 1 and n != ostium_node),
            "length_mm": float(sum(d for _, _, d in sub.edges(data="distance")) * spacing_mm),
            "n_targets_accepted": len(accepted),
            "n_targets_rejected": len(rejected),
            "accepted": [{"target_ijk": list(t), "length_mm": l} for t, l in accepted[:_MAX_TARGETS_RECORDED]],
            "rejected": [{"target_ijk": list(t), "length_mm": l} for t, l in rejected[:_MAX_TARGETS_RECORDED]],
            "n_shell_voxels_used": int((tree_volume & domain & ~mask & ~gap_bridges).sum()),
            "n_gap_bridge_voxels_used": int((tree_volume & gap_bridges).sum()),
            "centredness_median": float(numpy.median(centredness)) if centredness.size else None,
            "centredness_p5": float(numpy.percentile(centredness, 5)) if centredness.size else None,
            "fraction_nodes_centredness_below_0.7": float((centredness < 0.7).mean()) if centredness.size else None,
            "acyclic": networkx.is_forest(sub),
            "unexplained": unexplained_regions(
                mask, s["domain"], tree_volume, dt_voxels, ijk_to_ras, spacing_mm,
                cover_radius_scale, cover_tol_mm, suspect_extent_mm,
            ),
        }
        debug["targets"][side] = {"accepted": accepted, "rejected": rejected}

    tree_ijk = numpy.argwhere(sides["L"]["tree_volume"] | sides["R"]["tree_volume"])
    qc["unreachable"] = _unreachable_pieces(mask, domain_l | domain_r, tree_ijk, ijk_to_ras, spacing_mm, suspect_extent_mm)

    # Per-case flag for possibly missed branches; review.py ranks cases with it.
    reasons = [
        f"{side}: {qc['sides'][side]['unexplained']['n_suspect']} unexplained region(s) up to "
        f"{qc['sides'][side]['unexplained']['max_extent_mm']:.0f} mm long"
        for side in ("L", "R") if qc["sides"][side]["unexplained"]["n_suspect"]
    ]
    if qc["unreachable"]["n_suspect"]:
        reasons.append(
            f"{qc['unreachable']['n_suspect']} unreachable mask piece(s) up to "
            f"{qc['unreachable']['max_extent_mm']:.0f} mm long"
        )
    qc["suspect_missed_branch"] = {"flag": bool(reasons), "extent_threshold_mm": suspect_extent_mm, "reasons": reasons}
    return graph, qc, debug
