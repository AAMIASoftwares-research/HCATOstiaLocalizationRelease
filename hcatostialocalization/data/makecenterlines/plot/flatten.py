"""
Flat (2D) coordinates for a coronary centerline graph, as seen from superior: anterior up, the patient's right on the
right. Geometry only: no drawing, and nothing beyond numpy and networkx, so the module can be reused as it is (e.g. by
HCATNetwork for interactive visualization). The drawing is in draw.py, the command line in __main__.py, and
README.md explains the whole thing.

Methods:
  "projection"   plain axial projection, flat = (R, A). Screen right = R and screen up = A, so out of the screen is
                 R x A = S: a true view from superior.
  "epicardial"   the coronaries lie on the epicardium. A surface is fitted to the nodes (SURFACES):
                   "sphere"    least-squares sphere; its pole is the direction from the centre to the midpoint of the
                               two ostia (the aortic root);
                   "spheroid"  ellipsoid of revolution about the line from the sphere's centre to the aortic root:
                               the centre may slide along it, radius b along it and a across, all by linear least
                               squares, so the heart can come out elongated or flattened toward the apex; the pole
                               stays on the aortic root side, so the ostia stay at the centre of the map.
                 Every point is mapped by its meridian position: distance from the map centre = arc length along the
                 meridian from the pole to the point's (parametric) latitude, angle = its longitude around the axis,
                 with the projected A axis up and R right. For the sphere this is the azimuthal equidistant projection.
                 A bull's-eye map: aortic root at the centre, apex toward the rim, the far side unwrapped instead of
                 overlapping. The map is singular where the lower half of the axis leaves the surface: if a node lies
                 closer than axis_clearance_mm (10 mm) to it, the axis is tilted away from the aortic root by the
                 smallest angle that clears it (the ostia then sit slightly off the centre). FITS: "least-squares" keeps the fitted size; "inner" shrinks the surface about its centre
                 until no node lies inside it (same shape, so the map is only scaled: lengths come out shorter, never
                 longer).
  "length-true"  the epicardial layout rebuilt from each ostium, every edge keeping its 2D direction but taking its
                 true 3D length: distances along the vessels are exact. Points carried along (carry_points) keep their
                 true distance from their nearest node, so the vessel thickness of a mask is not deformed either.

    g2, unfolding = unfold_graph(g, "epicardial", surface="spheroid")   # g2 nodes get flat_x, flat_y
    mask_xy = carry_points(mask_ras, g2, unfolding)                     # (N, 2)
"""
from dataclasses import dataclass, field

import networkx
import numpy

METHODS = ("projection", "epicardial", "length-true")
SURFACES = ("sphere", "spheroid")
FITS = ("least-squares", "inner")


@dataclass
class Unfolding:
    """A fitted map from RAS (mm) to flat coordinates (mm). Fit once with fit_unfolding(), re-apply to anything.
    The surface is a spheroid with centre `center`, axis `pole` (unit, toward the aortic root), radius `a` across the
    axis and `b` along it (a sphere when a == b)."""
    method: str
    surface: str = "sphere"
    center: numpy.ndarray = field(default_factory=lambda: numpy.zeros(3))
    a: float = 0.0
    b: float = 0.0
    pole: numpy.ndarray = field(default_factory=lambda: numpy.array([0.0, 0.0, 1.0]))
    right: numpy.ndarray = field(default_factory=lambda: numpy.array([1.0, 0.0, 0.0]))
    up: numpy.ndarray = field(default_factory=lambda: numpy.array([0.0, 1.0, 0.0]))
    fit: str = "least-squares"
    fit_residual_mm: float = 0.0   # median distance of the nodes from the fitted (least-squares) surface
    scale: float = 1.0             # inner fit: shrink factor applied to the least-squares surface
    tilt_deg: float = 0.0          # how far the axis was tilted away from the aortic root (axis clearance)
    axis_clearance_mm: float = 0.0 # distance of the closest node from the lower half of the axis
    _meridian: tuple = None        # (t table, arc length table) of the meridian, filled on first use


def node_positions(graph, keys=("x", "y", "z")):
    """(node ids, (N, 3) array) of the graph's node positions."""
    nodes = list(graph.nodes)
    return nodes, numpy.array([[float(graph.nodes[n][k]) for k in keys] for n in nodes])


def ostium_nodes(graph, topology_key="topology", ostium_value="OSTIUM"):
    """Node ids whose topology is the ostium (accepts strings or enums whose name/str ends with OSTIUM)."""
    return [n for n, d in graph.nodes(data=True) if str(d.get(topology_key, "")).split(".")[-1] == ostium_value]


def fit_sphere(points):
    """Least-squares sphere through (N, 3) points: (center, radius, absolute radial residuals)."""
    points = numpy.asarray(points, float)
    a = numpy.hstack([2 * points, numpy.ones((len(points), 1))])
    b = (points ** 2).sum(axis=1)
    sol, *_ = numpy.linalg.lstsq(a, b, rcond=None)
    center = sol[:3]
    radius = float(numpy.sqrt(sol[3] + center @ center))
    return center, radius, numpy.abs(numpy.linalg.norm(points - center, axis=1) - radius)


def fit_spheroid(points, center, axis):
    """Least-squares spheroid whose axis of revolution is the line through `center` along unit `axis`: the centre may
    slide along the axis (offset h), with radius a across the axis and b along it. With r the distance from the axis
    and z the position along it, r^2/a^2 + (z - h)^2/b^2 = 1 is r^2 = P z^2 + Q z + R, linear in (P, Q, R):
    beta = a^2/b^2 = -P, h = Q / (2 beta), a^2 = R + beta h^2. Returns (center, a, b), or None when the fit is not a
    spheroid."""
    v = numpy.asarray(points, float) - center
    z = v @ axis
    r2 = (v ** 2).sum(axis=1) - z ** 2
    (P, Q, R), *_ = numpy.linalg.lstsq(numpy.column_stack([z ** 2, z, numpy.ones(len(z))]), r2, rcond=None)
    beta = -P
    if beta <= 0:
        return None
    h = Q / (2 * beta)
    a2 = R + beta * h * h
    if a2 <= 0:
        return None
    return center + h * axis, float(numpy.sqrt(a2)), float(numpy.sqrt(a2 / beta))


def _basis(pole, up_hint):
    up = numpy.asarray(up_hint, float) - (numpy.asarray(up_hint, float) @ pole) * pole
    up /= numpy.linalg.norm(up)
    return numpy.cross(up, pole), up   # right x up = pole: the pole points out of the screen (seen from outside)


def _surface_coords(unfolding, points):
    """(along the axis, distance from the axis, parametric polar angle t in [0, pi]) of (N, 3) points."""
    v = numpy.atleast_2d(points) - unfolding.center
    along = v @ unfolding.pole
    across = numpy.linalg.norm(v - along[:, None] * unfolding.pole, axis=1)
    return along, across, numpy.arctan2(across / unfolding.a, along / unfolding.b)


def lower_axis_clearance(points, center, axis):
    """Distance (mm) of the closest point from the lower half of the axis: the half-line from `center` away from the
    pole (-axis). Where that half-line leaves the surface, the map is singular (the point opposite the pole maps to
    the whole rim), so a vessel close to it is blown up."""
    v = numpy.asarray(points, float) - center
    along = v @ axis
    across = numpy.linalg.norm(v - along[:, None] * axis, axis=1)
    below = along < 0
    return float(across[below].min()) if below.any() else numpy.inf


def clear_axis(points, center, axis, clearance_mm=10.0, max_tilt_deg=45.0, step_deg=1.0, n_azimuth=72):
    """The axis tilted by the smallest angle (cones of step_deg around it) for which no point lies within clearance_mm
    of its lower half; among the directions at that angle, the one with the largest clearance. Returns (axis, tilt in
    degrees, clearance in mm); the original axis if it is already clear, the best one found up to max_tilt_deg
    otherwise."""
    axis = numpy.asarray(axis, float) / numpy.linalg.norm(axis)
    current = lower_axis_clearance(points, center, axis)
    if clearance_mm <= 0 or current >= clearance_mm:
        return axis, 0.0, current
    e1 = numpy.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= numpy.linalg.norm(e1)
    e2 = numpy.cross(axis, e1)
    best = (current, axis, 0.0)
    for tilt in numpy.arange(step_deg, max_tilt_deg + 1e-9, step_deg):
        t = numpy.radians(tilt)
        found = None
        for phi in numpy.linspace(0.0, 2 * numpy.pi, n_azimuth, endpoint=False):
            candidate = numpy.cos(t) * axis + numpy.sin(t) * (numpy.cos(phi) * e1 + numpy.sin(phi) * e2)
            c = lower_axis_clearance(points, center, candidate)
            if c > best[0]:
                best = (c, candidate, tilt)
            if c >= clearance_mm and (found is None or c > found[0]):
                found = (c, candidate)
        if found is not None:
            return found[1], float(tilt), float(found[0])
    return best[1], float(best[2]), float(best[0])


def fit_unfolding(graph, method="epicardial", surface="spheroid", fit="inner", keys=("x", "y", "z"),
                  up_hint=(0.0, 1.0, 0.0), axis_clearance_mm=10.0):
    """Fits the map of `method` to a graph (positions under `keys`, RAS mm). For "epicardial" and "length-true" the
    graph needs its ostium nodes (topology OSTIUM). up_hint is the 3D direction shown up (anterior).
    axis_clearance_mm: if a node lies closer than this to the lower half of the axis (where the map is singular), the
    axis is tilted away from the aortic root by the smallest angle that clears it (0 disables)."""
    if method not in METHODS or surface not in SURFACES or fit not in FITS:
        raise ValueError(f"method in {METHODS}, surface in {SURFACES}, fit in {FITS}")
    if method == "projection":
        return Unfolding("projection")
    _, pos = node_positions(graph, keys)
    ostia = ostium_nodes(graph)
    if not ostia:
        raise ValueError("the epicardial unfolding needs the ostium nodes (topology OSTIUM)")
    root = numpy.mean([[float(graph.nodes[n][k]) for k in keys] for n in ostia], axis=0)

    sphere_center, radius, _ = fit_sphere(pos)
    pole = (root - sphere_center) / numpy.linalg.norm(root - sphere_center)
    tilt_total, clearance = 0.0, lower_axis_clearance(pos, sphere_center, pole)
    for attempt in range(4):   # the spheroid centre slides along the axis: re-check the clearance after each fit
        center, a, b = sphere_center, radius, radius
        if surface == "spheroid":
            spheroid = fit_spheroid(pos, sphere_center, pole)
            if spheroid is None:
                surface = "sphere"   # degenerate fit: fall back on the sphere
            else:
                center, a, b = spheroid
        if attempt == 3:
            clearance = lower_axis_clearance(pos, center, pole)
            break
        new_pole, tilt, clearance = clear_axis(pos, center, pole, axis_clearance_mm)
        if tilt == 0.0:
            break
        pole, tilt_total = new_pole, tilt_total + tilt
        sphere_center = center   # tilt about the current centre, then refit along the new axis
    right, up = _basis(pole, up_hint)
    unfolding = Unfolding(method, surface, center, a, b, pole, right, up, fit)
    unfolding.tilt_deg, unfolding.axis_clearance_mm = float(tilt_total), float(clearance)
    along, across, _ = _surface_coords(unfolding, pos)
    level = numpy.sqrt((across / a) ** 2 + (along / b) ** 2)   # 1 on the surface, < 1 inside
    unfolding.fit_residual_mm = float(numpy.median(numpy.abs(level - 1.0)) * numpy.sqrt(a * b))
    if fit == "inner":
        unfolding.scale = float(level.min())
        unfolding.a, unfolding.b = a * unfolding.scale, b * unfolding.scale
    return unfolding


def _meridian(unfolding, n=4001):
    """Arc length along the meridian from the pole, tabulated against the parametric angle t."""
    if unfolding._meridian is None:
        t = numpy.linspace(0.0, numpy.pi, n)
        speed = numpy.sqrt((unfolding.a * numpy.cos(t)) ** 2 + (unfolding.b * numpy.sin(t)) ** 2)
        s = numpy.concatenate([[0.0], numpy.cumsum(0.5 * (speed[1:] + speed[:-1]) * numpy.diff(t))])
        unfolding._meridian = (t, s)
    return unfolding._meridian


def unfold_points(unfolding, points):
    """(N, 2) flat coordinates (mm) of (N, 3) RAS points under a fitted map. For "length-true" this is the epicardial
    position: use carry_points() to follow the tree."""
    points = numpy.atleast_2d(numpy.asarray(points, float))
    if unfolding.method == "projection":
        return points[:, :2].copy()
    v = points - unfolding.center
    along = v @ unfolding.pole
    tangent = v - along[:, None] * unfolding.pole
    across = numpy.linalg.norm(tangent, axis=1)
    t = numpy.arctan2(across / unfolding.a, along / unfolding.b)   # parametric polar angle, 0 at the pole
    table_t, table_s = _meridian(unfolding)
    s = numpy.interp(t, table_t, table_s)
    direction = numpy.zeros((len(points), 2))
    ok = across > 1e-9
    direction[ok, 0] = (tangent[ok] @ unfolding.right) / across[ok]
    direction[ok, 1] = (tangent[ok] @ unfolding.up) / across[ok]
    return direction * s[:, None]


def _length_true(graph, flat, keys):
    """Rebuilds each tree from its ostium: every edge keeps its flat direction and takes its 3D length."""
    out = dict(flat)
    for root in ostium_nodes(graph) or [next(iter(graph.nodes))]:
        for parent, child in networkx.bfs_edges(graph, root):
            p3 = numpy.array([float(graph.nodes[parent][k]) for k in keys])
            c3 = numpy.array([float(graph.nodes[child][k]) for k in keys])
            step = numpy.asarray(flat[child]) - numpy.asarray(flat[parent])
            norm = numpy.linalg.norm(step)
            direction = step / norm if norm > 1e-9 else numpy.array([0.0, 1.0])
            out[child] = numpy.asarray(out[parent]) + direction * numpy.linalg.norm(c3 - p3)
    return out


def unfold_graph(graph, method="epicardial", surface="spheroid", fit="inner", keys=("x", "y", "z"),
                 up_hint=(0.0, 1.0, 0.0), axis_clearance_mm=10.0):
    """A copy of the graph whose nodes carry flat_x, flat_y (mm), and the fitted Unfolding. Under "length-true" the
    copy also keeps the epicardial flat positions as graph attribute flat_before (a dict), for carry_points()."""
    unfolding = fit_unfolding(graph, method, surface, fit, keys, up_hint, axis_clearance_mm)
    nodes, pos = node_positions(graph, keys)
    xy = unfold_points(unfolding, pos)
    flat = {n: xy[i] for i, n in enumerate(nodes)}
    out = graph.copy()
    if method == "length-true":
        out.graph["flat_before"] = flat
        flat = _length_true(graph, flat, keys)
    for n, (x, y) in flat.items():
        out.nodes[n]["flat_x"], out.nodes[n]["flat_y"] = float(x), float(y)
    return out, unfolding


def flat_positions(graph):
    """(node ids, (N, 2) array) of the flat coordinates set by unfold_graph()."""
    nodes = list(graph.nodes)
    return nodes, numpy.array([[graph.nodes[n]["flat_x"], graph.nodes[n]["flat_y"]] for n in nodes])


def nearest_nodes(points, graph, keys=("x", "y", "z"), chunk=20000, group=1):
    """(N,) index, in list(graph.nodes), of the node nearest to each (N, 3) point (3D, brute force in chunks).
    group > 1: consecutive groups of that many points share the nearest node of their first point."""
    points = numpy.atleast_2d(numpy.asarray(points, float))
    _, pos = node_positions(graph, keys)
    anchors = points[::group]
    nearest = numpy.empty(len(anchors), int)
    for start in range(0, len(anchors), chunk):
        block = anchors[start:start + chunk]
        nearest[start:start + chunk] = ((block[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return numpy.repeat(nearest, group)[:len(points)]


def point_sides(points, graph, keys=("x", "y", "z"), group=1, side_key="side"):
    """(N,) side ("LEFT", "RIGHT", ...) of each point: the side of its nearest node (e.g. to split a mask by tree)."""
    nodes = list(graph.nodes)
    sides = numpy.array([str(graph.nodes[n].get(side_key, "")).split(".")[-1] for n in nodes])
    return sides[nearest_nodes(points, graph, keys, group=group)]


def carry_points(points, flat_graph, unfolding, keys=("x", "y", "z"), chunk=20000, group=1):
    """Flat coordinates of (N, 3) RAS points (e.g. mask voxels). "projection" and "epicardial": the fitted map itself.
    "length-true": each point is placed from its nearest node, in the direction the map gives, at its true distance
    from the node (the component tangent to the surface), so the vessel thickness is not deformed. group > 1: the
    points come in consecutive groups of that size (e.g. the sub-samples of one voxel) that share the nearest node of
    their first point, which is much faster."""
    points = numpy.atleast_2d(numpy.asarray(points, float))
    xy = unfold_points(unfolding, points)
    if flat_graph.graph.get("flat_before") is None:
        return xy
    nodes, pos = node_positions(flat_graph, keys)
    nearest = nearest_nodes(points, flat_graph, keys, chunk, group)
    node_xy_map = unfold_points(unfolding, pos)[nearest]
    node_xy = numpy.array([[flat_graph.nodes[n]["flat_x"], flat_graph.nodes[n]["flat_y"]] for n in nodes])[nearest]
    offset = points - pos[nearest]
    normal = pos[nearest] - unfolding.center
    normal /= numpy.linalg.norm(normal, axis=1, keepdims=True)
    tangential = numpy.linalg.norm(offset - (offset * normal).sum(axis=1, keepdims=True) * normal, axis=1)
    direction = xy - node_xy_map
    length = numpy.linalg.norm(direction, axis=1)
    ok = length > 1e-9
    direction[ok] /= length[ok, None]
    return node_xy + direction * tangential[:, None]


def distance_from_ostium(graph, keys=("x", "y", "z")):
    """{node: 3D path length (mm) from the ostium of its tree}."""
    weights = {}
    for u, v in graph.edges:
        a = numpy.array([float(graph.nodes[u][k]) for k in keys])
        b = numpy.array([float(graph.nodes[v][k]) for k in keys])
        weights[(u, v)] = float(numpy.linalg.norm(a - b))
    networkx.set_edge_attributes(graph, weights, "_length_3d")
    out = {}
    for ostium in ostium_nodes(graph):
        out.update(networkx.single_source_dijkstra_path_length(graph, ostium, weight="_length_3d"))
    for _, _, d in graph.edges(data=True):
        d.pop("_length_3d", None)
    return out


def edge_distortion(flat_graph, keys=("x", "y", "z")):
    """(E,) flat length / 3D length of every edge (1 = length preserved)."""
    out = []
    for u, v in flat_graph.edges:
        a3 = numpy.array([float(flat_graph.nodes[u][k]) for k in keys])
        b3 = numpy.array([float(flat_graph.nodes[v][k]) for k in keys])
        a2 = numpy.array([flat_graph.nodes[u]["flat_x"], flat_graph.nodes[u]["flat_y"]])
        b2 = numpy.array([flat_graph.nodes[v]["flat_x"], flat_graph.nodes[v]["flat_y"]])
        length = numpy.linalg.norm(a3 - b3)
        if length > 1e-9:
            out.append(numpy.linalg.norm(a2 - b2) / length)
    return numpy.array(out)


def count_crossings(flat_graph, chunk=400):
    """Number of pairs of non-adjacent edges whose flat segments intersect (vessel crossings in the drawing)."""
    edges = list(flat_graph.edges)
    if len(edges) < 2:
        return 0
    index = {n: i for i, n in enumerate(flat_graph.nodes)}
    _, xy = flat_positions(flat_graph)
    u = numpy.array([index[a] for a, _ in edges])
    v = numpy.array([index[b] for _, b in edges])
    p, q = xy[u], xy[v]

    def orient(a, b, c):
        return numpy.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    total = 0
    for start in range(0, len(edges), chunk):
        i = numpy.arange(start, min(start + chunk, len(edges)))
        a1, a2 = p[i][:, None, :], q[i][:, None, :]
        b1, b2 = p[None, :, :], q[None, :, :]
        cross = (orient(a1, a2, b1) * orient(a1, a2, b2) < 0) & (orient(b1, b2, a1) * orient(b1, b2, a2) < 0)
        shared = ((u[i][:, None] == u[None, :]) | (u[i][:, None] == v[None, :]) |
                  (v[i][:, None] == u[None, :]) | (v[i][:, None] == v[None, :]))
        later = numpy.arange(len(edges))[None, :] > i[:, None]
        total += int((cross & ~shared & later).sum())
    return total
