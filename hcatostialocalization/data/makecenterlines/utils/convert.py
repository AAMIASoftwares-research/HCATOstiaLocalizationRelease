"""
Radius estimation and conversion from the plain networkx.Graph this pipeline
builds internally to hcatnetwork's typed SimpleCenterlineGraph (a
networkx.Graph subclass), plus saving.

Uses hcatnetwork's own save_graph()/load_graph() (GML) directly -- no custom
serializer is written here, per the package's own IO module.

Radius note: the prototype imported skimage.morphology.medial_axis for this,
but medial_axis is a 2D-only function in skimage; it would silently operate
slice-by-slice on a 3D array, which is not a meaningful lumen radius. The 3D
equivalent used here is scipy.ndimage.distance_transform_edt on the working-grid
mask, which gives, at every foreground voxel, the true Euclidean distance to
the nearest background voxel in physical mm (via `sampling`).
"""
import os

import networkx
import numpy
from scipy.ndimage import distance_transform_edt
from termcolor import colored

from hcatnetwork.node.node import SimpleCenterlineNodeAttributes, ArteryNodeTopology, ArteryNodeSide
from hcatnetwork.edge.edge import SimpleCenterlineEdgeAttributes
from hcatnetwork.graph.graph import SimpleCenterlineGraph, SimpleCenterlineGraphAttributes
from hcatnetwork.io.io import save_graph as hcatnetwork_save_graph
from hcatnetwork.io.io import load_graph as hcatnetwork_load_graph


_SIDE_MAP = {"L": ArteryNodeSide.LEFT, "R": ArteryNodeSide.RIGHT, "RL": ArteryNodeSide.RL}


def compute_radius_map_mm(mask: numpy.ndarray, isotropic_spacing_mm: float) -> numpy.ndarray:
    """Euclidean distance transform of the working-grid mask, in mm. One call per
    case; look up individual node radii with lookup_radius_mm().
    """
    return distance_transform_edt(mask, sampling=(isotropic_spacing_mm,) * 3)


def lookup_radius_mm(radius_map_mm: numpy.ndarray, ijk) -> float:
    """Nearest-voxel lookup into a radius map, for both exact skeleton-voxel
    positions and the continuous (non-integer) position of an inserted ostium
    node. Indices are clipped to the array bounds rather than raising, since a
    node's position is already known to lie inside/near the mask.
    """
    index = tuple(
        int(numpy.clip(round(c), 0, dim - 1)) for c, dim in zip(ijk, radius_map_mm.shape)
    )
    return float(radius_map_mm[index])


def classify_topology(graph: networkx.Graph) -> None:
    """Sets 'topology' (ArteryNodeTopology) on every node, in place: OSTIUM for
    nodes inserted as roots (v1: rooting.assign_components_and_root; v2: trace) (marked 'is_ostium'),
    else by degree (1 -> ENDPOINT, 2 -> SEGMENT, >=3 -> INTERSECTION).
    """
    for node_id, data in graph.nodes(data=True):
        if data.get("is_ostium", False):
            data["topology"] = ArteryNodeTopology.OSTIUM
            continue
        degree = graph.degree(node_id)
        if degree <= 1:
            data["topology"] = ArteryNodeTopology.ENDPOINT
        elif degree == 2:
            data["topology"] = ArteryNodeTopology.SEGMENT
        else:
            data["topology"] = ArteryNodeTopology.INTERSECTION


def assign_radii(graph: networkx.Graph, radius_map_mm: numpy.ndarray, fallback_radius_map_mm: numpy.ndarray = None) -> None:
    """Sets 'r' (float, mm) on every node, in place, via lookup_radius_mm().
    Where radius_map_mm gives 0 (node outside the mask it was computed from --
    e.g. a branch traced by coverage gap filling through a region conditioning
    removed), fallback_radius_map_mm is used instead, if given.
    """
    for _, data in graph.nodes(data=True):
        r = lookup_radius_mm(radius_map_mm, data["ijk"])
        if r == 0.0 and fallback_radius_map_mm is not None:
            r = lookup_radius_mm(fallback_radius_map_mm, data["ijk"])
        data["r"] = r
    # An ostium sits on the aortic wall, at the edge of (or just outside) the
    # coronary mask, so the distance transform there is ~0 and says nothing
    # about the vessel. It takes the radius of the skeleton node it is attached
    # to instead (its only neighbour).
    for node_id, data in graph.nodes(data=True):
        if data.get("is_ostium", False):
            neighbour_radii = [graph.nodes[n]["r"] for n in graph.neighbors(node_id)]
            if neighbour_radii:
                data["r"] = float(max(neighbour_radii))


def to_simple_centerline_graph(graph: networkx.Graph, image_id: str) -> SimpleCenterlineGraph:
    """Builds a hcatnetwork.graph.SimpleCenterlineGraph from the assembled plain
    networkx.Graph (v1: graph_ops.assemble_final_graph; v2: trace.trace_centerlines). Requires every node to
    already carry 'pos_ras', 'r', 'topology' (classify_topology/assign_radii)
    and 'side'; requires every edge's two
    endpoints to have 'pos_ras' set. Edge distances are computed fresh from
    physical positions here, not carried over from the internal voxel-space
    'distance' used for shortest-path weighting during loop removal.
    """
    are_left_right_disjointed = not any(
        data.get("side") == "RL" for _, data in graph.nodes(data=True)
    )
    out = SimpleCenterlineGraph(
        SimpleCenterlineGraphAttributes(image_id=image_id, are_left_right_disjointed=are_left_right_disjointed)
    )
    for node_id, data in graph.nodes(data=True):
        pos = numpy.asarray(data["pos_ras"], dtype=float)
        node_attrs = SimpleCenterlineNodeAttributes(
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
            r=float(data["r"]), t=0.0,
            topology=data["topology"],
            side=_SIDE_MAP[data["side"]],
        )
        out.add_node(str(node_id), node_attrs)
    for u, v in graph.edges:
        u_pos = numpy.asarray(graph.nodes[u]["pos_ras"], dtype=float)
        v_pos = numpy.asarray(graph.nodes[v]["pos_ras"], dtype=float)
        euclidean_distance = float(numpy.linalg.norm(u_pos - v_pos))
        edge_attrs = SimpleCenterlineEdgeAttributes(weight=euclidean_distance, euclidean_distance=euclidean_distance)
        out.add_edge(str(u), str(v), edge_attrs)
    return out


def _ostium_taper_weight(distance_mm: float, taper_mm: float) -> float:
    """Raised-cosine taper: 1 at the ostium, 0 from taper_mm onwards, with zero
    slope at both ends so the smoothing strength changes without a kink."""
    if taper_mm <= 0 or distance_mm >= taper_mm:
        return 0.0
    return 0.5 * (1.0 + numpy.cos(numpy.pi * distance_mm / taper_mm))


def smooth_anatomic_segments(
    graph: SimpleCenterlineGraph,
    window_vertices: int = 5,
    sigma_mm: float = 0.5,
    ostium_sigma_mm: float = None,
    ostium_taper_mm: float = 15.0,
) -> None:
    """In place. Gaussian-smooths x, y, z and r along each anatomic segment
    (one landmark -- ostium, intersection or endpoint -- to the next), using a
    window centered on each interior node, weighted by true physical distance
    from that node (not raw vertex position, so it stays correct even if spacing
    along the path is not perfectly uniform).

    Landmark nodes themselves are never moved or resized: only SEGMENT nodes
    strictly between two landmarks move, so branch topology and bifurcation
    positions are unaffected by smoothing -- it can only make the geometry
    between them less jagged.

    Away from the ostia this matches ImageCAS-X's own centerline-regeneration
    protocol (Gaussian smoothing, sigma=0.5mm, 5-vertex sliding window -- see
    references/summaries/imagecas_x_reference_summary.md).

    Near the ostia, if ostium_sigma_mm is given, sigma is raised: with d the
    geodesic distance (mm, along the tree) of a node from its ostium,
        sigma(d) = sigma_mm + w(d) * (ostium_sigma_mm - sigma_mm),
        w(d) = (1 + cos(pi * d / ostium_taper_mm)) / 2 for d < ostium_taper_mm, else 0,
    i.e. full strength at the ostium, back to sigma_mm at ostium_taper_mm. The
    proximal centerline is where the skeleton is least reliable: the vessel is
    widest there, and the inserted ostium node joins the skeleton at an angle.
    Wherever sigma(d) is larger, the window is widened to cover at least
    +/- 3 sigma(d), since a fixed 5-vertex window would clip a wide Gaussian
    and weaken it.

    Call this after graph.resample() has produced roughly evenly-spaced nodes
    (e.g. ~1mm apart), not on the raw voxel-derived graph: a 5-vertex window on
    ~0.3mm native spacing only spans ~1.2mm of real support either side of a
    node, likely too narrow to average out the skeletonization staircase it is
    meant to remove.
    """
    base_half_window = window_vertices // 2
    distance_from_ostium = {}
    if ostium_sigma_mm is not None and ostium_sigma_mm != sigma_mm:
        for ostium in graph.get_coronary_ostia_node_id():
            if ostium is None:
                continue
            for node_id, d in networkx.single_source_dijkstra_path_length(graph, ostium, weight="weight").items():
                distance_from_ostium[node_id] = min(d, distance_from_ostium.get(node_id, numpy.inf))

    updates = {}
    for start_id, end_id in graph.get_anatomic_segments():
        path = networkx.shortest_path(graph, start_id, end_id)
        if len(path) <= 2:
            continue  # no interior nodes on this segment
        positions = numpy.array([[graph.nodes[n]["x"], graph.nodes[n]["y"], graph.nodes[n]["z"]] for n in path])
        radii = numpy.array([graph.nodes[n]["r"] for n in path])
        spacing = float(numpy.mean(numpy.linalg.norm(numpy.diff(positions, axis=0), axis=1)))
        for i in range(1, len(path) - 1):  # landmarks at path[0] and path[-1] are never touched
            sigma_i = sigma_mm
            half_window = base_half_window
            if path[i] in distance_from_ostium:
                w = _ostium_taper_weight(distance_from_ostium[path[i]], ostium_taper_mm)
                sigma_i = sigma_mm + w * (ostium_sigma_mm - sigma_mm)
                if spacing > 0:
                    half_window = max(base_half_window, int(numpy.ceil(3.0 * sigma_i / spacing)))
            lo, hi = max(0, i - half_window), min(len(path), i + half_window + 1)
            distances = numpy.linalg.norm(positions[lo:hi] - positions[i], axis=1)
            weights = numpy.exp(-(distances**2) / (2.0 * sigma_i**2))
            weights /= weights.sum()
            smoothed_position = weights @ positions[lo:hi]
            smoothed_radius = float(weights @ radii[lo:hi])
            updates[path[i]] = (smoothed_position, smoothed_radius)
    # Deferred single mutation pass, from positions/radii snapshotted before any
    # smoothing: an in-place update mid-loop would let already-smoothed
    # neighbours bias the next node's window, biasing the result by iteration
    # order.
    for node_id, (position, radius) in updates.items():
        graph.nodes[node_id]["x"] = float(position[0])
        graph.nodes[node_id]["y"] = float(position[1])
        graph.nodes[node_id]["z"] = float(position[2])
        graph.nodes[node_id]["r"] = radius


def assert_ostia_present(graph, stage: str, case_id: str = "") -> None:
    """Raises a clear, diagnostic ValueError if graph does not have exactly one
    LEFT and one RIGHT node of topology OSTIUM.

    Exists because hcatnetwork's own SimpleCenterlineGraph.get_coronary_ostia_node_id()
    silently returns None for a missing side rather than raising, and every
    caller of it downstream (get_anatomic_segments(), resample(),
    convert.smooth_anatomic_segments()) then crashes several frames later with
    an opaque "networkx.exception.NetworkXError: The node None is not in the
    graph." -- which says nothing about which side is missing, or when it went
    missing. Call this at every stage boundary that could plausibly drop an
    ostium (after classify_topology, and after each resample() call in the
    smoothing step) so a failure points at the actual stage and side, not an
    external library's internals three calls downstream.
    """
    counts = {ArteryNodeSide.LEFT: 0, ArteryNodeSide.RIGHT: 0}
    for _, data in graph.nodes(data=True):
        if data.get("topology") == ArteryNodeTopology.OSTIUM:
            # The plain networkx graph (before to_simple_centerline_graph) still
            # carries 'side' as the string "L"/"R"; the hcatnetwork graph carries
            # the ArteryNodeSide enum. Normalize so both are counted.
            side = _SIDE_MAP.get(data.get("side"), data.get("side"))
            counts[side] = counts.get(side, 0) + 1
    if counts.get(ArteryNodeSide.LEFT) != 1 or counts.get(ArteryNodeSide.RIGHT) != 1:
        raise ValueError(
            f"[{case_id}] after '{stage}': expected exactly one LEFT and one RIGHT "
            f"OSTIUM node, found LEFT={counts.get(ArteryNodeSide.LEFT, 0)} "
            f"RIGHT={counts.get(ArteryNodeSide.RIGHT, 0)} "
            f"(graph has {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges, "
            f"{networkx.number_connected_components(graph)} connected component(s))."
        )


_FLOAT_NODE_ATTRIBUTES = ("x", "y", "z", "r", "t")
_FLOAT_EDGE_ATTRIBUTES = ("weight", "euclidean_distance")


def save(graph: SimpleCenterlineGraph, file_path: str) -> None:
    """Saves as GML. Every numeric attribute is cast to a built-in float first:
    hcatnetwork's resample() produces numpy scalars, which the GML writer
    serialises as the text 'NP.FLOAT64(...)' instead of a number, making the
    file unreadable as numbers by any other tool."""
    for _, data in graph.nodes(data=True):
        for key in _FLOAT_NODE_ATTRIBUTES:
            if key in data:
                data[key] = float(data[key])
    for _, _, data in graph.edges(data=True):
        for key in _FLOAT_EDGE_ATTRIBUTES:
            if key in data:
                data[key] = float(data[key])
    hcatnetwork_save_graph(graph, file_path)


def load(file_path: str) -> SimpleCenterlineGraph:
    return hcatnetwork_load_graph(file_path, output_type=SimpleCenterlineGraph)


def write_slicer_markups(hcat_graph, out_dir: str, case_id: str) -> None:
    """Exports a SimpleCenterlineGraph as 3D Slicer markups: one open curve per
    ostium-to-endpoint branch, plus one fiducial list holding only the anatomic
    landmark nodes (ostia, bifurcations, endpoints).

    Coordinates are passed through unchanged (identity affine), and that is
    correct in both directions of the chain: our node x/y/z are already RAS mm
    (grid.compute_physical_positions() builds them through the working grid's
    MONAI RAS affine, and each ostium node carries the annotation's
    own RAS coordinate verbatim), while HCATNetwork's writers hardcode
    '"coordinateSystem": "RAS"' into the markup JSON. Do not pass an affine here
    to "fix" an apparent offset without re-checking both ends of that chain
    first -- a mismatch would mean something upstream is wrong, not this call.

    The fiducial file deliberately carries only the landmarks: a full graph is
    ~1000 nodes per side, which is unusable as a Slicer marker list, whereas the
    landmarks are the points actually worth inspecting -- and bifurcations in
    particular are where the residual-cycle problem shows up.
    """
    from hcatnetwork.utils.slicer.slicer import (
        convert_graph_to_3dslicer_opencurve,
        convert_graph_to_3dslicer_fiducials,
    )

    convert_graph_to_3dslicer_opencurve(hcat_graph, save_directory=out_dir)

    # get_anatomic_subgraph() walks a BFS tree from each ostium; graphs that still
    # contain cycles are the norm right now, so a failure here must not cost the
    # case its (already written) graph/qc artifacts.
    try:
        landmarks = hcat_graph.get_anatomic_subgraph()
        convert_graph_to_3dslicer_fiducials(
            landmarks,
            save_filename=os.path.join(out_dir, f"{case_id}_landmarks.SlicerFiducial.mkr.json"),
        )
    except Exception as e:
        print(colored(
            f"[convert] WARNING ({case_id}): landmark fiducial export failed "
            f"({type(e).__name__}: {e}); open curves were still written.",
            "yellow",
        ))
