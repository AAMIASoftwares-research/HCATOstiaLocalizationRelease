"""
Per-case orchestration of the v2 pipeline: load the mask onto the working grid,
crop it to the arteries, trace both trees (trace.py), then finalise and export
with v1's own converters (radii, topology, SimpleCenterlineGraph, smoothing,
GML, 3D Slicer markups), so v1 and v2 graphs are directly comparable.
"""
import os
import json
import traceback

import networkx
import numpy
from termcolor import colored

from ..utils import checks, convert, figures, grid, smoothing
from . import trace


def crop_to_arteries(mask_metatensor, ostium_ijk: dict, pad_mm: float, spacing_mm: float):
    """MONAI SpatialCrop of the working-grid MetaTensor to the bounding box of
    the mask and both ostia, padded by pad_mm. The cropped MetaTensor carries
    its own affine, which becomes the working grid from here on.

    Returns (mask, affine, offset): the cropped bool mask, its 4x4 index -> RAS
    affine, and the crop's start index in the uncropped grid.
    """
    from monai.transforms import SpatialCrop

    full_mask = mask_metatensor[0].cpu().numpy() > 0
    full_affine = numpy.asarray(mask_metatensor.affine.cpu().numpy(), dtype=numpy.float64).reshape(4, 4)
    points = numpy.vstack([numpy.argwhere(full_mask), ostium_ijk["L"], ostium_ijk["R"]])
    pad = int(numpy.ceil(pad_mm / spacing_mm))
    shape = numpy.asarray(full_mask.shape)
    lo = numpy.maximum(numpy.floor(points.min(axis=0)) - pad, 0).astype(int)
    hi = numpy.minimum(numpy.ceil(points.max(axis=0)) + pad + 1, shape).astype(int)

    cropped = SpatialCrop(roi_start=lo.tolist(), roi_end=hi.tolist())(mask_metatensor)
    mask = cropped[0].cpu().numpy() > 0
    affine = numpy.asarray(cropped.affine.cpu().numpy(), dtype=numpy.float64).reshape(4, 4)

    # Frame checks: the crop must keep every mask voxel, and its affine must be
    # the uncropped one shifted by the crop offset, nothing else.
    if int(mask.sum()) != int(full_mask.sum()):
        raise AssertionError(f"crop lost mask voxels ({int(mask.sum())} of {int(full_mask.sum())} kept)")
    expected = full_affine.copy()
    expected[:3, 3] = grid.working_ijk_to_ras(lo, full_affine)
    if not numpy.allclose(affine, expected, atol=1e-5):
        raise AssertionError(f"cropped affine {affine.tolist()} differs from the expected {expected.tolist()}")
    return mask, affine, lo


def _inside_mask(mask, affine):
    """position (RAS mm) -> is its nearest working-grid voxel in the mask."""
    upper = numpy.array(mask.shape) - 1

    def inside(position):
        ijk = numpy.rint(grid.ras_to_working_ijk(numpy.asarray(position, dtype=float), affine)).astype(int)
        return bool(numpy.all(ijk >= 0) and numpy.all(ijk <= upper) and mask[tuple(ijk)])
    return inside


def enforce_ostial_trunk(graph, min_trunk_mm: float = 2.0) -> dict:
    """In place: guarantees, on each side, a single trunk of at least min_trunk_mm
    (straight-line distance) between the ostium and its first bifurcation. A
    bifurcation right at the ostium is anatomically implausible: it appears when the
    ostium sits where the mask already splits, or when the bridge from the ostium
    reaches two vessels. HCATNetwork also requires the ostium to have one neighbour
    (its get_anatomic_segments(), used by resample(), raises IndexError otherwise).

    If the first bifurcation is closer than min_trunk_mm, the trunk follows the main
    path (at every bifurcation, the child leading to the longest downstream vessel)
    up to its first node T at least min_trunk_mm from the ostium. Every other node
    nearer to the ostium than T along the tree is removed, and the subtrees that hung
    from them are re-attached to T, which becomes the first bifurcation; the link to
    T is a straight segment (smoothed later with the rest). The distance is straight-
    line, so smoothing (ends fixed) cannot shorten the trunk below min_trunk_mm, and
    resampling at 0.5 mm puts at least 10 nodes on it, T included.
    Returns per side QC: the first bifurcation's distance before, nodes removed,
    subtrees re-attached and the longest new link (mm)."""
    out = {}
    for side in ("L", "R"):
        root = f"ostium_{side}"
        component = graph.subgraph(networkx.node_connected_component(graph, root)).copy()
        for u, v in component.edges:
            component.edges[u, v]["_mm"] = float(numpy.linalg.norm(
                numpy.asarray(component.nodes[u]["pos_ras"]) - numpy.asarray(component.nodes[v]["pos_ras"])))
        origin = numpy.asarray(graph.nodes[root]["pos_ras"], dtype=float)
        straight = lambda n: float(numpy.linalg.norm(numpy.asarray(graph.nodes[n]["pos_ras"]) - origin))
        # first bifurcation from the ostium
        node, previous = root, None
        while True:
            onward = [n for n in component.neighbors(node) if n != previous]
            if len(onward) != 1:
                break
            previous, node = node, onward[0]
        qc = {"first_bifurcation_mm_before": straight(node) if len(onward) > 1 else None,
              "n_nodes_removed": 0, "n_subtrees_reattached": 0, "max_link_mm": 0.0}
        out[side] = qc
        if len(onward) < 2 or straight(node) >= min_trunk_mm:
            continue
        main_path = smoothing.branch_decomposition(component, root, length="_mm")[0]
        trunk_end = next((n for n in main_path if straight(n) >= min_trunk_mm), main_path[-1])
        trunk = main_path[: main_path.index(trunk_end) + 1]
        geodesic = networkx.single_source_dijkstra_path_length(component, root, weight="_mm")
        zone = {n for n, d in geodesic.items() if d < geodesic[trunk_end]} - set(trunk)
        reattach = {n for z in zone for n in component.neighbors(z) if n not in zone and n not in trunk}
        # branches leaving a trunk node directly, without any node in the zone
        for k, n_trunk in enumerate(trunk[:-1]):
            for n in list(graph.neighbors(n_trunk)):
                if n not in trunk and n not in zone:
                    graph.remove_edge(n_trunk, n)
                    reattach.add(n)
        graph.remove_nodes_from(zone)
        for n in reattach:
            link = float(numpy.linalg.norm(numpy.asarray(graph.nodes[n]["pos_ras"]) - numpy.asarray(graph.nodes[trunk_end]["pos_ras"])))
            graph.add_edge(trunk_end, n, distance=float(numpy.linalg.norm(
                numpy.asarray(graph.nodes[n]["ijk"], dtype=float) - numpy.asarray(graph.nodes[trunk_end]["ijk"], dtype=float))))
            qc["max_link_mm"] = max(qc["max_link_mm"], link)
        qc["n_nodes_removed"] = len(zone)
        qc["n_subtrees_reattached"] = len(reattach)
    return out


def _plot_targets(debug: dict, graph, qc: dict, spacing_mm: float, title: str):
    """Mask maximum projections, top view (S compressed) above side view (A
    compressed), each with the trees, every accepted (green) and rejected
    (orange) target, the annotated ostia (big blue dots), and the centroids of
    suspect regions (possibly missed branches, big magenta crosses). Axes in mm
    from the crop corner."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    # (array axis compressed, array axis on the y axis, y label, view title)
    views = ((2, 1, "A (mm)", "Top view (S compressed)"), (1, 2, "S (mm)", "Side view (A compressed)"))
    tree_colors = {"L": "tab:cyan", "R": "tab:red"}
    ostia = numpy.array([graph.nodes[f"ostium_{side}"]["ijk"] for side in ("L", "R")]) * spacing_mm
    suspect_regions = [qc["sides"][side]["unexplained"] for side in ("L", "R")] + [qc["unreachable"]]
    suspects = numpy.array([
        r["centroid_ijk"] for regions in suspect_regions for r in regions["largest"] if r["suspect"]
    ]).reshape(-1, 3) * spacing_mm

    fig, axes = plt.subplots(2, 1, figsize=(7, 13), constrained_layout=True)
    fig.suptitle(title)
    for ax, (compressed, y_axis, ylabel, view_title) in zip(axes, views):
        projection = debug["mask"].max(axis=compressed)
        extent = (0, projection.shape[0] * spacing_mm, 0, projection.shape[1] * spacing_mm)
        ax.imshow(projection.T, origin="lower", cmap="gray_r", alpha=0.35, extent=extent, interpolation="nearest")
        for side, color in tree_colors.items():
            ijk = numpy.array([d["ijk"] for _, d in graph.nodes(data=True) if d["side"] == side]) * spacing_mm
            if len(ijk):
                ax.scatter(ijk[:, 0], ijk[:, y_axis], s=0.5, color=color, label=f"{side} tree")
            for kind, marker, mcolor in (("accepted", "o", "tab:green"), ("rejected", "x", "tab:orange")):
                targets = numpy.array([t for t, _ in debug["targets"][side][kind]]) * spacing_mm
                if len(targets):
                    ax.scatter(targets[:, 0], targets[:, y_axis], s=8, marker=marker, color=mcolor,
                               label=f"{kind} targets" if side == "L" else None)
        ax.scatter(ostia[:, 0], ostia[:, y_axis], s=80, color="blue", edgecolors="white", linewidths=0.8,
                   zorder=5, label="ostia")
        if len(suspects):
            ax.scatter(suspects[:, 0], suspects[:, y_axis], s=150, marker="X", color="magenta",
                       edgecolors="black", linewidths=0.6, zorder=6, label="suspect missed branch")
        ax.set_title(view_title)
        ax.set_xlabel("R (mm)")
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize="small")
    return fig


def _plot_distance_map(dt_mm: numpy.ndarray, spacing_mm: float):
    """Maximum projections of the distance-to-wall map used by the path cost,
    stacked vertically for a one-column figure: top view (S compressed, R-A
    plane) above side view (A compressed, R-S plane). Background is left white.
    Axes are in mm from the crop corner; the grid is isotropic, so pixels are
    square and the two views share the R axis scale."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    views = (
        (dt_mm.max(axis=2), "A (mm)", "Top view (S compressed)"),
        (dt_mm.max(axis=1), "S (mm)", "Side view (A compressed)"),
    )
    vmax = float(dt_mm.max())
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 7.0), constrained_layout=True)
    for ax, (projection, ylabel, title) in zip(axes, views):
        image = numpy.ma.masked_less_equal(projection, 0).T  # rows = second axis, for origin="lower"
        extent = (0, projection.shape[0] * spacing_mm, 0, projection.shape[1] * spacing_mm)
        shown = ax.imshow(image, origin="lower", cmap=cmap, vmin=0, vmax=vmax, extent=extent, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("R (mm)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
    colorbar = fig.colorbar(shown, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02)
    colorbar.set_label("Distance to vessel wall (mm)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    return fig


def process_one(
    mask_path: str,
    ostium_path: str,
    case_id: str,
    output_dir: str,
    artifacts: set,
    isotropic_spacing_mm: float = 0.5,
    dilation_voxels: int = 1,
    centering_power: float = 4.0,
    shell_penalty: float = 1000.0,
    max_ostium_bridge_mm: float = 25.0,
    min_branch_length_mm: float = 5.0,
    cover_radius_scale: float = 1.5,
    cover_tol_mm: float = 1.0,
    radius_cap_scale: float = 1.2,
    radius_cap_window_mm: float = 5.0,
    trim_endpoints_mm: float = 2.0,
    min_ostial_trunk_mm: float = 2.0,
    max_gap_bridge_mm: float = 10.0,
    suspect_extent_mm: float = 10.0,
    smooth_graph: bool = False,
    smooth_sigma_mm: float = 2.0,
    smooth_max_shift_fraction: float = 0.4,
    smooth_max_shift_mm: float = 1.0,
    smooth_min_shift_mm: float = 0.0,
    smooth_shift_limit: str = "mask",
    smooth_final_spacing_mm: float = 0.5,
) -> dict:
    """Runs one case. Never raises: a failure becomes {"status": "error"}, so a
    batch does not stop on one bad case (as in v1)."""
    written = {}
    try:
        _, full_affine, mask_metatensor = grid.load_mask_ras_isotropic(mask_path, isotropic_spacing_mm)
        grid.selftest_roundtrip(full_affine)
        ostium_ras = grid.load_ostium_ras(ostium_path)
        full_ostium_ijk = {s: grid.ras_to_working_ijk(ostium_ras[s], full_affine) for s in ("L", "R")}

        # The bridge ball around each ostium must fit inside the crop.
        mask, affine, offset = crop_to_arteries(
            mask_metatensor, full_ostium_ijk, max_ostium_bridge_mm + 2.0, isotropic_spacing_mm
        )
        del mask_metatensor
        grid.selftest_roundtrip(affine)
        ostium_ijk = {s: grid.ras_to_working_ijk(ostium_ras[s], affine) for s in ("L", "R")}

        graph, trace_qc, debug = trace.trace_centerlines(
            mask, ostium_ijk, lambda ijk: grid.working_ijk_to_ras(ijk, affine), isotropic_spacing_mm,
            dilation_voxels=dilation_voxels, centering_power=centering_power, shell_penalty=shell_penalty,
            max_bridge_mm=max_ostium_bridge_mm, min_branch_length_mm=min_branch_length_mm,
            cover_radius_scale=cover_radius_scale, cover_tol_mm=cover_tol_mm,
            radius_cap_scale=radius_cap_scale, radius_cap_window_mm=radius_cap_window_mm,
            suspect_extent_mm=suspect_extent_mm, max_gap_bridge_mm=max_gap_bridge_mm,
        )
        # Before radii and topology, so the new tips are labelled ENDPOINT.
        n_trimmed = trace.trim_endpoints(graph, trim_endpoints_mm, isotropic_spacing_mm)
        # A side with no tree (no target far enough from its ostium) would otherwise only
        # surface later, as a missing ostium after resampling.
        for side in ("L", "R"):
            if graph.degree(f"ostium_{side}") == 0:
                s = trace_qc["sides"][side]
                raise ValueError(
                    f"[{case_id}] {side}: no tree traced from the ostium (traced length before trimming "
                    f"{s['length_mm']:.1f} mm, longest rejected candidate "
                    f"{max((t['length_mm'] for t in s['rejected']), default=0.0):.1f} mm, mask "
                    f"{s['ostium_to_mask_mm']:.1f} mm from the ostium, left and right trees touching: "
                    f"{trace_qc['merged_component']})"
                )
        qc = {
            "case_id": case_id,
            "pipeline": "v2",
            "working_grid": {
                "spacing_mm": isotropic_spacing_mm,
                "shape": [int(v) for v in mask.shape],
                "affine_ijk_to_ras": affine.tolist(),
                "crop_offset_in_resampled_grid": [int(v) for v in offset],
            },
            "parameters": {
                "dilation_voxels": dilation_voxels, "centering_power": centering_power,
                "shell_penalty": shell_penalty, "max_ostium_bridge_mm": max_ostium_bridge_mm,
                "min_branch_length_mm": min_branch_length_mm, "cover_radius_scale": cover_radius_scale,
                "cover_tol_mm": cover_tol_mm, "radius_cap_scale": radius_cap_scale,
                "radius_cap_window_mm": radius_cap_window_mm, "trim_endpoints_mm": trim_endpoints_mm, "min_ostial_trunk_mm": min_ostial_trunk_mm,
                "suspect_extent_mm": suspect_extent_mm, "max_gap_bridge_mm": max_gap_bridge_mm,
            },
        }
        qc.update(trace_qc)
        qc["n_endpoints_trimmed"] = n_trimmed

        grid.compute_physical_positions(graph, affine)
        for side in ("L", "R"):
            # Exactly the annotated point, not its round trip through the affine.
            graph.nodes[f"ostium_{side}"]["pos_ras"] = numpy.asarray(ostium_ras[side], dtype=float)
        qc["ostial_trunk"] = enforce_ostial_trunk(graph, min_trunk_mm=min_ostial_trunk_mm)
        qc["final_acyclic"] = checks.assert_no_cycles(graph, case_id=case_id)
        qc["ostia_present"] = checks.assert_ostia_present(graph, case_id=case_id)

        # Radii from the mask itself; the dilation shell (0 there) falls back to
        # the dilated domain's distance transform.
        radius_map_mm = convert.compute_radius_map_mm(mask, isotropic_spacing_mm)
        fallback_radius_map_mm = convert.compute_radius_map_mm(
            trace.connectivity_domain(mask, dilation_voxels), isotropic_spacing_mm
        )
        convert.assign_radii(graph, radius_map_mm, fallback_radius_map_mm)
        convert.classify_topology(graph)
        convert.assert_ostia_present(graph, stage="classify_topology", case_id=case_id)
        roots = ["ostium_L", "ostium_R"]
        qc["direction_roughness"] = smoothing.direction_roughness(graph, roots)

        released = graph
        if smooth_graph:
            # Branch-wise smoothing of a copy (the debug figures keep the traced
            # voxel path): see utils/smoothing.py for why and how.
            released = graph.copy()
            qc["smoothing"] = smoothing.smooth_tree_branches(
                released, roots, sigma_mm=smooth_sigma_mm,
                max_shift_fraction=smooth_max_shift_fraction, max_shift_mm=smooth_max_shift_mm,
                min_shift_mm=smooth_min_shift_mm,
                inside=_inside_mask(mask, affine) if smooth_shift_limit == "mask" else None,
            )
            qc["smoothing"]["direction_roughness_after"] = smoothing.direction_roughness(released, roots)
            # The mask is the reference: count smoothed nodes outside it (ostia excluded,
            # they sit on the aortic wall by definition), before and after.
            for key, g in (("n_nodes_outside_mask_before", graph), ("n_nodes_outside_mask_after", released)):
                ijk = numpy.rint(grid.ras_to_working_ijk(
                    numpy.array([g.nodes[n]["pos_ras"] for n in g.nodes if not g.nodes[n].get("is_ostium")]), affine
                )).astype(int)
                ijk = numpy.clip(ijk, 0, numpy.array(mask.shape) - 1)
                qc["smoothing"][key] = int((~mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]]).sum())
        hcat_graph = convert.to_simple_centerline_graph(released, image_id=case_id)
        convert.assert_ostia_present(hcat_graph, stage="to_simple_centerline_graph", case_id=case_id)
        qc["are_left_right_disjointed"] = hcat_graph.graph.get("are_left_right_disjointed")

        if smooth_graph and smooth_final_spacing_mm > 0:
            # Uniform node spacing for release (linear between the smoothed nodes).
            hcat_graph = hcat_graph.resample(mm_between_nodes=smooth_final_spacing_mm)
            convert.assert_ostia_present(hcat_graph, stage="resample(final)", case_id=case_id)
            hcat_graph.graph["image_id"] = case_id  # resample() appends "- resampled X mm"
            qc["smoothing"]["final_spacing_mm"] = smooth_final_spacing_mm
            qc["smoothing"]["n_nodes_after"] = hcat_graph.number_of_nodes()

        if "none" not in artifacts:
            os.makedirs(output_dir, exist_ok=True)

            if "graph" in artifacts:
                graph_dir = os.path.join(output_dir, "graph")
                os.makedirs(graph_dir, exist_ok=True)
                graph_path = os.path.join(graph_dir, f"{case_id}.centerlines.gml")
                convert.save(hcat_graph, graph_path)
                written["graph"] = graph_path

            if "qc" in artifacts:
                qc_dir = os.path.join(output_dir, "qc")
                os.makedirs(qc_dir, exist_ok=True)
                qc_path = os.path.join(qc_dir, f"{case_id}.qc.json")
                with open(qc_path, "w") as f:
                    json.dump(qc, f, indent=2)
                written["qc"] = qc_path

            if "markup-3d-slicer" in artifacts:
                markup_dir = os.path.join(output_dir, "markup-3d-slicer", case_id)
                os.makedirs(markup_dir, exist_ok=True)
                convert.write_slicer_markups(hcat_graph, markup_dir, case_id)
                written["markup-3d-slicer"] = markup_dir

            if "debug_figures" in artifacts:
                debug_dir = os.path.join(output_dir, "debug_figures", case_id)
                os.makedirs(debug_dir, exist_ok=True)
                figures.save_figure(_plot_targets(debug, graph, qc, isotropic_spacing_mm, f"{case_id} - trees, targets and ostia"),
                             os.path.join(debug_dir, "targets_mip.png"))
                fig_distance = _plot_distance_map(debug["dt_mm"], isotropic_spacing_mm)
                fig_distance.savefig(os.path.join(debug_dir, "distance_map.pdf"))  # vector copy for the paper
                figures.save_figure(fig_distance, os.path.join(debug_dir, "distance_map.png"))
                figures.save_figure(figures.plot_graph_2d(graph, f"{case_id} - centerlines (unsmoothed)"),
                             os.path.join(debug_dir, "graph_2d.png"))
                figures.save_figure(figures.plot_graph_3d(graph, f"{case_id} - centerlines (unsmoothed)"),
                             os.path.join(debug_dir, "graph_3d.png"))
                if smooth_graph:
                    smoothed = figures.hcat_graph_to_plottable(hcat_graph)
                    figures.save_figure(figures.plot_graph_2d(smoothed, f"{case_id} - centerlines (smoothed)"),
                                 os.path.join(debug_dir, "graph_2d_smoothed.png"))
                    figures.save_figure(figures.plot_graph_3d(smoothed, f"{case_id} - centerlines (smoothed)"),
                                 os.path.join(debug_dir, "graph_3d_smoothed.png"))
                written["debug_figures"] = debug_dir

        for side in ("L", "R"):
            s = qc["sides"][side]
            print(colored(
                f"    {side}: {s['length_mm']:.0f} mm, {s['n_endpoints']} endpoints, "
                f"{s['n_targets_accepted']} accepted / {s['n_targets_rejected']} rejected targets, "
                f"centredness median {s['centredness_median']}", "white",
            ))
        if qc["suspect_missed_branch"]["flag"]:
            print(colored(f"    SUSPECT missed branch: {'; '.join(qc['suspect_missed_branch']['reasons'])}", "yellow"))
        return {"case_id": case_id, "status": "ok", "qc": qc, "written": written}

    except Exception as e:
        traceback.print_exc()
        return {"case_id": case_id, "status": "error", "error": f"{type(e).__name__}: {e}"}
