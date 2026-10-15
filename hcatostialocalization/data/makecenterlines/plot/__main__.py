"""
Flattened views of coronary centerline graphs (and of their segmentation masks), seen from superior: anterior up,
the patient's right on the right. Geometry in flatten.py, drawing in draw.py; see README.md.

    python -m hcatostialocalization.data.makecenterlines.plot --gml <a.gml> [<b.gml> ...] --out <dir>
        [--mask <a.nii.gz> ...] [--mockup-mask] [--images steps tree left right]
        [--method {length-true,epicardial,projection}] [--surface {spheroid,sphere}] [--fit {inner,least-squares}]
        [--axis-clearance-mm 10] [--elements {all,graph,graph-ostia,graph-landmarks,mask,mask-ostia}]
        [--landmarks {ostium,intersection,endpoint} ...] [--empty]
        [--show-grid] [--distance-colormap] [--invert-mask] [--grid R,C]

Per case, in <out>/<case>/, by default four images:
  <case>_steps_<surface>-<fit>_<elements>[...].png          the three methods side by side (how the final image is built)
  <case>_<method>_tree_<surface>-<fit>_<elements>[...].png  both trees, with --method (default length-true)
  <case>_<method>_left_...png, <case>_<method>_right_...png each tree alone, same layout as the tree image
--images picks a subset. --landmarks replaces the landmarks of --elements with exactly the ones listed (none if
given empty), e.g. --elements mask --landmarks ostium endpoint; the file names then carry _lm-<landmarks>. --empty: the tree, left and right images show only the subject (no axes, ticks, grid,
compass, scale bar or title; tightly cropped). With several graphs, also
<out>/grid_epicardial_<surface>-<fit>_graph-ostia_distance.png (--grid R,C, default as square as possible).
--mask takes one mask per graph (same order), loaded with utils/grid.py (MONAI, so on the cluster); --mockup-mask
draws a clean, smooth mask built from the graph instead, even when --mask is given (no image needed). The mask of the
left / right images is split by the side of each voxel's nearest node. See README.md.
"""
import argparse
import os
import sys

import networkx
import numpy

from hcatostialocalization.data.makecenterlines.plot import flatten
from hcatostialocalization.data.makecenterlines.plot.draw import ELEMENT_SETS, draw_flat


def mockup_mask(graph, spacing_mm=0.5, min_radius_mm=0.5, step_mm=0.25):
    """(N, 3) RAS voxel centres of a clean, smooth mask built from the graph itself: a ball of the (smoothed) node
    radius every step_mm along every edge, on a spacing_mm grid. A tube that follows the centerline exactly, without
    the irregular surface, stumps and fragments of a real segmentation."""
    nodes, pos = flatten.node_positions(graph)
    index = {n: i for i, n in enumerate(nodes)}
    radii = numpy.array([max(float(graph.nodes[n].get("r", 0.0)), min_radius_mm) for n in nodes])
    centres, rads = [pos], [radii]
    for u, v in graph.edges:
        a, b = index[u], index[v]
        k = int(numpy.ceil(numpy.linalg.norm(pos[b] - pos[a]) / step_mm))
        if k > 1:
            w = numpy.linspace(0, 1, k + 1)[1:-1, None]
            centres.append(pos[a] + w * (pos[b] - pos[a]))
            rads.append(radii[a] + w[:, 0] * (radii[b] - radii[a]))
    cells = set()
    for p, r in zip(numpy.vstack(centres), numpy.concatenate(rads)):
        n = int(numpy.ceil(r / spacing_mm))
        grid = numpy.mgrid[-n:n + 1, -n:n + 1, -n:n + 1].reshape(3, -1).T
        offsets = grid[(grid ** 2).sum(axis=1) * spacing_mm ** 2 <= r ** 2]
        cells.update(map(tuple, numpy.round(p / spacing_mm).astype(int) + offsets))
    return numpy.array(sorted(cells), float) * spacing_mm


def real_mask(path):
    """(N, 3) RAS voxel centres of a segmentation mask (0.5 mm grid, as the extraction)."""
    from hcatostialocalization.data.makecenterlines.utils.grid import load_mask_ras_isotropic
    mask, affine, _ = load_mask_ras_isotropic(path, 0.5)
    ijk = numpy.argwhere(mask)
    return (affine @ numpy.hstack([ijk, numpy.ones((len(ijk), 1))]).T).T[:, :3]


DENSIFY = 2   # sub-samples per voxel edge


def densify(points, spacing_mm=0.5, factor=DENSIFY):
    """Each voxel centre replaced by factor**3 sub-voxel samples, so that the drawn mask stays solid where the
    unfolding stretches it."""
    offsets = (numpy.arange(factor) + 0.5) / factor - 0.5
    sub = numpy.stack(numpy.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3) * spacing_mm
    return (points[:, None, :] + sub[None, :, :]).reshape(-1, 3)


def flatten_case(graph, method, mask_ras=None, surface="spheroid", fit="inner", axis_clearance_mm=10.0):
    """(flat graph, flat mask points or None, title text) for one method."""
    flat_graph, unfolding = flatten.unfold_graph(graph, method, surface, fit, axis_clearance_mm=axis_clearance_mm)
    mask_xy = None if mask_ras is None else flatten.carry_points(mask_ras, flat_graph, unfolding, group=DENSIFY ** 3)
    distortion = flatten.edge_distortion(flat_graph)
    text = (f"{method}: {flatten.count_crossings(flat_graph)} crossings, length x{numpy.median(distortion):.2f} "
            f"(p95 {numpy.percentile(distortion, 95):.2f})")
    if method != "projection":
        shape = (f"sphere r {unfolding.a:.0f} mm" if unfolding.surface == "sphere"
                 else f"spheroid a {unfolding.a:.0f}, b {unfolding.b:.0f} mm")
        text += f"\n{shape} ({unfolding.fit}), fit residual {unfolding.fit_residual_mm:.1f} mm"
        if unfolding.tilt_deg:
            text += f", axis tilted {unfolding.tilt_deg:.0f}\N{DEGREE SIGN}"
    return flat_graph, mask_xy, text


def grid_shape(n, spec=None):
    """(rows, columns) for n panels: spec "R,C" if given, else as square as possible."""
    if spec:
        rows, cols = (int(v) for v in spec.replace("(", "").replace(")", "").split(","))
        if rows * cols < n:
            raise ValueError(f"--grid {spec} has room for {rows * cols} panels, {n} needed")
        return rows, cols
    cols = int(numpy.ceil(numpy.sqrt(n)))
    return int(numpy.ceil(n / cols)), cols


IMAGES = ("steps", "tree", "left", "right")
SIDES = {"left": "LEFT", "right": "RIGHT"}
LANDMARKS = {"ostium": "ostia", "intersection": "intersections", "endpoint": "endpoints"}   # --landmarks -> element


def save_single(plt, path, flat_graph, mask_xy, a, title, dpi):
    """One standalone image: the subject alone with --empty, otherwise with title, compass and scale bar."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes((0, 0, 1, 1)) if a.empty else fig.add_subplot(111)
    draw_flat(ax, flat_graph, mask_xy, a.elements, a.distance_colormap, a.show_grid and not a.empty, a.invert_mask,
              None if a.empty else title, empty=a.empty)
    if not a.empty:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight" if a.empty else None,
                pad_inches=0.05)
    plt.close(fig)
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gml", nargs="+", required=True)
    p.add_argument("--mask", nargs="+", default=None, help="one mask per graph, same order")
    p.add_argument("--mockup-mask", "--mock-mask", dest="mockup_mask", action="store_true",
                   help="draw a clean, smooth mockup mask built from the graph (radius along the centerline) instead of "
                        "the real mask, even when --mask is given; needs no image, so it also runs without MONAI")
    p.add_argument("--out", required=True)
    p.add_argument("--images", nargs="+", default=list(IMAGES), choices=IMAGES,
                   help="steps: the three methods side by side; tree: the whole tree; left / right: one tree each "
                        "(tree, left, right with --method). Default: all four")
    p.add_argument("--method", default="length-true", choices=flatten.METHODS,
                   help="method of the tree, left and right images (the steps image always shows all three)")
    p.add_argument("--surface", default="spheroid", choices=flatten.SURFACES)
    p.add_argument("--fit", default="inner", choices=flatten.FITS)
    p.add_argument("--axis-clearance-mm", type=float, default=10.0,
                   help="tilt the axis until no node is closer than this to its lower half (0: never tilt)")
    p.add_argument("--elements", default="graph-landmarks", choices=list(ELEMENT_SETS))
    p.add_argument("--landmarks", nargs="*", default=None, choices=list(LANDMARKS),
                   help="exactly these landmarks, replacing those of --elements (none if given without values)")
    p.add_argument("--empty", action="store_true",
                   help="tree, left, right images: only the subject, no axes, ticks, grid, compass, scale bar or title")
    p.add_argument("--show-grid", action="store_true")
    p.add_argument("--distance-colormap", action="store_true")
    p.add_argument("--invert-mask", action="store_true")
    p.add_argument("--grid", default=None, help="rows,columns of the summary grid image (several graphs); default auto")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a.mask and len(a.mask) != len(a.gml):
        p.error("--mask needs one mask per --gml")
    elements_name = a.elements
    a.elements = set(ELEMENT_SETS[a.elements])
    if a.landmarks is not None:
        chosen = [k for k in LANDMARKS if k in a.landmarks]
        a.elements = (a.elements - set(LANDMARKS.values())) | {LANDMARKS[k] for k in chosen}
        elements_name += "_lm-" + ("-".join(chosen) or "none")
    needs_mask = "mask" in a.elements
    suffix = f"{a.surface}-{a.fit}_{elements_name}" + ("_distance" if a.distance_colormap else "") + \
        ("_grid" if a.show_grid else "") + ("_inverted" if a.invert_mask else "")
    single_suffix = suffix + ("_empty" if a.empty else "")
    summary = []
    for k, path in enumerate(a.gml):
        case = os.path.basename(path).split(".")[0]
        case_dir = os.path.join(a.out, case)
        os.makedirs(case_dir, exist_ok=True)
        graph = networkx.read_gml(path, label="id")
        mask_ras = None
        if needs_mask:
            mask_ras = mockup_mask(graph) if a.mockup_mask else real_mask(a.mask[k]) if a.mask else None
            if mask_ras is not None:
                mask_ras = densify(mask_ras)
            else:
                print(f"{case}: no mask (use --mask or --mockup-mask), drawn without it")

        results = {}
        if "steps" in a.images:
            fig, axes = plt.subplots(1, len(flatten.METHODS), figsize=(6.5 * len(flatten.METHODS), 6.8), squeeze=False)
            for ax, method in zip(axes[0], flatten.METHODS):
                results[method] = flatten_case(graph, method, mask_ras, a.surface, a.fit, a.axis_clearance_mm)
                draw_flat(ax, *results[method][:2], a.elements, a.distance_colormap, a.show_grid, a.invert_mask,
                          results[method][2])
            fig.suptitle(f"{case}: flattened coronary trees, seen from superior", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            out = os.path.join(case_dir, f"{case}_steps_{suffix}.png")
            fig.savefig(out, dpi=a.dpi, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(out)

        if set(a.images) & {"tree", "left", "right"}:
            if a.method not in results:
                results[a.method] = flatten_case(graph, a.method, mask_ras, a.surface, a.fit, a.axis_clearance_mm)
            flat_graph, mask_xy, text = results[a.method]
            if "tree" in a.images:
                save_single(plt, os.path.join(case_dir, f"{case}_{a.method}_tree_{single_suffix}.png"), flat_graph,
                            mask_xy, a, f"{case}: both trees, {text}", a.dpi)
            mask_sides = None if mask_xy is None else flatten.point_sides(mask_ras, graph, group=DENSIFY ** 3)
            for image in ("left", "right"):
                if image not in a.images:
                    continue
                side = SIDES[image]
                sub = flat_graph.subgraph([n for n, d in flat_graph.nodes(data=True)
                                           if str(d.get("side", "")).split(".")[-1] == side]).copy()
                if not sub.number_of_nodes():
                    print(f"{case}: no {image} tree, image skipped")
                    continue
                save_single(plt, os.path.join(case_dir, f"{case}_{a.method}_{image}_{single_suffix}.png"), sub,
                            None if mask_xy is None else mask_xy[mask_sides == side], a,
                            f"{case}: {image} tree ({a.method})", a.dpi)

        if len(a.gml) > 1:
            flat_graph, _, text = flatten_case(graph, "epicardial", None, a.surface, a.fit, a.axis_clearance_mm)
            summary.append((case, flat_graph, text))

    # Several graphs: one summary grid, epicardial, edges and ostia only, coloured by distance from the ostium.
    if summary:
        rows, cols = grid_shape(len(summary), a.grid)
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.9 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (case, flat_graph, text) in zip(axes.flat, summary):
            ax.axis("on")
            draw_flat(ax, flat_graph, None, "graph-ostia", True, False, False, f"{case}\n{text}")
        fig.tight_layout()
        out = os.path.join(a.out, f"grid_epicardial_{a.surface}-{a.fit}_graph-ostia_distance.png")
        fig.savefig(out, dpi=a.dpi)
        plt.close(fig)
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
