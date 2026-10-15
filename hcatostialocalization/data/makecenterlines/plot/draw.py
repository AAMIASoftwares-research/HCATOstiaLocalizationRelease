"""
Drawing of a flattened coronary graph (flat_x, flat_y from flatten.unfold_graph) with matplotlib, optionally over the
flattened segmentation mask. Drawing only: the geometry is in flatten.py.

Elements: "graph" (edges), "ostia", "intersections", "endpoints", "mask". The ELEMENT_SETS give the CLI choices.
"""
import numpy
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from hcatostialocalization.data.makecenterlines.plot import flatten

ELEMENT_SETS = {
    "graph": {"graph"},
    "graph-ostia": {"graph", "ostia"},
    "graph-landmarks": {"graph", "ostia", "intersections", "endpoints"},
    "mask": {"mask"},
    "mask-ostia": {"mask", "ostia"},
    "all": {"mask", "graph", "ostia", "intersections", "endpoints"},
}
STYLE = {
    "edge_color": "black",
    "edge_width_pt": (0.3, 2.0),         # min, max line width; proportional to the radius in between
    "edge_width_per_mm": 0.6,            # pt per mm of radius
    "ostium_face": "forestgreen",
    "ostium_edge": {"RIGHT": "firebrick", "LEFT": "navy"},
    "ostium_size": 70,
    "intersection_face": "#7DF9FF",      # electric blue
    "intersection_edge": "#0892D0",
    "intersection_size": 18,
    "endpoint_color": "firebrick",
    "endpoint_size": 8,
    "mask_gray": (0.45, 0.45, 0.45, 0.35),
    "mask_bin_mm": 0.5,
    "mask_closing_bins": 1,              # closes the small gaps where the unfolding stretches the voxels apart
}
# Distance from the ostium: red ramp for the right tree, blue ramp for the left, both starting away from white.
CMAP = {
    "RIGHT": LinearSegmentedColormap.from_list("right_tree", ["#f4a582", "#d6604d", "#b2182b", "#67001f"]),
    "LEFT": LinearSegmentedColormap.from_list("left_tree", ["#92c5de", "#4393c3", "#2166ac", "#053061"]),
}


def _side(d):
    return str(d.get("side", "")).split(".")[-1]


def _topology(d):
    return str(d.get("topology", "")).split(".")[-1]


def _shift_or(image, radius):
    """Binary dilation by a (2 radius + 1) square, with array shifts only (no scipy)."""
    out = numpy.zeros_like(image)
    padded = numpy.pad(image, radius)
    h, w = image.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def _closing(image, radius):
    """Binary closing (dilation, then erosion) by a (2 radius + 1) square."""
    if radius <= 0:
        return image
    padded = numpy.pad(image, radius)
    closed = ~_shift_or(~_shift_or(padded, radius), radius)
    return closed[radius:-radius, radius:-radius]


def draw_mask(ax, mask_xy, alone=False, invert=False, bin_mm=None):
    """The flattened mask as a 2D occupancy image: transparent gray under other elements, or black on white when
    alone (white on black with invert)."""
    bin_mm = bin_mm or STYLE["mask_bin_mm"]
    if mask_xy is None or not len(mask_xy):
        return
    lo = mask_xy.min(axis=0) - 2 * bin_mm
    hi = mask_xy.max(axis=0) + 2 * bin_mm
    bins = numpy.maximum(((hi - lo) / bin_mm).astype(int), 1)
    hist, xe, ye = numpy.histogram2d(mask_xy[:, 0], mask_xy[:, 1], bins=bins, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    occupied = _closing(hist.T > 0, STYLE["mask_closing_bins"])
    rgba = numpy.zeros(occupied.shape + (4,))
    if alone:
        fg = (1.0, 1.0, 1.0, 1.0) if invert else (0.0, 0.0, 0.0, 1.0)
        ax.set_facecolor("black" if invert else "white")
    else:
        fg = STYLE["mask_gray"]
    rgba[occupied] = fg
    ax.imshow(rgba, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), interpolation="nearest", zorder=1)


def draw_flat(ax, flat_graph, mask_xy=None, elements=("graph",), distance_colormap=False, show_grid=False,
              invert_mask=False, title=None, empty=False):
    """Draws a flattened graph (and mask) on ax. elements: a set of "graph", "ostia", "intersections", "endpoints",
    "mask" (or a key of ELEMENT_SETS). empty: only the subject (no axes, ticks, grid, compass, scale bar or title);
    the figure background takes the colour of the drawing's background (black for an inverted mask alone)."""
    if isinstance(elements, str):
        elements = ELEMENT_SETS[elements]
    elements = set(elements)
    if "mask" in elements:
        # Black (or white, inverted) whenever no centerline is drawn over it; transparent gray under the edges.
        draw_mask(ax, mask_xy, alone=("graph" not in elements), invert=invert_mask)
    xy = {n: (d["flat_x"], d["flat_y"]) for n, d in flat_graph.nodes(data=True)}

    if "graph" in elements:
        segments, widths, colors = [], [], []
        lo, hi = STYLE["edge_width_pt"]
        distance = flatten.distance_from_ostium(flat_graph) if distance_colormap else {}
        dmax = {side: max([distance.get(n, 0.0) for n, d in flat_graph.nodes(data=True) if _side(d) == side] or [1.0])
                for side in CMAP}
        for a, b in flat_graph.edges:
            segments.append([xy[a], xy[b]])
            r = 0.5 * (float(flat_graph.nodes[a].get("r", 0.0)) + float(flat_graph.nodes[b].get("r", 0.0)))
            widths.append(min(hi, max(lo, STYLE["edge_width_per_mm"] * r)))
            side = _side(flat_graph.nodes[a])
            if distance_colormap and side in CMAP:
                t = 0.5 * (distance.get(a, 0.0) + distance.get(b, 0.0)) / max(dmax[side], 1e-9)
                colors.append(CMAP[side](t))
            else:
                colors.append(STYLE["edge_color"])
        ax.add_collection(LineCollection(segments, linewidths=widths, colors=colors, capstyle="round", zorder=3))

    def scatter(selector, **kw):
        pts = [xy[n] for n, d in flat_graph.nodes(data=True) if selector(d)]
        if pts:
            pts = numpy.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], **kw)

    if "endpoints" in elements:
        scatter(lambda d: _topology(d) == "ENDPOINT", s=STYLE["endpoint_size"], c=STYLE["endpoint_color"], zorder=4,
                linewidths=0)
    if "intersections" in elements:
        scatter(lambda d: _topology(d) == "INTERSECTION", s=STYLE["intersection_size"], c=STYLE["intersection_face"],
                edgecolors=STYLE["intersection_edge"], linewidths=0.8, zorder=5)
    if "ostia" in elements:
        for side, edge in STYLE["ostium_edge"].items():
            scatter(lambda d, side=side: _topology(d) == "OSTIUM" and _side(d) == side, s=STYLE["ostium_size"],
                    c=STYLE["ostium_face"], edgecolors=edge, linewidths=2.0, zorder=6)

    pts = numpy.array(list(xy.values()))
    if mask_xy is not None and len(mask_xy) and "mask" in elements:
        pts = numpy.vstack([pts, mask_xy])
    lo_xy, hi_xy = pts.min(axis=0), pts.max(axis=0)
    pad = 0.06 * (hi_xy - lo_xy).max()
    ax.set_xlim(lo_xy[0] - pad, hi_xy[0] + pad)
    ax.set_ylim(lo_xy[1] - pad, hi_xy[1] + pad)
    ax.set_aspect("equal")
    dark = "mask" in elements and "graph" not in elements and invert_mask
    if empty:
        ax.axis("off")
        ax.figure.patch.set_facecolor("black" if dark else "white")
        return
    ax.set_xticks([]) if not show_grid else None
    ax.set_yticks([]) if not show_grid else None
    if show_grid:
        ax.grid(True, color="0.85", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(show_grid)
    _compass(ax, dark=dark)
    _scale_bar(ax, dark=dark)
    if title:
        ax.set_title(title, fontsize=9)


def _compass(ax, dark=False):
    color = "white" if dark else "0.3"
    kw = dict(transform=ax.transAxes, ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    ax.text(0.5, 0.985, "A", **kw)
    ax.text(0.5, 0.015, "P", **kw)
    ax.text(0.985, 0.5, "R", **kw)
    ax.text(0.015, 0.5, "L", **kw)


def _scale_bar(ax, length_mm=20.0, dark=False):
    color = "white" if dark else "black"
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x1 - 0.06 * (x1 - x0) - length_mm
    y = y0 + 0.05 * (y1 - y0)
    ax.plot([x, x + length_mm], [y, y], color=color, linewidth=2, zorder=7, solid_capstyle="butt")
    ax.text(x + length_mm / 2, y + 0.015 * (y1 - y0), f"{length_mm:g} mm", ha="center", va="bottom", fontsize=7,
            color=color, zorder=7)
