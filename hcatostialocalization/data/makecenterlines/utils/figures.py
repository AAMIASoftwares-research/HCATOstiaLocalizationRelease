"""
Debug plotting helpers shared by v1 and v2: deterministic figure-in, PNG-out
functions on real RAS mm positions, safe in a headless batch job.
hcatnetwork.draw is an interactive viewer; use it separately on a saved .gml.
"""
import numpy
import networkx


def save_figure(fig, file_path: str) -> None:
    """Saves and closes a figure. pyplot keeps every figure alive until closed,
    so without the close a 1000-case batch accumulates hundreds of them."""
    import matplotlib.pyplot as plt
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


def plot_graph_2d(graph: networkx.Graph, title: str, ax=None):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots()
    ax.set_title(title)
    side_colors = {"L": "tab:blue", "R": "tab:red", "RL": "tab:green"}
    for side, color in side_colors.items():
        segments = [
            [graph.nodes[u]["pos_ras"][:2], graph.nodes[v]["pos_ras"][:2]]
            for u, v in graph.edges
            if graph.nodes[u].get("side", graph.nodes[v].get("side")) == side
        ]
        if segments:
            ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6, alpha=0.8, label=side))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize="small")
    return ax.figure


def plot_graph_3d(graph: networkx.Graph, title: str, highlight_node=None, ax=None):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    segments = [[graph.nodes[u]["pos_ras"], graph.nodes[v]["pos_ras"]] for u, v in graph.edges]
    ax.add_collection3d(Line3DCollection(segments, colors="tab:blue", linewidths=0.6, alpha=0.7))
    if highlight_node is not None and highlight_node in graph:
        p = graph.nodes[highlight_node]["pos_ras"]
        ax.scatter([p[0]], [p[1]], [p[2]], color="tab:green", s=40)
    positions = numpy.array([graph.nodes[n]["pos_ras"] for n in graph.nodes])
    ax.set_xlabel("R (mm)")
    ax.set_ylabel("A (mm)")
    ax.set_zlabel("S (mm)")
    ax.set(
        xlim=[positions[:, 0].min(), positions[:, 0].max()],
        ylim=[positions[:, 1].min(), positions[:, 1].max()],
        zlim=[positions[:, 2].min(), positions[:, 2].max()],
    )
    return ax.figure


_SIDE_ENUM_TO_LETTERS = {"LEFT": "L", "RIGHT": "R", "RL": "RL"}


def hcat_graph_to_plottable(hcat_graph) -> networkx.Graph:
    """Adapts a SimpleCenterlineGraph (attributes x, y, z, ArteryNodeSide) into
    the plain networkx.Graph shape plot_graph_2d/plot_graph_3d expect
    ('pos_ras', string 'side'), so the same plotting helpers work on both the
    pipeline's own intermediate graphs and the final typed/possibly-smoothed
    one -- needed because smoothing (convert.smooth_anatomic_segments) only
    runs on the typed graph, after the plain final_graph used for every other
    debug figure has already been built.
    """
    out = networkx.Graph()
    for node_id, data in hcat_graph.nodes(data=True):
        out.add_node(
            node_id,
            pos_ras=numpy.array([data["x"], data["y"], data["z"]], dtype=float),
            side=_SIDE_ENUM_TO_LETTERS[data["side"].name],
        )
    for u, v in hcat_graph.edges:
        out.add_edge(u, v)
    return out
