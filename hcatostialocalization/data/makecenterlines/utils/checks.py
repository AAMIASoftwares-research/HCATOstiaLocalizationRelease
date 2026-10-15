"""
Warning-only consistency checks on the plain networkx graphs both pipelines build
(nodes with 'is_ostium' and string 'side'). The typed-graph counterpart, which
raises, is convert.assert_ostia_present().
"""
import networkx


def assert_no_cycles(graph: networkx.Graph, case_id: str = "") -> bool:
    """Checks that graph has no cycles. Warns (prints) and returns False if any
    are found, rather than raising, so a batch run can log the offending case
    and keep going.

    remove_loops_from_root() now returns a tree by construction, so a per-side
    failure here means a real bug. It stays a meaningful guard on the *assembled*
    graph though: in the merged-component case the two sides can share nodes, and
    the union of two trees that share nodes is not necessarily acyclic.
    """
    cycles = networkx.cycle_basis(graph)
    if cycles:
        print(
            f"[checks] WARNING{f' ({case_id})' if case_id else ''}: "
            f"{len(cycles)} residual cycle(s) after loop removal, "
            f"e.g. {cycles[0]}."
        )
        return False
    return True


def assert_ostia_present(graph: networkx.Graph, case_id: str = "") -> bool:
    """Checks that graph has exactly one node with is_ostium=True and side='L',
    and exactly one with is_ostium=True and side='R'. Warns and returns False
    rather than raising, matching assert_no_cycles()'s convention.

    This is the plain-networkx-graph counterpart of convert.assert_ostia_present()
    (that one works on the typed SimpleCenterlineGraph's topology/side enums,
    for use after conversion); call this one on final_graph itself, right after
    assembly (and after reconnection, if enabled), so a missing ostium is caught
    at the earliest point it could plausibly go missing -- before conversion,
    smoothing, or anything else downstream has a chance to -- rather than
    surfacing later as an opaque networkx error inside hcatnetwork's own
    get_anatomic_segments()/resample(), several calls away from the real cause.
    """
    counts = {"L": 0, "R": 0}
    for _, data in graph.nodes(data=True):
        if data.get("is_ostium", False):
            counts[data.get("side")] = counts.get(data.get("side"), 0) + 1
    if counts.get("L") != 1 or counts.get("R") != 1:
        print(
            f"[checks] WARNING{f' ({case_id})' if case_id else ''}: "
            f"expected exactly one L and one R ostium node, found "
            f"L={counts.get('L', 0)} R={counts.get('R', 0)} "
            f"(graph has {graph.number_of_nodes()} nodes, "
            f"{networkx.number_connected_components(graph)} connected component(s))."
        )
        return False
    return True
