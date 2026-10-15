"""
Batch entry point for the v2 centerline extraction (root-driven tracing, no
skeletonisation; see README.md in this folder).

    python -m hcatostialocalization.data.makecenterlines.v2 \\
        --masks 1.label.nii.gz 2.label.nii.gz ... \\
        --ostia 1_final_ostia.json 2_final_ostia.json ... \\
        --output-dir <outdir> [-a graph qc]

--masks/--ostia (and --images) may each be a single .txt/.list/.lst file, one
path per line, as in v1. Run from the repository root (repository root).
"""
import os
import sys
import json
import argparse

from termcolor import colored

from ..utils.cli import case_id_from_image_path, expand_paths
from . import pipeline


_ARTIFACT_CHOICES = ["none", "graph", "qc", "markup-3d-slicer", "debug_figures", "all"]
_ALL_ARTIFACTS = {"graph", "qc", "markup-3d-slicer", "debug_figures"}


def _case_id(path: str) -> str:
    """'1.img.nii.gz' / '1.label.nii.gz' (ImageCAS) / '1.coronary.nii.gz' (ImageCAS-X) -> '1';
    'Normal_1.nrrd' (ASOCA) -> 'Normal_1'."""
    case_id = case_id_from_image_path(path)
    for suffix in (".label", ".coronary"):
        if case_id.endswith(suffix):
            return case_id[: -len(suffix)]
    return case_id


# CLI flag -> where the value was recorded in a case's qc.json, for --resume
_RESUME_CHECKED = {
    "dilation_voxels": ("parameters", "dilation_voxels"), "centering_power": ("parameters", "centering_power"),
    "shell_penalty": ("parameters", "shell_penalty"), "max_ostium_bridge_mm": ("parameters", "max_ostium_bridge_mm"),
    "min_branch_length_mm": ("parameters", "min_branch_length_mm"), "cover_radius_scale": ("parameters", "cover_radius_scale"),
    "cover_tol_mm": ("parameters", "cover_tol_mm"), "radius_cap_scale": ("parameters", "radius_cap_scale"),
    "radius_cap_window_mm": ("parameters", "radius_cap_window_mm"), "trim_endpoints_mm": ("parameters", "trim_endpoints_mm"),
    "min_ostial_trunk_mm": ("parameters", "min_ostial_trunk_mm"), "suspect_extent_mm": ("parameters", "suspect_extent_mm"),
    "max_gap_bridge_mm": ("parameters", "max_gap_bridge_mm"), "smooth_min_shift_mm": ("smoothing", "min_shift_mm"), "smooth_shift_limit": ("smoothing", "shift_limit"),
    "smooth_sigma_mm": ("smoothing", "sigma_mm"), "smooth_max_shift_fraction": ("smoothing", "max_shift_fraction"),
    "smooth_max_shift_mm": ("smoothing", "max_shift_mm"), "smooth_final_spacing_mm": ("smoothing", "final_spacing_mm"),
}


def _done(output_dir: str, case_id: str) -> bool:
    """A case is processed if it left a qc file (success) or an error file (failure)."""
    return (os.path.isfile(os.path.join(output_dir, "qc", f"{case_id}.qc.json"))
            or os.path.isfile(os.path.join(output_dir, "errors", f"{case_id}.error.json")))


def _resume_start(output_dir: str, case_ids: list, overlap: int) -> int:
    """Index from which to resume: `overlap` cases before the last processed one in
    list order (they are redone, in case the interrupted run was writing them)."""
    last = max((k for k, c in enumerate(case_ids) if _done(output_dir, c)), default=-1)
    return max(0, last + 1 - overlap) if last >= 0 else 0


def _resume_mismatches(output_dir: str, case_ids: list, args) -> list:
    """Differences between this run's parameters and those recorded in the qc.json of
    the last successful case: resuming with different parameters would mix two
    configurations in one dataset."""
    for case_id in reversed(case_ids):
        path = os.path.join(output_dir, "qc", f"{case_id}.qc.json")
        if os.path.isfile(path):
            with open(path) as f:
                qc = json.load(f)
            break
    else:
        return []
    out = []
    smoothed = "smoothing" in qc
    if smoothed != args.smooth_graph:
        out.append(f"--smooth-graph was {'on' if smoothed else 'off'} in the previous run")
    for name, (section, key) in _RESUME_CHECKED.items():
        if section == "smoothing" and not (smoothed and args.smooth_graph):
            continue
        flag = "--" + name.replace("_", "-")
        if key not in qc.get(section, {}):
            out.append(f"{flag}: not recorded by the previous run (older code), now {getattr(args, name)}")
        elif qc[section][key] != getattr(args, name):
            out.append(f"{flag}: previous run {qc[section][key]}, now {getattr(args, name)} (pass {flag} {qc[section][key]} to match)")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract coronary centerline graphs by growing each tree from its annotated ostium through the segmentation mask (no skeletonisation)."
    )
    parser.add_argument("--masks", nargs="+", required=True, help="Coronary segmentation mask path(s), or a single .txt/.list/.lst file (one path per line). Loaded with MONAI (RAS) and resampled to --isotropic-spacing-mm.")
    parser.add_argument("--ostia", nargs="+", required=True, help="Ostium ground-truth JSON path(s), same count/order as --masks, or a single .txt/.list/.lst file.")
    parser.add_argument("--images", nargs="+", default=None, help="Optional CCTA path(s), same count/order as --masks, used only to name the cases (as in v1). Without it, the case id comes from the mask file name.")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory (one subdirectory per artifact type).")
    parser.add_argument("-a", "--artifacts", nargs="+", choices=_ARTIFACT_CHOICES, default=["all"], help="'graph' (GML), 'qc' (JSON), 'markup-3d-slicer', 'debug_figures'; 'none' (smoke test, nothing written) or 'all', alone.")
    parser.add_argument("--isotropic-spacing-mm", type=float, default=0.5, help="Working-grid spacing (mm).")
    parser.add_argument("--dilation-voxels", type=int, default=1, help="The mask is dilated by this many voxels to repair connections broken by resampling; paths use the added voxels only where the mask itself is broken. 0 = off.")
    parser.add_argument("--centering-power", type=float, default=4.0, help="Exponent p of the path cost (DT + 0.5)^-p, DT = distance to the vessel wall. Higher = paths stay closer to the vessel axis in bends, at the price of length.")
    parser.add_argument("--shell-penalty", type=float, default=1000.0, help="Cost multiplier for voxels added by the dilation; also the per-voxel cost of the bridge from the ostium to the mask.")
    parser.add_argument("--max-ostium-bridge-mm", type=float, default=25.0, help="The case fails if the mask is farther than this from an annotated ostium (mm). The gap is crossed in a straight line, so a long one is worth a look: see qc sides.{L,R}.ostium_to_mask_mm.")
    parser.add_argument("--min-branch-length-mm", type=float, default=5.0, help="A traced branch is kept only if it runs at least this far outside the tube already covered by the tree (10 mm until 2026-10-05: rejected candidates split into wall bumps < 2 mm and real short branches at 5-10 mm).")
    parser.add_argument("--cover-radius-scale", type=float, default=1.5, help="Tube around each traced path marked as covered: scale * local radius + --cover-tol-mm.")
    parser.add_argument("--cover-tol-mm", type=float, default=1.0, help="See --cover-radius-scale.")
    parser.add_argument("--radius-cap-scale", type=float, default=1.2, help="Second pass with the wall distance capped at this x the vessel radius measured along the first tree, so paths are not pulled into the centre of bulky junctions. 0 = single pass, no cap.")
    parser.add_argument("--radius-cap-window-mm", type=float, default=5.0, help="The vessel radius at a tree voxel is the median wall distance of the tree voxels within this distance along the tree.")
    parser.add_argument("--trim-endpoints-mm", type=float, default=2.0, help="Length removed from every endpoint of the finished tree (never past a bifurcation). 0 = off.")
    parser.add_argument("--min-ostial-trunk-mm", type=float, default=2.0, help="Minimum straight-line distance between each ostium and its first bifurcation (mm): branches leaving closer are re-attached to the main path at this distance (a bifurcation at the ostium is anatomically implausible). 0 = off.")
    parser.add_argument("--max-gap-bridge-mm", type=float, default=10.0, help="Mask pieces disconnected from the tree by a gap up to this (mm) are joined to it by a straight line (costed like the dilation shell, recorded in qc gap_bridges); farther pieces stay unreachable. 0 = off.")
    parser.add_argument("--suspect-extent-mm", type=float, default=10.0, help="QC only: a mask region the final tree does not explain (or a mask piece no ostium reaches) at least this long flags the case as a possibly missed branch.")
    parser.add_argument("--smooth-graph", action="store_true", help="Smooth the geometry branch by branch (main vessel through its bifurcations, then each side branch from its junction): Gaussian over arc length, ostia, junctions and tips fixed, each node's shift limited by its radius. Removes the voxel-staircase direction jitter that breaks curved planar reformats. Then resampled to --smooth-final-spacing-mm.")
    parser.add_argument("--smooth-sigma-mm", type=float, default=2.0, help="Gaussian sigma along the branch (mm). 2 mm: median turn 1.5 deg per 0.5 mm on the first 5 cases, nodes moved by a median 9%% of the radius.")
    parser.add_argument("--smooth-max-shift-fraction", type=float, default=0.4, help="A node moves at most this fraction of its radius.")
    parser.add_argument("--smooth-max-shift-mm", type=float, default=1.0, help="... and at most this far (mm).")
    parser.add_argument("--smooth-shift-limit", choices=["mask", "radius"], default="mask", help="'mask' (default): a node inside the mask may move up to --smooth-max-shift-mm as long as it stays inside the mask, else its shift is halved until it does (falling back to the radius limit). 'radius': at most --smooth-max-shift-fraction x its radius (floored at --smooth-min-shift-mm); on thin masks this leaves ~20 deg of staircase jitter at p95 (ASOCA, 2026-10-04).")
    parser.add_argument("--smooth-min-shift-mm", type=float, default=0.0, help="A node may always move at least this far (mm), whatever its radius (still at most --smooth-max-shift-mm): in thin vessels a fraction of the radius is smaller than the voxel staircase. 0 = off.")
    parser.add_argument("--smooth-final-spacing-mm", type=float, default=0.5, help="Node spacing after smoothing (0 = keep the traced nodes).")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run in the same --output-dir: skip the cases up to the last one processed (a qc or error file exists), redoing the last --resume-overlap of them. Refuses if the parameters differ from those recorded by the previous run.")
    parser.add_argument("--resume-overlap", type=int, default=2, help="Cases redone before the resume point (--resume).")
    parser.add_argument("--resume-anyway", action="store_true", help="With --resume, continue even if the parameters differ from the previous run (mixes two configurations in one dataset).")
    parser.add_argument("--shard", default=None, help="'i/n': process only the cases whose position in the list is i modulo n (0 <= i < n), to split a batch over n parallel jobs writing to the same --output-dir. Each job writes batch_summary.shard<i>of<n>.json.")
    args = parser.parse_args()

    for keyword in ("none", "all"):
        if keyword in args.artifacts and len(args.artifacts) != 1:
            parser.error(f"if '{keyword}' is specified in -a/--artifacts, nothing else must be included.")
    artifacts = _ALL_ARTIFACTS if "all" in args.artifacts else set(args.artifacts)

    masks = expand_paths(args.masks)
    ostia = expand_paths(args.ostia)
    names = expand_paths(args.images) if args.images else masks
    if not (len(masks) == len(ostia) == len(names)):
        parser.error(f"--masks ({len(masks)}), --ostia ({len(ostia)}) and --images ({len(names)}) must expand to the same count.")

    case_ids = [_case_id(p) for p in names]
    selected = list(range(len(masks)))
    if args.shard:
        try:
            shard_index, shard_count = (int(v) for v in args.shard.split("/"))
            assert 0 <= shard_index < shard_count
        except (ValueError, AssertionError):
            parser.error("--shard must be 'i/n' with 0 <= i < n")
        selected = [k for k in selected if k % shard_count == shard_index]
    if args.resume:
        mismatches = _resume_mismatches(args.output_dir, case_ids, args)
        if mismatches and not args.resume_anyway:
            parser.error("cannot resume, the parameters differ from the previous run:\n  " + "\n  ".join(mismatches)
                         + "\nPass the previous values, or --resume-anyway to mix the two configurations.")
        ids = [case_ids[k] for k in selected]
        start = _resume_start(args.output_dir, ids, args.resume_overlap)
        print(colored(f"Resuming at case {start + 1} of {len(ids)} ({ids[start] if start < len(ids) else 'none left'}): "
                      f"{start} skipped, the last {args.resume_overlap} processed ones redone.", "cyan"))
        selected = selected[start:]

    results = []
    for position, k in enumerate(selected, start=1):
        mask_path, ostium_path, case_id = masks[k], ostia[k], case_ids[k]
        idx = k + 1
        print(colored(f"[{idx}/{len(masks)}] {case_id}  ({position}/{len(selected)} in this job)", "cyan"))
        result = pipeline.process_one(
            mask_path, ostium_path, case_id, args.output_dir, artifacts,
            isotropic_spacing_mm=args.isotropic_spacing_mm,
            dilation_voxels=args.dilation_voxels,
            centering_power=args.centering_power,
            shell_penalty=args.shell_penalty,
            max_ostium_bridge_mm=args.max_ostium_bridge_mm,
            min_branch_length_mm=args.min_branch_length_mm,
            cover_radius_scale=args.cover_radius_scale,
            cover_tol_mm=args.cover_tol_mm,
            radius_cap_scale=args.radius_cap_scale,
            radius_cap_window_mm=args.radius_cap_window_mm,
            trim_endpoints_mm=args.trim_endpoints_mm,
            min_ostial_trunk_mm=args.min_ostial_trunk_mm,
            max_gap_bridge_mm=args.max_gap_bridge_mm,
            smooth_min_shift_mm=args.smooth_min_shift_mm,
            smooth_shift_limit=args.smooth_shift_limit,
            suspect_extent_mm=args.suspect_extent_mm,
            smooth_graph=args.smooth_graph,
            smooth_sigma_mm=args.smooth_sigma_mm,
            smooth_max_shift_fraction=args.smooth_max_shift_fraction,
            smooth_max_shift_mm=args.smooth_max_shift_mm,
            smooth_final_spacing_mm=args.smooth_final_spacing_mm,
        )
        error_path = os.path.join(args.output_dir, "errors", f"{case_id}.error.json")
        if result["status"] == "ok":
            print(colored(f"    ok: {result['written']}", "green"))
            if os.path.isfile(error_path):
                os.remove(error_path)  # an earlier failure of this case, now fixed
        else:
            print(colored(f"    FAILED: {result['error']}", "red"))
            if "none" not in artifacts:  # kept per case, so an interrupted run still records its failures
                os.makedirs(os.path.dirname(error_path), exist_ok=True)
                with open(error_path, "w") as f:
                    json.dump(result, f, indent=2)
        results.append(result)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(colored(f"Done: {n_ok} ok, {len(results) - n_ok} failed, {len(results)} total.", "cyan"))
    if "none" not in artifacts:
        os.makedirs(args.output_dir, exist_ok=True)
        name = "batch_summary"
        if args.shard:
            name += f".shard{shard_index}of{shard_count}"
        if args.resume:
            name += f".resumed_at_{selected[0] + 1 if selected else 'end'}"
        summary_path = os.path.join(args.output_dir, name + ".json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(colored(f"Batch summary written to {summary_path}", "cyan"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
