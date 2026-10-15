"""
2D and 3D scatter plots of the right and left coronary ostia of ImageCAS, CAT08 and ASOCA.

Each patient's right and left ostium are first centered on that patient's own midpoint
(R+L)/2: raw RAS coordinates are a world frame tied to each volume's own scanner/export
origin, which differs by tens to hundreds of mm between patients (and between datasets) for
reasons unrelated to anatomy, so plotting raw positions across patients mixes real anatomy
with this arbitrary per-patient offset. Centering removes it and keeps only the bilateral
(right vs. left) geometry, which is what this plot is about.

All the (now per-patient-centered) points share one common rotation:
  - x axis: the mean direction, over all patients, of the vector from left to right,
    i.e. R_p - L_p for patient p (this equals twice the centered R, or -2x the centered L,
    for every single patient, by construction -- rotation only, no further centering needed);
  - y axis: the RAS anterior axis, made orthogonal to x (Gram-Schmidt);
  - z axis: x cross y (approximately superior).

Three permutation tests (PERMANOVA-style: the test statistic is the between-group sum of
squared distances from the grand mean, its null distribution built by reshuffling the dataset
labels) check whether any dataset sits at a systematically different position, distance or
orientation than the other two -- see `permutation_location_test` below for the exact
statistic and `--n-permutations`/`--seed` for reproducibility. Holm-Bonferroni correction is
applied across the three tests.

Colors: ASOCA forest green, CAT08 orange, ImageCAS firebrick (right) and navy (left). ASOCA is
drawn as small, fully opaque triangles and CAT08 as small, fully opaque squares, neither with a
confidence ellipse (few points, single scanner/protocol each -- a raw scatter already reads
cleanly). ImageCAS is drawn as small, semi-transparent circles (many points, would otherwise
dominate the plot), with a 2-sigma confidence ellipse per side (~86% of a bivariate normal)
drawn under the points.

Usage (numpy + matplotlib only; no torch/monai needed unless a label file needs the image header):

    python hcatostialocalization/analysis/make_ostia_scatterplot.py -o <output_dir>

The ostium JSON files are found through the lists <dataset>_all_labels.txt in
hcatostialocalization/test/collections (one JSON path per row). When a list does not exist, the
JSON files are searched in <annotations root>/OstiumAnnotations_<dataset>/OSTIAv2/JSON/, where the
annotations root is deduced from the ImageCAS list (or given with --annotations-root).
Outputs: ostia_scatter_2d / ostia_scatter_3d (pdf and png), ostia_transformed.csv, ostia_frame.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent

DATASETS = ("ASOCA", "CAT08", "ImageCAS")
COLOR_ASOCA = "mediumseagreen"  # good contrast against both firebrick and navy, without being neon
COLOR_CAT08 = "orange"
COLOR_IMAGECAS_R = "firebrick"
COLOR_IMAGECAS_L = "navy"


def point_color(dataset: str, side: str) -> str:
    if dataset == "ASOCA":
        return COLOR_ASOCA
    if dataset == "CAT08":
        return COLOR_CAT08
    return COLOR_IMAGECAS_R if side == "R" else COLOR_IMAGECAS_L


# ------------------------------------------------------------------------------------------
# finding and reading the label files
# ------------------------------------------------------------------------------------------

def read_list(path: Path) -> list[Path]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(Path(line))
    return rows


def deduce_annotations_root(imagecas_labels: list[Path]) -> Path | None:
    # <root>/OstiumAnnotations_ImageCAS/Ostia[/...]/ImageCAS__<id>.ostium.json (or OSTIAv2/JSON/...):
    # the root is the folder that contains OstiumAnnotations_ImageCAS
    if imagecas_labels:
        for parent in imagecas_labels[0].parents:
            if parent.name.startswith("OstiumAnnotations_"):
                return parent.parent
    return None


def find_label_files(dataset: str, collections_dir: Path, override: str | None,
                     annotations_root: Path | None) -> tuple[list[Path], str]:
    if override:
        return read_list(Path(override)), f"list {override}"
    list_path = collections_dir / f"{dataset}_all_labels.txt"
    if list_path.exists():
        return read_list(list_path), f"list {list_path}"
    if annotations_root is not None:
        folder = annotations_root / f"OstiumAnnotations_{dataset}" / "OSTIAv2" / "JSON"
        files = sorted(folder.glob(f"{dataset}__*.ostium.json"))
        if files:
            return files, f"folder {folder} (no {dataset}_all_labels.txt list)"
    return [], "not found"


def sample_id(dataset: str, path: Path) -> str:
    name = path.name
    for suffix in (".ostium.json", "_final_ostia.json", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    prefix = f"{dataset}__"
    return name[len(prefix):] if name.startswith(prefix) else name


def _valid_point(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3


def read_ras(path: Path) -> dict | None:
    """Returns {"R": (3,), "L": (3,)} in RAS mm, or None if a side is missing."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: cannot read {path}: {e}")
        return None

    points = {}
    if "pos" in data:                       # blueprint format
        for side in ("R", "L"):
            value = (data["pos"].get(side) or {}).get("ras")
            if _valid_point(value):
                points[side] = value
    elif "ostia" in data:                   # ImageCAS / CAT08 format
        for side in ("R", "L"):
            value = (data["ostia"].get(side) or {}).get("coordinate_ras")
            if _valid_point(value):
                points[side] = value

    if len(points) < 2:
        # other formats (e.g. raw ASOCA points, Slicer markups): use the package loader, which
        # needs torch/monai and, for the raw ASOCA format, the image header (not available here)
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from hcatostialocalization.io.io import extract_simplified_ras
            simple = extract_simplified_ras(str(path))
        except Exception as e:                # noqa: BLE001
            print(f"WARNING: skipping {path}: unsupported format ({e})")
            return None
        if not simple:
            return None
        points = {s: simple[s]["ras"] for s in ("R", "L") if _valid_point(simple[s].get("ras"))}
        if len(points) < 2:
            print(f"WARNING: skipping {path}: missing side")
            return None

    return {s: np.asarray(points[s], dtype=float) for s in ("R", "L")}


# ------------------------------------------------------------------------------------------
# anatomical frame
# ------------------------------------------------------------------------------------------

def build_frame(right: np.ndarray, left: np.ndarray) -> dict:
    """right, left: (N, 3) RAS positions, same patient order. Returns the common rotation
    (rows = new axes in RAS) and its supporting statistics. No origin: every patient is
    centered on their own (R+L)/2 (see `transform`), so the frame only needs to rotate."""
    centers = (right + left) / 2.0
    half_diff = right - centers                      # = (R - L) / 2, per patient
    half_distance = np.linalg.norm(half_diff, axis=1)  # half of the R-L distance, per patient
    unit_dirs = half_diff / half_distance[:, None]

    mean_dir = unit_dirs.mean(axis=0)
    mean_resultant_length = float(np.linalg.norm(mean_dir))  # 1 = all patients share the same R-L direction
    ex = mean_dir / mean_resultant_length
    anterior = np.array([0.0, 1.0, 0.0])
    ey = anterior - np.dot(anterior, ex) * ex
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)

    rotation = np.stack([ex, ey, ez])       # rows are the new axes, expressed in RAS
    return {
        "rotation": rotation,
        "mean_axis_direction_ras": ex,
        "mean_resultant_length": mean_resultant_length,
        "half_distance_mean_mm": float(half_distance.mean()),
        "half_distance_std_mm": float(half_distance.std()),
        # per-patient arrays (RAS, pre-rotation), same order as `right`/`left`: reused for the
        # consistency tests below. The between-group test statistic used there is invariant to a
        # fixed rotation applied identically to every patient, so RAS-space values are equivalent
        # to using the rotated/plotted ones.
        "unit_dir_per_patient": unit_dirs,
        "half_distance_per_patient": half_distance,
    }


def transform(point: np.ndarray, center: np.ndarray, frame: dict) -> np.ndarray:
    """point, center: (3,) RAS mm; center is that same patient's (R+L)/2."""
    return (point - center) @ frame["rotation"].T


# ------------------------------------------------------------------------------------------
# permutation tests: does any dataset sit at a systematically different position/distance/
# orientation than the other two? (H0 below, per test)
# ------------------------------------------------------------------------------------------

def _between_group_ss(values: np.ndarray, labels: np.ndarray, groups: list[str]) -> float:
    """values: (N, d); labels: (N,) dataset name per row. Sum over groups of n_g * ||mean_g -
    grand_mean||^2 -- the between-group sum of squares of a one-way (P)MANOVA."""
    grand_mean = values.mean(axis=0)
    ss = 0.0
    for g in groups:
        rows = values[labels == g]
        if len(rows) == 0:
            continue
        diff = rows.mean(axis=0) - grand_mean
        ss += len(rows) * float(np.dot(diff, diff))
    return ss


def permutation_location_test(values: np.ndarray, labels: np.ndarray, n_perm: int, seed: int) -> dict:
    """Non-parametric (PERMANOVA-style) test of H0: the datasets share the same mean `values`
    (no systematic difference in position/distance/direction) vs. H1: at least one dataset's
    mean differs. `values` can be 1D (e.g. a scalar distance, reshaped to (N, 1)) or multi-D
    (e.g. a 3D position or direction). The null distribution is built by randomly reassigning
    the dataset labels (keeping their counts fixed) `n_perm` times; the p-value is the fraction
    of permutations whose statistic is at least as extreme as the one actually observed."""
    labels = np.asarray(labels)
    groups = sorted(set(labels.tolist()))
    observed = _between_group_ss(values, labels, groups)

    rng = np.random.default_rng(seed)
    permuted = labels.copy()
    at_least_as_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(permuted)
        if _between_group_ss(values, permuted, groups) >= observed - 1e-9:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (n_perm + 1)  # +1: the observed labeling is itself one permutation

    group_stats = {g: {"n": int((labels == g).sum()), "mean": values[labels == g].mean(axis=0)} for g in groups}
    return {"statistic": observed, "p_value": p_value, "n_perm": n_perm, "groups": group_stats}


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction for a family of named p-values."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running_max = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (m - rank) * p))
        adjusted[name] = running_max
    return adjusted


# ------------------------------------------------------------------------------------------
# plots
# ------------------------------------------------------------------------------------------

MARKER = {"ASOCA": "^", "CAT08": "s", "ImageCAS": "o"}


def _legend_handles():
    from matplotlib.lines import Line2D
    def handle(color, marker, label):
        return Line2D([], [], marker=marker, linestyle="", markersize=4, markerfacecolor=color,
                      markeredgecolor=color, label=label)
    return [handle(COLOR_IMAGECAS_R, MARKER["ImageCAS"], "ImageCAS right"),
            handle(COLOR_IMAGECAS_L, MARKER["ImageCAS"], "ImageCAS left"),
            handle(COLOR_ASOCA, MARKER["ASOCA"], "ASOCA"),
            handle(COLOR_CAT08, MARKER["CAT08"], "CAT08")]


ELLIPSE_ALPHA = 0.45      # ImageCAS only
ELLIPSE_N_STD = 2.0       # ~86% of a bivariate normal (Mahalanobis radius 2 in 2D)


def _scatter_style(dataset: str) -> dict:
    # ImageCAS (many points, would otherwise dominate): small, semi-transparent circles, with an
    # ellipse under them. ASOCA/CAT08 (few points, single scanner/protocol): small, fully opaque
    # triangles/squares, no ellipse.
    if dataset == "ImageCAS":
        return dict(marker=MARKER[dataset], s=4, alpha=0.35, edgecolors="none")
    # a square reads visually bigger than a triangle at the same `s`, so size them separately
    size = 10 if dataset == "ASOCA" else 6
    return dict(marker=MARKER[dataset], s=size, alpha=1.0, edgecolors="none")


def _add_confidence_ellipse(ax, points_2d: np.ndarray, color: str) -> None:
    """2-sigma covariance ellipse of a (N, 2) point cloud, drawn under the scatter points."""
    from matplotlib.patches import Ellipse
    if len(points_2d) < 3:
        return
    mean = points_2d.mean(axis=0)
    cov = np.cov(points_2d, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * ELLIPSE_N_STD * np.sqrt(np.clip(eigvals, 0, None))
    ax.add_patch(Ellipse(mean, width, height, angle=angle, facecolor=color,
                         edgecolor="none", alpha=ELLIPSE_ALPHA, zorder=1))


def plot_2d(samples: list[dict], out_stem: Path, formats: list[str], dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})

    # stacked vertically (single-column width, two panels tall) to fit a double-column layout
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 6.4))
    panels = [(0, 1, "x: left to right (mm)", "y: anterior (mm)"),
              (0, 2, "x: left to right (mm)", "z: superior (mm)")]
    for ax, (i, j, xlabel, ylabel) in zip(axes, panels):
        groups = [(dataset, side, np.array([s["xyz"] for s in samples if s["dataset"] == dataset and s["side"] == side]))
                  for dataset in ("ImageCAS", "ASOCA", "CAT08") for side in ("R", "L")]
        for dataset, side, pts in groups:
            if dataset == "ImageCAS" and len(pts):
                _add_confidence_ellipse(ax, pts[:, [i, j]], point_color(dataset, side))
        for dataset, side, pts in groups:
            if len(pts):
                ax.scatter(pts[:, i], pts[:, j], color=point_color(dataset, side), zorder=2, **_scatter_style(dataset))
        ax.axhline(0, color="gray", linewidth=0.4, zorder=0)
        ax.axvline(0, color="gray", linewidth=0.4, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.6)
        ax.relim()
        ax.autoscale_view()
    axes[0].legend(handles=_legend_handles(), fontsize=6, loc="best", frameon=True)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{out_stem}.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_3d(samples: list[dict], out_stem: Path, formats: list[str], dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})

    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    for dataset in ("ImageCAS", "ASOCA", "CAT08"):
        for side in ("R", "L"):
            pts = np.array([s["xyz"] for s in samples if s["dataset"] == dataset and s["side"] == side])
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=point_color(dataset, side), **_scatter_style(dataset))
    all_xyz = np.array([s["xyz"] for s in samples])
    ax.set_box_aspect(np.ptp(all_xyz, axis=0))
    ax.set_xlabel("x: left to right (mm)")
    ax.set_ylabel("y: anterior (mm)")
    ax.set_zlabel("z: superior (mm)")
    ax.view_init(elev=20, azim=-60)
    ax.legend(handles=_legend_handles(), fontsize=6, loc="upper left", frameon=True)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{out_stem}.{fmt}", dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="2D and 3D scatter plot of the coronary ostia in a common anatomical frame.")
    parser.add_argument("--collections-dir", default=str(PACKAGE_DIR / "test" / "collections"),
                        help="folder with the <dataset>_all_labels.txt lists")
    for dataset in DATASETS:
        parser.add_argument(f"--{dataset.lower()}-labels", default=None,
                            help=f"list file with the ostium JSON paths of {dataset} (overrides the default list)")
    parser.add_argument("--annotations-root", default=None,
                        help="folder that contains OstiumAnnotations_<dataset>/ (used when a list is missing)")
    parser.add_argument("-o", "--output-dir", default="ostia_scatter")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-permutations", type=int, default=9999,
                        help="permutations for the dataset-consistency tests (excludes the observed labeling)")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for the permutation tests")
    args = parser.parse_args()

    collections_dir = Path(args.collections_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imagecas_list = collections_dir / "ImageCAS_all_labels.txt"
    annotations_root = Path(args.annotations_root) if args.annotations_root else (
        deduce_annotations_root(read_list(imagecas_list)) if imagecas_list.exists() else None)

    # read the ostia
    records = []   # one per patient: dataset, id, R (3,), L (3,)
    for dataset in DATASETS:
        override = getattr(args, f"{dataset.lower()}_labels")
        files, source = find_label_files(dataset, collections_dir, override, annotations_root)
        print(f"{dataset}: {len(files)} label files from {source}")
        skipped = 0
        for path in files:
            pos = read_ras(path)
            if pos is None:
                skipped += 1
                continue
            records.append({"dataset": dataset, "id": sample_id(dataset, path), "R": pos["R"], "L": pos["L"]})
        if skipped:
            print(f"  skipped {skipped} files (unreadable or missing side)")
    if not records:
        raise SystemExit("No ostium annotation found: check --collections-dir / --annotations-root.")

    # common frame (rotation only; each patient is centered on their own (R+L)/2, see build_frame/transform)
    right = np.array([r["R"] for r in records])
    left = np.array([r["L"] for r in records])
    frame = build_frame(right, left)

    samples = []
    for r in records:
        center = (r["R"] + r["L"]) / 2.0
        for side in ("R", "L"):
            samples.append({"dataset": r["dataset"], "id": r["id"], "side": side,
                            "ras": r[side], "xyz": transform(r[side], center, frame)})

    # summary
    np.set_printoptions(precision=3, suppress=True)
    print(f"\nMean left-to-right axis direction (RAS): {frame['mean_axis_direction_ras']}")
    print(f"Mean resultant length of that direction across patients: {frame['mean_resultant_length']:.4f} "
          f"(1.0 = every patient's R-L axis points exactly the same way)")
    print(f"Half of the R-L distance, over all patients: {frame['half_distance_mean_mm']:.2f} +- {frame['half_distance_std_mm']:.2f} mm")
    print(f"New axes in RAS (rows x, y, z):\n{frame['rotation']}\n")
    print(f"{'dataset':10s} {'patients':>8s} {'R-L distance mm (mean +- SD)':>30s}   mean R (x,y,z)          mean L (x,y,z)")
    for dataset in DATASETS:
        recs = [r for r in records if r["dataset"] == dataset]
        if not recs:
            continue
        dist = np.array([np.linalg.norm(r["R"] - r["L"]) for r in recs])
        centers = [(r["R"] + r["L"]) / 2.0 for r in recs]
        mr = np.mean([transform(r["R"], c, frame) for r, c in zip(recs, centers)], axis=0)
        ml = np.mean([transform(r["L"], c, frame) for r, c in zip(recs, centers)], axis=0)
        print(f"{dataset:10s} {len(recs):8d} {dist.mean():14.2f} +- {dist.std():5.2f}        {mr}   {ml}")

    # -------------------------------------------------------------------------------------
    # dataset-consistency tests: does any dataset sit at a systematically different position,
    # distance or orientation than the other two? Each test below shares the same H0/H1:
    #   H0: the three datasets have the same mean value (no systematic difference).
    #   H1: at least one dataset's mean value differs from the other two.
    # -------------------------------------------------------------------------------------
    labels_per_patient = np.array([r["dataset"] for r in records])
    tests = {}
    for side in ("R", "L"):
        centers = (right + left) / 2.0
        raw = right if side == "R" else left
        xyz = np.array([transform(p, c, frame) for p, c in zip(raw, centers)])
        tests[f"position_{side}"] = permutation_location_test(xyz, labels_per_patient, args.n_permutations, args.seed)
    tests["half_distance"] = permutation_location_test(
        frame["half_distance_per_patient"][:, None], labels_per_patient, args.n_permutations, args.seed)
    tests["axis_direction"] = permutation_location_test(
        frame["unit_dir_per_patient"], labels_per_patient, args.n_permutations, args.seed)

    holm_p = holm_bonferroni({name: t["p_value"] for name, t in tests.items()})
    for name, t in tests.items():
        t["p_value_holm"] = holm_p[name]

    print(f"\nConsistency tests (H0: the three datasets share the same mean value; "
          f"H1: at least one differs), {args.n_permutations} permutations, seed {args.seed}, "
          f"Holm-corrected across {len(tests)} tests:")
    print(f"{'test':16s} {'statistic':>10s} {'p (raw)':>10s} {'p (Holm)':>10s}")
    for name, t in tests.items():
        print(f"{name:16s} {t['statistic']:10.4f} {t['p_value']:10.4f} {t['p_value_holm']:10.4f}")
        for g, s in t["groups"].items():
            print(f"    {g:10s} n={s['n']:5d}  mean={np.array2string(s['mean'], precision=3, suppress_small=True)}")

    # outputs
    with open(output_dir / "ostia_transformed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "id", "side", "x_mm", "y_mm", "z_mm", "ras_r_mm", "ras_a_mm", "ras_s_mm"])
        for s in samples:
            writer.writerow([s["dataset"], s["id"], s["side"], *[f"{v:.4f}" for v in s["xyz"]], *[f"{v:.4f}" for v in s["ras"]]])

    def _jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    with open(output_dir / "ostia_frame.json", "w") as f:
        json.dump({"frame": _jsonable({k: v for k, v in frame.items() if not k.endswith("_per_patient")}),
                   "consistency_tests": _jsonable(tests)}, f, indent=2)

    plot_2d(samples, output_dir / "ostia_scatter_2d", args.formats, args.dpi)
    plot_3d(samples, output_dir / "ostia_scatter_3d", args.formats, args.dpi)
    print(f"\nFigures and tables saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
