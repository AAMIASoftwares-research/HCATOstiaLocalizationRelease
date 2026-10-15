"""
Ensemble of ostium predictions already produced by the inference of N models.

For every image, the predictions of the N models are merged into ONE prediction (one left and one right ostium).
Nothing is run on a GPU. Two ways of ensembling, chosen per image (--from):

  maps         the per-class PROBABILITY NIfTI volumes written by `infer -a probability-nifti` are merged voxel by
               voxel (median across models), then each ostium is located on the merged map exactly like the inference
               does (peak + local centroid, --window-size voxels). The NIfTI files carry the affine of the image, so
               the position in RAS (mm) comes straight from them, no image is needed.
  coordinates  the final positions (`.ostium.json` / `.mrk.json`) are merged: median of the N positions, independently
               for each ostium.
  auto         (default) maps for the images for which every model has its two probability volumes (same grid),
               coordinates for the others (a warning is printed).

Usage, as a module, from the folder that contains `hcatostialocalization/`:

    python -m hcatostialocalization.infer.ensamble_predictions \
        -p results/test_set__SwinUNETRv2_DSNT/<run> results/test_set__nnUNetResidualUNet_DSNT/<run> ... \
        [-i hcatostialocalization/test/collections_test_set/all_images.txt] \
        [-l hcatostialocalization/test/collections_test_set/all_labels.txt] \
        [-o results/ensemble_<name>]

-p/--predictions   N sources (one per model), 2 or more. Each one can be:
                     - a folder (for instance the output folder of infer/__main__.py): it is searched for
                       `probability-nifti/*_probability_class00?*.nii.gz` (or the same files directly in it) and for
                       positions: `markers/*.ostium.json`, else `*.ostium.json`, else `markups-3d-slicer/*.mrk.json`,
                       else `*.mrk.json`;
                     - a single `.ostium.json` (or 3D Slicer `.mrk.json`) file, or a single probability NIfTI: one
                       image of one model;
                     - any mix of the above. Each -p argument is ONE model; predictions are grouped by image, so
                       a list of single files (one per model, all of the same image) gives a one-image ensemble.
-i/--images        optional. Images (or a .txt list, one path per line), as in infer/__main__.py. If given, only these
                   images are ensembled, and they are needed to read the labels.
-l/--label         optional. Ground truth, one per image in the same order (or a .txt list), any format supported by
                   io.extract_simplified_ras. Requires -i. If present, the distances of the ensemble AND of every
                   single model are computed and compared; if not, the new predictions are just saved.
-a/--artifacts    what to save, like infer/__main__.py but only what exists for an ensemble (default: marker, the
                   ensemble prediction itself; use `none` for the metrics only):
                     marker             markers/<name>.ostium.json          ensemble prediction (same schema as infer)
                     markup-3d-slicer   markups-3d-slicer/<name>.mrk.json   same, for 3D Slicer
                     probability-nifti  probability-nifti/<name>_probability_class00?.nii.gz   merged maps, heavy;
                                        only for the images merged from maps
                     all                the three above        none   nothing
                   (segmentation masks, logits and the .pt tensors of infer are not produced here.)
-o/--output        output folder. Besides the artifacts, with labels it holds
                     metrics__Ensemble.json                same layout as infer/__main__.py, so
                                                           test/split_metrics_by_dataset.py works on it

Matching predictions to images. The key of an image is its file name without extension (`Normal_5` for
`Normal_5.nrrd`). For an `.ostium.json` it is read from `image_filename`; for a `.mrk.json` or a probability NIfTI it is
the file name without the `case_<index>_` prefix (and `_probability_class...` suffix) written by infer/__main__.py.
The index is NOT used: different runs may have a different order.

Maps. Class 1 is the right and class 2 the left ostium, as in infer/__main__.py. All the maps of one image must be on
the same grid (same shape and affine); otherwise that image falls back to coordinates. The voxel-wise MEDIAN of
probability maps is degenerate when the models disagree by more than the width of their peaks (with 3 or more models
whose peaks do not overlap the merged map is about zero there): if that happens, use `--map-combine mean`, or check
the printed distance between the two ensembles.
Positions located on a probability map are not bit-identical to the ones in the `.ostium.json` of a single model,
because infer/__main__.py locates the ostia on the LOGITS (before the spatial softmax), while the NIfTI files hold the
probabilities.

Affines of the blueprint. A blueprint `.ostium.json` already contains the affines of the image, so the ensemble
blueprint is taken from it without opening any image. Probability NIfTI files only hold the MONAI RAS affine, not the
SimpleITK one: if no source has a blueprint, the affines are computed from the image given with -i (needs
SimpleITK/MONAI); without it, only the 3D Slicer markup is written.

Missing predictions. By default an image must be predicted by ALL the N models, otherwise it is skipped with a warning.
Use --min-models K to also ensemble images predicted by at least K models.

Median. "median" (default) is the coordinate-wise median (same definition as numpy.median on each of x, y, z): with an odd
number of models the result is one of the predictions in each coordinate, with an even number the average of the two
central values (for N = 2 it is the mean). "geometric-median" is the point minimizing the sum of the Euclidean distances
to the predictions (Weiszfeld), which is a single vector rather than a mix of coordinates.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import sys
import warnings
from pathlib import Path

import numpy as np

from hcatostialocalization.infer.thresholds import BUCKETS, bucket_summary, classify_pair

SIDES = ("R", "L")  # order of every per-class list, as in infer/__main__.py and test/split_metrics_by_dataset.py
CLASS_OF_SIDE = {"R": 1, "L": 2}  # probability/logits channel of each ostium
STAT_KEYS = ["min", "p05", "p25", "median", "p75", "p95", "max", "mean", "std"]
NIFTI_EXT = (".nii.gz", ".nii")


# ---------------------------------------------------------------------------------------------------------------
# Keys and readers
# ---------------------------------------------------------------------------------------------------------------

def image_key(path_or_name: str) -> str:
    """File name without extension (.nii.gz, .nrrd, ...), as infer/__main__.py builds it."""
    name = os.path.basename(str(path_or_name))
    for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".ostium.json", ".mrk.json", ".json"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


_CASE_PREFIX = re.compile(r"^case_\d+_")
_MAP_NAME = re.compile(r"^(?P<key>.+)_probability_class(?P<cls>\d{3})(?:_.*)?$")


def parse_map_name(path: Path):
    """(image key, class index) of a probability NIfTI written by infer/__main__.py, else None."""
    m = _MAP_NAME.match(image_key(path.name))
    return (_CASE_PREFIX.sub("", m.group("key")), int(m.group("cls"))) if m else None


def read_prediction(path: Path) -> dict:
    """{'key', 'ras': {'R': array(3), 'L': array(3)}, 'blueprint': dict|None} from a .ostium.json or a Slicer .mrk.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "pos" in data:  # blueprint written by infer/__main__.py
        key = image_key(data.get("image_filename") or _CASE_PREFIX.sub("", image_key(path.name)))
        ras = {s: np.array(data["pos"][s]["ras"], dtype=float) for s in SIDES}
        return {"key": key, "ras": ras, "blueprint": data}
    if "markups" in data:  # 3D Slicer
        markup = data["markups"][0]
        flip = markup.get("coordinateSystem", "LPS") == "LPS"
        ras = {}
        for pt in markup.get("controlPoints", []):
            label = pt.get("label", "").upper()
            if label in SIDES:
                p = np.array(pt["position"], dtype=float)
                ras[label] = p * np.array([-1.0, -1.0, 1.0]) if flip else p
        if set(ras) != set(SIDES):
            raise ValueError("needs one 'R' and one 'L' control point")
        return {"key": _CASE_PREFIX.sub("", image_key(path.name)), "ras": ras, "blueprint": None}
    raise ValueError("neither a blueprint (.ostium.json) nor a 3D Slicer markup (.mrk.json)")


def find_prediction_files(source: Path) -> tuple:
    """(position files, {key: {class index: probability NIfTI path}}) of one model."""
    if source.is_file():
        if source.name.lower().endswith(NIFTI_EXT):
            parsed = parse_map_name(source)
            if parsed is None:
                raise SystemExit(f"{source}: not a probability NIfTI written by infer (..._probability_class00?...)")
            return [], {parsed[0]: {parsed[1]: source}}
        return [source], {}
    if not source.is_dir():
        raise SystemExit(f"prediction source not found: {source}")
    files = []
    for pattern in ("markers/*.ostium.json", "*.ostium.json", "markups-3d-slicer/*.mrk.json", "*.mrk.json"):
        files = sorted(source.glob(pattern))
        if files:
            break
    maps = {}
    for pattern in ("probability-nifti/*_probability_class*", "*_probability_class*"):
        for f in sorted(source.glob(pattern)):
            parsed = f.name.lower().endswith(NIFTI_EXT) and parse_map_name(f)
            if parsed:
                maps.setdefault(parsed[0], {}).setdefault(parsed[1], f)
        if maps:
            break
    if not files and not maps:
        raise SystemExit(f"no *.ostium.json / *.mrk.json / probability NIfTI found in {source}")
    return files, maps


def source_name(source: Path, taken: set) -> str:
    """A readable, unique label for a model: the run folder, or the folder above `markers/`."""
    base = source.parent if source.is_file() else source
    name = base.name
    if name in ("markers", "markups-3d-slicer", "probability-nifti"):
        name = base.parent.name
    name = name or str(source)
    if name in taken:
        name = f"{name}#{len(taken)}"
    taken.add(name)
    return name


def expand_list(items) -> list:
    """Paths given on the command line, or one .txt file with one path per line (as infer/__main__.py)."""
    if items is None:
        return []
    if len(items) == 1 and items[0].lower().endswith(".txt"):
        with open(items[0], encoding="utf-8") as f:
            return [os.path.abspath(os.path.expanduser(l.strip())) for l in f if l.strip() and not l.startswith("#")]
    return [os.path.abspath(os.path.expanduser(i)) for i in items]


# ---------------------------------------------------------------------------------------------------------------
# Median of positions, and of probability maps
# ---------------------------------------------------------------------------------------------------------------

def geometric_median(points: np.ndarray, iterations: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Weiszfeld's algorithm; starts from the coordinate-wise median. points: (N, 3)."""
    y = np.median(points, axis=0)
    for _ in range(iterations):
        d = np.linalg.norm(points - y, axis=1)
        if np.any(d < 1e-12):  # y coincides with a point: that point is the answer to working precision
            return points[np.argmin(d)].copy()
        w = 1.0 / d
        y_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


def combine(points: np.ndarray, method: str) -> np.ndarray:
    return np.median(points, axis=0) if method == "median" else geometric_median(points)


def locate_on_map(volume: np.ndarray, affine: np.ndarray, window_size: int) -> np.ndarray:
    """
    Same as infer.utils.get_location_from_probability_one_channel('centroid') followed by the IJK -> RAS step of
    infer.__main__.locate_ostia: global peak, then centroid (weights = values minus their minimum) in a cubic window
    of `window_size` voxels around it; sub-voxel index, then RAS (mm) through the NIfTI affine.
    """
    peak = np.unravel_index(int(np.argmax(volume)), volume.shape)
    half = window_size // 2
    lo = [max(0, p - half) for p in peak]
    hi = [min(n, p + half + 1) for p, n in zip(peak, volume.shape)]
    window = volume[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(np.float64)
    window = window - window.min()
    weights = window.sum() + 1e-6
    grids = np.meshgrid(*[np.arange(b - a, dtype=np.float64) for a, b in zip(lo, hi)], indexing="ij")
    ijk = np.array([a + (window * g).sum() / weights for a, g in zip(lo, grids)])
    return (affine @ np.append(ijk, 1.0))[:3]


class MapLoader:
    """Loads probability NIfTI files with nibabel (keeps the affine), one class at a time to limit memory."""

    def __init__(self):
        import nibabel  # shipped with MONAI's environment
        self.nib = nibabel

    def load(self, path: Path):
        img = self.nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.float32).squeeze(), np.asarray(img.affine, dtype=np.float64)

    def save(self, volume: np.ndarray, affine: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.nib.save(self.nib.Nifti1Image(volume.astype(np.float32), affine), str(path))


def ensemble_from_maps(loader: MapLoader, map_paths: dict, models: list, map_combine: str, method: str, window_size: int,
                       out_maps: dict = None):
    """
    map_paths: {model: {class index: path}}. Returns (ras {side: array(3)}, ras_of_model {model: {side: array(3)}}) or
    raises ValueError if the maps are not on the same grid. If `out_maps` is a dict, the merged volumes
    {class index: (volume, affine)} are stored in it.
    """
    ras, ras_of_model = {}, {m: {} for m in models}
    affine_ref = None
    for side in SIDES:
        c = CLASS_OF_SIDE[side]
        volumes = []
        for m in models:
            vol, aff = loader.load(map_paths[m][c])
            if affine_ref is None:
                affine_ref = aff
            if vol.shape != (volumes[0].shape if volumes else vol.shape) or not np.allclose(aff, affine_ref, atol=1e-3):
                raise ValueError(f"the probability maps of the models are not on the same grid ({map_paths[m][c].name})")
            volumes.append(vol)
            ras_of_model[m][side] = locate_on_map(vol, aff, window_size)
        stack = np.stack(volumes)
        merged = np.median(stack, axis=0) if map_combine == "median" else stack.mean(axis=0)
        del stack, volumes
        ras[side] = locate_on_map(merged, affine_ref, window_size)
        if out_maps is not None:
            out_maps[c] = (merged, affine_ref)
    return ras, ras_of_model


# ---------------------------------------------------------------------------------------------------------------
# Statistics (same definitions as test/split_metrics_by_dataset.py)
# ---------------------------------------------------------------------------------------------------------------

def stats(values) -> dict:
    a = np.array([v for v in values if v is not None], dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0, **{k: None for k in STAT_KEYS}}
    out = {"min": a.min(), "p05": np.percentile(a, 5), "p25": np.percentile(a, 25), "median": np.median(a),
           "p75": np.percentile(a, 75), "p95": np.percentile(a, 95), "max": a.max(), "mean": a.mean(), "std": a.std()}
    return {"n": int(a.size), **{k: round(float(v), 6) for k, v in out.items()}}


def evaluate(distances_rl: list) -> dict:
    """distances_rl: list of (d_R, d_L) or None, one per image -> summary like the 'Overall Metrics' of infer."""
    valid = [d for d in distances_rl if d is not None]
    per_class = {s: stats([d[i] for d in valid]) for i, s in enumerate(SIDES)}
    both = stats([float(np.mean(d)) for d in valid])
    triples = [classify_pair(*d) for d in valid]
    return {"distance": both, "distance_per_class": per_class,
            "buckets": bucket_summary([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])}


def fmt(x, w=6):
    return f"{x:{w}.2f}" if x is not None else " " * (w - 1) + "-"


def print_comparison(rows: dict, n_single: int):
    """rows: {name: summary from evaluate()}; the first n_single rows are the single models, the others ensembles."""
    width = max(len(n) for n in rows) + 2
    print("\n" + "=" * (width + 92))
    print(f"{'':{width}s}|{'Right ostium':^21s}|{'Left ostium':^21s}|{'Both':^21s}| patient % (opt/good/acc/fail)")
    print(f"{'model':{width}s}| {'med':>6s} {'mean':>6s} {'max':>6s} | {'med':>6s} {'mean':>6s} {'max':>6s} | {'med':>6s} {'mean':>6s} {'max':>6s} |")
    print("-" * (width + 92))
    for i, (name, s) in enumerate(rows.items()):
        if i == n_single:
            print("-" * (width + 92))
        dc, d, b = s["distance_per_class"], s["distance"], s["buckets"]["patient"]
        pcts = " / ".join(f"{b['pct'][k]:.1f}" for k in BUCKETS) if b["n"] else "-"
        cells = " | ".join(f"{fmt(x['median'])} {fmt(x['mean'])} {fmt(x['max'])}" for x in (dc["R"], dc["L"], d))
        print(f"{name:{width}s}| {cells} | {pcts}")
    print("=" * (width + 92))


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------

def blueprint_from_template(template: dict, image_name: str, ras: dict, comment: str, other: dict) -> dict:
    """Copy of a blueprint with new positions (ijk recomputed with its own affines)."""
    pos = {}
    for s in SIDES:
        h = np.append(ras[s], 1.0)
        pos[s] = {"ras": ras[s].tolist(),
                  "ijk": {k: (np.array(template["affine_from_ras_to_ijk"][k]) @ h)[:3].tolist()
                          for k in ("sitk", "monai-ras")}}
    return {"version": template["version"], "image_id": template.get("image_id", image_name),
            "image_filename": template.get("image_filename", image_name), "pos": pos,
            "affine_from_ijk_to_ras": template["affine_from_ijk_to_ras"],
            "affine_from_ras_to_ijk": template["affine_from_ras_to_ijk"], "comment": comment, "other": other}


def main():
    p = argparse.ArgumentParser(description="Median ensemble of the ostium predictions of N models.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("-p", "--predictions", nargs="+", required=True,
                   help="N >= 2 model predictions: folders (inference output) and/or single files, one argument per model.")
    p.add_argument("-i", "--images", nargs="+", default=None, help="Images or a .txt list of images (optional; required with -l).")
    p.add_argument("-l", "--label", nargs="+", default=None, help="Ground truth, one per image in the same order, or a .txt list (optional).")
    p.add_argument("-o", "--output", default=os.path.join(os.getcwd(), "ensemble_predictions"), help="Output folder.")
    p.add_argument("--names", nargs="+", default=None, help="Names of the N models in the tables (default: run folder names).")
    p.add_argument("--from", dest="source_kind", choices=["auto", "maps", "coordinates"], default="auto",
                   help="Merge the probability NIfTI maps or the final coordinates (default auto: maps where every model has them).")
    p.add_argument("--method", choices=["median", "geometric-median"], default="median",
                   help="How to merge coordinates. Default: coordinate-wise median.")
    p.add_argument("--map-combine", choices=["median", "mean"], default="median", help="How to merge probability maps voxel by voxel. Default: median.")
    p.add_argument("--window-size", type=int, default=11, help="Voxels of the cubic window of the centroid on a map (default 11, as infer).")
    p.add_argument("-a", "--artifacts", nargs="+", choices=["none", "marker", "markup-3d-slicer", "probability-nifti", "all"], default=["marker"],
                   help="What to save (default marker: the ensemble .ostium.json). probability-nifti saves the merged maps (heavy; only for images merged from maps). 'none': metrics only.")
    p.add_argument("--min-models", type=int, default=None, help="Ensemble an image if at least this many models predicted it (default: all N).")
    args = p.parse_args()
    if ("none" in args.artifacts or "all" in args.artifacts) and len(args.artifacts) != 1:
        p.error("if 'none' or 'all' is specified in -a/--artifacts, nothing else must be included.")
    want = lambda name: "all" in args.artifacts or name in args.artifacts
    want_marker, want_slicer, want_maps = want("marker"), want("markup-3d-slicer"), want("probability-nifti")

    if len(args.predictions) < 2:
        raise SystemExit("an ensemble needs at least 2 prediction sources (-p)")
    sources = [Path(s).expanduser() for s in args.predictions]
    n_models = len(sources)
    min_models = n_models if args.min_models is None else args.min_models
    if not 1 <= min_models <= n_models:
        raise SystemExit(f"--min-models must be between 1 and {n_models}")

    taken = set()
    names = args.names or [source_name(s, taken) for s in sources]
    if len(names) != n_models:
        raise SystemExit(f"--names has {len(names)} entries for {n_models} prediction sources")

    # ---- read predictions: predictions[key][model] = {'ras', 'blueprint'} (positions), maps[key][model] = {class: path}
    predictions, maps, record_paths = {}, {}, {}
    for m, src in enumerate(sources):
        files, src_maps = find_prediction_files(src)
        for f in files:
            try:
                rec = read_prediction(f)
            except Exception as e:
                print(f"[!] skipped {f}: {e}")
                continue
            slot = predictions.setdefault(rec["key"], {})
            if m in slot:
                print(f"[!] model '{names[m]}' has more than one prediction for '{rec['key']}', keeping the first ({record_paths[(rec['key'], m)]})")
                continue
            slot[m] = rec
            record_paths[(rec["key"], m)] = f
        for key, per_class in src_maps.items():
            if set(per_class) >= set(CLASS_OF_SIDE.values()):
                maps.setdefault(key, {})[m] = per_class
            else:
                print(f"[!] model '{names[m]}', '{key}': probability maps found for classes {sorted(per_class)}, both 1 and 2 are needed, ignored")
    for m, name in enumerate(names):
        print(f"[*] model {m + 1}/{n_models} '{name}': {sum(1 for v in predictions.values() if m in v)} positions, "
              f"{sum(1 for v in maps.values() if m in v)} probability-map pairs")

    # ---- images and labels
    images = expand_list(args.images)
    labels = expand_list(args.label)
    if labels and not images:
        raise SystemExit("-l requires -i (labels are matched to predictions through the image names)")
    if labels and len(labels) != len(images):
        raise SystemExit(f"{len(images)} images but {len(labels)} labels: they must be paired in the same order")
    image_of = {image_key(i): i for i in images}
    label_of = {image_key(i): l for i, l in zip(images, labels)}
    known = set(predictions) | set(maps)
    if images:
        extra = sorted(known - set(image_of))
        if extra:
            print(f"[!] {len(extra)} predicted image(s) not in -i, ignored: {extra[:5]}{' ...' if len(extra) > 5 else ''}")
        keys = [image_key(i) for i in images]
    else:
        keys = sorted(known)

    out_dir = Path(args.output)
    markers_dir, slicer_dir = out_dir / "markers", out_dir / "markups-3d-slicer"
    out_dir.mkdir(parents=True, exist_ok=True)
    if want_marker:
        markers_dir.mkdir(parents=True, exist_ok=True)
    if want_slicer:
        slicer_dir.mkdir(parents=True, exist_ok=True)
    if not (want_marker or want_slicer or want_maps) and not labels:
        print("[!] artifacts 'none' and no labels: nothing will be saved, only the comparison of the models is printed")
    # imported here (not at the top) so that --help works without torch/MONAI installed
    from hcatostialocalization.io.io import compute_image_metadata, extract_simplified_ras, save_slicer_json
    loader = MapLoader() if maps and args.source_kind != "coordinates" else None

    label_tag = {"maps": f"probability maps ({args.map_combine})", "coordinates": f"coordinates ({args.method})"}
    start = datetime.datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
    per_image, dist_single, dist_ens, dist_coord = [], [[] for _ in range(n_models)], [], []
    n_saved, n_maps, skipped = 0, 0, []

    def model_ras(recs, ras_from_map, m):
        """Position of one model: the official one of its .ostium.json if present, else the one located on its map."""
        if m in recs:
            return recs[m]["ras"]
        return (ras_from_map or {}).get(m)

    for idx, key in enumerate(keys):
        recs = predictions.get(key, {})
        key_maps = maps.get(key, {})
        use_maps = False
        ras, ras_from_map, merged_maps = None, None, {}
        # models usable in each way
        if args.source_kind != "coordinates" and len(key_maps) >= min_models:
            used = sorted(key_maps)
            try:
                ras, ras_from_map = ensemble_from_maps(loader, key_maps, used, args.map_combine, args.method, args.window_size,
                                                       merged_maps if want_maps else None)
                use_maps = True
            except ValueError as e:
                print(f"[!] {key}: {e}" + ("; using coordinates" if args.source_kind == "auto" else ""))
                if args.source_kind == "maps":
                    skipped.append((key, 0))
                    continue
        elif args.source_kind == "maps":
            skipped.append((key, len(key_maps)))
            continue
        if not use_maps:
            if len(recs) < min_models:
                skipped.append((key, len(recs)))
                continue
            used = sorted(recs)
            ras = {s: combine(np.stack([recs[m]["ras"][s] for m in used]), args.method) for s in SIDES}
            if args.source_kind == "auto" and key_maps:
                print(f"[!] {key}: maps of only {len(key_maps)}/{n_models} models, using coordinates")
        used_positions = {m: model_ras(recs, ras_from_map, m) for m in used}
        used_positions = {m: v for m, v in used_positions.items() if v is not None}
        spread = {s: round(float(max(np.linalg.norm(used_positions[m][s] - ras[s]) for m in used_positions)), 6)
                  for s in SIDES} if used_positions else None
        other = {"ensemble": {"from": "probability maps" if use_maps else "coordinates",
                              "combine": args.map_combine if use_maps else args.method,
                              "n_models": len(used), "models": [names[m] for m in used],
                              "max_distance_of_a_model_from_ensemble_mm": spread}}
        # coordinate-median of the same models, for comparison with the map-based one
        coord_ras = None
        if use_maps and len(used_positions) == len(used):
            coord_ras = {s: combine(np.stack([used_positions[m][s] for m in used]), args.method) for s in SIDES}
            other["ensemble"]["distance_from_coordinate_ensemble_mm"] = {s: round(float(np.linalg.norm(ras[s] - coord_ras[s])), 6) for s in SIDES}
        comment = f"Ensemble of {len(used)} models ({', '.join(names[m] for m in used)}) from " + \
                  (f"probability maps ({args.map_combine})" if use_maps else f"coordinates ({args.method})")
        unique_name = f"case_{idx:06d}_{key}"

        # blueprint: affines of the first source that has a blueprint, else computed from the image
        template = next((recs[m]["blueprint"] for m in sorted(recs) if recs[m]["blueprint"]), None)
        blueprint, full_pos = None, None
        if not want_marker:
            pass  # the blueprint is only needed for the marker artifact (the Slicer markup needs the RAS positions only)
        elif template is not None:
            blueprint = blueprint_from_template(template, key, ras, comment, other)
            full_pos = blueprint["pos"]
        elif key in image_of:
            try:
                full_pos, a_ijk_ras, a_ras_ijk = compute_image_metadata(image_of[key], {s: {"ras": ras[s].tolist()} for s in SIDES})
                blueprint = {"version": "1.0", "image_id": os.path.basename(image_of[key]), "image_filename": os.path.basename(image_of[key]),
                             "pos": full_pos, "affine_from_ijk_to_ras": a_ijk_ras, "affine_from_ras_to_ijk": a_ras_ijk,
                             "comment": comment, "other": other}
            except Exception as e:
                print(f"[!] {key}: could not compute the affines from the image ({e}); only the Slicer markup is saved")
        if blueprint is not None:
            with open(markers_dir / f"{unique_name}.ostium.json", "w") as f:
                json.dump(blueprint, f, indent=4)
        elif want_marker and (idx == 0 or n_saved == 0):
            print(f"[!] no blueprint (.ostium.json) written: no model has one and no image is given with -i; only 3D Slicer markups are saved")
        if want_slicer:
            save_slicer_json(str(slicer_dir / f"{unique_name}.mrk.json"), full_pos or {s: {"ras": ras[s].tolist()} for s in SIDES})
        if want_maps and not use_maps and idx == 0:
            print("[!] probability-nifti requested, but maps are only saved for images merged from maps")
        for c, (vol, aff) in merged_maps.items():
            cname = "right_ostium" if c == CLASS_OF_SIDE["R"] else "left_ostium"
            loader.save(vol, aff, out_dir / "probability-nifti" / f"{unique_name}_probability_class{c:03d}_{cname}.nii.gz")
        n_saved += 1
        n_maps += int(use_maps)

        # ---- metrics (only with labels)
        if labels:
            gt = extract_simplified_ras(label_of[key], image_of[key]) if key in label_of else None
            if gt and all(len(gt[s]["ras"]) == 3 for s in SIDES):
                g = {s: np.array(gt[s]["ras"], dtype=float) for s in SIDES}
                dist = lambda pos: tuple(float(np.linalg.norm(pos[s] - g[s])) for s in SIDES)
                d_ens = dist(ras)
                for m in range(n_models):
                    pos = model_ras(recs, ras_from_map, m)
                    dist_single[m].append(dist(pos) if pos is not None else None)
                dist_coord.append(dist(coord_ras) if coord_ras is not None else d_ens if not use_maps else None)
            else:
                print(f"[!] {key}: label missing or unreadable, no metrics for this image")
                d_ens = None
                for m in range(n_models):
                    dist_single[m].append(None)
                dist_coord.append(None)
            dist_ens.append(d_ens)
            b_r, b_l, b_p = classify_pair(*(d_ens if d_ens else (None, None)))
            per_image.append({"image": image_of.get(key, key), "label": label_of.get(key),
                              "distance": round(float(np.mean(d_ens)), 6) if d_ens else None,
                              "distance_per_class": [round(x, 6) for x in d_ens] if d_ens else None,
                              "peak_confidence": None, "peak_confidence_per_class": None,
                              "outlier": int("failed" in (b_r, b_l)) if d_ens else None,
                              "outlier_per_class": [int(b_r == "failed"), int(b_l == "failed")] if d_ens else None,
                              "bucket": b_p, "bucket_per_class": [b_r, b_l] if d_ens else None})

    print(f"\n[*] {n_saved} ensemble prediction(s) computed, artifacts ({', '.join(args.artifacts)}) in {out_dir} ({n_maps} from probability maps, {n_saved - n_maps} from coordinates)")
    if skipped:
        print(f"[!] {len(skipped)} image(s) skipped, predicted by fewer than {min_models} models (key: n predictions): {skipped[:8]}{' ...' if len(skipped) > 8 else ''}")

    if labels and per_image:
        summaries = {names[m]: evaluate(dist_single[m]) for m in range(n_models)}
        ensemble_summary = evaluate(dist_ens)
        summaries["ENSEMBLE" + (" (maps)" if n_maps else f" ({args.method})")] = ensemble_summary
        if n_maps and any(d is not None for d in dist_coord):
            summaries[f"ensemble of coordinates ({args.method})"] = evaluate(dist_coord)
        print_comparison(summaries, n_models)
        overall = {"cli": f"{sys.executable} {shlex.join(sys.argv)}", "date and time": start,
                   "model nickname": "Ensemble", "ensemble of": {n: str(s) for n, s in zip(names, sources)},
                   "images processed": len(per_image), "images from probability maps": n_maps, "total images": len(keys),
                   **ensemble_summary, "single models": {n: summaries[n] for n in names}}
        with open(out_dir / "metrics__Ensemble.json", "w") as f:
            json.dump(["Overall Metrics", overall, "Per Image Metrics", *per_image, "Errors", {"errors": []}], f, indent=4)
        print(f"[*] metrics written to {out_dir / 'metrics__Ensemble.json'} "
              f"(per-dataset split: python -m hcatostialocalization.test.split_metrics_by_dataset --metrics <that file>)")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
