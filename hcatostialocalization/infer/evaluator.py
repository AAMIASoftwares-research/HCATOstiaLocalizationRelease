"""
Evaluation of predicted coronary ostia against reference ostia: setup and accumulation of the metrics, final
reporting, printing and saving (also while the run is in progress).

This module holds everything that does not depend on HOW the predictions were obtained, so that it can be used by:
  - infer/evaluate_predictions.py  (predicted ostia already on disk vs. label ostia), and, in the future,
  - infer/__main__.py              (which today keeps the same code inline: accumulator lists, per-image metrics,
                                    metrics file, final log). Replacing it with `OstiaEvaluator` should not change a
                                    single output file: the layouts below are the ones written by __main__.py.

Conventions (same as __main__.py and test/split_metrics_by_dataset.py)
  - Positions are RAS world coordinates in mm: {"R": [x, y, z], "L": [x, y, z]} (or an array of shape (2, 3), rows
    [right, left]).
  - Every "per class" list is [right ostium, left ostium].
  - Distances are Euclidean, in mm. Accuracy buckets and their thresholds are in infer/thresholds.py.

Files written in the output directory
  - `_test_log <start date and time>.txt`   one JSON line per processed image, appended on the go
  - `metrics__<run name>.json`               rewritten after every image (the previous version is kept as a hidden backup
                                             until the new one is on disk); layout:
                                             ["Overall Metrics", {...}, "Per Image Metrics", {row}, ..., "Errors", {"errors": [...]}]
                                             which is what test/split_metrics_by_dataset.py reads.

Minimal use
    ev = OstiaEvaluator(output_dir, run_name="MyModel", total_images=len(pairs))
    for image, label_path, prediction_path in pairs:
        try:
            ev.add_sample(image, label_path, prediction_ras=load_ostia_ras(prediction_path),
                          label_ras=load_ostia_ras(label_path))
        except Exception as e:
            ev.add_error(image, label_path, str(e))
    ev.finalize()
"""
import json
import os
import shutil
import sys
import shlex
import time
import warnings
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy

from hcatostialocalization.infer.thresholds import (
    BUCKETS, THRESHOLDS_MM, bucket_summary, classify_pair,
)

try:  # colours are cosmetic
    from termcolor import colored
except Exception:  # pragma: no cover
    def colored(text, *args, **kwargs):
        return text

SIDES = ("R", "L")  # order of every per-class list
GREEN = (167, 201, 87)
RED = (200, 20, 50)
ALL_METRICS = ("distance", "peak-confidence", "outliers")


# ---------------------------------------------------------------------------------------------------------------
# Small helpers (same behavior as the private helpers of __main__.py)
# ---------------------------------------------------------------------------------------------------------------

def collect_input_paths(items: Iterable, are_labels: bool = False) -> List[str]:
    """
    Expand command-line arguments into a list of absolute paths, keeping the order.

    - a .txt/.list/.lst file is read, one path per line; empty lines and lines starting with '#' are skipped, except
      for labels and predictions (are_labels=True), where an empty line means "nothing for this image" and is kept as
      '' so that the alignment with the other list is preserved;
    - a directory is replaced by its `*.ostium.json` files (else its `*.mrk.json`), sorted by name;
    - any other item is a path.
    Not deduplicated (would break the alignment between lists).
    """
    collected = []
    for it in items:
        if it is None:
            continue
        it = os.path.expanduser(str(it))
        if os.path.isfile(it) and it.lower().endswith(('.txt', '.list', '.lst')):
            try:
                with open(it, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if line.startswith('#'):
                            continue
                        if not line:
                            if are_labels:
                                collected.append('')
                            continue
                        collected.append(os.path.abspath(os.path.expanduser(line)))
            except Exception:
                raise RuntimeError(f"Could not read input list file: {it}")
        elif os.path.isdir(it):
            files = sorted(f for f in os.listdir(it) if f.lower().endswith('.ostium.json'))
            if not files:
                files = sorted(f for f in os.listdir(it) if f.lower().endswith('.mrk.json'))
            if not files:
                raise RuntimeError(f"No *.ostium.json / *.mrk.json found in directory {it}")
            collected.extend(os.path.join(os.path.abspath(it), f) for f in files)
        elif len(it) == 0 and are_labels:
            collected.append('')
        else:
            collected.append(os.path.abspath(it))
    return collected


def load_ostia_ras(path: str, image_path: Optional[str] = None) -> Optional[numpy.ndarray]:
    """
    Ostia positions of a file, in any format supported by io.extract_simplified_ras (standard .ostium.json, old ostia
    JSON, ASOCA, 3D Slicer .mrk.json, ...), as a float array (2, 3) with rows [right, left] (RAS, mm).
    Returns None if `path` is empty or does not exist. Raises ValueError if the file has no usable R and L point.
    image_path is only needed for the raw ASOCA formats (their points can be converted with the image header only).
    """
    if not path or not os.path.exists(path):
        return None
    from hcatostialocalization.io.io import extract_simplified_ras  # heavy imports (torch/MONAI) only when needed
    simple_pos = extract_simplified_ras(path, image_path=image_path)
    if simple_pos is None or any(len(simple_pos[side]['ras']) != 3 for side in SIDES):
        raise ValueError(f"could not read both R and L ostium RAS coordinates from {path}")
    return numpy.array([[float(c) for c in simple_pos[side]['ras']] for side in SIDES], dtype=float)


def _as_positions(pos) -> Optional[numpy.ndarray]:
    """{"R": [...], "L": [...]} / {"R": {"ras": [...]}, ...} / array (2, 3) -> float array (2, 3) [R, L]; None -> None."""
    if pos is None:
        return None
    if isinstance(pos, dict):
        rows = [pos[s]["ras"] if isinstance(pos[s], dict) else pos[s] for s in SIDES]
        pos = rows
    arr = numpy.array(pos, dtype=float)
    if arr.shape != (2, 3):
        raise ValueError(f"positions must have shape (2, 3) [right, left] x (r, a, s), got {arr.shape}")
    return arr


def process_none_lists(_list: list) -> numpy.ndarray:
    """
    Per-class metrics list (one row per image) -> float array (N, C), None -> nan. Rows that are None or shorter than the
    longest row are padded with nan, so that a ragged list (e.g. an image that failed) never breaks the conversion.
    """
    rows = [list(r) if isinstance(r, (list, tuple, numpy.ndarray)) else [r] for r in _list]
    if len(rows) == 0:
        return numpy.array([], dtype=float)
    width = max(len(r) for r in rows)
    rows = [[numpy.nan if v is None else float(v) for v in r] + [numpy.nan] * (width - len(r)) for r in rows]
    return numpy.array(rows, dtype=float)


def round_dict_if_float(dict_: dict, significant_digits: int = 6) -> dict:
    if not isinstance(dict_, dict):
        raise TypeError("Input must be a dict.")
    return {
        k: (round(v, significant_digits) if isinstance(v, float)
            else [round(_x, significant_digits) if isinstance(_x, float) else _x for _x in v] if isinstance(v, list)
            else v)
        for k, v in dict_.items()
    }


def get_metrics_statistics(arr) -> dict:
    """min, p05, p25, median, p75, p95, max, mean, std of a 1D array or of each column of a 2D one (nan ignored; all-nan -> None)."""
    arr = numpy.array(arr)
    if arr.ndim not in (1, 2):
        raise ValueError("Input array must be 1D or 2D.")

    def safe_stat(fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = fn(arr, axis=0)
            if isinstance(result, numpy.ndarray):
                result = result.tolist()
            else:
                result = float(result)
            if (isinstance(result, float) and numpy.isnan(result)) or \
               (isinstance(result, list) and all(numpy.isnan(x) for x in result)):
                return None
            return result
        except Exception:
            return None

    return {
        "min": safe_stat(numpy.nanmin),
        "p05": safe_stat(lambda a, axis: numpy.nanpercentile(a, 5, axis=axis)),
        "p25": safe_stat(lambda a, axis: numpy.nanpercentile(a, 25, axis=axis)),
        "median": safe_stat(numpy.nanmedian),
        "p75": safe_stat(lambda a, axis: numpy.nanpercentile(a, 75, axis=axis)),
        "p95": safe_stat(lambda a, axis: numpy.nanpercentile(a, 95, axis=axis)),
        "max": safe_stat(numpy.nanmax),
        "mean": safe_stat(numpy.nanmean),
        "std": safe_stat(numpy.nanstd),
    }


# ---------------------------------------------------------------------------------------------------------------
# Metrics of ONE image
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class SampleMetrics:
    """Metrics of one image. Every field is None when it was not computed (no label, metric not requested, error)."""
    distance: Optional[float] = None                        # mean of the two distances
    distance_per_class: Optional[list] = None               # [right, left]
    peak_confidence: Optional[float] = None                 # mean of the two
    peak_confidence_per_class: Optional[list] = None        # [right, left]
    outlier: Optional[int] = None                           # 1 if any ostium failed (kept for backward compatibility)
    outlier_per_class: Optional[list] = None                # [right, left] of 0/1
    bucket: Optional[str] = None                            # patient bucket: the worse of the two
    bucket_per_class: Optional[list] = None                 # [right, left]

    def as_row_fields(self) -> dict:
        return {
            "distance": self.distance, "distance_per_class": self.distance_per_class,
            "peak_confidence": self.peak_confidence, "peak_confidence_per_class": self.peak_confidence_per_class,
            "outlier": self.outlier, "outlier_per_class": self.outlier_per_class,
            "bucket": self.bucket, "bucket_per_class": self.bucket_per_class,
        }


def compute_sample_metrics(prediction_ras, label_ras, metrics: Sequence[str] = ('all',),
                           peak_confidence_per_class: Optional[Sequence[float]] = None) -> SampleMetrics:
    """
    Metrics of one image from the predicted and the reference positions (RAS, mm; dict or array, see _as_positions).
    `metrics` is a subset of {'all', 'none', 'distance', 'peak-confidence', 'outliers'} like infer's --metrics.
    Without a label (label_ras None) or with 'none' nothing is computed. The peak confidence (probability at the
    predicted position) cannot be derived from positions: it must be passed, [right, left], if wanted.
    """
    m = SampleMetrics()
    prediction_ras, label_ras = _as_positions(prediction_ras), _as_positions(label_ras)
    if label_ras is None or prediction_ras is None or 'none' in metrics:
        return m
    want = lambda name: 'all' in metrics or name in metrics
    # rows are [right ostium, left ostium]
    distances = numpy.linalg.norm(prediction_ras - label_ras, axis=1)
    d_r, d_l = float(distances[0]), float(distances[1])
    if want('distance'):
        m.distance = float(distances.mean())
        m.distance_per_class = [d_r, d_l]
    if want('peak-confidence') and peak_confidence_per_class is not None:
        conf_r, conf_l = (float(c) for c in peak_confidence_per_class)
        m.peak_confidence = 0.5 * (conf_r + conf_l)
        m.peak_confidence_per_class = [conf_r, conf_l]
    if want('outliers'):
        bucket_r, bucket_l, bucket_patient = classify_pair(d_r, d_l)
        # "outlier" is kept for backward compatibility: it means "failed" (above the failure threshold)
        failed_r, failed_l = bucket_r == "failed", bucket_l == "failed"
        m.outlier = int(failed_r or failed_l)
        m.outlier_per_class = [int(failed_r), int(failed_l)]
        m.bucket = bucket_patient
        m.bucket_per_class = [bucket_r, bucket_l]
    return m


# ---------------------------------------------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------------------------------------------

class OstiaEvaluator:
    """
    Accumulates the metrics of a run image by image and handles logging, reporting and saving.

    Parameters
      output_dir          where the log and the metrics file are written (created if missing).
      run_name            tag used in the metrics file name (`metrics__<run_name>.json`), like `models_tag` in __main__.py.
      total_images        expected number of images (for "total images" and the progress bar).
      metrics             which metrics to compute: subset of {'all', 'none', 'distance', 'peak-confidence', 'outliers'}.
      overview            extra key/values for the 'Overall Metrics' block (e.g. inference spacing, weights path, ...). They
                          go before the computed statistics, like in __main__.py. May be updated later: `self.overview`.
      cli                 the command line to store; default: the current one.
      start_datetime      default: now, formatted as in __main__.py.
      save_every_sample   rewrite the metrics file after every image (default). If False, only at `finalize()`.
    """

    def __init__(self, output_dir: str, run_name: str, total_images: Optional[int] = None,
                 metrics: Sequence[str] = ('all',), overview: Optional[dict] = None, cli: Optional[str] = None,
                 start_datetime: Optional[str] = None, save_every_sample: bool = True):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.run_name = run_name
        self.total_images = total_images
        self.metrics = tuple(metrics)
        self.overview = dict(overview or {})
        self.cli = cli if cli is not None else f"{sys.executable} {shlex.join(sys.argv)}"
        self.start_datetime = start_datetime or time.strftime("%Y.%m.%d - %H:%M:%S")
        self.save_every_sample = save_every_sample
        self.t0 = time.time()

        self.logfile = os.path.join(self.output_dir, f"_test_log {self.start_datetime}.txt")
        self.metrics_file = os.path.join(self.output_dir, f"metrics__{run_name}.json")

        # accumulators: one entry per image (None when not available), same names as in __main__.py
        self.rows: List[dict] = []                      # 'Per Image Metrics'
        self.errors: List[tuple] = []                   # (image, label, message)
        self.distance_list, self.distance_per_class_list = [], []
        self.peak_confidence_list, self.peak_confidence_per_class_list = [], []
        self.outlier_list, self.outlier_per_class_list = [], []
        self.bucket_list, self.bucket_per_class_list = [], []
        self.metrics_dict: dict = {}                    # statistics of the last update, see _compute_statistics

    # ----------------------------------------------------------------- adding images

    def add_sample(self, image: str, label: str, prediction_ras, label_ras,
                   peak_confidence_per_class: Optional[Sequence[float]] = None) -> SampleMetrics:
        """Metrics of one image; label_ras None means "no label": the image is counted but has no metrics."""
        return self._add(image, label, compute_sample_metrics(prediction_ras, label_ras, self.metrics, peak_confidence_per_class))

    def add_error(self, image: str, label: str, message: str) -> None:
        """An image that could not be processed: counted, without metrics, reported in the 'Errors' block."""
        error = (image, label, message)
        self.errors.append(error)
        with open(self.logfile, 'a') as f:
            f.write(f"{error}\n")
        self._add(image, label, SampleMetrics())

    def _add(self, image: str, label: str, m: SampleMetrics) -> SampleMetrics:
        self.distance_list.append(m.distance)
        self.distance_per_class_list.append(m.distance_per_class if m.distance_per_class is not None else [None] * len(SIDES))
        self.peak_confidence_list.append(m.peak_confidence)
        self.peak_confidence_per_class_list.append(m.peak_confidence_per_class if m.peak_confidence_per_class is not None else [None] * len(SIDES))
        self.outlier_list.append(m.outlier)
        self.outlier_per_class_list.append(m.outlier_per_class if m.outlier_per_class is not None else [None] * len(SIDES))
        self.bucket_list.append(m.bucket)
        self.bucket_per_class_list.append(m.bucket_per_class if m.bucket_per_class is not None else [None] * len(SIDES))
        row = {"image": image, "label": label}
        row.update({k: (round(v, 6) if isinstance(v, float) else
                        [round(x, 6) if isinstance(x, float) else x for x in v] if isinstance(v, list) else v)
                    for k, v in m.as_row_fields().items()})
        self.rows.append(row)
        with open(self.logfile, 'a') as f:  # on the go, in case the program is interrupted
            f.write(json.dumps(row) + "\n")
        if self.save_every_sample:
            self.write_metrics_file()
        return m

    # ----------------------------------------------------------------- statistics

    def _compute_statistics(self) -> dict:
        """Statistics of everything accumulated so far (the `metrics_dict` of __main__.py)."""
        metrics_dict = {}
        raw_metrics_2d = {
            "distance": process_none_lists(self.distance_per_class_list),
            "peak_confidence": process_none_lists(self.peak_confidence_per_class_list),
            "outlier": process_none_lists(self.outlier_per_class_list),
        }
        for name, arr_2d in raw_metrics_2d.items():
            if arr_2d.size == 0:
                continue
            metrics_dict[f"{name}_per_class"] = round_dict_if_float(get_metrics_statistics(arr_2d))
            # overall: average across the two ostia first -> one score per image
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                arr_1d = numpy.nanmean(arr_2d, axis=1)
            metrics_dict[name] = round_dict_if_float(get_metrics_statistics(arr_1d))
        # accuracy buckets (optimal / good / acceptable / failed), per ostium and per patient (worse of the two)
        pairs = [p for p in self.bucket_per_class_list if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(pairs) > 0:
            metrics_dict["buckets"] = bucket_summary([p[0] for p in pairs], [p[1] for p in pairs], self.bucket_list)
        self.metrics_dict = metrics_dict
        return metrics_dict

    # ----------------------------------------------------------------- saving

    def write_metrics_file(self) -> Optional[str]:
        """(Re)writes `metrics__<run_name>.json`. Safe: the previous version is restored if the write fails."""
        metrics_dict = self._compute_statistics()
        to_write = [
            'Overall Metrics',
            {
                "cli": self.cli,
                "date and time": self.start_datetime,
                "model nickname": self.run_name,
                **self.overview,
                "total time (s)": round(time.time() - self.t0, 2),
                "images processed": len(self.distance_list),
                "total images": self.total_images if self.total_images is not None else len(self.distance_list),
                "Number of errors": len(self.errors),
                **metrics_dict,
            },
            'Per Image Metrics',
            *self.rows,
            'Errors',
            {"errors": self.errors},
        ]
        hidden_backup = os.path.join(self.output_dir, f".metrics__{self.run_name}.json.bak")
        try:
            if os.path.exists(self.metrics_file):
                shutil.copy2(self.metrics_file, hidden_backup)
            with open(self.metrics_file, 'w') as f:
                json.dump(to_write, f, indent=4)
            if os.path.exists(hidden_backup):
                os.remove(hidden_backup)
            return self.metrics_file
        except Exception as e:
            if os.path.exists(hidden_backup):
                shutil.move(hidden_backup, self.metrics_file)
            print(colored(f"\nWarning: Could not update intermediate metrics file. Error: {e}", RED))
            return None

    # ----------------------------------------------------------------- printing

    def print_progress(self, done: int, width: int = 40) -> None:
        total = self.total_images or done
        filled = int(width * done / total) if total else width
        print("[" + "#" * filled + "." * (width - filled) + f"] {100 * done / total if total else 100: 3.0f}% ({done}/{total})", end='\r')

    def print_report(self) -> None:
        """Final summary: where the files are, distance statistics, accuracy buckets, errors."""
        md = self._compute_statistics()
        print(' ' * 80)
        print(colored("Evaluation complete! Results saved in:", GREEN))
        print(self.output_dir)
        print(colored("Final Metrics:", GREEN))
        print(self.metrics_file)
        print()
        d_pc, d = md.get("distance_per_class"), md.get("distance")
        if d_pc and d and d.get("median") is not None:
            f = lambda x: f"{x:7.2f}" if x is not None else f"{'-':>7s}"
            print(colored("Euclidean distance (mm):", GREEN))
            print(f"{'':8s}" + "".join(f"{k:>8s}" for k in ("p25", "median", "mean", "p75", "max")))
            for name, stats_ in (("Right", {k: v[0] for k, v in d_pc.items() if isinstance(v, list)}),
                                 ("Left", {k: v[1] for k, v in d_pc.items() if isinstance(v, list)}),
                                 ("Both", d)):
                print(f"{name:8s}" + "".join(f(stats_.get(k)) + " " for k in ("p25", "median", "mean", "p75", "max")))
            print()
        buckets = md.get("buckets")
        if buckets is not None:
            print(colored("Accuracy buckets (count, % of images with a label):", GREEN))
            print("thresholds (mm)  " + "  ".join(f"{name}: R {vals['R']:.2f} / L {vals['L']:.2f}" for name, vals in THRESHOLDS_MM.items()))
            print(f"{'bucket':12s} " + " ".join(f"{col:>16s}" for col in ("Right", "Left", "Patient")))
            for b in BUCKETS:
                cells = []
                for col in ("R", "L", "patient"):
                    c, p = buckets[col]["counts"][b], buckets[col]["pct"][b]
                    cells.append(f"{c:5d} ({p:6.2f}%)" if p is not None else f"{'-':>16s}")
                print(f"{b:12s} " + " ".join(f"{cell:>16s}" for cell in cells))
            print()
        if len(self.errors) > 0:
            print(colored("ERRORS", RED))
            print(f"Some errors ({len(self.errors)}) were encountered during the execution of the program.")
            print("Please check the metrics file to see which images presented some errors and try again.")

    def finalize(self) -> Optional[str]:
        """Writes the metrics file one last time and prints the final report. Returns the metrics file path."""
        path = self.write_metrics_file()
        self.print_report()
        return path
