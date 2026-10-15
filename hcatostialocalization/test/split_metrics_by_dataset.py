"""
Per-dataset statistics from ONE inference run over all the test images.

Usage (numpy only, no torch/monai needed), as a module, from the folder that contains `hcatostialocalization/`:

    python -m hcatostialocalization.test.split_metrics_by_dataset \
        --metrics <output_dir>/metrics__<model_nickname>.json \
        [--lists-dir hcatostialocalization/test/collections_test_set] \
        [-o <output_dir>/metrics_by_dataset.json]

Every `<NAME>_images.txt` in --lists-dir (except all_images.txt) defines a dataset. Each row of the
'Per Image Metrics' section of the metrics file is assigned to the dataset whose list contains its
image path (paths are compared as absolute paths, like infer/__main__.py stores them).

Per-class values are [right ostium, left ostium], in this order (as built in infer/__main__.py).
Statistics use the same definitions as infer/__main__.py: min, p05, p25, median, p75, p95, max, mean, std,
ignoring images without a value (failed images / images without label).

Accuracy buckets (optimal / good / acceptable / failed, thresholds in infer/thresholds.py) are ALWAYS recomputed here
from the per-class distances of each row, so metrics files written before the buckets existed (or with older
thresholds) give the same result as new ones; the 'outlier' / 'bucket' fields stored in the file are never used.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from hcatostialocalization.infer.thresholds import BUCKETS, bucket_summary, classify_pair

STAT_KEYS = ["min", "p05", "p25", "median", "p75", "p95", "max", "mean", "std"]
SIDES = {"R": 0, "L": 1}


def stats(values) -> dict:
    a = np.array([v for v in values if v is not None], dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0, **{k: None for k in STAT_KEYS}}
    out = {
        "min": a.min(), "p05": np.percentile(a, 5), "p25": np.percentile(a, 25), "median": np.median(a),
        "p75": np.percentile(a, 75), "p95": np.percentile(a, 95), "max": a.max(), "mean": a.mean(), "std": a.std(),
    }
    return {"n": int(a.size), **{k: round(float(v), 6) for k, v in out.items()}}


def read_list(path: Path) -> list[str]:
    return [os.path.abspath(os.path.expanduser(l.strip())) for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]


def per_image_rows(metrics: list) -> list[dict]:
    """Rows of the 'Per Image Metrics' section (a list mixing section titles (str) and dicts)."""
    start = metrics.index("Per Image Metrics") + 1
    rows = []
    for item in metrics[start:]:
        if isinstance(item, str):
            break
        rows.append(item)
    return rows


def group_stats(rows: list[dict]) -> dict:
    def per_class(key, side):
        return [(r.get(key) or [None, None])[SIDES[side]] for r in rows]
    # buckets recomputed from the distances (per-class order: [right, left])
    triples = [classify_pair(*(list(r.get("distance_per_class") or []) + [None, None])[:2]) for r in rows]
    buckets = bucket_summary([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])
    # "outlier" = failed (above the failure threshold); an image counts if ANY ostium failed
    failed_per_class = {s: buckets[s]["counts"]["failed"] for s in SIDES}
    n_failed_any = sum(1 for t in triples if t[0] == "failed" or t[1] == "failed")
    valid = [r for r in rows if r.get("distance") is not None]
    return {
        "n_images": len(rows),
        "n_with_metrics": len(valid),
        "n_without_metrics": len(rows) - len(valid),   # failed to load, or no label
        "distance": stats([r.get("distance") for r in rows]),
        "distance_per_class": {s: stats(per_class("distance_per_class", s)) for s in SIDES},
        "peak_confidence": stats([r.get("peak_confidence") for r in rows]),
        "peak_confidence_per_class": {s: stats(per_class("peak_confidence_per_class", s)) for s in SIDES},
        "buckets": buckets,
        "outlier_images": n_failed_any,
        "outlier_fraction": round(n_failed_any / len(valid), 6) if valid else None,
        "outlier_per_class": failed_per_class,
    }


def f(x, w=7):
    return f"{x:{w}.2f}" if x is not None else " " * (w - 1) + "-"


def bucket_line(b, w=28) -> str:
    if not b["n"]:
        return " " * (w - 1) + "-"
    pcts = " / ".join("%.1f" % b["pct"][k] for k in BUCKETS)
    return f"{pcts:>{w}s}"


def main():
    default_lists = Path(__file__).parent / "collections_test_set"
    p = argparse.ArgumentParser(description="Split the metrics of one inference run into per-dataset statistics.")
    p.add_argument("--metrics", required=True, help="metrics__<model>.json written by infer/__main__.py")
    p.add_argument("--lists-dir", default=str(default_lists), help="folder with <NAME>_images.txt files")
    p.add_argument("-o", "--output", default=None, help="output json (default: metrics_by_dataset.json next to --metrics)")
    args = p.parse_args()

    metrics = json.load(open(args.metrics, encoding="utf-8"))
    rows = per_image_rows(metrics)

    lists = {q.name[:-len("_images.txt")]: read_list(q) for q in sorted(Path(args.lists_dir).glob("*_images.txt")) if q.name != "all_images.txt"}
    dataset_of = {}
    for name, paths in lists.items():
        for path in paths:
            if path in dataset_of:
                raise SystemExit(f"image in more than one dataset list: {path} ({dataset_of[path]}, {name})")
            dataset_of[path] = name

    groups = {name: [] for name in lists}
    unassigned = []
    for r in rows:
        name = dataset_of.get(os.path.abspath(r["image"]))
        (groups[name] if name else unassigned).append(r)
    missing = {name: [p_ for p_ in paths if p_ not in {os.path.abspath(r["image"]) for r in groups[name]}] for name, paths in lists.items()}

    result = {"ALL": group_stats(rows), **{name: group_stats(g) for name, g in groups.items()}}

    print(f"{'dataset':10s} {'images':>6s} {'w/metr':>6s} | {'dist median':>11s} {'mean':>7s} {'p95':>7s} {'max':>7s} | {'R median':>8s} {'L median':>8s} | {'patient % opt/good/acc/fail':>28s}")
    for name, s in result.items():
        d, dc = s["distance"], s["distance_per_class"]
        print(f"{name:10s} {s['n_images']:6d} {s['n_with_metrics']:6d} | {f(d['median'], 11)} {f(d['mean'])} {f(d['p95'])} {f(d['max'])} | {f(dc['R']['median'], 8)} {f(dc['L']['median'], 8)} | {bucket_line(s['buckets']['patient'])}")

    if unassigned:
        print(f"\n[!] {len(unassigned)} row(s) of the metrics file do not belong to any list, ignored: {[r['image'] for r in unassigned]}")
    for name, paths in missing.items():
        if paths:
            print(f"[!] {len(paths)} image(s) of {name}_images.txt are not in the metrics file (run stopped early?): {paths}")

    # sanity check: ALL computed here must equal the 'Overall Metrics' written by infer/__main__.py
    overall = next((m for m in metrics if isinstance(m, dict) and "distance" in m and "cli" in m), None)
    if overall and overall["distance"] and result["ALL"]["distance"]["median"] is not None:
        ok = all(np.isclose(overall["distance"][k], result["ALL"]["distance"][k], atol=1e-5) for k in ("median", "mean", "max"))
        print(f"\ncheck: ALL matches the 'Overall Metrics' of the metrics file: {ok}")

    out = Path(args.output) if args.output else Path(args.metrics).with_name("metrics_by_dataset.json")
    out.write_text(json.dumps({"source_metrics": os.path.abspath(args.metrics), "datasets": result}, indent=4), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
