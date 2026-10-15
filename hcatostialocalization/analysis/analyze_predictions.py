from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hcatostialocalization.analysis.serve_viewer import write_viewer_files
from hcatostialocalization.analysis.volume_io import (
    distance_mm,
    heatmap_metrics,
    parse_coord,
    parse_shape,
    parse_spacing,
    parse_triplet,
    read_volume,
    read_volume_metadata,
)


STAT_KEYS = ("mean", "std", "min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max")


def looks_like_existing_or_expected_path(value: str | None) -> bool:
    if not value:
        return False
    text = str(value).strip()
    lower = text.lower()
    suffixes = (".nii.gz", ".nrrd", ".mhd", ".npy", ".npz", ".json", ".txt", ".csv", ".tsv", ".mrk")
    return "/" in text or "\\" in text or any(lower.endswith(suffix) for suffix in suffixes)


def path_status(value: str | None, base_dir: Path) -> str:
    if not looks_like_existing_or_expected_path(value):
        return ""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return "ok" if path.exists() else "file_not_found"


def read_input_table(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    sample = "\n".join(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    if not sample:
        return []
    try:
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(sample.splitlines(), dialect=dialect)
    return expand_side_rows([dict(row) for row in reader])


def expand_side_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    for row in rows:
        explicit_side = str(row.get("side") or row.get("ostium") or "").strip()
        has_side_columns = any(
            key in row
            for key in (
                "label_R",
                "label_L",
                "gt_R",
                "gt_L",
                "prediction_R",
                "prediction_L",
                "pred_R",
                "pred_L",
                "heatmap_R",
                "heatmap_L",
            )
        )
        if explicit_side or not has_side_columns:
            expanded.append(row)
            continue

        for side in ("R", "L"):
            side_row = dict(row)
            side_row["side"] = side
            for generic, candidates in {
                "label": (f"label_{side}", f"gt_{side}"),
                "prediction": (f"prediction_{side}", f"pred_{side}"),
                "heatmap": (f"heatmap_{side}", f"heatmap_path_{side}"),
            }.items():
                for key in candidates:
                    if key in row and str(row[key]).strip():
                        side_row[generic] = row[key]
                        break
            expanded.append(side_row)
    return expanded


def nonempty(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def maybe_path(value: str | None, base_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def resolve_geometry(row: dict[str, Any], base_dir: Path) -> tuple[tuple[int, int, int] | None, tuple[float, float, float] | None]:
    image_path = maybe_path(nonempty(row, "image", "image_path"), base_dir)
    heatmap_path = maybe_path(nonempty(row, "heatmap", "heatmap_path"), base_dir)
    fallback_shape = None
    fallback_spacing = None
    for path in (image_path, heatmap_path):
        if path and Path(path).exists():
            try:
                fallback_shape, fallback_spacing = read_volume_metadata(path)
                break
            except Exception:
                pass
    shape = parse_shape(row, fallback=fallback_shape)
    spacing = parse_spacing(row, fallback=fallback_spacing)
    return shape, spacing


def analyze_row(row: dict[str, str], base_dir: Path, outlier_threshold_mm: float) -> dict[str, Any]:
    side = (nonempty(row, "side", "ostium", "class") or "").upper()
    if side not in {"R", "L"}:
        side = ""

    image_path = maybe_path(nonempty(row, "image", "image_path"), base_dir)
    label_ref = nonempty(row, "label", "label_path", "gt", "gt_path")
    pred_ref = nonempty(row, "prediction", "prediction_path", "pred", "pred_path")
    heatmap_path = maybe_path(nonempty(row, "heatmap", "heatmap_path"), base_dir)
    label_ref_status = path_status(label_ref, base_dir)
    prediction_ref_status = path_status(pred_ref, base_dir)
    image_path_status = path_status(image_path, base_dir)
    heatmap_path_status = path_status(heatmap_path, base_dir)
    case_id = nonempty(row, "case_id", "sample_id", "id")
    if not case_id:
        source = image_path or heatmap_path or label_ref or pred_ref or "case"
        case_id = Path(str(source)).stem.replace(".nii", "")

    shape, spacing = resolve_geometry(row, base_dir)
    label = parse_triplet(row, "label") or parse_triplet(row, "gt") or parse_coord(label_ref, side=side or None)
    prediction = parse_triplet(row, "prediction") or parse_triplet(row, "pred") or parse_coord(pred_ref, side=side or None)
    dist = distance_mm(prediction, label, spacing)

    out: dict[str, Any] = {
        "case_id": case_id,
        "dataset": nonempty(row, "dataset") or "",
        "side": side,
        "image_path": image_path or "",
        "label_ref": label_ref or "",
        "prediction_ref": pred_ref or "",
        "heatmap_path": heatmap_path or "",
        "image_path_status": image_path_status,
        "label_ref_status": label_ref_status,
        "prediction_ref_status": prediction_ref_status,
        "heatmap_path_status": heatmap_path_status,
        "shape_z": shape[0] if shape else "",
        "shape_y": shape[1] if shape else "",
        "shape_x": shape[2] if shape else "",
        "spacing_z": spacing[0] if spacing else "",
        "spacing_y": spacing[1] if spacing else "",
        "spacing_x": spacing[2] if spacing else "",
        "gt_z": label[0] if label else "",
        "gt_y": label[1] if label else "",
        "gt_x": label[2] if label else "",
        "pred_z": prediction[0] if prediction else "",
        "pred_y": prediction[1] if prediction else "",
        "pred_x": prediction[2] if prediction else "",
        "distance_to_label_mm": dist if dist is not None else "",
        "is_outlier": bool(dist is not None and dist > outlier_threshold_mm),
        "outlier_reason": f"distance>{outlier_threshold_mm:g}mm" if dist is not None and dist > outlier_threshold_mm else "",
    }
    if label_ref and label is None:
        out["label_error"] = "label file not found" if label_ref_status == "file_not_found" else "label not parsed"
    if pred_ref and prediction is None:
        out["prediction_error"] = (
            "prediction file not found" if prediction_ref_status == "file_not_found" else "prediction not parsed"
        )

    if heatmap_path and Path(heatmap_path).exists():
        try:
            heatmap, heatmap_spacing = read_volume(heatmap_path)
            metrics_spacing = spacing or heatmap_spacing
            out.update(heatmap_metrics(heatmap, metrics_spacing))
        except Exception as exc:
            out["heatmap_error"] = str(exc)
    elif heatmap_path:
        out["heatmap_error"] = "file not found"
    return out


def stats(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {key: None for key in STAT_KEYS} | {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "distance_to_label_mm": {
            "all": stats([float(r["distance_to_label_mm"]) for r in rows if r.get("distance_to_label_mm") != ""]),
            "R": stats([float(r["distance_to_label_mm"]) for r in rows if r.get("side") == "R" and r.get("distance_to_label_mm") != ""]),
            "L": stats([float(r["distance_to_label_mm"]) for r in rows if r.get("side") == "L" and r.get("distance_to_label_mm") != ""]),
        }
    }
    datasets = sorted({str(r.get("dataset", "")) for r in rows if r.get("dataset")})
    if datasets:
        summary["distance_to_label_mm"]["by_dataset"] = {
            d: stats([float(r["distance_to_label_mm"]) for r in rows if r.get("dataset") == d and r.get("distance_to_label_mm") != ""])
            for d in datasets
        }
    for metric in ("heatmap_entropy_norm", "heatmap_spread_mm", "heatmap_peak_prominence_p95", "heatmap_peak_prominence_median"):
        values = [float(r[metric]) for r in rows if r.get(metric) not in ("", None)]
        if values:
            summary[metric] = stats(values)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ostia predictions from a CSV/TSV manifest without running inference.")
    parser.add_argument("input_table", type=Path, help="CSV/TSV with optional image, label, prediction, heatmap columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    parser.add_argument("--outlier-threshold-mm", type=float, default=10.0)
    parser.add_argument("--serve", action="store_true", help="Start the local viewer after writing outputs.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    base_dir = args.input_table.resolve().parent
    input_rows = read_input_table(args.input_table)
    analyzed = [analyze_row(row, base_dir, args.outlier_threshold_mm) for row in input_rows]
    outliers = [row for row in analyzed if row.get("is_outlier")]
    summary = build_summary(analyzed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in analyzed:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(args.output_dir / "predictions.csv", analyzed, fieldnames=fieldnames)
    write_csv(args.output_dir / "outliers.csv", outliers, fieldnames=fieldnames)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_viewer_files(args.output_dir, analyzed, summary)

    viewer_cmd = (
        f"python -m hcatostialocalization.analysis.serve_viewer "
        f"{args.output_dir} --host 0.0.0.0 --port {args.port}"
    )
    print("")
    print("Analysis completed.")
    print(f"- Predictions: {(args.output_dir / 'predictions.csv').resolve()}")
    print(f"- Outliers:    {(args.output_dir / 'outliers.csv').resolve()}")
    print(f"- Summary:     {(args.output_dir / 'summary.json').resolve()}")
    print("")
    print("Viewer tutorial")
    print("===============")
    print("")
    print("1. Keep this terminal on the remote machine/VM and start the viewer:")
    print(f"   {viewer_cmd}")
    print("")
    print("2. If you are running the viewer on your own computer, open:")
    print(f"   http://localhost:{args.port}")
    print("")
    print("3. If you are running the viewer on cluster, open a NEW terminal on your own computer")
    print("   and start the SSH tunnel:")
    print(f"   ssh -L {args.port}:<node-ip>:{args.port} <user>@<login-host>")
    print("")
    print("4. Then open this same link in your browser:")
    print(f"   http://localhost:{args.port}")
    print("")
    print("Notes:")
    print("- The viewer terminal and the SSH tunnel terminal must stay open.")
    print("- If the viewer is running on a different cluster node, replace <node-ip> with that node IP.")

    if args.serve:
        from hcatostialocalization.analysis.serve_viewer import serve

        serve(args.output_dir, host="0.0.0.0", port=args.port, open_browser=True)


if __name__ == "__main__":
    main()
