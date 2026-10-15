"""
Evaluate predicted coronary ostia (already on disk) against reference ostia.

Use it when you have the predictions but not a model to run: the output folder of an inference run, the output of
infer/ensamble_predictions.py, the ostia of another method, ... The metrics and the files written are the ones of
infer/__main__.py (see infer/evaluator.py), so `test/split_metrics_by_dataset.py` works on the result.

Usage, as a module, from the folder that contains `hcatostialocalization/`:

    python -m hcatostialocalization.infer.evaluate_predictions \
        -p results/ensemble_swin_nnres/markers \
        -l hcatostialocalization/test/collections_test_set/all_labels.txt \
        [-i hcatostialocalization/test/collections_test_set/all_images.txt] \
        [-o results/ensemble_swin_nnres_evaluation] [--name Ensemble]

-p/--predictions   predicted ostia: files, a .txt list (one path per line), or a folder (all its `*.ostium.json`, else
                   `*.mrk.json`, sorted by name). Any format understood by io.extract_simplified_ras: standard
                   `.ostium.json` blueprint, the older ostia JSON of ImageCAS/CAT08/ASOCA, 3D Slicer `.mrk.json`.
-l/--label         reference ostia, same kinds of input. In a .txt list an empty line means "no label for this image":
                   the image is counted but has no metrics (as in infer/__main__.py).
-i/--images        optional. Images (files or a .txt list). They fill the `image` field of the per-image rows (needed by
                   split_metrics_by_dataset.py, which assigns each row to a dataset through the image path), give the
                   header that raw ASOCA label files need, and are required by `--match name`.
--match            how predictions and labels are paired:
                     order  (default) the i-th prediction with the i-th label (and i-th image): the lists must have
                            the same length, like -i/-l of infer/__main__.py. Predictions in a folder are sorted by
                            name, which is the inference order (`case_000000_...`).
                     name   through the image names (requires -i): an image `Normal_5.nrrd` takes the file whose
                            `image_filename` (blueprints) or name, without `case_<n>_`, is `Normal_5` or ends/starts
                            with `_Normal_5` / `Normal_5_` (`ASOCA__Normal_5.ostium.json`). Images without a match are
                            reported as errors; an ambiguous match too.
-o/--output        output folder (default: ./evaluation_<date and time>).
--name             run name, used in `metrics__<name>.json` (default: the name of the predictions folder).
--metrics          distance / outliers / all (default all). `peak-confidence` needs the probability map and does not
                   exist for predictions given as positions.

Written in the output folder: `metrics__<name>.json` (rewritten after every image) and `_test_log <date>.txt`.
"""
import argparse
import json
import os
import re
import sys
import time

from hcatostialocalization.infer.evaluator import (
    OstiaEvaluator, collect_input_paths, load_ostia_ras,
)

_CASE_PREFIX = re.compile(r"^case_\d+_")


def image_key(path_or_name: str) -> str:
    """File name without extension (.nii.gz, .nrrd, .ostium.json, ...), as infer/__main__.py builds it."""
    name = os.path.basename(str(path_or_name))
    for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".ostium.json", ".mrk.json", ".json"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def candidate_keys(path: str) -> set:
    """Names under which a prediction/label file can be matched to an image."""
    stem = image_key(path)
    keys = {stem, _CASE_PREFIX.sub("", stem)}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("image_filename"):  # blueprint written by infer
            keys.add(image_key(data["image_filename"]))
    except Exception:
        pass
    return keys


def match_by_name(images: list, files: list, what: str) -> tuple:
    """For each image, the file of `files` that belongs to it ('' if none) and a list of problems (None if fine)."""
    candidates = [(f, candidate_keys(f)) for f in files if f]
    matched, problems = [], []
    for image in images:
        k = image_key(image)
        exact = [f for f, keys in candidates if k in keys]
        loose = [f for f, keys in candidates if any(c.endswith("_" + k) or c.startswith(k + "_") for c in keys)]
        hits = exact or loose
        if len(hits) == 1:
            matched.append(hits[0])
            problems.append(None)
        elif not hits:
            matched.append('')
            problems.append(f"no {what} found for image '{k}'")
        else:
            matched.append('')
            problems.append(f"ambiguous {what} for image '{k}': {[os.path.basename(h) for h in hits[:4]]}")
    return matched, problems


def main():
    p = argparse.ArgumentParser(description="Evaluate predicted ostia against reference ostia.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("-p", "--predictions", nargs="+", required=True, help="Predicted ostia: files, a .txt list, or a folder.")
    p.add_argument("-l", "--label", nargs="+", required=True, help="Reference ostia: files, a .txt list, or a folder.")
    p.add_argument("-i", "--images", nargs="+", default=None, help="Images (files or a .txt list). Optional with --match order.")
    p.add_argument("--match", choices=["order", "name"], default="order", help="How predictions and labels are paired (default order).")
    p.add_argument("-o", "--output", default=None, help="Output folder (default: ./evaluation_<date and time>).")
    p.add_argument("--name", default=None, help="Run name used in the metrics file name.")
    p.add_argument("--metrics", nargs="+", choices=["distance", "outliers", "all"], default=["all"], help="Metrics to compute (default all).")
    args = p.parse_args()

    # ---- inputs
    predictions = collect_input_paths(args.predictions, are_labels=True)
    labels = collect_input_paths(args.label, are_labels=True)
    images = collect_input_paths(args.images, are_labels=False) if args.images else []
    if not predictions:
        raise SystemExit("no predictions found")
    if not labels:
        raise SystemExit("no labels found")

    problems = None
    if args.match == "order":
        n = len(predictions)
        if len(labels) != n or (images and len(images) != n):
            raise SystemExit(f"--match order needs lists of the same length, got {n} predictions, {len(labels)} labels"
                             + (f", {len(images)} images" if images else "") + ". Use --match name (with -i) to pair through the image names.")
        images_out = images or [pred or lab for pred, lab in zip(predictions, labels)]
        pairs = list(zip(images_out, labels, predictions))
    else:
        if not images:
            raise SystemExit("--match name requires the images (-i)")
        pred_matched, pred_problems = match_by_name(images, predictions, "prediction")
        lab_matched, _ = match_by_name(images, labels, "label")
        pairs = list(zip(images, lab_matched, pred_matched))
        problems = pred_problems

    # ---- run setup
    start = time.strftime("%Y.%m.%d - %H:%M:%S")
    output = args.output or os.path.join(os.getcwd(), "evaluation_" + time.strftime("%Y_%m_%d_T_%H_%M_%S"))
    name = args.name or (os.path.basename(os.path.normpath(args.predictions[0])) if len(args.predictions) == 1 and os.path.isdir(os.path.expanduser(args.predictions[0])) else "predictions")
    name = re.sub(r"[^\w.+-]+", "_", name)
    evaluator = OstiaEvaluator(
        output_dir=output, run_name=name, total_images=len(pairs), metrics=args.metrics, start_datetime=start,
        overview={"evaluated from": "predicted ostia files", "predictions": args.predictions, "labels": args.label,
                  "match": args.match},
    )
    print(f"Found {len(pairs)} image(s) to evaluate, {sum(1 for _, l, _ in pairs if l)} with a label, "
          f"{sum(1 for _, _, pr in pairs if pr)} with a prediction.")
    print("The program will now execute disregarding errors.")

    # ---- cycle
    for i, (image, label, prediction) in enumerate(pairs):
        evaluator.print_progress(i + 1)
        if not prediction:
            evaluator.add_error(image, label, (problems[i] if problems and problems[i] else "no prediction for this image"))
            continue
        try:
            prediction_ras = load_ostia_ras(prediction, image_path=image if os.path.exists(image) else None)
            if prediction_ras is None:
                raise FileNotFoundError(f"prediction file not found: {prediction}")
            label_ras = load_ostia_ras(label, image_path=image if os.path.exists(image) else None) if label else None
            evaluator.add_sample(image, label, prediction_ras=prediction_ras, label_ras=label_ras)
        except Exception as e:
            evaluator.add_error(image, label, f"Could not read prediction or label. {e}")

    metrics_file = evaluator.finalize()
    if images and metrics_file:
        print("Per-dataset statistics:\n  python -m hcatostialocalization.test.split_metrics_by_dataset --metrics " + f"'{metrics_file}'")


if __name__ == "__main__":
    main()
