# infer/

Inference, ensembling and evaluation of the coronary ostia localization models.

Run every command **from the repository root** (the folder that contains `hcatostialocalization/`). The examples use the test
set lists in `hcatostialocalization/test/collections_test_set/` (`all_images.txt`, `all_labels.txt`: one path per
line, same order). Only the necessary arguments are given; spacing, patch size, overlap and orientation come from the
model in the hub (`model/hub/hub.py`), and `-w best` is the default.

**User scripts** (the ones you call):

| File | What it does |
|---|---|
| `__main__.py` | runs 1 to 3 models on the images, saves the predicted ostia, computes the metrics if labels are given (section 1, 2) |
| `ensamble_predictions.py` | merges the predictions of N models into one (median) (section 3) |
| `evaluate_predictions.py` | evaluates predicted ostia already on disk against labels (section 3) |

**Support files** (imported by the scripts, not run directly):

| File | What it holds | Used by |
|---|---|---|
| `evaluator.py` | `OstiaEvaluator`: accumulates the metrics image by image, writes the log and `metrics__<name>.json`, prints the final report; plus `load_ostia_ras` and `compute_sample_metrics` | `evaluate_predictions.py` (intended also for `__main__.py`, which still has this code inline) |
| `thresholds.py` | accuracy thresholds (Anatomical / Interoperator / Failure, mm) and the four buckets Optimal / Good / Acceptable / Failed | `__main__.py`, `ensamble_predictions.py`, `evaluator.py`, `test/split_metrics_by_dataset.py` |
| `utils.py` | sliding-window prediction engine (`predict_minibatches_inplace`) and the ostium location on a heatmap (`get_location_from_probability_one_channel`: peak + local centroid) | `__main__.py` |

`__legacy/` holds old, unused scripts (`infer.py`, `batch_infer.py`, `validation_like_imagecas10.py`): do not use them.

```
images + labels
      |
  __main__.py -- utils.py (sliding window, ostium location)
      |  \__ thresholds.py + evaluator.py (metrics)  [today inline in __main__.py]
      v
  markers/*.ostium.json  --->  ensamble_predictions.py  --->  markers/*.ostium.json
      |                                                              |
      +----------------> evaluate_predictions.py (evaluator.py) <----+
                                  |
                      metrics__<name>.json --> test/split_metrics_by_dataset.py
```

## 1. One model: `__main__.py`

```bash
python -m hcatostialocalization.infer \
    -i hcatostialocalization/test/collections_test_set/all_images.txt \
    -l hcatostialocalization/test/collections_test_set/all_labels.txt \
    -m SwinUNETRv2_DSNT --gpu \
    -o results/test_set__SwinUNETRv2_DSNT/run1
```

- `-i` images, or a `.txt` list; `-l` labels (optional: without it no metrics are computed; in a list an empty
  line means "no label for this image"); `-m` model nickname (`--show-models` lists them); `-o` output folder.
- `--gpu` uses the GPU if available. `-a` chooses the artifacts: the default is `none` (only the metrics and the
  log are saved, as in the command above), because the tensors and NIfTI volumes are very heavy. Typical values: `-a marker` (predicted ostia), `-a markup-3d-slicer`, `-a marker probability-nifti`;
  `-a all` produces everything.
- Output in `-o`: `metrics__<model>.json` (overall statistics, per-image rows, errors), `_test_log ....txt`, and the
  requested artifacts, for instance `markers/*.ostium.json` (predicted ostia, the format used by everything below) and
  `markups-3d-slicer/*.mrk.json` (to look at them in 3D Slicer).
- Per-dataset statistics (ASOCA / CAT08 / ImageCAS) of a finished run:

```bash
python -m hcatostialocalization.test.split_metrics_by_dataset \
    --metrics results/test_set__SwinUNETRv2_DSNT/run1/metrics__SwinUNETRv2_DSNT.json
```

## 2. Multi-stage prediction (cascade) and evaluation

Give 2 or 3 models to `-m`: the first predicts the whole volume, the second refines a patch around each predicted
ostium (with 3 models: the second refines the left ostium, the third the right one). `-w` takes one weights file per
model, or `best` for all. Metrics are computed in the same run because `-l` is given.

```bash
python -m hcatostialocalization.infer \
    -i hcatostialocalization/test/collections_test_set/all_images.txt \
    -l hcatostialocalization/test/collections_test_set/all_labels.txt \
    -m SwinUNETRv2_DSNT nnUNetResidualUNet_DSNT --gpu \
    -a marker -o results/cascade_swin_nnres
```

The output is tagged `SwinUNETRv2_DSNT+nnUNetResidualUNet_DSNT` (`metrics__...json`). The cascade code has not been run
on the cluster yet.

## 3. Ensemble of predictions and evaluation

**a. Predict with each model**, saving only the ostia (`-a marker`). To also merge the probability maps, add
`probability-nifti`: `-a marker probability-nifti` (without `-a` nothing is saved, and the ensemble would have no
predictions to merge). Without `-l` no metrics are computed.

```bash
for M in SwinUNETRv2_DSNT nnUNetResidualUNet_DSNT ResNet101_DSNT; do
  python -m hcatostialocalization.infer \
      -i hcatostialocalization/test/collections_test_set/all_images.txt \
      -m $M --gpu -a marker -o results/ens/$M
done
```

**b. Merge** (median of the N predictions, per ostium). Each `-p` is one model (its output folder of step a). Without
`-l` it only saves the new predictions in `-o/markers/` (default `-a marker`; `-a markup-3d-slicer` and
`-a probability-nifti` (merged maps, heavy) are also available, `-a none` saves nothing but the metrics):

```bash
python -m hcatostialocalization.infer.ensamble_predictions \
    -p results/ens/SwinUNETRv2_DSNT results/ens/nnUNetResidualUNet_DSNT results/ens/ResNet101_DSNT \
    -i hcatostialocalization/test/collections_test_set/all_images.txt \
    -o results/ens/ensemble
```

Adding `-l hcatostialocalization/test/collections_test_set/all_labels.txt` also prints a table with every single model
and the ensemble. If the models were run with `probability-nifti`, the merge is done on those maps (`--from coordinates`
forces the positions).

**c. Evaluate** the ensemble against the labels (works for any predicted ostia: files, a list, or a folder):

```bash
python -m hcatostialocalization.infer.evaluate_predictions \
    -p results/ens/ensemble/markers \
    -l hcatostialocalization/test/collections_test_set/all_labels.txt \
    -i hcatostialocalization/test/collections_test_set/all_images.txt \
    -o results/ens/ensemble_evaluation --name Ensemble
```

The output is `metrics__Ensemble.json`, in the same format as `__main__.py`, so `split_metrics_by_dataset` works on it
(the command is printed at the end; `-i` is needed for that). Predictions and labels are paired by order; if the lists
are not aligned, add `--match name`.

## 4. Per-dataset statistics: `test/split_metrics_by_dataset.py`

The test set mixes three datasets (ASOCA 4, CAT08 4, ImageCAS 50 volumes). Every `metrics__<name>.json` (from
`__main__.py` or from `evaluate_predictions.py`) reports the whole test set; this script splits it by dataset and writes
`metrics_by_dataset.json` (blocks `ALL`, `ASOCA`, `CAT08`, `ImageCAS`), which is the file the tables of the paper are
filled from. It needs numpy only (no GPU, no torch).

```bash
python -m hcatostialocalization.test.split_metrics_by_dataset \
    --metrics results/test_set__SwinUNETRv2_DSNT/run1/metrics__SwinUNETRv2_DSNT.json
```

- A row belongs to the dataset whose `<NAME>_images.txt` (in `test/collections_test_set/`, `--lists-dir` to change it)
  contains the row's `image` path, compared as an absolute path. So the metrics must come from the image paths of those
  lists: with `evaluate_predictions.py` pass `-i` (otherwise the `image` field is not an image path).
- The output is written next to the metrics file (`-o` to change it) and a summary table is printed.
- Accuracy buckets are always recomputed from the per-class distances with the thresholds of `thresholds.py`, so old
  metrics files give the same result as new ones.
