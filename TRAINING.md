# Training guide

This is a **short guide to repeat the training of the paper's models**. The training code is the one used by the authors, on
their cluster; it was not generalised to run on any machine. Expect to adapt paths and job scripts to your system.
For using the released models you do not need any of this (see the [README](README.md)).

## 0. What you need

- The install of the README (sections 1.1-1.4). `mlflow` is **required** by the trainers (it is in `requirements.txt`).
- A CUDA GPU. The models were trained with 128x128x128 patches at 0.5 mm, batch size 1 with gradient accumulation; a GPU
  with 40 GB of memory and 64-128 GB of RAM was used for the larger models. Smaller GPUs need a smaller patch size or
  fewer workers (`config/model.yaml`).
- The CT images of the three source datasets, from their original sources, and the ostium annotations (section 1).

## 1. Data layout

The training and validation cases, and the test cases, are fixed by three split files, one per dataset. **They are the
single place that decides which files a case uses**; each entry is an `(image, label)` pair with paths **relative to
`data_root`**:

| Dataset | Split file | Image (relative to `data_root`) | Label (relative to `data_root`) |
|---|---|---|---|
| ImageCAS | `hcatostialocalization/data/dataset/ImageCAS/imagecas_config.yaml` (880 train / 70 val / 50 test) | `ImageCAS/Data/<id>.img.nii.gz` | `Annotations/OstiumAnnotations_ImageCAS/OSTIAv2/JSON/ImageCAS__<id>.ostium.json` |
| ASOCA | `.../ASOCA/asoca_config.yaml` (32 / 4 / 4) | `ASOCA/Normal/CTCA/Normal_<n>.nrrd` | `Annotations/OstiumAnnotations_ASOCA/OSTIAv2/JSON/ASOCA__Normal_<n>.ostium.json` |
| CAT08 | `.../CAT08/cat08_config.yaml` (25 / 3 / 4) | `CAT08/dataset<nn>_volume.nii.gz` | `Annotations/OstiumAnnotations_CAT08/OSTIAv2/JSON/CAT08__image<nn>.ostium.json` |

- **ImageCAS**: download the CT volumes from the ImageCAS authors (see the README), rename or link them to
  `<data_root>/ImageCAS/Data/<id>.img.nii.gz`, and unzip the released ImageCAS ostia (`Ostia.zip` on Zenodo,
  `JSON/` folder) to `<data_root>/Annotations/OstiumAnnotations_ImageCAS/OSTIAv2/JSON/`.
- **ASOCA and CAT08**: obtain the images from the ASOCA challenge and the CAT08 website. The **ostium references of these
  two datasets are not part of the Zenodo release**. The trainer builds its training pool from all three datasets, so
  to repeat the paper's training you must create them in the same `.ostium.json` format: for CAT08 from the challenge's
  centerline start point (`hcatostialocalization/data/dataset/CAT08/CAT08_Label&Conversion.py`), for ASOCA from the first point
  of each centerline branch, after visual inspection. (Training on ImageCAS alone is not supported by the provided
  entry points; it needs a change in `data/dataset/General_Class_Dataset_Multi.py`.)
- The file format is described in the dataset README on Zenodo; `hcatostialocalization/io/io.py` reads and writes it.

If your files live elsewhere, edit the `image:`/`label:` lines in the split files (they also accept absolute paths) rather
than moving the data.

## 2. Tell the code where things are

```bash
export HCAT_DATA_ROOT=/path/to/data          # the folder that contains ImageCAS/, ASOCA/, CAT08/, Annotations/
export HCAT_REPO_ROOT=$(pwd)                 # this repository (checkpoints/ and mlruns/ are created here)
# optional: cache of the resampled CT volumes (large, speeds up later epochs)
# export HCAT_CACHE_DIR=/path/to/cache
python -m hcatostialocalization.config.config | head      # prints the resolved configuration: check the paths
```

These variables are read by `hcatostialocalization/config/machine.yaml`; you can also edit the `default_user` block there.

## 3. Choose the model and train

Every experiment is a block of `hcatostialocalization/config/model.yaml` (architecture, loss, optimiser, schedule,
augmentation). The one that is trained is set by `active_experiment` at the top of the file:

| `active_experiment` | Model of the paper |
|---|---|
| `swin_unetr_v2_dsnt` | SwinUNETRv2 |
| `nnunet_residualunet_dsnt` | nnResUNet |
| `resnet50_dsnt`, `resnet101_dsnt` | ResNet50, ResNet101 |
| `mednext_s_dsnt`, `mednext_b_dsnt` | MedNeXt-S/B (needs `mednextv1`, see `requirements-extra.txt`; reported in the paper, weights not released) |

(The file also contains the experiments of the ablations and a few variants; read the comments in the file.) Then:

```bash
python -m hcatostialocalization.train.trainer                      # new run
python -m hcatostialocalization.train.trainer --resume-auto        # continue the latest run of this experiment
python -m hcatostialocalization.train.trainer --resume-checkpoint checkpoints/<model>/<run>/last_model.pth
```

Outputs:

- `checkpoints/<model>/<timestamp>_<run name>/best_model.pth` (best validation error) and `last_model.pth`;
- an MLflow store in `mlruns/` (curves, parameters, example images). To browse it: `mlflow ui --backend-store-uri file:./mlruns`.

The weights file is a PyTorch checkpoint, `{"model_state_dict": ...}`. To use a model you trained with the inference
command, pass it with `-w`:

```bash
python -m hcatostialocalization.infer -i ct.nii.gz -m SwinUNETRv2_DSNT -w checkpoints/swin_unetr/<run>/best_model.pth --gpu -o results/
```

## 4. On a SLURM cluster

`slurm/sbatch-train.sh` is the job script of the authors' cluster: it activates a conda environment, goes to the repository
root and runs the trainer, forwarding its arguments (`sbatch slurm/sbatch-train.sh --resume-auto`). Edit the `#SBATCH`
lines (partition, QoS, GPU, memory, time), the conda activation line and the paths to match your cluster, and create
`slurm/logs/` before submitting (SLURM does not create it). `slurm/sbatch-train-ablation-*.sh` are the same for the two
ablation trainers (`train/trainer_ablation_training_spacing.py`, `train/trainer_ablation_imagecas_size.py`).

## 5. After training

- Evaluate on the paper's test set as in README section 3.5 (`infer` with `-l`, then `test/split_metrics_by_dataset.py`).
- The accuracy buckets and the interoperator thresholds used in the paper are defined in
  `hcatostialocalization/infer/thresholds.py`.
