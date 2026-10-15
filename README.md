# HCATOstiaLocalization

[![Project website](https://img.shields.io/badge/website-Awakening%20ImageCAS-1e788c?style=flat-square&logo=googlechrome&logoColor=white)](https://hal9000-lab.github.io/Awakening-ImageCAS/)
[![Dataset DOI](https://img.shields.io/badge/dataset-Zenodo-1682d4?style=flat-square&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.23243093)
[![Weights DOI](https://img.shields.io/badge/model%20weights-Zenodo-1682d4?style=flat-square&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.23399623)
[![Paper](https://img.shields.io/badge/paper-coming%20soon-lightgrey?style=flat-square)](https://hal9000-lab.github.io/Awakening-ImageCAS/)
[![Code licence: MIT](https://img.shields.io/badge/code-MIT-yellow?style=flat-square)](LICENSE)
[![Data and weights licence: CC BY-NC 4.0](https://img.shields.io/badge/data%20%26%20weights-CC%20BY--NC%204.0-lightgrey?style=flat-square)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](#1-installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-3d8f5b?style=flat-square)](https://monai.io/)

**All the resources of this work (the coronary ostium annotations and centerline graphs, the pretrained model weights,
the code and the documentation) can be reached from the project website,
[Awakening ImageCAS: project page, datasets, models and code](https://hal9000-lab.github.io/Awakening-ImageCAS/).**

Automatic localization of the **left and right coronary ostia** in 3D cardiac CT angiography (CCTA).
Code, benchmark models and tools of **"Awakening ImageCAS: Automated Coronary Ostia Localization Framework and
Centerlines Open Dataset"**: manual and semi-automated coronary ostia annotations for all 1000 CCTA scans of ImageCAS,
and open-weight models trained on them.

| Resource | Where |
|---|---|
| 🌐 Project website | [hal9000-lab.github.io/Awakening-ImageCAS](https://hal9000-lab.github.io/Awakening-ImageCAS/) |
| 🗂️ Dataset: ostia and centerlines | [doi:10.5281/zenodo.23243093](https://doi.org/10.5281/zenodo.23243093) (Zenodo, all versions) |
| 🧠 Model weights | [doi:10.5281/zenodo.23399623](https://doi.org/10.5281/zenodo.23399623) (Zenodo, all versions) |
| 📄 Paper | to be added when published |
| 💻 Code | this repository (MIT, see [LICENSE](LICENSE)) |

The **ostium annotations, the centerline graphs and the model weights are distributed separately** (links above) and
under their own licence, which also has to respect the terms of the original ImageCAS, ASOCA and CAT08 datasets.
**The CT images themselves are not part of any release**: get them from their original sources.

This README has three parts:

1. [Installation](#1-installation) and [a smoke test](#2-check-the-installation-smoke-tests) that needs no data.
2. [Inference with the released models](#3-inference-with-the-released-models) (what most people want).
3. [Training](#6-training-your-own-models) (short guide; the training code is tuned to the authors' cluster, see below).

Always run the commands **from the root of this repository** (the folder that contains `hcatostialocalization/`).

---

## 1. Installation

Requirements: Linux, macOS or Windows (WSL recommended); **Python 3.10 or newer** (the authors use 3.13); a CUDA GPU is
strongly recommended (the code also runs on CPU, much more slowly). A few GB of free disk space for the environment,
plus the weights.

### 1.1 Get the code

```bash
git clone https://github.com/AAMIASoftwares-research/HCATOstiaLocalizationRelease.git
cd HCATOstiaLocalizationRelease
```

### 1.2 Create an environment

With conda (or miniconda, <https://www.anaconda.com/docs/getting-started/miniconda/install/overview>):

```bash
conda create -n hcatostia python=3.13
conda activate hcatostia
```

or with `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
```

### 1.3 Install PyTorch (choose the build for your hardware)

Pick the command for your system on <https://pytorch.org/get-started/locally/>. Typical choices:

```bash
# NVIDIA GPU (example: CUDA 12.9; use the CUDA version that matches your driver)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# CPU only (Linux/Windows) or Apple silicon
python -m pip install torch torchvision
```

### 1.4 Install the other dependencies

```bash
python -m pip install -r requirements.txt
```

This installs MONAI, nnU-Net (for the nnResUNet architecture), SimpleITK, nibabel, scikit-image, MLflow (used only by
the trainers), pytest and the other libraries listed in [requirements.txt](requirements.txt).

Optional, only for the centerline tools, 3D viewers and figures:

```bash
python -m pip install -r requirements-extra.txt
```

> No `pip install .` is needed: the package is used in place, which is why the commands are run from the repository root.

---

## 2. Check the installation (smoke tests)

These tests need **no dataset and no trained weights**: they build the released architectures with random weights, run the
whole inference command on a small synthetic volume, and check the output files. They verify the installation, not the
accuracy of the models.

```bash
# quick checks (a few minutes on CPU): imports, configuration, split files, ResNet50, post-processing, end-to-end inference
python -m pytest tests/ -m "not slow" -v

# everything, including the larger architectures (SwinUNETRv2, ResNet101, nnResUNet) and the training imports;
# ~16 GB of RAM or a GPU are recommended
python -m pytest tests/ -v
```

If the quick checks pass, inference will run on your machine. If a test fails, see [Troubleshooting](#8-troubleshooting).

---

## 3. Inference with the released models

### 3.1 Download the weights

Download the weights from the Zenodo record linked at the top (`https://doi.org/10.5281/zenodo.23399623`) and put the
`.pth` files in a folder named `checkpoints/` that you create in the repository root (`mkdir checkpoints`):

| Model (`-m`) | Architecture | Weights file | Default location |
|---|---|---|---|
| `SwinUNETRv2_DSNT` | SwinUNETR v2 (MONAI), feature size 24 | `SwinUNETRv2_DSNT.pth` | `checkpoints/SwinUNETRv2_DSNT.pth` |
| `nnUNetResidualUNet_DSNT` | nnU-Net residual U-Net (nnResUNet) | `nnUNetResidualUNet_DSNT.pth` | `checkpoints/nnUNetResidualUNet_DSNT.pth` |
| `ResNet101_DSNT` | 3D ResNet101 encoder + heatmap head | `ResNet101_DSNT.pth` | `checkpoints/ResNet101_DSNT.pth` |
| `ResNet50_DSNT` | 3D ResNet50 encoder + heatmap head | `ResNet50_DSNT.pth` | `checkpoints/ResNet50_DSNT.pth` |
| `SwinUNETRv2_DSNT_ablation_training_spacing_1.0mm` | SwinUNETR v2 trained and run at **1.0 mm** isotropic spacing (from the training-spacing ablation of the paper) | `SwinUNETRv2_DSNT_ablation_training_spacing_1.0mm.pth` | `checkpoints/SwinUNETRv2_DSNT_ablation_training_spacing_1.0mm.pth` |

Files placed at the default location are found automatically. Weights stored anywhere else are passed with
`-w /path/to/file.pth`. All the models were trained with the DSNT-based loss on 128x128x128 patches of CT volumes
resampled to 0.5 mm isotropic spacing (1.0 mm for the last model, which resamples your CT to 1.0 mm accordingly), and predict three channels: background, right ostium, left ostium.

### 3.2 Run on your own CT

```bash
python -m hcatostialocalization.infer \
    -i /path/to/ct.nii.gz \
    -m SwinUNETRv2_DSNT \
    --gpu \
    -a marker markup-3d-slicer \
    -o results/
```

- `-i`: one or more CT volumes (NIfTI, NRRD, or any format MONAI/SimpleITK reads), **or one text file with one path per
  line** (to process many scans in one go).
- `-m`: the model, one of the nicknames of the table above.
- `-w`: weights file, only if not in `checkpoints/` (see above).
- `--gpu`: use the GPU if available (omit to run on CPU).
- `-a`: the files to write. `marker` = the ostia as `*.ostium.json`; `markup-3d-slicer` = the same two points as a 3D
  Slicer point list; the others (`probability-nifti`, `logits-nifti`, ...) are heavy volumes for visual inspection.
- `-o`: output folder.

You do **not** need to resample or normalise the CT: the program reorients it to RAS, resamples it to the spacing the
model was trained at (0.5 mm), clips the intensities to [-200, 800] HU and rescales them to [-1, 1], runs a sliding window
(128^3 patches, 50 % overlap), and brings the result back to the **native space** of your image. The input should be in
Hounsfield units (a volume stored with an offset of 1024 is detected and corrected). One scan with one heart is expected.

### 3.3 What you get

In `-o` (names are taken from the image file name):

- `markers/<name>.ostium.json`: the two predicted ostia. In `pos.R` and `pos.L` you find `ras` (mm, world coordinates of
  your image) and `ijk` (continuous voxel indices, for the two index conventions `sitk` and `monai-ras`), plus the 4x4
  affine matrices between them. The format is the one of the released annotations (see the dataset README on Zenodo).
- `markups-3d-slicer/<name>.mrk.json`: open it in 3D Slicer together with the CT.
- `metrics__<model>.json` and a log file (error statistics, only meaningful if you gave labels).

### 3.4 Evaluate against labels

If you have reference ostia in the same `.ostium.json` format (for instance the released ImageCAS annotations), pass
them with `-l` (a file list in the same order as the images, or the label files themselves):

```bash
python -m hcatostialocalization.infer -i images.txt -l labels.txt -m SwinUNETRv2_DSNT --gpu -a marker -o results/
```

The report gives the error in mm per ostium (median, mean, percentiles) and the percentage of cases in four accuracy
buckets (Optimal / Good / Acceptable / Failed, defined in `hcatostialocalization/infer/thresholds.py`).

### 3.5 Reproduce the benchmark on the paper's test set

The 58 test volumes (50 ImageCAS, 4 ASOCA, 4 CAT08) are listed in `hcatostialocalization/test/collections_test_set/`.
The lists contain paths **relative to your data folder**; make absolute lists with:

```bash
export HCAT_DATA_ROOT=/path/to/your/data      # the folder that contains ImageCAS/, ASOCA/, CAT08/, Annotations/
for f in all_images all_labels; do
  sed "s#^#$HCAT_DATA_ROOT/#" hcatostialocalization/test/collections_test_set/$f.txt > /tmp/$f.txt
done
python -m hcatostialocalization.infer -i /tmp/all_images.txt -l /tmp/all_labels.txt \
    -m SwinUNETRv2_DSNT --gpu --metrics all -a marker -o results/test_set
python -m hcatostialocalization.test.split_metrics_by_dataset --metrics results/test_set/metrics__SwinUNETRv2_DSNT.json
```

The ASOCA and CAT08 reference ostia are not part of the released dataset, so with only the ImageCAS annotations you can
run the 50 ImageCAS volumes (`ImageCAS_images.txt`, `ImageCAS_labels.txt`).

More inference options (cascades of models, ensembles, sliding-window overlap, metrics) are described in
[hcatostialocalization/infer/README.md](hcatostialocalization/infer/README.md).

---

## 4. The dataset

The annotations extend the public [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT) dataset (1000 CCTA scans and
coronary segmentations) with the left and right coronary ostia of every scan, and (in the same Zenodo record, derived from
the ImageCAS masks) centerline graphs. How the ostia were annotated (multi-annotator consensus for the first 500 scans,
model-assisted single-annotator correction for the other 500), their format and a quick start are in the dataset README
on Zenodo and on the project page. The ostium JSON format is read and written by `hcatostialocalization/io/io.py`.

## 5. Repository layout

```
hcatostialocalization/
  infer/       inference, ensembling and evaluation (the entry point you need)  -> infer/README.md
  model/       architectures (model_factory.py), released-model constructors (hub/hub.py), DSNT layer
  config/      machine.yaml (paths), model.yaml (all training experiments), dataset.yaml
  io/          reading/writing of the .ostium.json format and of images
  train/       trainers (main training and the two ablation trainers)
  loss/ layers/ logger/ data/ analysis/ test/   losses, CT preprocessing, MLflow logging, dataset
                                                 and centerline tools, analysis scripts, test-set lists
slurm/         example SLURM job scripts of the authors' cluster
tests/         smoke tests (section 2)
checkpoints/   (you create it; git-ignored) downloaded weights; training also writes its checkpoints here
```

## 6. Training your own models

Retraining is **not turnkey**: the training code is tuned to the authors' cluster and data layout and was not generalised.
It is still the code that produced the released models, and the steps below are everything needed to repeat it. See
**[TRAINING.md](TRAINING.md)** for the full guide. In short:

1. Obtain the CT images (ImageCAS, ASOCA, CAT08) from their original sources and the ostium annotations, and arrange them
   as described in TRAINING.md. The split files with the exact train/validation/test cases are in
   `hcatostialocalization/data/dataset/*/*_config.yaml`.
2. Point the code to your folders with environment variables (`HCAT_DATA_ROOT`, `HCAT_REPO_ROOT`; see
   `hcatostialocalization/config/machine.yaml`).
3. Choose the model in `hcatostialocalization/config/model.yaml` (`active_experiment`), then
   `python -m hcatostialocalization.train.trainer` (on a SLURM cluster, adapt `slurm/sbatch-train.sh`).

Training logs everything with MLflow (a local `mlruns/` folder is created) and writes the checkpoints to
`checkpoints/<model>/<run>/`.

## 7. Centerlines

The centerline graphs of the dataset are derived from the ImageCAS segmentation masks with the code in
`hcatostialocalization/data/makecenterlines/` (see its README and `release_README.md`; install `requirements-extra.txt`).
This part is documented separately and is not needed for ostia localization.

## 8. Troubleshooting

| Problem | What to try |
|---|---|
| `ModuleNotFoundError: hcatostialocalization` | run from the repository root, not from inside `hcatostialocalization/` |
| `Provided weights path is invalid` | the file is not at `checkpoints/<Model>.pth`: pass `-w /full/path/file.pth` |
| `KeyError` / unknown `active_paths` / config errors | unset `HCAT_HOST` (or set it to `default_user`); see `config/machine.yaml` |
| CUDA out of memory | use a GPU with more memory, or run on CPU (slow); smaller `--patch-size`/`--overlap` values exist but were not validated and may reduce accuracy |
| `torch.load` / pickle errors with weights | check that the downloaded file is complete (compare the checksum on Zenodo) and unmodified; report your PyTorch version in an issue |
| Results look shifted or mirrored | the image must be a standard CT with a valid affine/direction; check it in 3D Slicer |
| Smoke tests fail while installing | run `python -m pytest tests/ -x -v` and open an issue with the output |

## 9. Citation

If you use the code, the annotations or the models, please cite the paper (reference to be added when published) and
the dataset record: Leccardi M. *et al.*, "Awakening ImageCAS", Zenodo, https://doi.org/10.5281/zenodo.23243093
(see also [CITATION.cff](CITATION.cff)). Please also cite ImageCAS, which this work builds upon.

## 10. Licence and acknowledgements

The code is released under the [MIT licence](LICENSE). Funding: Italian Ministry of University and Research (MUR),
PE00000013 FAIR; Fondo di Beneficenza Intesa Sanpaolo, grant B/2023/0201. Developed at Politecnico di Milano.
