# ImageCAS coronary ostia

The left and right coronary ostium of all 1000 CCTA scans of ImageCAS: the point where each coronary artery originates
from the aorta. Part of the dataset [10.5281/zenodo.23243094](https://doi.org/10.5281/zenodo.23243094); see the
top-level README for the licence (CC BY-NC 4.0) and the citations.

| Path | Content |
|---|---|
| `JSON/ImageCAS__<id>.ostium.json` | the two ostia of scan `<id>`, in world and voxel coordinates, with the affine matrices of the image (below) |
| `Slicer/ImageCAS__<id>.mrk.json` | the same two points as a 3D Slicer point list (labels `R` and `L`), to be loaded together with `<id>.img.nii.gz` |

## How the ostia were annotated

- **Scans 1-500**: annotated independently by six annotators in rotating groups of three (each scan by three
  annotators); the released point is the mean of the three annotations of each side, and every scan and consensus point
  was then checked visually.
- **Scans 501-1000**: a deep learning model, trained on the first 500 scans together with CAT08 and ASOCA, proposed the
  two points; one annotator inspected every proposal against the image and corrected it where needed. This is a
  lighter, model-assisted procedure, not an independent manual annotation.

`other.source` in each file says which procedure applies (`consensus of 3 annotator files`, or the reviewed markup
file). The test set of the paper is drawn from scans 1-500 only; the training, validation and test lists are in the
code repository (project page: https://hal9000-lab.github.io/Awakening-ImageCAS).

## JSON format

```json
{
  "version": "1.0",
  "image_id": "ImageCAS__1",
  "image_filename": "1.img.nii.gz",
  "pos": {
    "L": {"ras": [x, y, z], "ijk": {"sitk": [i, j, k], "monai-ras": [i, j, k]}},
    "R": {"ras": [x, y, z], "ijk": {"sitk": [i, j, k], "monai-ras": [i, j, k]}}
  },
  "affine_from_ijk_to_ras": {"sitk": 4x4, "monai-ras": 4x4},
  "affine_from_ras_to_ijk": {"sitk": 4x4, "monai-ras": 4x4},
  "comment": "...",
  "other": {"source": "...", "checks": [...]}
}
```

| Field | Meaning |
|---|---|
| `pos.<side>.ras` | the ostium (`L` left, `R` right), mm, RAS world coordinates of the image. **This is the reference value**; the others are derived from it. |
| `pos.<side>.ijk.sitk` | continuous voxel index (i, j, k) in the volume as stored in the file, i.e. as SimpleITK indexes it (`image.TransformPhysicalPointToContinuousIndex`); note that `sitk.GetArrayFromImage` returns the array as `[k, j, i]` |
| `pos.<side>.ijk.monai-ras` | continuous voxel index (i, j, k) in the volume loaded with MONAI and reoriented to RAS (`LoadImage` + `Orientation("RAS")`, below), indexed `volume[i, j, k]` |
| `affine_from_ijk_to_ras`, `affine_from_ras_to_ijk` | 4x4 matrices between those indices (homogeneous, column vector `[i, j, k, 1]`) and RAS mm, for each of the two index conventions |
| `comment`, `other.source` | provenance (see above) |
| `other.checks` | automatic remarks raised while assembling the file, kept for transparency (e.g. a left-right distance outside the usual range, or an annotator's left/right labels found swapped and corrected); empty for almost every scan |

Indices are continuous: round them (`numpy.rint`) to get the nearest voxel.

## Quick start (Python, MONAI)

```python
import json
import numpy as np
from monai.transforms import LoadImage, Orientation

ostia = json.load(open("JSON/ImageCAS__1.ostium.json"))
image = Orientation(axcodes="RAS")(LoadImage(image_only=True, ensure_channel_first=True)("ImageCAS/1.img.nii.gz"))
volume = image[0].numpy()                                    # indexed volume[i, j, k]
ras_to_ijk = np.linalg.inv(image.affine.numpy().astype(float))

for side in ("L", "R"):
    ras = np.array(ostia["pos"][side]["ras"])
    ijk = (ras_to_ijk @ np.append(ras, 1.0))[:3]               # same as ostia["pos"][side]["ijk"]["monai-ras"]
    i, j, k = np.rint(ijk).astype(int)
    print(side, ras.round(1), ijk.round(1), volume[i, j, k])

# or, without MONAI, with the matrices stored in the file:
A = np.array(ostia["affine_from_ras_to_ijk"]["monai-ras"])
```

With SimpleITK, use the `sitk` index and affine instead (the image is then not reoriented).
