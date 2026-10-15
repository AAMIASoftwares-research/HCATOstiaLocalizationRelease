# Awakening ImageCAS: coronary ostia and centerline annotations for ImageCAS

DOI: [10.5281/zenodo.23243094](https://doi.org/10.5281/zenodo.23243094)

Annotations that extend [ImageCAS](https://doi.org/10.1016/j.compmedimag.2023.102287), the public dataset of 1000
coronary CT angiography (CCTA) scans with coronary artery segmentation masks. They are released with the paper
*Awakening ImageCAS: Automated Coronary Ostia Localization Framework and Centerlines Open Dataset*. Code, model weights
and documentation are linked from the project page: https://hal9000-lab.github.io/Awakening-ImageCAS

| File | Content |
|---|---|
| `README.md` | this file |
| `Ostia.zip` | the left and right coronary ostium of all 1000 ImageCAS scans: one JSON file per scan (world and voxel coordinates, affine matrices) and one 3D Slicer markup per scan. Details in its `README.md`. |
| `Centerlines.zip` | the left and right coronary centerline trees, grown from the annotated ostia: the official set, traced on the ImageCAS masks (999 of 1000 scans), and an alternative set traced on the ImageCAS-X masks (800 scans); GML graphs (networkx, HCATNetwork), VTK polydata and 3D Slicer markups, plus `cases.csv` with the quality checks and the recommended graph of every scan. Details in its `README.md`. |

**Images and masks are not included.** Download the CCTA volumes and the segmentation masks from ImageCAS (and, for
the alternative centerlines, the masks from [ImageCAS-X](https://doi.org/10.5281/zenodo.21887809)), under their own
terms. Case `<id>` here is ImageCAS case `<id>` (`<id>.img.nii.gz`, `<id>.label.nii.gz`).

**Coordinates.** Every position is in millimetres, in the RAS world frame of the ImageCAS image of the same scan (the
frame of nibabel, MONAI and 3D Slicer; ITK and SimpleITK use LPS, i.e. x and y negated). Each README has a short
Python example to load an image with MONAI and convert between RAS and voxel indices.

## Licence

The annotations (ostia and centerlines, with their tables) are released under the Creative Commons
Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/).
The code linked from the project page is released under the MIT licence.

## Citation

If you use these annotations, please cite:
- this dataset: Leccardi M., Moglia A., Morandini M., Brembilla M., Nicosia A., Penserini A., Brandini C., Marcon M.,
  Mainardi L., Cerveri P., *Awakening ImageCAS: coronary ostia and centerline annotations for ImageCAS*, Zenodo, 2026,
  doi:10.5281/zenodo.23243094;
- the paper: Leccardi M. et al., *Awakening ImageCAS: Automated Coronary Ostia Localization Framework and Centerlines
  Open Dataset* (under review; the up-to-date reference is on the project page);
- ImageCAS: Zeng A. et al., *ImageCAS: A large-scale dataset and benchmark for coronary artery segmentation based on
  computed tomography angiography images*, Computerized Medical Imaging and Graphics 109, 102287, 2023,
  doi:10.1016/j.compmedimag.2023.102287;
- if you use the centerlines traced on the ImageCAS-X masks, also ImageCAS-X: Bransby K. M. et al., *ImageCAS-X: a
  dataset and benchmark for coronary artery segmentation and centerline extraction in coronary CT angiography*,
  arXiv:2608.30404, 2026; data: Bransby K. M., Paulsen R. R., ImageCAS-X, Zenodo, doi:10.5281/zenodo.21887809.

BibTeX entries are on the project page.

## Contact

Matteo Leccardi, Politecnico di Milano. Questions and corrections are welcome through the project page.
