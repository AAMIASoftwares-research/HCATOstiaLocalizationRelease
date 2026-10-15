# ImageCAS coronary centerlines

Centerline graphs of the left and right coronary trees for the 1000 CCTA scans of ImageCAS. Part of the dataset
[10.5281/zenodo.23243094](https://doi.org/10.5281/zenodo.23243094) (`Centerlines.zip`); the ostia are in `Ostia.zip`. Each tree is traced from
the annotated coronary ostium through the coronary segmentation mask.

| Folder | Traced on | Cases |
|---|---|---|
| `centerlines_from_ImageCAS_masks/` | the official ImageCAS segmentation masks | 1000 scans, 999 graphs |
| `centerlines_from_ImageCAS-X_masks/` | the curated ImageCAS-X masks | the 800 scans included in ImageCAS-X |

The **ImageCAS-mask centerlines are the official set**. The ImageCAS-X-mask ones are provided as an alternative, for
users who work with the ImageCAS-X masks, and as a fallback where the official mask is defective. `cases.csv` says, for
every scan, which graph to use (`recommended`) and why.

### Which set to use

The two sets are traced with the same procedure from the same ostia; they differ only in the mask they follow, so the
first rule is to **use the centerlines of the masks you work with**, so that trees and masks agree voxel by voxel:
- **ImageCAS masks** (all 1000 scans): use `centerlines_from_ImageCAS_masks/`. It is the only set covering the 200
  scans that ImageCAS-X excludes, and its trees follow the distal vessels further, especially on the inferior surface
  of the heart.
- **ImageCAS-X masks** (800 scans): use `centerlines_from_ImageCAS-X_masks/`. Its masks are more regular, and their
  trees tend to be shorter distally.
- **One graph per scan, whatever the mask** (e.g. for centerline or labelling methods that take the image only): use
  the `recommended` column of `cases.csv`, which keeps the official graph unless the quality checks or the visual review
  found a problem in it that the alternative graph does not have (96 scans).

Each scan has:
- `gml/<id>.centerlines.gml`: the full graph (GML), to be read with networkx or HCATNetwork (examples below);
- `vtk/<id>.graph.vtk` and `vtk/<id>.paths.vtk`: the same trees as VTK polydata, for ParaView, VMTK, 3D Slicer or any
  VTK-based tool (below);
- `slicer/<id>/`: the same trees as 3D Slicer markups, for viewing.

## Coordinates and conventions

- **Positions** are in **millimetres, in RAS world coordinates** of the ImageCAS image of the same scan, the
  coordinates 3D Slicer shows when the image is loaded. ImageCAS-X shares the geometry of the ImageCAS images.
- **Two trees per scan**, LEFT and RIGHT, each starting at its ostium node. Each tree is acyclic and the two never share
  a node, so a graph has exactly two connected components.
- **Nodes** are about 0.5 mm apart along every branch. The ostium node is the annotated ostium itself, on the aortic
  wall.

## GML content

Graph attributes: `image_id` (the ImageCAS case id), `are_left_right_disjointed`, `creation_datetime`, plus the
type-conversion dictionaries that HCATNetwork uses to restore its attribute types.

| Node attribute | Meaning |
|---|---|
| `x`, `y`, `z` | position, mm, RAS |
| `r` | lumen radius, mm: distance from the mask boundary. 0 at the ostium node. |
| `t` | time, always 0 (static image) |
| `topology` | `OSTIUM` (one per tree), `SEGMENT` (two neighbours), `INTERSECTION` (bifurcation), `ENDPOINT` (distal tip) |
| `side` | `LEFT` or `RIGHT` |

| Edge attribute | Meaning |
|---|---|
| `weight`, `euclidean_distance` | length of the edge, mm |

## Reading the graphs

### With networkx only

```python
import networkx as nx
import numpy as np

g = nx.read_gml("centerlines_from_ImageCAS_masks/gml/1.centerlines.gml", label="id")

ostia = {d["side"]: n for n, d in g.nodes(data=True) if d["topology"] == "OSTIUM"}
for side, ostium in ostia.items():
    tree = g.subgraph(nx.node_connected_component(g, ostium))
    length = sum(d["euclidean_distance"] for _, _, d in tree.edges(data=True))
    endpoints = [n for n, d in tree.nodes(data=True) if d["topology"] == "ENDPOINT"]
    print(f"{side}: {length:.0f} mm, {len(endpoints)} endpoints")

    # one ostium-to-tip path as an (N, 3) array of RAS points, with radii
    path = nx.shortest_path(tree, ostium, endpoints[0])
    xyz = np.array([[g.nodes[n]["x"], g.nodes[n]["y"], g.nodes[n]["z"]] for n in path])
    radius = np.array([g.nodes[n]["r"] for n in path])
```

### With HCATNetwork

[HCATNetwork](https://github.com/AAMIASoftwares-research/HCATNetwork) is the library the graphs were written with. It
restores the typed attributes (`ArteryNodeTopology`, `ArteryNodeSide`) and adds coronary-specific operations.

```bash
pip install git+https://github.com/AAMIASoftwares-research/HCATNetwork.git
```

```python
from hcatnetwork.io.io import load_graph
from hcatnetwork.graph.graph import SimpleCenterlineGraph
from hcatnetwork.node.node import ArteryNodeTopology, ArteryNodeSide

g = load_graph("centerlines_from_ImageCAS_masks/gml/1.centerlines.gml", output_type=SimpleCenterlineGraph)

print(g.get_coronary_ostia_node_id())          # ids of the two ostium nodes
for start, end in g.get_anatomic_segments():    # ostium/bifurcation/endpoint to the next one
    print(start, "->", end)

left = [n for n, d in g.nodes(data=True) if d["side"] == ArteryNodeSide.LEFT]
tips = [n for n, d in g.nodes(data=True) if d["topology"] == ArteryNodeTopology.ENDPOINT]

g1 = g.resample(mm_between_nodes=1.0)           # the same trees, one node per mm
```

## Loading the images with MONAI (quick start)

The graphs live in the RAS world coordinates of the ImageCAS images (`<id>.img.nii.gz`; masks `<id>.label.nii.gz`).
To relate them to voxels, load the image with MONAI, reorient it to RAS and use its affine:

```python
import numpy as np
import networkx as nx
from monai.transforms import LoadImage, Orientation

image = LoadImage(image_only=True, ensure_channel_first=True)("ImageCAS/1.img.nii.gz")
image = Orientation(axcodes="RAS")(image)        # MetaTensor (1, I, J, K): i grows towards R, j towards A, k towards S
volume = image[0].numpy()                        # HU, indexed volume[i, j, k]
ijk_to_ras = image.affine.numpy().astype(float)  # 4x4: voxel index (i, j, k) -> RAS mm
ras_to_ijk = np.linalg.inv(ijk_to_ras)           # 4x4: RAS mm -> continuous voxel index

def ijk2ras(ijk):
    ijk = np.atleast_2d(ijk).astype(float)
    return (ijk_to_ras @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]

def ras2ijk(ras):
    ras = np.atleast_2d(ras).astype(float)
    return (ras_to_ijk @ np.c_[ras, np.ones(len(ras))].T).T[:, :3]

g = nx.read_gml("centerlines_from_ImageCAS_masks/gml/1.centerlines.gml", label="id")
nodes = list(g.nodes)
ras = np.array([[g.nodes[n]["x"], g.nodes[n]["y"], g.nodes[n]["z"]] for n in nodes])
ijk = ras2ijk(ras)                               # continuous indices into volume
voxel = np.rint(ijk).astype(int)                 # nearest voxel of each node
hu_on_centerline = volume[voxel[:, 0], voxel[:, 1], voxel[:, 2]]
assert np.allclose(ijk2ras(ijk), ras)            # and back
```

The indices refer to the RAS-reoriented array. For the array exactly as stored in the file, skip `Orientation` and use
the affine returned by `LoadImage` (MONAI also gives it in RAS). The same RAS coordinates are those of nibabel and of
3D Slicer, so no sign flip is needed; tools working in LPS (ITK, SimpleITK) need x and y negated.

## VTK

Legacy ASCII VTK polydata, positions in RAS mm (the header carries `SPACE=RAS`, so 3D Slicer loads them in the right
place).

| File | Content |
|---|---|
| `<id>.graph.vtk` | the undirected graph: one point per node, one 2-point line per edge. Point data: `radius_mm`, `topology` (1 ostium, 2 segment, 3 bifurcation, 4 endpoint), `side` (1 left, 2 right), `node_id` (the node id in the GML file) |
| `<id>.paths.vtk` | one polyline per ostium-to-endpoint path, like the Slicer open curves (the shared proximal part of a tree is repeated in each of its paths). Point data: `radius_mm`, `topology`, `node_id`, `distance_from_ostium_mm`; cell data: `side`, `endpoint_node_id`, `length_mm` |

```python
import pyvista as pv   # or vtk.vtkPolyDataReader, with ReadAllScalarsOn() to get every point/cell array

paths = pv.read("centerlines_from_ImageCAS_masks/vtk/1.paths.vtk")
for i in range(paths.n_cells):
    cell = paths.get_cell(i)
    xyz = cell.points                                  # (N, 3) RAS points, ostium first
    radius = paths["radius_mm"][cell.point_ids]
    print(i, paths.cell_data["side"][i], f"{paths.cell_data['length_mm'][i]:.0f} mm")
```

## 3D Slicer

`slicer/<id>/` holds:
- one open curve (`*.SlicerOpenCurve.mkr.json`) for every ostium-to-endpoint path, so curves of the same tree overlap
  along their shared proximal part;
- `<id>_notable_landmarks.mrk.json`, a point list of the ostia, bifurcations and endpoints. Each point is named with
  its side, its kind, its rank along the tree (ordered by distance from the ostium along the tree) and, in
  parentheses, the id of the corresponding node in the GML, e.g. `Left ostium (924)`, `Left bifurcation 2 (412)`,
  `Right endpoint 3 (1383)`.

Drag and drop the files into 3D Slicer together with the ImageCAS image. The curves can be used directly, for example
in the Curved Planar Reformat module.

## cases.csv

One row per ImageCAS scan:

| Column(s) | Content |
|---|---|
| `case_id` | ImageCAS case id |
| `recommended`, `why_recommended` | `ImageCAS` (default) or `ImageCAS-X`. ImageCAS-X is recommended only where the official graph is missing, failed the quality checks, may miss branches, or has its left and right trees touching in the mask, and the ImageCAS-X graph is free of all three; the visual review can override this (`why_recommended` = `expert review`). |
| `in_imagecas_x` | whether the scan is one of the 800 in ImageCAS-X |
| `gml_*`, `slicer_*` | paths of the files of each set, relative to this folder; empty when there is no graph |
| `imagecas_*`, `imagecas_x_*` | for each set: `status` (`ok`, `extraction failed`, `not in ImageCAS-X`); `qc` from the automatic quality checks (`clean`; `check`, a minor point worth a look; `left and right trees touch`, the two trees meet in the mask, which is anatomically wrong; `possible missed branch`, part of the mask not reached by the tree; `failed`); `qc_notes`, the details; `short_branches`, the accepted branches shorter than 10 mm (length and tip position, RAS), for users who prefer to prune them; and the length of each tree, mm |
| `recall_vs_imagecas_x_reference`, `precision_vs_imagecas_x_reference`, `missing_length_vs_imagecas_x_reference_mm` | agreement of the ImageCAS-mask graph with the expert centerlines distributed with ImageCAS-X. Disagreement mostly reflects the differences between the two masks. |
| `comment` | notes from the visual review of specific scans |

Every scan flagged by the quality checks was reviewed visually; `comment` records what was found (for example a
confirmed missed branch, or which part of one tree the other one took where the masks touch). A random sample of 50 unflagged scans
was also reviewed, and no missed branch was found in it. As a general guide: the official ImageCAS masks often carry irregular protrusions, short spurious
stubs and implausible fragments, which the extraction keeps out of the trees, and most of their problems lie in the
distal vessels of the inferior part of the heart. The ImageCAS-X masks are more regular, but follow those distal
vessels less often, so their trees tend to be shorter there.

## Flattened views of the trees (code)

The code of the paper, including the extraction and a tool that unfolds each coronary tree (and its mask) onto a plane
seen from superior, is linked from the project page, https://hal9000-lab.github.io/Awakening-ImageCAS. Install it
from the repository and run, for example:

```bash
pip install "git+<repository URL from the project page>"
# black mask on white, split into left and right, with the ostia and the endpoints only
python -m hcatostialocalization.data.makecenterlines.plot \
    --gml centerlines_from_ImageCAS_masks/gml/1.centerlines.gml --mask ImageCAS/1.label.nii.gz \
    --out flat/ --images left right --elements mask --landmarks ostium endpoint --empty
```

`--mockup-mask` draws a smooth mask built from the graph instead of the real one (no image needed). The geometry is
usable on its own, with numpy and networkx only:

```python
import networkx as nx
from hcatostialocalization.data.makecenterlines.plot import flatten

g = nx.read_gml("centerlines_from_ImageCAS_masks/gml/1.centerlines.gml", label="id")
flat, unfolding = flatten.unfold_graph(g, "length-true")   # nodes gain flat_x, flat_y (mm)
```

See `data/makecenterlines/plot/README.md` in the repository for every option.

## Licence and citation

The centerlines (both sets), `cases.csv` and the landmark files are released under the
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** licence; the code is released under the
MIT licence. The CCTA images and the segmentation masks are not included: obtain them from ImageCAS and ImageCAS-X,
under their own terms. If you use these centerlines, please cite:
- this dataset: Leccardi M., Moglia A., Morandini M., Brembilla M., Nicosia A., Penserini A., Brandini C., Marcon M.,
  Mainardi L., Cerveri P., *Awakening ImageCAS: coronary ostia and centerline annotations for ImageCAS*, Zenodo, 2026,
  doi:10.5281/zenodo.23243094;
- the paper: Leccardi M., Moglia A., Morandini M., Brembilla M., Nicosia A., Penserini A., Brandini C., Marcon M.,
  Mainardi L., Cerveri P., *Awakening ImageCAS: Automated Coronary Ostia Localization Framework and Centerlines Open
  Dataset* (under review; the up-to-date reference is on the project page);
- ImageCAS: Zeng A. et al., *ImageCAS: A large-scale dataset and benchmark for coronary artery segmentation based on
  computed tomography angiography images*, Computerized Medical Imaging and Graphics, 2023,
  doi:10.1016/j.compmedimag.2023.102287;
- for the ImageCAS-X-mask set, also ImageCAS-X: Bransby K. M. et al., *ImageCAS-X: a dataset and benchmark for coronary
  artery segmentation and centerline extraction in coronary CT angiography*, arXiv:2608.30404, 2026; data:
  Bransby K. M., Paulsen R. R., ImageCAS-X, Zenodo, doi:10.5281/zenodo.21887809.
