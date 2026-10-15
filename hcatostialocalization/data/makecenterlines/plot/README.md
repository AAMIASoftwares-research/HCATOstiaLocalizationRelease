# makecenterlines/plot: flattened coronary trees

This module draws a patient's coronary centerline graph, and optionally its segmentation mask, unfolded onto a plane.
The view is from superior: **anterior up, the patient's right on the right**. Each tree starts at its ostium, the
branches spread out without overlapping, and vessel lengths stay meaningful. It is meant for figures (paper, website)
and for interactive visualization (HCATNetwork).

| File | Content | Depends on |
|---|---|---|
| `flatten.py` | **Geometry only**: surface fit, the three unfolding methods, carrying 3D points (mask voxels) along, metrics | numpy, networkx |
| `draw.py` | **Drawing only**: one matplotlib axis per flattened graph, with its elements and style | matplotlib, `flatten.py` |
| `__main__.py` | **Command line**: reads GML files, masks or mockup masks; writes four images per case (the three steps, the whole tree, the left and the right tree) and a summary grid | everything above; `utils/grid.py` (MONAI) only for real masks |

`flatten.py` imports nothing beyond numpy and networkx, and takes the position attribute names as parameters. It can
therefore be copied into another project (HCATNetwork, a web exporter) unchanged.

---

## 1. Conventions

- **Input:** a `networkx` graph whose nodes carry a position in **RAS millimetres** (`x`, `y`, `z` by default), a
  `topology` (`OSTIUM`, `SEGMENT`, `INTERSECTION`, `ENDPOINT`) and a `side` (`LEFT`, `RIGHT`), and optionally a
  radius `r` (mm).
  - These are the attributes of the HCATNetwork `SimpleCenterlineGraph` that the v2 extraction writes, and of its
    GML files read with `networkx.read_gml(path, label="id")`.
  - Enum values (`ArteryNodeTopology.OSTIUM`) are accepted as well as strings.
  - The unfolding needs exactly the ostium nodes; one per tree is expected.
- **Output:** every node of the returned graph copy gets `flat_x`, `flat_y` (mm).
- **Orientation:** screen right = R and screen up = A, so out of the screen is R × A = S. This is a true view from
  superior, not a mirrored one. The plain projection is literally `(flat_x, flat_y) = (x, y)`.
- **Units:** millimetres everywhere, including the flat plane, so a scale bar in mm is meaningful.

## 2. How the flattening is made (`flatten.py`)

### 2.1 `projection`
The axial projection, `(x, y)`. Exact for vessels lying in a horizontal plane, foreshortened for vessels running up
or down the heart, and branches cross (0 to 13 crossings on the test cases, 7 and 13 in the two worst). It is the baseline the other two methods
are compared against; the "Top view" panel of the extraction's `targets_mip.png` debug figure is the same thing.

### 2.2 `epicardial` (main method)
The coronaries run on the epicardium, so the trees lie roughly on a closed surface around the heart. The method fits
that surface, then unrolls it about the aortic root.

1. **Surface fit** (`fit_unfolding`, `surface=`):
   - `"sphere"`: linear least-squares sphere through all the nodes (`fit_sphere`), giving centre `c` and radius `ρ`.
     The **pole** (axis direction) is the unit vector from `c` to the midpoint of the two ostia, i.e. the aortic root.
   - `"spheroid"` (default): an ellipsoid of revolution about the same line (from the sphere's centre toward the
     aortic root), so the heart can come out elongated or flattened along the base–apex direction (`fit_spheroid`).
     - With `r` the distance of a node from the axis and `z` its position along it, the spheroid
       `r²/a² + (z − h)²/b² = 1` is rewritten as `r² = P z² + Q z + R`, which is linear in `(P, Q, R)`. The solution
       gives `β = a²/b² = −P`, the centre offset `h = Q / (2β)` and `a² = R + β h²`.
     - So the centre may slide along the axis, `a` is the radius across the axis and `b` the radius along it.
     - The axis itself is fixed, so the pole stays on the aortic-root side and the ostia stay at the centre of the
       map.
     - If the fit is not a spheroid (`β ≤ 0` or `a² ≤ 0`), the sphere is used.
     - A fully free ellipsoid was tried and dropped: on near-spherical hearts its axis can land far from the aortic
       root, which breaks the map.
2. **Fit size** (`fit=`):
   - `"least-squares"` keeps the fitted surface.
   - `"inner"` (default) shrinks it about its centre until no node lies inside it: the factor is the smallest
     normalized radius `√((r/a)² + (z/b)²)` over the nodes, 0.62–0.78 on the test cases.
   - Note: the map depends only on each point's *angular* position (see 3. below), so the inner fit **does not change
     the shape of the drawing**. It only scales it, so that flat lengths come out shorter than the 3D ones, never
     longer.
3. **Meridian map** (`unfold_points`):
   - Each point `p` gets the parametric polar angle `t = atan2(r/a, z/b)` (0 at the pole, π at the opposite end of
     the axis) and its longitude `φ` around the axis.
   - It is drawn at distance `s(t)` from the centre of the map, where `s` is the **arc length along the meridian from
     the pole**, `s(t) = ∫₀ᵗ √(a² cos²τ + b² sin²τ) dτ` (tabulated numerically, `_meridian`). The angle is `φ`.
   - `φ` is measured in the plane perpendicular to the axis, in the basis `up` = the anterior direction (+A)
     projected onto that plane, and `right = up × pole`. So, seen from outside the pole, A is up and R is right.
   - For the sphere this is exactly the azimuthal equidistant projection, `s = ρ θ`.
   - The result is a bull's-eye map: the aortic root at the centre, the apex toward the rim. Vessels on the far side
     of the heart unwrap instead of overlapping, giving 0 or 1 crossing on the test cases.
4. **Distortion:**
   - Along the meridians, lengths are exact on the surface.
   - Across them, the map stretches by `s(t) / (a sin t)`. This is 1 near the pole and grows toward the opposite end
     of the axis, where the map is **singular**: that single point becomes the whole rim.
   - Vessels close to the lower half of the axis are therefore blown up. In test case 4, the distal RCA passed 0.8 mm
     from it and turned into a ring, with an edge stretched 95 times.
5. **Axis clearance** (`axis_clearance_mm=10`, `clear_axis`):
   - If a node lies closer than this to the **lower half of the axis** (the half-line from the centre away from the
     aortic root, `lower_axis_clearance`), the axis is tilted away from the aortic root by the smallest angle that
     clears it.
   - The search tries cones of 1°, 2°, … (up to 45°) around the original axis, 72 directions each. At the first angle
     where some direction clears, it keeps the direction with the largest clearance.
   - The spheroid is then refitted along the new axis, and the check is repeated, since the centre may slide (at most
     3 times).
   - The ostia then sit slightly off the centre of the map.
   - In case 4: a 16° tilt, and the largest stretch went from 95× to 6.6×. The other test cases already had
     10–17 mm of clearance and were left untouched.
   - Set `axis_clearance_mm=0` to disable it.

### 2.3 `length-true`
The epicardial layout rebuilt from each ostium (`_length_true`): for every edge, in breadth-first order from the
ostium, the child is placed at the parent's flat position plus the **edge's flat direction** times its **3D length**.

- Vessel lengths and the distances along the trees are exact: the edge distortion is 1.00.
- The branching pattern and directions come from the epicardial map.
- Since lengths are restored, a few crossings may reappear (1 in test case 4).
- The epicardial positions are kept as the graph attribute `flat_before`, which `carry_points` needs.

### 2.4 Carrying the mask (or any 3D point set) along (`carry_points`)
- **`projection`, `epicardial`:** every point goes through the same map as the nodes, so the mask is deformed
  exactly as the vessels are. Where the map stretches, a vessel looks wider (distally, toward the rim).
- **`length-true`:** each point is placed from its **nearest node**, in the direction the epicardial map gives, at
  its **true distance** from the node. That distance is the component of the 3D offset tangent to the surface (the
  part along the radius from the centre is dropped). So **the vessel thickness is not deformed**: the mask is a band
  of the true vessel width everywhere.
- **`group=k`:** the points come in consecutive groups of `k` that share the nearest node of the first one. The
  command line uses it for the 8 sub-samples of each voxel (§3.3), which makes the nearest-node search 8 times
  faster.

### 2.5 Metrics (in every panel title)
- `count_crossings`: pairs of non-adjacent edges whose flat segments intersect.
- `edge_distortion`: flat length / 3D length of every edge. The title gives the median and the 95th percentile.
- Fit residual: the median distance of the nodes from the least-squares surface, mm.
- `Unfolding.tilt_deg` and `Unfolding.axis_clearance_mm`.

### 2.6 API summary
```python
from hcatostialocalization.data.makecenterlines.plot import flatten

flat_graph, unfolding = flatten.unfold_graph(graph, method="epicardial", surface="spheroid", fit="inner",
                                             keys=("x", "y", "z"), up_hint=(0, 1, 0), axis_clearance_mm=10.0)
nodes, xy = flatten.flat_positions(flat_graph)                         # (N, 2), mm
mask_xy = flatten.carry_points(mask_ras, flat_graph, unfolding)        # (M, 2) for (M, 3) RAS points
xy_any = flatten.unfold_points(unfolding, points_ras)                  # the bare map, any point set
sides = flatten.point_sides(mask_ras, graph)                           # "LEFT"/"RIGHT" of each point's nearest node
left = flat_graph.subgraph([n for n, d in flat_graph.nodes(data=True) if d["side"] == "LEFT"])   # one tree
flatten.count_crossings(flat_graph), flatten.edge_distortion(flat_graph), flatten.distance_from_ostium(graph)
```
- `Unfolding` is a dataclass (centre, `a`, `b`, pole, right, up, fit, scale, tilt). Fit it once and apply it to
  anything in the same RAS space: other graphs of the same patient, the mask, landmarks, a fluid-simulation domain.
- `up_hint` is the 3D direction shown up. It defaults to anterior; `(0, 0, 1)` shows superior up, for example.

## 3. How it is plotted (`draw.py`)

`draw_flat(ax, flat_graph, mask_xy=None, elements=..., distance_colormap=False, show_grid=False, invert_mask=False,
title=None, empty=False)` draws on any matplotlib axis. With `empty=True`, only the subject is drawn: no axes, ticks,
grid, compass, scale bar or title. The figure background is white, or black for an inverted mask drawn alone. `elements` is a key of `ELEMENT_SETS`, or a set of `"graph"`, `"ostia"`,
`"intersections"`, `"endpoints"` and `"mask"`.

### 3.1 Elements and style (`STYLE`, `CMAP`)
| Element | Look |
|---|---|
| edges (`graph`) | black, thin; width proportional to the mean radius of the two nodes (0.6 pt per mm), clipped to 0.3–2.0 pt |
| ostia | forest green dot (70 pt²), edge **firebrick** for the right ostium and **navy** for the left |
| intersections (bifurcations) | electric-blue fill `#7DF9FF` with a `#0892D0` edge, 18 pt² |
| endpoints | small firebrick dots, 8 pt² |
| mask | transparent gray `(0.45, 0.45, 0.45, 0.35)` under the centerline edges; **black on white** when no edges are drawn (`mask`, `mask-ostia`); white on black with `invert_mask` |
| `distance_colormap` | edges coloured by 3D path distance from their ostium, normalized per tree: a red ramp for the right tree (`#f4a582` → `#67001f`), a blue ramp for the left (`#92c5de` → `#053061`), both starting away from white |
| `show_grid` | mm ticks and light grid lines; otherwise no axes |
| always | compass letters A (top), P, R (right), L, and a 20 mm scale bar |

To change the look, edit `STYLE` or `CMAP`; they are module-level dictionaries.

### 3.2 Drawing the mask
- The flat mask points are binned into a 2D occupancy image with 0.5 mm bins (`numpy.histogram2d`).
- The image is then closed with a 3×3 square (one bin: dilation, then erosion, done with array shifts and no scipy).
  This fills the small gaps the map opens where it stretches the voxels apart.
- The image is drawn with `imshow` under the vector elements.

### 3.3 Dense sampling of the mask (command line)
Before unfolding, every 0.5 mm voxel centre is replaced by 2×2×2 sub-samples at ±0.125 mm (`densify`). The stretched
parts of the map then still receive enough points to look solid. Combined with the one-bin closing, the mask stays
solid even under the epicardial map's distal stretch.

## 4. Command line (`__main__.py`)

Run from the repository root:
```bash
python -m hcatostialocalization.data.makecenterlines.plot --gml <a.gml> [<b.gml> ...] --out <dir> [options]
```
| Option | Default | Meaning |
|---|---|---|
| `--gml` | (required) | one or more centerline graphs (v2 output, `<run>/<dataset>/graph/<id>.centerlines.gml`) |
| `--mask` | none | one segmentation mask per graph, same order; loaded with `utils/grid.py::load_mask_ras_isotropic` at 0.5 mm, RAS (MONAI: cluster) |
| `--mockup-mask` (alias `--mock-mask`) | off | draws a **clean, smooth mockup mask** built from the graph instead of the real one, even when `--mask` is given (a ball of the node radius every 0.25 mm along every edge, radius interpolated, 0.5 mm grid). Needs no image and no MONAI |
| `--images` | `steps tree left right` | which images to write (any subset), see Outputs |
| `--method` | `length-true` | method of the `tree`, `left` and `right` images: `length-true`, `epicardial` or `projection` (the `steps` image always shows all three) |
| `--surface` | `spheroid` | `spheroid` or `sphere` |
| `--fit` | `inner` | `inner` or `least-squares` |
| `--axis-clearance-mm` | 10 | tilt the axis away from nodes closer than this to its lower half; 0 never tilts |
| `--elements` | `graph-landmarks` | `graph` (edges), `graph-ostia`, `graph-landmarks` (+ bifurcations and endpoints), `mask`, `mask-ostia` (black mask + ostia, no edges), `all` (mask + `graph-landmarks`) |
| `--landmarks` | from `--elements` | exactly these landmarks, any of `ostium`, `intersection`, `endpoint`, replacing those of `--elements` (`--landmarks` alone = none); file names get `_lm-<landmarks>`, e.g. `mask_lm-ostium-endpoint` |
| `--empty` | off | `tree`, `left`, `right` images: only the subject, no axes, ticks, grid, compass, scale bar or title, tightly cropped (the `steps` image keeps its titles) |
| `--distance-colormap` | off | colour the edges by distance from the ostium (with `--empty` too: the colours belong to the subject) |
| `--show-grid` | off | mm grid and ticks |
| `--invert-mask` | off | mask alone: white on black instead of black on white |
| `--grid R,C` | auto | layout of the summary grid when several graphs are given (default: as square as possible) |
| `--dpi` | 200 | resolution |

**Outputs**

Each case gets four images by default, in `<out>/<case>/`:

| Image | File | Content |
|---|---|---|
| `steps` | `<case>_steps_<surface>-<fit>_<elements>[...].png` | the three methods side by side, with their metrics: how the final image is built |
| `tree` | `<case>_<method>_tree_<surface>-<fit>_<elements>[...].png` | both trees, with `--method` |
| `left`, `right` | `<case>_<method>_left_...png`, `<case>_<method>_right_...png` | each tree alone, in the same flat coordinates as the `tree` image (same fit, same orientation and scale). The mask is split by the side of each voxel's nearest node |

- The file names end with `_distance`, `_grid`, `_inverted` and `_empty` when those options are on.
- With more than one `--gml`, there is also `<out>/grid_epicardial_<surface>-<fit>_graph-ostia_distance.png`: every
  case side by side, epicardial, edges and ostia only, coloured by distance from the ostium.

**Examples**
```bash
# five cases, the four images each (steps, tree, left, right) with the clean mockup mask; plus the summary grid
python -m hcatostialocalization.data.makecenterlines.plot \
    --gml <run>/ImageCAS/graph/{1,2,3,4,5}.centerlines.gml --out flat/ --elements all --mockup-mask

# subject-only images for a figure or the website: tree, left, right, nothing else on them
python -m hcatostialocalization.data.makecenterlines.plot --gml <...>/1.centerlines.gml --out flat/ \
    --images tree left right --elements all --mockup-mask --empty

# the real masks (on the cluster), length-true, mask only: black vessels on white, true vessel width
python -m hcatostialocalization.data.makecenterlines.plot \
    --gml <run>/ImageCAS/graph/1.centerlines.gml --mask <data>/ImageCAS/Data/1.label.nii.gz \
    --out flat/ --images tree --elements mask

# black mask split into left and right, only the ostia on it, nothing else
python -m hcatostialocalization.data.makecenterlines.plot --gml <...>/{1,2,3}.centerlines.gml --out flat/ \
    --images left right --elements mask-ostia --mockup-mask --empty

# black mask split into left and right, with the ostia and the endpoints only
python -m hcatostialocalization.data.makecenterlines.plot --gml <...>/{1,2,3}.centerlines.gml --out flat/ \
    --images left right --elements mask --landmarks ostium endpoint --mockup-mask --empty

# a figure-style tree: edges coloured by distance, ostia only, grid on
python -m hcatostialocalization.data.makecenterlines.plot --gml <...>/7.centerlines.gml --out flat/ \
    --elements graph-ostia --distance-colormap --show-grid

# summary grid of 12 cases on 3 rows and 4 columns
python -m hcatostialocalization.data.makecenterlines.plot --gml <...>/{1..12}.centerlines.gml --out flat/ --grid 3,4
```

**From Python, e.g. with an HCATNetwork graph:**
```python
import matplotlib.pyplot as plt
from hcatnetwork.io.io import load_graph
from hcatnetwork.graph.graph import SimpleCenterlineGraph
from hcatostialocalization.data.makecenterlines.plot import flatten, draw

g = load_graph("1.centerlines.gml", output_type=SimpleCenterlineGraph)
flat, unfolding = flatten.unfold_graph(g, "length-true")
fig, ax = plt.subplots(figsize=(6, 6))
draw.draw_flat(ax, flat, elements="graph-landmarks", distance_colormap=True)
fig.savefig("1_flat.png", dpi=200)
```

## 5. Status, validation and limitations

- **Prototype, 2026-10-07.** Tested on the laptop only, on five real v2 graphs (ImageCAS cases 1–5) with mockup
  masks. The real-mask path (`--mask`) still has to be run on the cluster.

| Measure on the test cases | Projection | Epicardial | Length-true |
|---|---|---|---|
| Crossings | 0–13 | 0–1 | 0–1 |
| Edge length (median) | ×0.87–0.95 | ×0.63–0.81 (inner fit; about ×1.1 with least-squares) | ×1.00 |

- **Spheroid vs sphere:** every fitted spheroid is elongated along the base–apex axis (`b/a` 1.08–1.31). It fits
  the trees better than the sphere in 3 of 5 cases (residual 3.8–5.8 mm against 3.6–5.9 mm), but changes the
  distortion little. The axis clearance and the length-true thickness are what
  matter for the look.
- **Real masks** contain stumps, fragments and touching trees that the mockup does not have. A mask voxel is always
  attached to its nearest node, so fragments far from the tree are drawn near that node under `length-true`.
- **Speed:** a few seconds per case and method on the laptop. The nearest-node search is brute force in chunks; use
  `group` for dense point sets.
- Any map centred on the aortic root has a singular point opposite it (§2.2, 4.); the axis clearance keeps the
  vessels away from it, but cannot remove it.

## 6. Ideas for later
- An export of the flat coordinates and radii (JSON or SVG) for the website, and a 2D vessel domain for a fluid or
  flow animation along the length-true trees.
- Moving `flatten.py` into HCATNetwork (e.g. a visualization module of `SimpleCenterlineGraph`); this project would
  then import it from there.
- Before citing in the paper: literature on coronary flattening and bull's-eye maps (e.g. CoViCAD, Termeer et al.,
  2007) still needs verifying.
