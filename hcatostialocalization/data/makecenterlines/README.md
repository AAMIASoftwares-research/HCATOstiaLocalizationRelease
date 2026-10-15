# makecenterlines

Coronary centerline graphs from segmentation masks, rooted at the annotated ostia, and their
comparison against reference centerlines.

| Path | What it is |
|---|---|
| [`v2/`](v2/README.md) | **current pipeline**: root-driven minimal-path tracing, no skeletonisation |
| [`__v1/`](__v1/README.md) | legacy: first pipeline (skeleton + repairs), superseded by v2 (2026-10-01: simpler, much faster, finds the branches out of the box), kept for reference |
| `utils/` | shared code: `grid.py` (loading, frames, ostia), `convert.py` (radii, topology, HCATNetwork conversion, v1 smoothing, GML, Slicer markups), `smoothing.py` (v2 branch-wise smoothing, direction roughness), `checks.py`, `figures.py`, `cli.py` |
| `compare.py` | comparison of our graphs against ASOCA / ImageCAS-X (or any) reference centerlines, below |
| `collections/` | input list files, and `make_lists.sh` which builds them for ASOCA, ImageCAS and ImageCAS-X |
| `scripts/` | `run_extraction.sh`, `run_review.sh`, `run_compare.sh`: complete runs (see `v2/README.md`, Usage) |
| `release.py` | public release from a complete run: graphs (GML + Slicer) traced on the ImageCAS masks (official) and on the ImageCAS-X masks, `cases.csv` with the recommended graph per case, landmarks, networkx validation; see below |
| `inspection.py` | inspection report of a whole run, `<run>/inspection.html` (+ `.csv`): every case worth a visual check before the release (failed, touching trees, possibly missed branches, far ostia, bridged gaps, radius cap, ASOCA extras, flagged ImageCAS-X fallbacks), with sizes, RAS positions and the debug figures inline, linking the per-dataset reviews and comparisons. Run automatically at the end of `run_review.sh` and `run_compare.sh` |
| [`plot/`](plot/README.md) | flattened coronary trees (and masks) seen from superior: `projection`, `epicardial` (meridian map on a spheroid fitted about the aortic root) or `length-true` (exact vessel lengths, mask at its true width). `python -m hcatostialocalization.data.makecenterlines.plot`; geometry (`plot/flatten.py`, numpy + networkx only) separate from drawing (`plot/draw.py`). See its README |
| `release_README.md`, `release_case_notes.csv` | the README copied into the release; the expert notes per case (comment + optional recommendation override) |

All commands run from the repository root:
```bash
python -m hcatostialocalization.data.makecenterlines.v2      ...   # see v2/README.md
python -m hcatostialocalization.data.makecenterlines.__v1    ...   # legacy, see __v1/README.md
python -m hcatostialocalization.data.makecenterlines.compare ...
python -m hcatostialocalization.data.makecenterlines.release --run <run> --out <release folder>
```
**Release (2026-10-06).** The ImageCAS-mask graphs are the official set (the one described in the paper); the
ImageCAS-X-mask graphs are released too, as a fallback. `cases.csv` recommends the ImageCAS graph unless it is
missing, or the review found it failed, at risk of missed branches or with touching left and right trees, while the
ImageCAS-X one is free of all three;
`release_case_notes.csv` can override this. Every released GML is re-read with plain networkx (two ostia, two trees,
numeric positions); problems go to `validation.json` and make the script exit 1. Two VTK files per case
(`vtk/<id>.graph.vtk`, undirected graph; `vtk/<id>.paths.vtk`, one polyline per ostium-to-endpoint path). The Slicer
folder is the one written by the extraction, plus `<id>_notable_landmarks.mrk.json` (ostia, bifurcations, endpoints, named e.g.
"Left bifurcation 2 (412)", with the GML node id in parentheses).

**Viewing the inspection report from the cluster.** The HTML files use relative links, so serve the run folder and
open it in a local browser (Ctrl-C stops the server; pick another port if 8766 is taken):
```bash
cd <run> && python3 -m http.server 8766 --bind 0.0.0.0     # on login-node
# VPN connected:  http://cluster:8766/inspection.html
# otherwise, from the laptop:  ssh -L 8766:localhost:8766 <user>@<login node>
#                              then http://localhost:8766/inspection.html
```

Layout change (2026-10-01): `utils/` now holds only what v2 also uses; v1-only code moved to `v1/`, then renamed
`__v1/` (legacy) once v2 was chosen.
`python -m ...makecenterlines` alone no longer runs v1 (use `.v1`). The old `validate.py` stub was
replaced by `compare.py`.

---

## compare.py

**Status: written 2026-10-01. Tested on synthetic data and on the real reference files in
`data/makecenterlines/example_reference_centerlines/` (ASOCA Normal_1/2, ImageCAS-X 1/2), each against
an exact copy and against a copy with one segment deleted (all as expected). Not yet run against our own
v2 graphs: all tests ran on the laptop, without `vtk`, `scipy` or HCATNetwork, through the fallbacks.**

### The reference files (checked 2026-10-01 on the examples)

| | ASOCA (`<case>.vtp`) | ImageCAS-X (`<case>.coronary_{left,right}_centerline.vtk`) |
|---|---|---|
| encoding | XML PolyData, zlib-compressed binary | legacy VTK 5.1, binary |
| trees | both in one file | one file per side: join them on one list line, `left.vtk;right.vtk` |
| polylines | one per ostium-to-endpoint path (VMTK), no shared points, so proximal vessels repeat; merged by `--overlap-mm` (Normal_1: 10 lines, 1205 mm summed, 643 mm after merging) | one per segment, sharing the bifurcation points |
| root | the point most lines start from (= the ostium) | `start_points` flag |
| side | not stored: from `--ostia`, else the R coordinate of the two roots | from the file name |
| radius | `MaximumInscribedSphereRadius` → radius tolerance, OT | none → fixed 2 mm tolerance, no OT |
| other | point spacing 0.06 mm | point spacing ~0.4 mm; `segment_label` + `segment_name` per point → recall per anatomical segment |
| frame | positive coordinates on the scale of ASOCA's raw ostium frame: pass `--images` so `asoca-raw` is a candidate | to be confirmed by `auto` on the first real case |

### Usage

```bash
python -m hcatostialocalization.data.makecenterlines.compare \
    --pred  <ours.txt> --ref <reference.txt> \
    [--ostia <ostia.txt>] [--images <images.txt>] [--case-ids <ids.txt>] \
    -o ./logs/compare_centerlines/<run>/
```

Every list is one path per line (or paths inline), same case order. Case ids default to the
reference file names (`Diseased_1.vtk` → `Diseased_1`).

- `--pred`: our graphs. A path that does not exist (e.g. `-` on that line) is a **missing case**:
  it is flagged, never skipped silently.
- `--ref`: the ground truth. One line may join several files with `;` (e.g. one VTK per side).
- `--ostia` (optional, recommended for references without L/R labels): any ostium JSON that
  `io.io.extract_simplified_ras` reads. Used to give every component of an unlabelled graph its
  side and root, and to report root-to-ostium distances.
- `--images` (optional): only the headers are read. Needed for ASOCA's raw frame (below).

Example for ASOCA (our v2 run on the ASOCA masks, with the ASOCA ostia; `asoca_v2_graphs.txt`
lists our `.../graph/<case>.centerlines.gml`, `asoca_centerlines.txt` the ASOCA reference files,
same order; both lists still to be written):
```bash
python -m hcatostialocalization.data.makecenterlines.compare \
    --pred  asoca_v2_graphs.txt \
    --ref   asoca_centerlines.txt \
    --ostia hcatostialocalization/test/collections/ASOCA_all_labels.txt \
    --images hcatostialocalization/test/collections/ASOCA_all_images.txt \
    -o ./logs/compare_centerlines/asoca_v2/
```

### Formats (`io.io.load_centerline_graph`)

Both sides accept any of:

| Format | Read as |
|---|---|
| `.gml` | our `SimpleCenterlineGraph` (HCATNetwork reader when installed, else networkx): positions `x,y,z`, radius `r`, side, ostium nodes (`topology == OSTIUM`) |
| `.vtk` / `.vtp` | PolyData line cells (vtk package, installed with pyvista; a small ASCII legacy parser is the fallback). Radius from a point array named `MaximumInscribedSphereRadius`, `Radius`, `radius`, `r`. Side guessed from the file name (`left`, `LAD`, `RCA`, `_L_`, ...) |
| folder of `*.SlicerOpenCurve.mkr.json` (or one markup file) | one polyline per curve, LPS converted to RAS; side guessed from the markup / file name |

**Overlapping polylines.** ASOCA-style files store one polyline per ostium-to-endpoint path, so
the proximal vessels appear several times. `--overlap-mm` (default 1 mm) merges them into one
tree: polylines are added longest first, the parts within 1 mm of geometry already added are
dropped, and what remains is attached where it leaves the existing geometry. Without it, shared
trunks count several times (in the synthetic test recall rose from 83.9 % to 87.9 % and a missed
branch was mislabelled).

### Coordinate frames

Our graphs are RAS mm. A VTK file has no frame. `--ref-frame auto` (default) tries RAS, LPS and,
with `--images`, ASOCA's raw frame (the same one the ASOCA ostium files use: see
`io.io.asoca_raw_point_to_ras`), and keeps the one whose points lie closest to our graph (median
distance). Every candidate's median is written to `summary.json` (`cases[].frame`), and a case is
flagged `frame_warning` if the median is still above `--frame-warn-mm` (5 mm) afterwards. GML
and Slicer inputs are taken as RAS. Nothing is registered or shifted: a residual offset shows up
in the distances and is not hidden.

### Sides and roots

Per connected component, in order of preference: ostium nodes / side labels in the file; the
annotated ostia (`--ostia`: the component endpoint nearest an ostium is its root, the component
takes that side); for VTK, the node most polylines start from (ASOCA: every line starts at the
ostium); otherwise the two largest components are told apart by the root's R coordinate (right
ostium = larger R, `side_source = heuristic-x`). Small unlabelled pieces take the side of the
nearest labelled node. `summary.json` records the rule used (`ref_side_source`,
`pred_side_source`).

### Metrics

Both graphs are resampled every `--step-mm` (0.25 mm). Each sample stands for its share of
centerline length, so sums are exact lengths and every percentage below is **of length**. Each
sample is measured against the other graph resampled every `--fine-step-mm` (0.05 mm; error ≤
0.025 mm). Everything is reported for `ALL`, `L` and `R`; per-side reference samples are
measured against all of our graph (a vessel found but labelled with the wrong side still
counts as matched; `side_agreement` measures the labels separately).

**Matching rule.** A reference point is matched if our centerline passes within the reference
lumen radius there, floored at `--min-tolerance-mm` (1 mm); one of ours is matched if it lies
within the radius of its nearest reference point. This is the radius rule of the CAT08 evaluation
framework (Schaap et al. 2009, `.bib` key `cat08`, which defines OV, OF, OT and AI). Without radii in
the reference, a fixed `--tolerance-mm` (2 mm) is used (`tolerance_mode` in the output).

| Metric | Definition |
|---|---|
| `recall` | matched reference length / reference length: **coverage of the ground truth** |
| `precision` | matched length of ours / our length |
| `overlap` | (matched ref + matched ours) / (ref + ours), CAT08's OV |
| `f1` | harmonic mean of recall and precision |
| `overlap_clinical` | OV restricted to reference vessels at least `--clinical-diameter-mm` (1.5 mm) wide, CAT08's OT (references with radii only) |
| `of_mean` | per reference ostium-to-endpoint path, ignoring its first `--of-skip-mm` (40 mm, where ostium definitions disagree), fraction of the rest matched until the first unmatched point (CAT08's OF, which skips 5 mm), averaged over paths; paths not longer than 40 mm are left out |
| `ref_to_pred_{p25,mean,median,p75,p95,max}` | distance (mm) from every reference point to our centerline: includes missed vessels |
| `pred_to_ref_{...}` | the same from ours to the reference: includes extra vessels |
| `matched_{...}` | `ref_to_pred` over matched points only: **spatial accuracy where both have the vessel** (CAT08's AI) |
| `assd_mm`, `hd_mm`, `hd95_mm` | symmetric mean distance (length-weighted), Hausdorff, 95th-percentile Hausdorff |
| `recall_at_Xmm`, `precision_at_Xmm` | the same with fixed tolerances (`--tolerance-curve`, default 0.5–5 mm) |
| `endpoint_*` | one-to-one greedy matching of reference and our endpoints within 10 mm (v2 trims 2 mm from every tip): recall, precision, mean error |
| `bifurcation_pct_{0-5,5-10,10-15,>15}mm` | every reference bifurcation paired one-to-one (greedy, by distance, no cap) with one of ours; share of reference bifurcations per distance bucket (`--bifurcation-buckets-mm 5 10 15`). Unpaired ones (we have fewer) count as >15 mm. Also `bifurcation_error_median_mm`, `n_ref`, `n_pred` |
| segments (`segments.csv`) | references with segment labels only (ImageCAS-X): per segment id **and name** (LM, LAD, LCX, D1, D2, OM1, OM2, RCA, R-PDA, R-PLA, Other, read from the file's `segment_name` array), length, recall, median distance; `summary.json` → `segments_over_cases.by_id` / `.by_name` (statistics over cases) |
| `root_distance_mm` | distance between the reference root and ours, per side; `*_root_to_annotated_ostium_mm` with `--ostia` |
| `length_ratio` | our length / reference length |
| `radius_error_{mean,median,mae}_mm` | our radius minus the reference radius, on matched points |
| `side_agreement` | fraction of matched reference length whose nearest point of ours has the same side |

`summary.json` gives, for every metric and scope, the statistics over cases (n, p25, mean,
median, p75, max), plus the distance statistics pooled over all points of all cases.

### Missing centerlines (flags)

Every connected unmatched stretch of the reference at least `--missing-min-length-mm` (10 mm)
long is listed and classified:

| Kind | Meaning |
|---|---|
| `missing_case` | no graph of ours for this case |
| `missing_tree` | a side's recall below `--missing-tree-recall` (5 %) |
| `missing_branch` | reaches a reference endpoint and starts within `--branch-origin-mm` (5 mm, along the tree) of a bifurcation: a whole side branch is missing. Its first mm lie inside the parent lumen, so they count as matched |
| `truncated` | reaches a reference endpoint but starts mid-vessel: we stop early |
| `gap` | interior stretch (we leave the reference and come back) |
| `extra` | a stretch of ours, as long, with no reference counterpart (a vessel the reference does not have, or a wrong path) |
| `frame_warning` | see Coordinate frames |

Each comes with its side, length, distance from the ostium along the tree, and start/end RAS
points (to open in Slicer).

### Outputs

```
<output-dir>/
├── per_case.csv              one row per case and scope (ALL/L/R), every metric
├── flags.csv                 every flag, most severe first, with segment (ImageCAS-X) and RAS positions
├── discrepancies.html / .csv cases by severity (not compared / major / minor / agreement), each with what differs,
│                             where (side, segment, distance from the ostium, RAS) and how much; major = a missing
│                             tree, a discrepancy >= 20 mm (--major-mm), recall < 90 % (--major-recall), a frame warning
├── segments.csv              recall per reference segment label (only when the reference has labels)
├── summary.json              options, counts, statistics over cases, pooled distances, every case in full
├── report.html               cases sorted by recall (worst first), flags, figure thumbnails
├── coverage_vs_tolerance.png recall and precision against a fixed tolerance, one line per case + median
├── distance_ecdf.png         pooled distance distributions, both directions
├── per_case_metrics.png      recall, precision, median distance, OF per case, by side
└── figures/<case>.png        top (R-A) and front (R-S) views: reference matched (gray) / missed (red),
                              ours matched (blue) / extra (orange), roots, ostia, labelled flags
```

### Known limitations

- **Not run against our own graphs yet.** The reference formats are verified on the examples; the
  frame choice and the ostium/root agreement are not, and need a first real batch.
- **Slow without scipy.** On the laptop the brute-force neighbour search takes ~1 min per ASOCA case;
  on the cluster scipy's KD-tree is used.
- **Length depends on jaggedness.** Lengths, recall and precision are measured on the geometry as
  given; an unsmoothed voxel staircase is ~5–10 % longer than the vessel. Compare smoothed graphs
  (v2 `--smooth-graph`, branch-wise, σ 2 mm), or both unsmoothed.
- **Different masks, different extents.** ImageCAS-X centerlines come from corrected masks; ours from
  the ImageCAS masks (or the ImageCAS-X ones). Distal extent differs by construction, so `extra` and
  `truncated` near the tips are partly a mask difference, not a tracing error.
- **Ostium definitions differ** (our annotated ostium vs ASOCA's first centerline point vs
  ImageCAS-X's aorta-proximity roots), so the first mm near the root can be unmatched; this is why OF
  skips the first 40 mm.
- **Overlap merging** at 1 mm can cut the first mm of a branch that runs within 1 mm of its parent;
  increase `--branch-origin-mm` if such branches are classified as `truncated`.
