# makecenterlines v2: root-driven tracing

Extracts the same output as v1 (one `SimpleCenterlineGraph` per case with the left and
right trees, rooted at the annotated ostia, GML on disk) by growing each tree from its
ostium through the mask instead of skeletonising it. Status (2026-10-01), on the first 5
ImageCAS cases only:
- **First run:** no missing branches, good centerlines. Two defects were seen: paths leaving
  the artery at bulky junctions, and a distal vessel missed in case 3 (running parallel to a
  larger branch).
- **Second run (`first5_cap`, radius cap + 5 mm trimming):** centerlines centred and smooth in
  3D Slicer. The radius cap capped only 0–0.04 % of the domain voxels, so at 1.2 it barely acts
  (see Known risks). That run's QC has no `suspect_missed_branch` / `unexplained`, so it predates
  the missed-branch QC.
- **Branch-wise smoothing (step 9):** checked on those graphs. Curved planar reformats in Slicer
  are now straight.
- **Not run yet:** the current code end to end, the 1000-case batch, and the comparison against
  ASOCA / ImageCAS-X (`../compare.py`).

## Why

v1 starts from a Lee skeleton, and most of its 14 stages repair skeleton artefacts:
spurious branches (pruning), loops (shortest-path trees), rough surfaces (image-guided
conditioning), and lost branches (reconnection, coverage gap filling). v2 never builds a
skeleton, so none of these stages exist.

## Method

1. **Loading.** MONAI, RAS, 0.5 mm isotropic, nearest neighbour (v1's
   `utils/grid.load_mask_ras_isotropic`). Then a MONAI `SpatialCrop` to the bounding box
   of the mask and both ostia, padded by the ostium bridge radius + 2 mm. The cropped
   MetaTensor's affine is the working grid. A check confirms that no mask voxel was lost
   and that the affine is only shifted by the crop offset.
2. **Domain and cost.** `M` is the mask (the source of truth). `D` is `M` dilated by 1 voxel,
   which repairs connections broken by resampling. `DT` is the distance transform of `D`
   (voxels). The per-voxel cost is
   ```
   c = (DT + 0.5)^-p           in M                 (p = 4)
   c = P (DT + 0.5)^-p         in D \ M  ("shell", P = 1000)
   c = P                       in the ostium bridge
   c = inf                     elsewhere
   ```
   A minimal path minimises `∫ ds / DT^p`: short **and** central in one integral.
   Moving δ voxels off-axis in a vessel of radius R multiplies the local cost by
   `(R/(R−δ))^p`. A high p keeps paths on the axis through bends: a 1-voxel shortcut in a
   6-voxel-radius vessel costs +44 % with p = 2 and +107 % with p = 4. The **ostium
   bridge** is the ball around the ostium just large enough to reach `D` (+1 mm). The
   ostium sits on the aortic wall, mostly outside the mask, and enters it by the shortest
   route. A case fails if the mask is more than 25 mm from an ostium (10 mm until 2026-10-01: an ImageCAS
   case had a 14.6 mm gap, i.e. a mask that stops short of the aorta). The gap is crossed in a straight line.
3. **Fronts.** Per side, two single-source minimal-cost fronts from the ostium
   (`skimage.graph.MCP_Geometric`, 26-connected):
   - `G`, with unit cost: the geodesic distance through the mask;
   - `C`, with the cost above: its traceback is one predecessor map rooted at the ostium.
4. **Left/right.** Each side's domain is what its front reaches. If both fronts reach a
   voxel (trees touching), it goes to the side with the smaller `G`, and QC sets
   `merged_component`. Mask pieces that neither front reaches are reported in `unreachable`.
   **Gap bridging** (2026-10-04, `--max-gap-bridge-mm`, default 10 mm): a mask should be one tree
   per side, but segmentations can be interrupted for a few mm (the first full run lost 145–334 mm
   of left tree in ImageCAS-X cases 957/910/134, and 10 ImageCAS cases had a piece of 10 mm or more
   2–23 mm away). Before the fronts, the pieces nearest to the ostia form the connected set. The
   nearest remaining piece (at least 10 mm³) is joined to it by a straight one-voxel line between
   their closest voxels if the gap is ≤ 10 mm; this repeats, so chains of pieces are joined in
   order. Line voxels cost like the dilation shell, so paths use them only to cross the gap.
   Recorded in qc `gap_bridges` and per side as `n_gap_bridge_voxels_used`; `review.py` lists the
   case. This is the one place where a path may leave the mask.
5. **Tree growth (TEASAR farthest-point loop).** Repeat until every mask voxel of the
   side is covered:
   1. pick the uncovered voxel with the largest `G` (the tip of the longest vessel not yet
      traced);
   2. trace its path from the predecessor map, cut at the last voxel already in the tree
      (the junction);
   3. measure the path's longest run outside the covered tube;
   4. mark as covered every mask voxel within `1.5 · r + 1 mm` of the new path **whose
      segment to its nearest path node stays inside the mask** (line of sight). A vessel
      running next to the path, separated from it by background or joined to it only
      through the dilation shell, is not swallowed by the tube and is still traced;
   5. keep the path if that run is ≥ `L_min` = 5 mm (10 mm until 2026-10-05), otherwise discard it. Its tube
      stays covered, so it is never picked again.

   All paths come from the same predecessor map, so **the result is a tree by
   construction**. Bumps and wall pockets get picked as targets, but their paths leave
   the covered tube for only a few mm, so they are rejected. This is what makes endpoint
   detection reliable without looking for local maxima.
6. **Radius cap, second pass (default on).** `(DT + 0.5)^-p` rewards depth without
   limit, so where the mask bulges (an over-segmented bifurcation) the deep centre of
   the bulge pulls the path off the real lumen. From the pass-1 tree, the vessel radius
   at each tree voxel is the median DT of the tree voxels within 5 mm along the tree,
   which the short DT spike of a bulge does not shift. Every domain voxel takes the
   radius of its nearest tree voxel (same side), DT is capped at 1.2 × that radius
   (at least 1 voxel), and step 5 is repeated with the capped cost. Targets and domains
   are unchanged. In a bulge the cost becomes flat beyond one vessel radius of depth, so
   the path takes the shortest route that is still a full radius deep, near the line
   joining the vessels that meet there. Elsewhere DT does not exceed ~1 radius and the
   cost is unchanged. Coverage and centredness still use the uncapped DT.
7. **Endpoint trimming.** Tips run to the very end of the mask; 2 mm (5 mm until 2026-10-02) is
   removed from every endpoint of the finished tree, never past a bifurcation or an ostium.
7b. **Minimum ostial trunk** (2026-10-01). A bifurcation at, or within a few mm of, the ostium
   is anatomically implausible, but the traced tree can have one: when the annotated ostium sits
   where the mask already splits, or when its bridge reaches two vessels. HCATNetwork also
   requires the ostium to have a single neighbour: `get_anatomic_segments()`, used by `resample()`,
   raised `IndexError` on an ImageCAS case. If the first bifurcation lies closer than
   `--min-ostial-trunk-mm` (2 mm, straight line; 5 mm on 2026-10-01) to the ostium:
   - the trunk follows the main path (the child with the longest downstream vessel at every
     bifurcation) up to its first node at least 2 mm from the ostium;
   - every other node nearer the ostium along the tree is removed, and the subtrees that hung from
     them are re-attached to that node with a straight link, which becomes the first bifurcation;
   - smoothing (step 9) later rounds the link.

   The distance is straight-line, so smoothing cannot shorten the trunk below the threshold, and
   resampling at 0.5 mm leaves at least 2 nodes per mm on it (5 mm: 10 nodes, the 10th being the
   bifurcation). Checked (at 5 mm) on synthetic
   trees: a split at the ostium, a 3 mm left main whose side branch splits again, and a 10 mm left
   main (left untouched). All tips are kept.
8. **Finalisation.** v1 unchanged: positions in RAS (the ostium node keeps its exact
   annotated coordinate), radii from the distance transform of `M` (the shell falls back
   to the one of `D`), degree-based topology, `SimpleCenterlineGraph`, optional smoothing
   (`--smooth-graph`, step 9), GML, 3D Slicer markups.
9. **Branch-wise smoothing** (`--smooth-graph`, `utils/smoothing.py`, 2026-10-01). A voxel path
   is centred, but its direction jumps at every voxel. A curved planar reformat builds each slice
   normal to the curve, so on the first 5 cases it showed a wiggly mosaic instead of a straight
   artery. v1's smoothing (σ = 0.5 mm on 1 mm-spaced nodes, which barely acts) left a median turn
   of 5° per 0.5 mm, a 6 mm radius of curvature almost everywhere. Now:
   1. The tree is split by longest-path decomposition: from the ostium, the main branch follows
      the child with the longest path to a tip at every bifurcation; every other child starts a
      side branch at that bifurcation. Parents are smoothed first.
   2. Each branch is smoothed as one curve: Gaussian over arc length, σ = 2 mm, with
      point-reflection padding. The ostium, the tips and each side branch's attachment point stay
      exactly in place, and the main vessel passes smoothly through its bifurcations.
   3. A node moves at most 1 mm. With `--smooth-shift-limit mask` (default since 2026-10-04), a node
      inside the mask may move that far as long as it stays inside; otherwise its shift is halved until
      it does, falling back to 0.4 × its radius. With `radius` (the previous behaviour), the limit is
      always 0.4 × radius. On ASOCA the radius limit left a p95 turn of 20° on thin vessels (about 230
      limited shifts per case). A 0.35 mm floor on it (`--smooth-min-shift-mm`) brought that to 12°,
      but pushed 290 extra nodes out of the mask over 40 cases. On a synthetic thin vessel (radius
      0.6–1.5 mm, staircase path), the mask limit gives a p95 turn of 1.6° at every radius, with no
      node outside the mask.

   Radii are smoothed the same way. Then the graph is resampled every 0.5 mm. On the 5 cases
   (applied to the old smoothed graphs):

   | | median turn per 0.5 mm | p95 |
   |---|---|---|
   | before | 5.4° | 16.5° |
   | after | 1.6° | 4.3° |

   Shifts were at most 0.77 mm (median 9 % of the radius), and 34 of ~6400 nodes were limited by
   the radius rule. The few remaining sharp turns are at curve ends: tip hooks and the ostium
   bridge. σ sweep: 1 mm → 2.4°, 1.5 mm → 1.8°, 2 mm → 1.5°, 3 mm → 1.3° median turn, with
   growing shifts.

## Usage

**Complete runs (recommended):** three scripts in `../scripts/`, launchable from any folder, with the same
options: `-o <run folder>`, `--use-shards [K]` (K parallel jobs, 2 if K is omitted), `--resume`.
```bash
S=hcatostialocalization/data/makecenterlines/scripts        # from the repository root
bash $S/run_extraction.sh -o ./logs/makecenterlines_v2_final --use-shards 8   # lists + v2 on ASOCA, ImageCAS, ImageCAS-X
bash $S/run_extraction.sh -o ./logs/makecenterlines_v2_final --resume         # after an interruption (same shards/options)
bash $S/run_review.sh     -o ./logs/makecenterlines_v2_final --use-shards 3   # review.csv/html per dataset
bash $S/run_compare.sh    -o ./logs/makecenterlines_v2_final --use-shards 3   # the three comparisons, in <run>/compare/
```
- `run_extraction.sh` writes `<run>/lists/` (through `collections/make_lists.sh`), `<run>/run_config.env`
  (shards, datasets, data root, extra options; `--resume` reads it), one folder per dataset, and
  `<run>/logs/<dataset>.shard<i>of<K>.log`. Options after `--` go to every v2 job.
- `run_review.sh` and `run_compare.sh` run up to K datasets / comparisons at once; `--resume` skips those already
  done. `--help` on any script lists every option.

**Input lists** for ASOCA, ImageCAS and ImageCAS-X (extraction and comparison): from the repository root,
`bash hcatostialocalization/data/makecenterlines/collections/make_lists.sh` (data root and output root are
optional arguments). Lists are built case by case from the `Ostia` folders, so all lists of a dataset share the same
order. ImageCAS-X uses the ImageCAS ostia (same case ids), and its reference is `left.vtk;right.vtk` on one line.
Our graphs traced on the ImageCAS masks and on the ImageCAS-X masks are both compared with ImageCAS-X.

from the repository root:
```bash
python -m hcatostialocalization.data.makecenterlines.v2 \
    --masks hcatostialocalization/data/makecenterlines/collections/ImageCAS_first_5_masks.txt \
    --ostia hcatostialocalization/data/makecenterlines/collections/ImageCAS_first_5_ostia.txt \
    --output-dir ./logs/makecenterlines_v2/first5/ \
    -a graph qc markup-3d-slicer debug_figures --smooth-graph
```
After a batch, rank the cases for manual review:
```bash
python -m hcatostialocalization.data.makecenterlines.v2.review ./logs/makecenterlines_v2/first5/ --audit 50
```
It writes `review.csv` (every case with its category and reasons) and `review.html`. Every case falls in the most
severe category that applies:
1. **failed**: the extraction raised an error, a mask piece of 20 mm or more could not be attached (`--failed-extent-mm`),
   or a side's tree is shorter than 20 mm;
2. **likely missed branch**: a mask region of at least the suspect extent (10 mm) that the tree does not explain, or an
   unattached piece of 10–20 mm;
3. **other checks**: a gap bridged, touching trees, an ostium more than 3 mm from the mask, a side below the batch's
   5th percentile in length or endpoints;
4. **clean**: the rest. 50 of them (`--audit`) are drawn at random for the audit; the misses found among them give the
   miss rate to quote in the paper.

A separate section lists the **suspiciously short branches that were accepted** (run outside the covered tube under
`--short-branch-mm`, 10 mm), with side, length and tip position in RAS, to check for spurious stubs now that the
branch threshold is 5 mm. Each card also shows where the case's Slicer curves are.

To measure the graphs against reference centerlines (ASOCA, ImageCAS-X), pass the `graph/`
outputs to `compare.py` (coverage, distances, missing-branch flags; see [`../README.md`](../README.md)):
```bash
python -m hcatostialocalization.data.makecenterlines.compare \
    --pred <list of graph/<case>.centerlines.gml> --ref <list of reference .vtk/.vtp> \
    --ostia <list of ostium JSONs> -o ./logs/compare_centerlines/<run>/
```

**Long batches.** A failed case writes `errors/<case>.error.json` as it happens, so an interrupted run
keeps its failures (`review.py` reads them, plus every `batch_summary*.json`). Restart an interrupted run with
the same command plus `--resume`. Split a long batch over n parallel jobs with `--shard 0/n` ... `--shard n-1/n`
(same output folder).

`-a none` is a smoke test that writes nothing. `--images` is optional and only names the
cases. Without it, `1.label.nii.gz` (ImageCAS) and `1.coronary.nii.gz` (ImageCAS-X) give case `1`, and
`Normal_1.nrrd` (ASOCA) gives `Normal_1`.

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--isotropic-spacing-mm` | 0.5 | working grid |
| `--dilation-voxels` | 1 | connectivity repair after resampling (0 = off) |
| `--centering-power` | 4 | p: higher = more central, less corner cutting |
| `--shell-penalty` | 1000 | P: cost factor in the shell; also the bridge cost |
| `--max-ostium-bridge-mm` | 25 | case fails if the mask is farther from an ostium |
| `--min-branch-length-mm` | 5 | L_min: run outside the covered tube for a branch to be kept |
| `--cover-radius-scale`, `--cover-tol-mm` | 1.5, 1 mm | covered tube = scale · r + tol |
| `--radius-cap-scale` | 1.2 | DT cap in pass 2, × the vessel radius along the pass-1 tree (0 = single pass, no cap) |
| `--radius-cap-window-mm` | 5 | window along the tree for the vessel radius (median) |
| `--max-gap-bridge-mm` | 10 | mask pieces up to this far from the tree are joined by a straight link (0 = off) |
| `--smooth-shift-limit` | mask | `mask`: up to `--smooth-max-shift-mm` while staying inside the mask; `radius`: 0.4 × radius |
| `--smooth-min-shift-mm` | 0 | a node may always move at least this far, whatever its radius (thin vessels; 0 = off) |
| `--trim-endpoints-mm` | 2 | length removed from every endpoint (0 = off) |
| `--min-ostial-trunk-mm` | 2 | minimum straight-line distance from each ostium to its first bifurcation (0 = off) |
| `--resume`, `--resume-overlap`, `--resume-anyway` | off, 2 | continue an interrupted run in the same output folder: skips the cases up to the last one processed (a `qc/` or `errors/` file exists) and redoes the last 2; refuses if the parameters differ from those in the previous run's `qc.json` |
| `--shard i/n` | – | process only the cases at list positions ≡ i (mod n): n parallel jobs on the same output folder, each writing `batch_summary.shard<i>of<n>.json` |
| `--suspect-extent-mm` | 10 | QC only: unexplained or unreachable mask regions at least this long flag the case |
| `--smooth-graph` | off | branch-wise smoothing (step 9) |
| `--smooth-sigma-mm` | 2 | Gaussian σ along each branch |
| `--smooth-max-shift-fraction`, `--smooth-max-shift-mm` | 0.4, 1 mm | a node moves at most this fraction of its radius, and at most this far |
| `--smooth-final-spacing-mm` | 0.5 | node spacing after smoothing (0 = keep the traced nodes) |

## QC (`qc/<case>.qc.json`)

| Key | Meaning |
|---|---|
| `sides.{L,R}.ostium_to_mask_mm` | bridge length: distance from the annotated ostium to the mask |
| `sides.{L,R}.length_mm`, `n_endpoints` | tree size |
| `sides.{L,R}.n_targets_accepted/_rejected`, `accepted`, `rejected` | branches kept / stubs discarded (first 50, target voxel and run length) |
| `sides.{L,R}.centredness_median`, `_p5`, `fraction_nodes_centredness_below_0.7` | per node, DT / max DT within a ball of radius DT: 1 = on the axis, low = corner cutting |
| `sides.{L,R}.n_shell_voxels_used` | tree voxels outside the mask (dilation shell): where the mask was broken |
| `merged_component` | the two trees touch in the mask (split by geodesic Voronoi) |
| `sides.{L,R}.unexplained` | mask voxels the final (accepted, untrimmed) tree does not explain, with the same tube and line-of-sight rule as growth: count, volume, fraction, `n_suspect`, `max_extent_mm`, and the 20 longest regions (volume, extent along the first principal axis, distance to the tree, centroid RAS / ijk) |
| `unreachable` | mask pieces reached by neither ostium: same fields as `unexplained`, plus fraction of the mask |
| `suspect_missed_branch` | `flag` and human-readable `reasons`: any unexplained or unreachable region at least `--suspect-extent-mm` long |
| `radius_cap` | scale, window, fraction of domain voxels whose DT was capped, pass-1 tree sizes; `changes.{L,R}` (`n_voxels_moved`: pass-2 tree voxels not on the pass-1 tree; `max_distance_from_pass1_mm`) and `tree_changed` (some pass-2 path more than one voxel from pass 1): the share of cases where the cap matters |
| `n_endpoints_trimmed` | endpoints shortened by `--trim-endpoints-mm` (the `sides` lengths are before trimming) |
| `final_acyclic`, `ostia_present` | consistency checks |
| `gap_bridges` | `max_gap_mm`, `n`, and per bridge `gap_mm`, `piece_ijk`, `tree_side_ijk`, `piece_volume_mm3`; per side `n_gap_bridge_voxels_used` |
| `ostial_trunk.{L,R}` | `first_bifurcation_mm_before` (straight-line distance of the traced tree's first bifurcation from the ostium), and when it was below `--min-ostial-trunk-mm`: `n_nodes_removed`, `n_subtrees_reattached`, `max_link_mm` (longest straight link to the new bifurcation; large = sharply diverging branches, worth a look) |
| `direction_roughness` | turn angle per 0.5 mm along the branches of the traced (unsmoothed) tree: median, p95, max, fraction above 5°, implied radius of curvature |
| `smoothing` | parameters, `n_branches`, `max_shift_mm_applied`, `n_shifts_limited`, `direction_roughness_after`, `n_nodes_outside_mask_before` / `_after` (ostia excluded; must not grow), `n_nodes_after` |

Debug figures (`debug_figures/<case>/`):

| Figure | Content |
|---|---|
| `distance_map.png` / `.pdf` | maximum projections of the distance-to-wall map used by the cost (of `D`, in mm), no graph: top view (S compressed) above side view (A compressed), one-column size (3.5 × 7 in) for the Methods section |
| `targets_mip.png` | mask maximum projections, top view (S compressed) above side view (A compressed), with the trees (L cyan, R red), accepted (green) and rejected (orange) targets, and the annotated ostia (big blue dots) |
| `graph_2d.png`, `graph_3d.png` | trees before smoothing (2D = R-A plane) |
| `graph_2d_smoothed.png`, `graph_3d_smoothed.png` | trees after smoothing (`--smooth-graph` only) |

## Code map

| File | Contents |
|---|---|
| `trace.py` | steps 2–7: `connectivity_domain`, `centred_cost`, `ostium_bridge`, `single_source_front`, `split_left_right`, `grow_tree` (+ `_line_of_sight`), `vessel_radius_along_tree`, `capped_distance`, `trim_endpoints`, QC helpers, `trace_centerlines` |
| `pipeline.py` | loading, `crop_to_arteries` (MONAI `SpatialCrop` + frame checks), finalisation, writers |
| `__main__.py` | CLI and batch loop, `batch_summary.json` |
| `review.py` | ranks a finished batch for manual review: `review.csv`, `review.html` |

Shared with v1 (`../utils/`): `grid.py` (loader, ostia, frames, `longest_uncovered_stretch`),
`smoothing.py` (branch-wise smoothing, direction roughness), `convert.py` (radii, topology,
conversion, v1 smoothing, save, Slicer markups), `checks.py`
(cycle / ostium checks), `figures.py` (2D/3D graph plots), `cli.py` (path lists, case ids).
Comparison with reference centerlines: `../compare.py` (see `../README.md`).

## Known risks / open work

- **The radius cap rarely acts, and is kept as a safety net** (decided 2026-10-01). In
  `first5_cap` it capped 0 % of the domain voxels in 3 cases and 0.004–0.04 % in the other 2. It
  cannot change the cost where DT ≤ 1.2 × the vessel radius, so it cannot harm regular vessels;
  it only costs a second growth pass. The share of cases where it changes the tree
  (`qc.radius_cap.tree_changed`) is to be reported from the 1000-case run.
- **Line-of-sight coverage** (case 3's parallel distal vessel) and the missed-branch QC: run
  the current code on the first 5 cases and check case 3. If it is still missed, also try
  `--dilation-voxels 0`. Line of sight does not help when the two vessels touch inside the
  mask itself.
- **Centredness at bulges** is low by design after the cap (the path no longer goes
  through the bulge's deepest voxel); read it away from junctions.
- **Corner cutting** in tortuous segments: watch `centredness_p5`. If it is low on the
  proximal RCA/LAD, the planned fix is a cross-sectional re-centring of each node (move
  it to the max-DT voxel in the plane orthogonal to the tangent, skipping bifurcations).
  It would change positions only, never topology.
- **Shortcuts where two branches touch** in the mask itself (the shell penalty only covers
  contacts created by the dilation). If seen, add v1's core preference (erosion by 1
  voxel) to the cost.
- **Disconnected mask pieces** farther than `--max-gap-bridge-mm` are reported in `unreachable`, not bridged (decided
  2026-10-01).
- `L_min` was lowered from 10 to 5 mm on 2026-10-05. In the 2026-10-04 run the rejected candidates split into wall bumps
  (< 2 mm, 90–95 % of them), a valley at 2–5 mm, and a second group at 5–10 mm (about 1.2 per case on ASOCA, 0.7 on
  ImageCAS, 0.3 on ImageCAS-X), which looked like real short branches in Slicer. Check the new run for spurious stubs
  on the bulkier ImageCAS masks; tune it with the
  accepted/rejected lists.
