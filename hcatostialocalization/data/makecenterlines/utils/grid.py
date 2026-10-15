"""
The working grid and its coordinate frames, shared by v1 and v2.

  - Ostium ground truth (hcatostialocalization.io.io.extract_simplified_ras) is a
    RAS physical-mm coordinate.
  - The working grid is the mask as loaded by MONAI (io.io.load_image_ras:
    LoadImage + Orientation("RAS")) and resampled by MONAI Spacing to an
    isotropic spacing (load_mask_ras_isotropic). Its 4x4 affine maps an array
    index (i, j, k), in the array's own axis order, to RAS mm.
  - A graph node's 'ijk' attribute is an array index on that working grid.
So every conversion is a single affine, applied in one direction or the other.
"""
import numpy

from hcatostialocalization.io.io import extract_simplified_ras


def load_mask_ras_isotropic(mask_path: str, spacing_mm: float):
    """Loads a segmentation mask onto the working grid: MONAI LoadImage +
    Orientation("RAS") via io.io.load_image_ras(), then MONAI Spacing to
    spacing_mm isotropic with nearest-neighbour interpolation (labels stay
    labels). Any non-zero label counts as vessel.

    Returns (mask, affine, mask_metatensor):
      mask: bool numpy array, axes (i, j, k) of the RAS-oriented grid;
      affine: 4x4 float64, working-grid index (i, j, k) -> RAS mm;
      mask_metatensor: the resampled MetaTensor, the reference grid for
      load_image_on_grid().
    """
    from monai.transforms import Spacing
    from hcatostialocalization.io.io import load_image_ras

    mask_ras = load_image_ras(mask_path)
    resampled = Spacing(pixdim=(float(spacing_mm),) * 3, mode="nearest")(mask_ras)
    affine = numpy.asarray(resampled.affine.cpu().numpy(), dtype=numpy.float64).reshape(4, 4)
    mask = resampled[0].cpu().numpy() > 0
    return mask, affine, resampled


def load_ostium_ras(ostium_json_path: str) -> dict:
    """Returns {"L": numpy.ndarray(3,), "R": numpy.ndarray(3,)}, the ground-truth
    ostium RAS physical coordinates, via the existing, format-agnostic loader.
    """
    simple_pos = extract_simplified_ras(ostium_json_path)
    if simple_pos is None:
        raise ValueError(f"Could not read ostium ground truth from {ostium_json_path}")
    out = {}
    for side_letter in ("L", "R"):
        ras = simple_pos.get(side_letter, {}).get("ras")
        if not ras or len(ras) != 3:
            raise ValueError(f"Missing/invalid '{side_letter}' ostium RAS point in {ostium_json_path}")
        out[side_letter] = numpy.asarray(ras, dtype=float)
    return out


def working_ijk_to_ras(ijk, affine_ijk_to_ras: numpy.ndarray) -> numpy.ndarray:
    """Working-grid array index (i, j, k), continuous, -> RAS mm. Accepts a (3,)
    point or an (N, 3) array."""
    ijk = numpy.asarray(ijk, dtype=float)
    affine = numpy.asarray(affine_ijk_to_ras, dtype=float)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def ras_to_working_ijk(ras, affine_ijk_to_ras: numpy.ndarray) -> numpy.ndarray:
    """RAS mm -> working-grid array index (i, j, k), continuous (not rounded).
    Accepts a (3,) point or an (N, 3) array."""
    ras = numpy.asarray(ras, dtype=float)
    affine = numpy.asarray(affine_ijk_to_ras, dtype=float)
    return (ras - affine[:3, 3]) @ numpy.linalg.inv(affine[:3, :3]).T


def selftest_roundtrip(affine_ijk_to_ras: numpy.ndarray, atol_mm: float = 1e-6) -> None:
    """Round-trips a handful of arbitrary points through RAS -> working ijk ->
    RAS and raises AssertionError if they don't come back within floating-point
    tolerance. Called once per case (cheap): a singular or corrupted affine
    stops the case instead of silently misplacing it.
    """
    rng = numpy.random.default_rng(0)
    for _ in range(5):
        probe_ras = rng.uniform(-50.0, 50.0, size=3)
        ijk = ras_to_working_ijk(probe_ras, affine_ijk_to_ras)
        back_to_ras = working_ijk_to_ras(ijk, affine_ijk_to_ras)
        if not numpy.allclose(probe_ras, back_to_ras, atol=atol_mm):
            raise AssertionError(
                f"Coordinate round-trip failed: {probe_ras} -> {ijk} -> {back_to_ras} "
                f"(max abs diff {numpy.max(numpy.abs(probe_ras - back_to_ras)):.6g} mm)"
            )


def compute_physical_positions(graph, affine_ijk_to_ras: numpy.ndarray) -> None:
    """Sets 'pos_ras' (RAS mm, numpy.ndarray(3,)) on every node of graph, from its
    working-grid 'ijk' attribute. In place. Call once the graph is small
    (post-pruning), not on the raw ~10^5-node skeleton graph.
    """
    for node_id, data in graph.nodes(data=True):
        data["pos_ras"] = working_ijk_to_ras(numpy.array(data["ijk"], dtype=float), affine_ijk_to_ras)


def longest_uncovered_stretch(path, covered_volume) -> float:
    """Longest run of consecutive path voxels not in covered_volume, as an arc
    length in voxel units (the step into the first uncovered voxel counts)."""
    best = run = 0.0
    for previous, voxel in zip(path[:-1], path[1:]):
        if covered_volume[voxel]:
            run = 0.0
            continue
        run += float(numpy.linalg.norm(numpy.subtract(voxel, previous)))
        best = max(best, run)
    return best
