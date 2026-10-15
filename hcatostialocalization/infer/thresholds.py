"""
Localization-accuracy thresholds and buckets, shared by infer/__main__.py and test/split_metrics_by_dataset.py.

Pure Python: no torch/monai imports, so it can be imported from numpy-only scripts.

Three per-side thresholds (mm) split the Euclidean error between predicted and reference ostium into four buckets:

    0 mm -> [optimal] -> ANATOMICAL -> [good] -> INTEROPERATOR -> [acceptable] -> FAILURE -> [failed]

A distance exactly equal to a threshold falls in the better bucket (d <= threshold).
Sides are "R" (right ostium, per-class index 0) and "L" (left ostium, per-class index 1), as in infer/__main__.py.
"""
from typing import Optional

SIDES = ("R", "L")

# Anatomical: lower bound mu - 2*SD of the internal ostial radius (half-diameter) reported for FEMALE hearts by
# Nasr and El Tahlawi (2018). Within it, ~95% of women (hence even more men, whose coronaries are larger) are covered.
ANATOMICAL_MM = {"R": 1.08, "L": 1.45}

# Interoperator: 75th percentile of the leave-one-out distance between one annotator and the others
# (Pooled row of Tables_Images/table_annotator_statistics.tex). At or below it, a prediction is no worse than
# 3 out of 4 human placements.
INTEROPERATOR_MM = {"R": 1.85, "L": 2.59}

# Failure: threshold from the state of the art (Mostafa et al., EMBC 2021), same for both sides.
FAILURE_MM = {"R": 5.0, "L": 5.0}

# ordered, best first
BUCKETS = ("optimal", "good", "acceptable", "failed")

THRESHOLDS_MM = {
    "anatomical": ANATOMICAL_MM,
    "interoperator": INTEROPERATOR_MM,
    "failure": FAILURE_MM,
}

assert all(ANATOMICAL_MM[s] < INTEROPERATOR_MM[s] < FAILURE_MM[s] for s in SIDES)


def classify(distance_mm: Optional[float], side: str) -> Optional[str]:
    """Bucket of one distance for one side ("R" or "L"). None (or NaN) in, None out."""
    if distance_mm is None or distance_mm != distance_mm:
        return None
    if distance_mm <= ANATOMICAL_MM[side]:
        return "optimal"
    if distance_mm <= INTEROPERATOR_MM[side]:
        return "good"
    if distance_mm <= FAILURE_MM[side]:
        return "acceptable"
    return "failed"


def classify_pair(distance_r_mm: Optional[float], distance_l_mm: Optional[float]) -> tuple:
    """(bucket_R, bucket_L, bucket_patient). The patient bucket is the worse of the two sides; None if either is missing."""
    b_r, b_l = classify(distance_r_mm, "R"), classify(distance_l_mm, "L")
    return b_r, b_l, worst_bucket(b_r, b_l)


def worst_bucket(*buckets: Optional[str]) -> Optional[str]:
    """The worst of the given buckets; None if any of them is None."""
    if len(buckets) == 0 or any(b is None for b in buckets):
        return None
    return max(buckets, key=BUCKETS.index)


def bucket_counts(buckets) -> dict:
    """{'n': valid count, 'counts': {bucket: int}, 'pct': {bucket: float % over the valid ones}}; None entries are ignored."""
    valid = [b for b in buckets if b is not None]
    counts = {b: sum(1 for v in valid if v == b) for b in BUCKETS}
    n = len(valid)
    return {
        "n": n,
        "counts": counts,
        "pct": {b: (round(100.0 * c / n, 6) if n else None) for b, c in counts.items()},
    }


def bucket_summary(buckets_r, buckets_l, buckets_patient) -> dict:
    """Counts and percentages for right ostium, left ostium and patient (worse of the two)."""
    return {
        "thresholds_mm": THRESHOLDS_MM,
        "R": bucket_counts(buckets_r),
        "L": bucket_counts(buckets_l),
        "patient": bucket_counts(buckets_patient),
    }
