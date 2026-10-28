"""
plot_ostium_from_mrk.py
=======================
Visualizes coronary ostia annotations (R and L) directly from the 3 annotators' .mrk.json
files, without requiring the final consensus JSON.

Shows:
  - Axial slice at the height of the consensus centroid (mean of the 3 annotators)
  - The 3 points from individual annotators (different colors)
  - The consensus centroid (white star)

USAGE:
  Modify PATHs at the bottom of the file and run:
      python plot_ostium_from_mrk.py
  or paste it into a Jupyter notebook cell.
"""

import os
import glob
import json
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# ==============================================================================
# PATH CONFIGURATION 
# ==============================================================================
current_dir = os.path.abspath(os.getcwd())
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)

from hcatostialocalization.config.config import config
from hcatostialocalization.data.dataset.Base_Classes_Dataset import LabeledDataset
from hcatostialocalization.data.dataset.Heatmap_Config import HeatmapParams

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def lps_to_ras(lps: list) -> list:
    """Converts from LPS (3D Slicer coordinateSystem) to RAS."""
    return [-lps[0], -lps[1], lps[2]]

def get_ijk_from_lps(img: sitk.Image, lps: list) -> tuple:
    """
    Converts LPS physical coordinates to IJK voxel indices.
    SimpleITK uses LPS internally, so TransformPhysicalPointToIndex
    accepts LPS directly.
    """
    return img.TransformPhysicalPointToIndex(lps)

def load_points_from_mrk(mrk_path: str) -> dict:
    """
    Reads a 3D Slicer .mrk.json file and returns a dict:
      {"R": [x, y, z], "L": [x, y, z]} in LPS coordinates.
    Returns None for side if not found.
    """
    points = {"R": None, "L": None}
    with open(mrk_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    control_points = data.get("markups", [{}])[0].get("controlPoints", [])
    for pt in control_points:
        label = pt.get("label", "").strip().upper()
        pos   = pt.get("position")
        if pos is None:
            continue
        # Robust recognition: "R", "RCA", "RIGHT" -> side R
        #                     "L", "LCA", "LEFT"  -> side L
        # NOTE: the check 'R' in label must come BEFORE 'L' in label
        # because "RL" or "LR" could create ambiguity (uncommon).
        if label in ("R", "RCA", "RIGHT") or (label.startswith("R") and "L" not in label):
            points["R"] = pos
        elif label in ("L", "LCA", "LEFT") or (label.startswith("L") and "R" not in label):
            points["L"] = pos
        # Generic fallback for non-standard labels
        elif "R" in label and "L" not in label:
            points["R"] = pos
        elif "L" in label and "R" not in label:
            points["L"] = pos

    return points


def plot_ostium_from_annotators(
    patient_id: str,
    mrk_files: list[str],
    image_path: str,
    side: str = "both",
    window_min: int = -200,
    window_max: int = 800,
):
    """
    Plots the axial slice with annotator points and consensus centroid.

    Parameters
    ----------
    patient_id : str
        Patient ID (e.g. "387"), used only for the title.
    mrk_files : list[str]
        List of absolute paths to annotators' .mrk.json files.
    image_path : str
        Path to the patient's .nii.gz (or .nii) image file.
    side : str
        "R", "L", or "both" (default). If "both", plots both sides
        on two side-by-side subplots.
    window_min, window_max : int
        HU range for windowing (default: [-200, 800] good for coronary contrast).
    """
    if not os.path.exists(image_path):
        print(f"[!] Image not found: {image_path}")
        return

    # --- Load image ---
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X) = (K, J, I)

    # --- Collect points from all annotators ---
    annotator_colors = ["#FF6B6B", "#FFD93D", "#6BCB77"]  # red, yellow, green
    annotator_labels = [f"Annotator {i+1}" for i in range(len(mrk_files))]

    points_per_side = {"R": [], "L": []}

    for mrk_f in mrk_files:
        pts = load_points_from_mrk(mrk_f)
        for s in ["R", "L"]:
            if pts[s] is not None:
                points_per_side[s].append(pts[s])
            else:
                print(f"[!] Side {s} not found in: {os.path.basename(mrk_f)}")

    # --- Compute consensus centroids (mean = equivalent to 1-cluster K-means) ---
    consensus = {}
    ijk_consensus = {}
    for s in ["R", "L"]:
        if len(points_per_side[s]) == 0:
            print(f"[!] No points found for side {s} — will not be plotted.")
            consensus[s] = None
            ijk_consensus[s] = None
        else:
            mean_lps = np.mean(points_per_side[s], axis=0).tolist()
            consensus[s] = mean_lps
            ijk_consensus[s] = get_ijk_from_lps(img, mean_lps)
            print(f"[+] Side {s} — LPS centroid: {[round(c, 2) for c in mean_lps]}")
            print(f"    → IJK: {ijk_consensus[s]}")
            print(f"    → Annotator points ({len(points_per_side[s])}): OK")

    # --- Determine sides to plot ---
    sides_to_plot = []
    if side == "both":
        sides_to_plot = [s for s in ["R", "L"] if ijk_consensus[s] is not None]
    elif side in ["R", "L"] and ijk_consensus.get(side) is not None:
        sides_to_plot = [side]
    else:
        print(f"[!] Side '{side}' not available.")
        return

    n_plots = len(sides_to_plot)
    if n_plots == 0:
        print("[!] No side to plot.")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    for ax, s in zip(axes, sides_to_plot):
        i_c, j_c, k_c = ijk_consensus[s]   # IJK (X, Y, Z)
        k_c = int(np.clip(k_c, 0, img_array.shape[0] - 1))

        # Extract and apply windowing to slice
        slice_2d = img_array[k_c, :, :]
        slice_2d = np.clip(slice_2d, window_min, window_max)

        ax.imshow(slice_2d, cmap="gray", origin="upper")

        # Plot individual annotators
        for idx, (lps_pt, color) in enumerate(zip(points_per_side[s], annotator_colors)):
            ijk_pt = get_ijk_from_lps(img, lps_pt)
            i_pt, j_pt, k_pt_ann = int(ijk_pt[0]), int(ijk_pt[1]), int(ijk_pt[2])
            # Note: imshow uses (X=cols, Y=rows) -> (i, j) = (X, Y)
            ax.scatter(i_pt, j_pt, color=color, marker="o", s=120, zorder=4,
                       label=annotator_labels[idx] if s == sides_to_plot[0] else "")
            # Small arrow if point is on a different slice than consensus
            if abs(k_pt_ann - k_c) > 2:
                ax.annotate(
                    f"z={k_pt_ann}", (i_pt, j_pt),
                    textcoords="offset points", xytext=(5, -10),
                    color=color, fontsize=7
                )

        # Plot consensus centroid (white star)
        ax.scatter(int(i_c), int(j_c), color="white", marker="*", s=400,
                   edgecolors="black", linewidths=0.8, zorder=5, label="Consensus")

        side_name = "Right (RCA)" if s == "R" else "Left (LCA)"
        ax.set_title(
            f"Patient {patient_id} — {side_name} Ostium\n"
            f"Slice K={k_c}  |  Consensus IJK=({int(i_c)}, {int(j_c)}, {k_c})",
            fontsize=12, pad=10
        )
        ax.axis("off")

    # Common legend
    legend_patches = [
        mpatches.Patch(color=annotator_colors[i], label=annotator_labels[i])
        for i in range(len(mrk_files))
    ]
    legend_patches.append(
        mpatches.Patch(color="white", label="Consensus (mean)")
    )
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               fontsize=10, framealpha=0.8)

    fig.suptitle(f"Ostia Annotation Verification — Patient {patient_id}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    from hcatostialocalization.config.config import config

    PATIENT_ID = "387"

    # Directory with annotators' .mrk.json files
    MRK_DIR = config.active_paths.data_root + "/OstiumAnnotations_ImageCAS/SlicerObjects"

    # Patient image
    IMAGE_PATH = config.active_paths.data_root + f"/Datasets/ImageCAS/Extracted_Images/{PATIENT_ID}.nii.gz"
    if not os.path.exists(IMAGE_PATH):
        IMAGE_PATH = config.active_paths.data_root + f"/Datasets/ImageCAS/Extracted_Images/{PATIENT_ID}.nii"

    # Search all .mrk.json files for patient 387
    # Pattern: "387_*.mrk.json" or "0387_*.mrk.json"
    mrk_files = sorted(
        glob.glob(os.path.join(MRK_DIR, f"{PATIENT_ID}_*.mrk.json")) +
        glob.glob(os.path.join(MRK_DIR, f"{int(PATIENT_ID):04d}_*.mrk.json"))
    )

    if not mrk_files:
        print(f"[!] No .mrk.json files found for patient {PATIENT_ID} in:\n    {MRK_DIR}")
    else:
        print(f"[+] Found {len(mrk_files)} annotation files:")
        for f in mrk_files:
            print(f"    {os.path.basename(f)}")
        print()

        plot_ostium_from_annotators(
            patient_id=PATIENT_ID,
            mrk_files=mrk_files,
            image_path=IMAGE_PATH,
            side="both",         # "R", "L", or "both"
            window_min=-200,
            window_max=800,
        )
