import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from monai.transforms import LoadImage, Orientation, Compose

# Importing as requested, though standard json handles the reading in this verification context
try:
    import hcatostialocalization.io.io as hcat_io
except ImportError:
    print("Warning: hcatostialocalization.io.io could not be imported. Proceeding with standard built-ins.")

def parse_args():
    parser = argparse.ArgumentParser(description="Verify .ostium.json coordinates by plotting MONAI and SITK slices.")
    parser.add_argument("--image", type=str, required=True, help="Path to the original image (e.g., .nii.gz)")
    parser.add_argument("--json", type=str, required=True, help="Path to the .ostium.json blueprint file")
    parser.add_argument("--out_dir", type=str, required=False, default="", help="Directory to save the verification plot. Defaults to script directory.")
    return parser.parse_args()

def normalize_image_for_display(img_array):
    """Clips the image to the 1st and 99th percentiles for good CT contrast visualization."""
    p1 = np.percentile(img_array, 1)
    p99 = np.percentile(img_array, 99)
    img_clipped = np.clip(img_array, p1, p99)
    # Normalize to [0, 1]
    if (p99 - p1) > 1e-8:
        return (img_clipped - p1) / (p99 - p1)
    return img_clipped

def main():
    args = parse_args()

    # Determine output path
    if not args.out_dir:
        args.out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)
    
    image_basename = os.path.basename(args.image)
    out_filename = f"verify_{image_basename.replace('.nii.gz', '').replace('.nii', '')}.png"
    out_filepath = os.path.join(args.out_dir, out_filename)

    print(f"[*] Verifying Coordinates for: {image_basename}")
    print(f"[*] Reading JSON: {args.json}")

    try:
        with open(args.json, 'r') as f:
            blueprint = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    if "pos" not in blueprint:
        print("Error: JSON does not contain 'pos' key. Is this a valid blueprint?")
        sys.exit(1)

    pos_data = blueprint["pos"]

    print("[*] Loading image via SimpleITK (Native)...")
    img_sitk = sitk.ReadImage(args.image)
    arr_sitk = sitk.GetArrayFromImage(img_sitk).astype(np.float32) # Shape: (Z, Y, X)
    arr_sitk = normalize_image_for_display(arr_sitk)

    print("[*] Loading image via MONAI (RAS forced)...")
    monai_loader = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        Orientation(axcodes="RAS", labels=None)
    ])
    img_monai = monai_loader(args.image)
    arr_monai = img_monai[0].numpy().astype(np.float32) # Shape: (X, Y, Z)
    arr_monai = normalize_image_for_display(arr_monai)

    # 2 rows (MONAI, SITK), 2 cols (Left Ostium, Right Ostium)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Ostia Verification: {image_basename}", fontsize=16)

    sides = ["L", "R"]
    titles = ["Left Ostium", "Right Ostium"]

    for col, side in enumerate(sides):
        if side not in pos_data:
            print(f"Warning: {side} ostium not found in JSON.")
            continue
        
        # 1. MONAI-RAS (Row 0)
        ijk_monai = pos_data[side]["ijk"]["monai-ras"]
        x_m, y_m, z_m = [int(round(c)) for c in ijk_monai]
        
        # Extract Z slice. Array is (X, Y, Z). Transpose to (Y, X) for standard visualization.
        slice_monai = arr_monai[:, :, z_m].T
        
        ax_monai = axes[0, col]
        # origin='lower' handles the (X, Y) layout correctly once transposed
        ax_monai.imshow(slice_monai, cmap='gray', origin='lower')
        ax_monai.plot(x_m, y_m, marker='+', color='red', markersize=15, markeredgewidth=2)
        ax_monai.plot(x_m, y_m, marker='o', color='none', markeredgecolor='yellow', markersize=15, markeredgewidth=1.5)
        ax_monai.set_title(f"MONAI (RAS) - {titles[col]}\nSlice Z={z_m}")
        ax_monai.axis('off')

        # 2. SimpleITK (Row 1)
        ijk_sitk = pos_data[side]["ijk"]["sitk"]
        x_s, y_s, z_s = [int(round(c)) for c in ijk_sitk]
        
        # Extract Z slice. Array is (Z, Y, X).
        slice_sitk = arr_sitk[z_s, :, :]
        
        ax_sitk = axes[1, col]
        # SITK natural shape is (Y, X), default origin='upper' works well
        ax_sitk.imshow(slice_sitk, cmap='gray')
        ax_sitk.plot(x_s, y_s, marker='+', color='cyan', markersize=15, markeredgewidth=2)
        ax_sitk.plot(x_s, y_s, marker='o', color='none', markeredgecolor='yellow', markersize=15, markeredgewidth=1.5)
        ax_sitk.set_title(f"SimpleITK (Native) - {titles[col]}\nSlice Z={z_s}")
        ax_sitk.axis('off')

    plt.tight_layout()
    plt.savefig(out_filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[*] Verification complete. Plot saved to: {out_filepath}")

if __name__ == "__main__":
    main()



