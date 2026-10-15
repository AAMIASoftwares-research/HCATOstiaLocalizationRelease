import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import your custom MONAI pipeline loader
from hcatostialocalization.infer.__main__ import load_native_ras_pair


def parse_args():
    parser = argparse.ArgumentParser(description="Generate axial slice SVGs overlaying probability heatmaps on a medical image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the original image (e.g., .nii.gz)")
    parser.add_argument("--prob", type=str, required=True, help="Path to the probability tensor (.pt, .pth, or .npy) of shape (1, 3, R, A, S)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save the figures")
    return parser.parse_args()

def normalize_min_max(tensor):
    """Normalizes a tensor strictly between 0.0 and 1.0 using min-max scaling."""
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min > 1e-8:
        return (tensor - t_min) / (t_max - t_min)
    return torch.zeros_like(tensor)

def normalize_image_for_display(img_array):
    """Clips the image to the 1st and 99th percentiles to ensure good grayscale contrast (crucial for CT)."""
    p1 = np.percentile(img_array, 5)
    p99 = np.percentile(img_array, 95)
    img_clipped = np.clip(img_array, p1, p99)
    return (img_clipped - p1) / (p99 - p1 + 1e-8)

def main():
    args = parse_args()

    # 1. Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    image_basename = os.path.basename(args.image)
    print(f"Loading image via load_native_ras_pair: {args.image}")
    
    # 2. Load the image using your custom MONAI pipeline
    try:
        # Pass an empty string for the label path as we only need the image here
        image_metatensor_ras, label_json_dict, base_transforms = load_native_ras_pair(
            args.image, 
            ""
        )
        # Squeeze removes the channel dimension (1, R, A, S) -> (R, A, S)
        img_array = image_metatensor_ras.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    print(f"Loading probability tensor: {args.prob}")
    
    # 3. Load the probability tensor
    try:
        if args.prob.endswith('.npy'):
            prob_tensor = torch.from_numpy(np.load(args.prob))
        else:
            # map_location='cpu' ensures we don't crash if the tensor was saved on a GPU
            prob_tensor = torch.load(args.prob, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading probability tensor: {e}")
        sys.exit(1)

    # Validate shape (1, 3, R, A, S)
    if prob_tensor.ndim != 5 or prob_tensor.shape[:2] != (1, 3):
        print(f"Warning: Expected prob tensor shape (1, 3, R, A, S), got {prob_tensor.shape}. Attempting to proceed anyway.")

    # 4. Extract and normalize the channels
    # Channel 1: Right Ostium, Channel 2: Left Ostium
    prob_right = prob_tensor[0, 1, :, :, :]
    prob_left = prob_tensor[0, 2, :, :, :]

    print("Normalizing probabilities per channel [0, 1]...")
    prob_right_norm = normalize_min_max(prob_right).numpy()
    prob_left_norm = normalize_min_max(prob_left).numpy()

    print("Normalizing image contrast for visualization...")
    img_array_vis = normalize_image_for_display(img_array)

    S_dim = img_array.shape[2]
    
    print(f"Generating {S_dim} axial slice visualizations. This may take a moment...")

    # 5. Iterate through the axial slices (Z-axis / S-axis)
    for s in range(S_dim):
        # Extract slices. Transposing (.T) aligns I (X) horizontally and J (Y) vertically 
        img_slice = img_array_vis[:, :, s].T
        pr_slice = prob_right_norm[:, :, s].T
        pl_slice = prob_left_norm[:, :, s].T

        height, width = img_slice.shape

        # ==========================================
        # MANUALLY CONSTRUCT RGBA ARRAYS FOR HEATMAPS
        # ==========================================
        
        # alpha transparency multiplier
        alpha_multiplier = 1.0 

        # Right -> Red overlay
        rgba_right = np.zeros((height, width, 4), dtype=np.float32)
        rgba_right[..., 0] = 1.0  # Red channel
        rgba_right[..., 3] = pr_slice * alpha_multiplier  # Alpha channel

        # Left -> Blue overlay
        rgba_left = np.zeros((height, width, 4), dtype=np.float32)
        rgba_left[..., 2] = 1.0  # Blue channel
        rgba_left[..., 3] = pl_slice * alpha_multiplier  # Alpha channel

        # ==========================================
        # PLOT AND SAVE FIGURE
        # ==========================================
        
        # 1x3 horizontal layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Subplot 1: Raw Image
        axes[0].imshow(img_slice, cmap='gray', origin='lower')
        axes[0].set_title(f"Image Slice {s}")
        axes[0].axis('off')

        # Subplot 2: Heatmaps Only
        # Set a black background so the transparency colors pop vividly
        axes[1].set_facecolor('black')
        axes[1].imshow(rgba_right, origin='lower')
        axes[1].imshow(rgba_left, origin='lower')
        axes[1].set_title(f"Normalized Probabilities (R=Red, L=Blue)")
        axes[1].axis('off')

        # Subplot 3: Overlay (Image + Heatmaps)
        axes[2].imshow(img_slice, cmap='gray', origin='lower')
        axes[2].imshow(rgba_right, origin='lower') 
        axes[2].imshow(rgba_left, origin='lower')
        axes[2].set_title(f"Overlay")
        axes[2].axis('off')

        plt.tight_layout()

        # Construct output path and save
        out_filename = f"{image_basename}__normalized_probability_overlay_slice_{s:06d}.svg"
        out_filepath = os.path.join(args.out_dir, out_filename)
        
        plt.savefig(out_filepath, format='svg', bbox_inches='tight')
        plt.close(fig)

        # Simple progress tracker
        if (s + 1) % 50 == 0:
            print(f"  Processed {s + 1}/{S_dim} slices...")

    print(f"\nDone! All figures saved to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
