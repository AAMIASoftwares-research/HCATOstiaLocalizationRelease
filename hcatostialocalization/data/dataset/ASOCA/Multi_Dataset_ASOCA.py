import os
import json
import torch
import yaml
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
# MULTI-CLASS DATASET CLASS (3 GAUSSIAN CHANNELS)
# ==============================================================================
class Dataset_ASOCA_Ostia_Multi(LabeledDataset):
    """
    Multi-Class Multi-Channel version for ASOCA.
    Generates a [3, R, A, S] tensor (volume in RAS orientation).
    Channel 0 = Background, Channel 1 = Right Ostium (R), Channel 2 = Left Ostium (L).
    """
    labels: dict[int, str] = {
        0: "background",
        1: "RCA_ostium",
        2: "LCA_ostium",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma: float = HeatmapParams.sigma

    def _init_data_pool(self) -> list[tuple[str, str]]:
        # Explicit (image, label) pairs defined in the YAML split
        # (asoca_config.yaml, path in config/dataset.yaml -> dataset.ASOCA.yaml_path)
        return self._data_pool_from_split_yaml("ASOCA")

    def _get_patient_id(self, current_patient: tuple[str, str]) -> str:
        id = os.path.basename(current_patient[0]).split('.')[0].strip()
        return f"{self.split}-ASOCA-Multi-{id}"

    def _load_data(self, image_label_f: tuple[str, str], new_spacing: float|tuple|None = None) -> dict:
        # Image loaded in RAS (1, 1, R, A, S) with MONAI; target [3, R, A, S] constructed
        # from the JSON RAS world coordinates (see LabeledDataset._build_ostia_target).
        return self._load_ostia_data(image_label_f, new_spacing=new_spacing)

    def __len__(self):
        return len(self.data_pool)
    
# ==============================================================================
# RANDOM TEST LOGIC (Multi-Slice L & R Visualization)
# ==============================================================================
def run_random_test():
    print("[*] Initializing Multi-Class visual test...")
    
    ds = Dataset_ASOCA_Ostia_Multi(split='val', nsdim=3)
    if not ds.data_pool: 
        print("[!] Empty dataset.")
        return

    img_f, json_f = random.choice(ds.data_pool)
    patient_id = ds._get_patient_id((img_f, json_f))
    print(f"[*] --- RANDOM EXTRACTION: {patient_id} ---")


    print("[*] Loading 3D volume. This may take a few seconds...")
    sample = ds._load_data((img_f, json_f))
    if float(sample['ostia_coords_R'][0]) < 0 or float(sample['ostia_coords_L'][0]) < 0:
        print('[!] Missing R or L ostium in label. Skipping.')
        return
    # Axial slice index = S axis (k) of the loaded RAS grid
    ostium_z_R = int(round(float(sample["ostia_coords_R"][2])))
    ostium_z_L = int(round(float(sample["ostia_coords_L"][2])))
    
    image_tensor = sample['image'].squeeze().numpy()  # [R, A, S]
    label_tensor = sample['label'].numpy()            # [3, R, A, S]
    
    img_slice_R = image_tensor[:, :, ostium_z_R].T
    heat_slice_R = label_tensor[1, :, :, ostium_z_R].T # Channel 1: Right

    img_slice_L = image_tensor[:, :, ostium_z_L].T
    heat_slice_L = label_tensor[2, :, :, ostium_z_L].T # Channel 2: Left
    
    print("[*] Opening plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"ASOCA Patient Check: {patient_id}", fontsize=16)

    # Right Ostium (Channel 1 -> Green)
    axes[0].imshow(img_slice_R, cmap='gray', vmin=-500, vmax=800, origin='lower')
    axes[0].imshow(np.ma.masked_where(heat_slice_R < 0.05, heat_slice_R), cmap='Greens', alpha=0.7, origin='lower')
    axes[0].set_title(f"RIGHT Ostium (Slice S: {ostium_z_R})")
    axes[0].axis('off')
    
    # Left Ostium (Channel 2 -> Red)
    axes[1].imshow(img_slice_L, cmap='gray', vmin=-500, vmax=800, origin='lower')
    axes[1].imshow(np.ma.masked_where(heat_slice_L < 0.05, heat_slice_L), cmap='Reds', alpha=0.7, origin='lower')
    axes[1].set_title(f"LEFT Ostium (Slice S: {ostium_z_L})")
    axes[1].axis('off')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_random_test()