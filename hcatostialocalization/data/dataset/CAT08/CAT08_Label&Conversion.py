import os
import sys
import json
from pathlib import Path
import SimpleITK as sitk

# ==============================================================================
# 0. PROJECT PATH RESOLUTION (To import config)
# ==============================================================================
current_dir = os.path.abspath(os.getcwd())
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)

# Import the config object
from hcatostialocalization.config.config import config

# ==============================================================================
# 1. DYNAMIC PATH CONFIGURATION (Via machine.yaml)
# ==============================================================================
# Instead of a fixed path, use the data_root specific to the machine running the code
DATA_ROOT = Path(config.active_paths.data_root)

# INPUT directory (where dataset00, dataset01... subfolders are located)
BASE_DIR = DATA_ROOT / "Datasets" / "CAT08"

# OUTPUT directories (where converted files will be saved)
OUTPUT_IMG_DIR = BASE_DIR / "Extracted_Images_CAT08"
OUTPUT_JSON_DIR = DATA_ROOT / "OstiumAnnotations_CAT08" / "JSON_Completi"

# Create directories if they do not exist
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[*] Starting CAT08 Standardization...")
    print(f"[*] Data Root Directory (Dynamic): {DATA_ROOT}")
    print(f"[*] Input Directory: {BASE_DIR}")
    
    dataset_folders = sorted([f for f in BASE_DIR.glob("dataset*") if f.is_dir()])
    
    if not dataset_folders:
        print("[!] ERROR: No 'datasetXX' folders found in Input.")
        print("    Verify that data_root in your machine.yaml is correct!")
        return

    print(f"[*] Found {len(dataset_folders)} patients. Starting processing...\n")

    for folder in dataset_folders:
        patient_id = folder.name
        mhd_files = list(folder.glob("*.mhd"))
        
        if not mhd_files:
            print(f"  [!] Skipping {patient_id}: no .mhd file found.")
            continue
            
        mhd_path = mhd_files[0]
        nifti_path = OUTPUT_IMG_DIR / f"{patient_id}_volume.nii.gz"
        json_path = OUTPUT_JSON_DIR / f"{patient_id}_labels.json"
        
        # --- PHASE 1: IMAGE CONVERSION TO RAS (.nii.gz) ---
        print(f" -> Processing {patient_id}...")
        img = sitk.ReadImage(str(mhd_path))
        img_ras = sitk.DICOMOrient(img, 'RAS')
        img_ras = sitk.Cast(img_ras, sitk.sitkInt16)
        sitk.WriteImage(img_ras, str(nifti_path), useCompression=True)
        
        # --- PHASE 2: COORDINATE EXTRACTION AND JSON CREATION ---
        patient_labels = {
            "id_immagine": patient_id,
            "spacing": list(img_ras.GetSpacing()),
            "orientation": list(img_ras.GetDirection()),
            "origin": list(img_ras.GetOrigin()),
            "ostia": {}, 
            "comment": "Coordinates converted to RAS and IJK",
            "other": {}
        }
        
        vessel_mapping = {0: "R", 1: "L"}
        
        for v_idx, side in vessel_mapping.items():
            vessel_dir = folder / f"vessel{v_idx}"
            target_files = ["pointS.txt", "reference.txt"]
            
            for filename in target_files:
                txt_path = vessel_dir / filename
                if txt_path.exists():
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                        if lines and lines[0].strip():
                            try:
                                # 1. Raw original coordinates (LPS space)
                                lps_coords = [float(c) for c in lines[0].strip().split()[:3]]
                                
                                # 2. Mathematical transformation to RAS space
                                ras_coords = [-lps_coords[0], -lps_coords[1], lps_coords[2]]
                                
                                # 3. Voxel (IJK) extraction on the newly oriented image!
                                voxel_idx = img_ras.TransformPhysicalPointToIndex(lps_coords)
                                
                                patient_labels["ostia"][side] = {
                                    "coordinate_ras": ras_coords,
                                    "coordinate_ijk": list(voxel_idx),
                                    "world_mm_original_lps": lps_coords, 
                                    "source_file": filename
                                }
                                break 
                            except Exception as e:
                                print(f"    [!] Error reading coordinates in {txt_path}: {e}")

        # Final JSON saving
        if patient_labels["ostia"]:
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(patient_labels, jf, indent=4)

    print("\n[SUCCESS] Conversion and alignment completed for the whole CAT08 dataset!")

if __name__ == "__main__":
    main()