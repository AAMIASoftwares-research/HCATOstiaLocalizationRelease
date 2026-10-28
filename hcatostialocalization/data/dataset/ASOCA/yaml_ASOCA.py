import yaml
import os
import shutil
import random
import re
from omegaconf import OmegaConf
## Find folders to work on ASOCA
path_to_env_config = "hcatostialocalization/config/machine.yaml"
conf = OmegaConf.load(path_to_env_config)

try:
        output_root = conf.active_paths.output_dir
        data_root = conf.active_paths.data_root
        print(f"[*] Detected machine: {conf.host_settings.current_host}")
        print(f"[*] Data Root: {data_root}")
        print(f"[*] Output Root: {output_root}")
except Exception as e:
    print(f"[!] ERROR: Unable to find paths for this machine. {e}")
    exit()

## DATASET SPLIT CODE
def prepare_asoca_normal_dataset(data_path, annotations_path_normal, annotations_path_diseased, output_yaml_path, data_root_placeholder):
    # --- READING ANNOTATED IDS ---
    print(f"[*] Reading annotations directory: {annotations_path_normal}")
    annotated_ids = set()
    
    if not os.path.exists(annotations_path_normal):
        print(f"[!] ERROR: Annotations directory not found: {annotations_path_normal}")
        return
    if not os.path.exists(annotations_path_diseased):
        print(f"[!] ERROR: Annotations directory not found: {annotations_path_diseased}")
        return
    
    for json_file in os.listdir(annotations_path_normal):
        if json_file.endswith('.json'):
            # Extract ID (e.g. "Normal_001")
            match = re.search('(Normal_\\d+)', json_file)
            if match:
                annotated_ids.add(match.group(1))
    for json_file in os.listdir(annotations_path_diseased):
        if json_file.endswith('.json'):
            # Extract ID (e.g. "Diseased_001")
            match = re.search('(Diseased_\\d+)', json_file)
            if match:
                annotated_ids.add(match.group(1))
                
    print(f"    -> Found {len(annotated_ids)} ASOCA cases with annotations.\n")

    # --- PREPARING OUTPUT DIRECTORY ---
    # Create a specific directory for extracted ASOCA Normal
    extracted_dir = output_yaml_path
    
    all_samples = []

    # --- SCANNING ASOCA DATA (Normal and Diseased) ---
    # Assume data_path is the root containing subfolders
    # or iterate over both paths.
    
    # Example: paths_to_scan = [".../ASOCA/Normal/CTCA", ".../ASOCA/Diseased/CTCA"]
    # If a single data_path is provided, handle it as:
    paths_to_scan = [
        os.path.join(data_path, "Normal", "CTCA"),
        os.path.join(data_path, "Diseased", "CTCA")
    ]

    for current_path in paths_to_scan:
        if not os.path.exists(current_path):
            print(f"[!] WARNING: Directory not found, skipping: {current_path}")
            continue

        print(f"[*] Filtering and copying .nrrd files from: {current_path}")
        
        for item in os.listdir(current_path):
            if item.endswith('.nrrd'):
                
                # REGEX MODIFICATION: Now accepts both 'Normal_...' and 'Diseased_...'
                match = re.search('((Normal|Diseased)_\\d+)', item)
                
                if not match:
                    continue
                    
                patient_id = match.group(1)
                
                # Cross-check with the set created in step 1
                if patient_id in annotated_ids:
                    # Avoid duplicates in case an ID appears twice by mistake
                    if patient_id not in all_samples:
                        all_samples.append(patient_id)
                        print(f"    [+] Added: {patient_id}.nrrd")

    if not all_samples:
        print("[!] WARNING: No .nrrd files found or no match with annotations.")
        return

    # --- CREATING SPLIT ---
    random.seed(42)
    random.shuffle(all_samples)
    
    total = len(all_samples)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)

    config_data = {
        'dataset': {
            'ASOCA': {
                'yaml_path': f"{extracted_dir}/asoca_config.yaml",
                'image_path_normal': f"{data_root_placeholder}/Datasets/ASOCA/Normal/CTCA",
                'image_path_diseased': f"{data_root_placeholder}/Datasets/ASOCA/Diseased/CTCA",
                'json_path_normal': f"{data_root_placeholder}/OstiumAnnotations_ASOCA/Normal_ijk_ras_json",                'json_path_diseased': f"{data_root_placeholder}/OstiumAnnotations_ASOCA/Diseased_ijk_ras_json"
            }
        },
        'ASOCA': {
            'split': {
                'train': all_samples[:train_end],
                'val': all_samples[train_end:val_end],
                'test': all_samples[val_end:]
            }
        }
    }

    # --- SAVING ---
    with open(f"{output_yaml_path}/asoca_config.yaml", 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)
    
    print(f"\n[+] Completed! YAML file created: {output_yaml_path}")
    print(f"    -> Split: {len(config_data['ASOCA']['split']['train'])} Train | "
          f"{len(config_data['ASOCA']['split']['val'])} Val | "
          f"{len(config_data['ASOCA']['split']['test'])} Test")

# --- EXECUTION ---
data_path = data_root + "/Datasets/ASOCA"
percorso_annotazioni_normal = data_root + "/OstiumAnnotations_ASOCA/Normal_ijk_ras_json"
percorso_annotazioni_diseased = data_root + "/OstiumAnnotations_ASOCA/Diseased_ijk_ras_json"
percorso_output_yaml = output_root + "/hcatostialocalization/data/dataset/ASOCA"
percorso_dati_normal = data_root + "/Datasets/ASOCA/Normal/CTCA"
percorso_dati_diseased = data_root + "/Datasets/ASOCA/Diseased/CTCA"
prepare_asoca_normal_dataset(data_path, percorso_annotazioni_normal, percorso_annotazioni_diseased, percorso_output_yaml, data_root)
