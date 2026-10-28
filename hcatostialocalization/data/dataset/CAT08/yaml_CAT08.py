import os
import random
import yaml
import re

def prepare_cat08_dataset_no_copy(data_path, annotations_path, output_yaml_path):
    # --- READING ANNOTATED IDS ---
    print(f"[*] Scanning annotations in: {annotations_path}")
    annotated_ids = set()
    
    if not os.path.exists(annotations_path):
        print(f"[!] ERROR: Annotations directory not found: {annotations_path}")
        return
        
    for json_file in os.listdir(annotations_path):
        if json_file.endswith('.json'):
            # Regex to extract 'datasetXX'
            match = re.search(r'(dataset\d+)', json_file)
            if match:
                annotated_ids.add(match.group(1))
                
    print(f"    -> Found {len(annotated_ids)} JSON annotations.\n")

    # --- CHECKING IMAGE EXISTENCE ---
    all_samples = []
    print("[*] Checking image folder matches...")

    for item in os.listdir(data_path):
        full_path = os.path.join(data_path, item)
        
        # Check if ID is among annotated and if folder exists
        if os.path.isdir(full_path) and item.startswith('dataset'):
            if item in annotated_ids:
                # Quick check: does at least one .mhd file exist inside?
                has_mhd = any(f.endswith('.mhd') for f in os.listdir(full_path))
                if has_mhd:
                    all_samples.append(item)
                else:
                    print(f"    [!] Warning: Folder {item} found but .mhd file is missing")

    print(f"[+] Verification completed. {len(all_samples)} volumes ready for splitting.")

    # --- CREATING SPLIT ---
    random.seed(42) 
    random.shuffle(all_samples)

    total = len(all_samples)
    train_perc, val_perc = 0.8, 0.1
    train_end = int(total * train_perc)
    val_end = train_end + int(total * val_perc)

    # Note: image_path now points to the main CAT08 directory
    config_data = {
        'dataset': {
            'CAT08': {
                'yaml_path': "${active_paths.output_dir}/hcatostialocalization/data/dataset/CAT08/cat08_config.yaml",
                'image_path': "${active_paths.data_root}/Datasets/CAT08",
                'json_path': "${active_paths.data_root}/OstiumAnnotations_CAT08"
            }
        },
        'CAT08': {
            'split': {
                'train': sorted(all_samples[:train_end]),
                'val': sorted(all_samples[train_end:val_end]),
                'test': sorted(all_samples[val_end:])
            }
        }
    }

    # --- SAVING YAML ---
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)
    
    print(f"\n[+] YAML file successfully created in: {output_yaml_path}")
    print(f"    -> Split: {len(config_data['CAT08']['split']['train'])} Train | "
          f"{len(config_data['CAT08']['split']['val'])} Val | "
          f"{len(config_data['CAT08']['split']['test'])} Test")


if __name__ == "__main__":
    
    data_root = "/path/to/data"
    output_dir = "/path/to/data"

    p_dati = os.path.join(data_root, "Datasets", "CAT08")
    p_ann = os.path.join(data_root, "OstiumAnnotations_CAT08")
    p_yaml = os.path.join(output_dir, "hcatostialocalization", "data", "dataset", "CAT08", "cat08_config.yaml")
    
    prepare_cat08_dataset_no_copy(p_dati, p_ann, p_yaml)