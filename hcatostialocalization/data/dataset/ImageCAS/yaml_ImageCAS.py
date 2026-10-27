import os
import shutil
import random
import yaml
import re
import argparse

def prepare_imagecas_dataset(data_path, annotations_path, output_yaml_path):
    # --- 1. READING ANNOTATED IDS ---
    print(f"[*] Reading annotations directory: {annotations_path}")
    annotated_ids = set()
    
    if not os.path.exists(annotations_path):
        print(f"[!] ERROR: Annotations directory not found: {annotations_path}")
        return
        
    for json_file in os.listdir(annotations_path):
        if json_file.endswith('.json'):
            # Extract ID (initial numbers) from JSON file name (e.g. "001.json" -> "001")
            match = re.match(r'^(\d+)', json_file)
            if match:
                annotated_ids.add(match.group(1))
                
    print(f"    -> Found {len(annotated_ids)} patients with valid annotations.\n")

    # --- 2. PREPARING OUTPUT DIRECTORY ---
    base_dir = os.path.dirname(data_path)
    extracted_dir = os.path.join(base_dir, "Extracted_Images")
    os.makedirs(extracted_dir, exist_ok=True)
    
    print(f"[*] Destination directory for filtered files: {extracted_dir}")
    print("[*] Starting scan, filter and copy...\n")
    
    all_samples = []

    # --- 3. SCANNING DATA (with Label and ID filters) ---
    for item in os.listdir(data_path):
        full_path = os.path.join(data_path, item)
        
        # CASE A: It is a directory (e.g. folder "001")
        if os.path.isdir(full_path):
            patient_id = item
            
            # Check if this ID has an annotation
            if patient_id not in annotated_ids:
                continue
                
            # Look for .nii or .nii.gz files, IGNORING those containing 'label'
            nii_files = [f for f in os.listdir(full_path) 
                         if (f.endswith('.nii') or f.endswith('.nii.gz')) and 'label' not in f.lower()]
            
            if nii_files:
                src_file = os.path.join(full_path, nii_files[0])
                ext = '.nii.gz' if src_file.endswith('.gz') else '.nii'
                dst_file = os.path.join(extracted_dir, f"{patient_id}{ext}")
                
                shutil.copy2(src_file, dst_file)
                all_samples.append(patient_id)

        # CASE B: It is directly a file (e.g. "001.image.nii.gz")
        elif item.endswith('.nii.gz') or item.endswith('.nii'):
            # Skip label files
            if 'label' in item.lower():
                continue
                
            # Extract ID from image file name
            match = re.match(r'^(\d+)', item)
            if not match:
                continue
                
            patient_id = match.group(1)
            
            # Check if this ID has an annotation
            if patient_id not in annotated_ids:
                continue
                
            dst_file = os.path.join(extracted_dir, f"{patient_id}.nii.gz" if item.endswith('.gz') else f"{patient_id}.nii")
            shutil.copy2(full_path, dst_file)
            all_samples.append(patient_id)

    print(f"[+] Filtering completed! Copied {len(all_samples)} images ('image' versions only with annotation).")

    if len(all_samples) == 0:
        print("\n[!] WARNING: 0 images copied. Check that JSON and image file names start with the same ID.")
        return

    # --- 4. CREATING SPLIT AND OMEGACONF CONFIGURATION ---
    random.seed(42) 
    random.shuffle(all_samples)

    total = len(all_samples)
    train_perc, val_perc = 0.8, 0.1
    train_end = int(total * train_perc)
    val_end = train_end + int(total * val_perc)

    # NEW DICTIONARY FORMAT (with OmegaConf variables)
    config_data = {
        'dataset': {
            'ImageCAS': {
                'yaml_path': "${active_paths.output_dir}/hcatostialocalization/data/dataset/ImageCAS/imagecas_config.yaml",
                'image_path': "${oc.select:active_paths.imagecas_image_path,${active_paths.data_root}/Datasets/ImageCAS/Extracted_Images}",
                'json_path': "${oc.select:active_paths.imagecas_json_path,${active_paths.data_root}/OstiumAnnotations_ImageCAS/Annotations ImageCAS_RAS_IJK}"
            }
        },
        'ImageCAS': {
            'split': {
                'train': all_samples[:train_end],
                'val': all_samples[train_end:val_end],
                'test': all_samples[val_end:]
            }
        }
    }

    # --- 5. SAVING YAML ---
    with open(output_yaml_path, 'w') as f:
        # sort_keys=False prevents yaml from sorting keys alphabetically, 
        # keeping 'dataset' before 'ImageCAS'
        yaml.dump(config_data, f, sort_keys=False)
    
    print(f"\n[+] YAML file successfully created in: {output_yaml_path}")
    print(f"    -> Split generated on {total} files: {len(config_data['ImageCAS']['split']['train'])} Train | "
          f"{len(config_data['ImageCAS']['split']['val'])} Val | "
          f"{len(config_data['ImageCAS']['split']['test'])} Test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ImageCAS images for training and regenerate imagecas_config.yaml."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Folder containing ImageCAS source images, e.g. .../ImageCAS/Data.",
    )
    parser.add_argument(
        "--annotations-path",
        required=True,
        help="Folder containing ImageCAS ostium annotation JSON files.",
    )
    parser.add_argument(
        "--output-yaml-path",
        required=True,
        help="Path where imagecas_config.yaml will be written.",
    )
    args = parser.parse_args()
    
    prepare_imagecas_dataset(args.data_path, args.annotations_path, args.output_yaml_path)
