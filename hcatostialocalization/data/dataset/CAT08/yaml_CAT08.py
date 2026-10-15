import os
import random
import yaml
import re

def prepare_cat08_dataset_no_copy(data_path, annotations_path, output_yaml_path):
    # --- 1. LETTURA DEGLI ID ANNOTATI ---
    print(f"[*] Scansione annotazioni in: {annotations_path}")
    annotated_ids = set()
    
    if not os.path.exists(annotations_path):
        print(f"[!] ERRORE: Cartella annotazioni non trovata: {annotations_path}")
        return
        
    for json_file in os.listdir(annotations_path):
        if json_file.endswith('.json'):
            # Regex per estrarre 'datasetXX'
            match = re.search(r'(dataset\d+)', json_file)
            if match:
                annotated_ids.add(match.group(1))
                
    print(f"    -> Trovate {len(annotated_ids)} annotazioni JSON.\n")

    # --- 2. VERIFICA ESISTENZA IMMAGINI  ---
    all_samples = []
    print("[*] Verifica corrispondenza cartelle immagini...")

    for item in os.listdir(data_path):
        full_path = os.path.join(data_path, item)
        
        # Verifichiamo se l'ID è tra quelli annotati e se la cartella esiste
        if os.path.isdir(full_path) and item.startswith('dataset'):
            if item in annotated_ids:
                # Controllo rapido: esiste almeno il file .mhd dentro?
                has_mhd = any(f.endswith('.mhd') for f in os.listdir(full_path))
                if has_mhd:
                    all_samples.append(item)
                else:
                    print(f"    [!] Avviso: Cartella {item} trovata ma manca il file .mhd")

    print(f"[+] Verifica completata. {len(all_samples)} volumi pronti per lo split.")

    # --- 3. CREAZIONE DELLO SPLIT ---
    random.seed(42) 
    random.shuffle(all_samples)

    total = len(all_samples)
    train_perc, val_perc = 0.8, 0.1
    train_end = int(total * train_perc)
    val_end = train_end + int(total * val_perc)

    # Nota: image_path ora punta alla cartella principale CAT08
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

    # --- 4. SALVATAGGIO YAML ---
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)
    
    print(f"\n[+] File YAML creato con successo in: {output_yaml_path}")
    print(f"    -> Split: {len(config_data['CAT08']['split']['train'])} Train | "
          f"{len(config_data['CAT08']['split']['val'])} Val | "
          f"{len(config_data['CAT08']['split']['test'])} Test")


if __name__ == "__main__":
    # --- PERCORSI ---
    data_root = "/path/to/data"
    output_dir = "/path/to/data"

    p_dati = os.path.join(data_root, "Datasets", "CAT08")
    p_ann = os.path.join(data_root, "OstiumAnnotations_CAT08")
    p_yaml = os.path.join(output_dir, "hcatostialocalization", "data", "dataset", "CAT08", "cat08_config.yaml")
    
    prepare_cat08_dataset_no_copy(p_dati, p_ann, p_yaml)