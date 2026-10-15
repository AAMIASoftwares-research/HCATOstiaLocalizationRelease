import yaml
import os
import shutil
import random
import re
from omegaconf import OmegaConf
##Tova le cartelle per poter lavorare su ASOCA
path_to_env_config = "hcatostialocalization/config/machine.yaml"
conf = OmegaConf.load(path_to_env_config)

try:
        output_root = conf.active_paths.output_dir
        data_root = conf.active_paths.data_root
        print(f"[*] Macchina rilevata: {conf.host_settings.current_host}")
        print(f"[*] Data Root: {data_root}")
        print(f"[*] Output Root: {output_root}")
except Exception as e:
    print(f"[!] ERRORE: Impossibile trovare i percorsi per questa macchina. {e}")
    exit()

## CODICE PER LA DIVISIONE DEL DATASET
def prepare_asoca_normal_dataset(data_path, annotations_path_normal, annotations_path_diseased, output_yaml_path, data_root_placeholder):
    # --- 1. LETTURA DEGLI ID ANNOTATI ---
    print(f"[*] Lettura cartella annotazioni: {annotations_path_normal}")
    annotated_ids = set()
    
    if not os.path.exists(annotations_path_normal):
        print(f"[!] ERRORE: Cartella annotazioni non trovata: {annotations_path_normal}")
        return
    if not os.path.exists(annotations_path_diseased):
        print(f"[!] ERRORE: Cartella annotazioni non trovata: {annotations_path_diseased}")
        return
    
    for json_file in os.listdir(annotations_path_normal):
        if json_file.endswith('.json'):
            # Estrae l'ID (es. "Normal_001")
            match = re.search('(Normal_\\d+)', json_file)
            if match:
                annotated_ids.add(match.group(1))
    for json_file in os.listdir(annotations_path_diseased):
        if json_file.endswith('.json'):
            # Estrae l'ID (es. "Diseased_001")
            match = re.search('(Diseased_\\d+)', json_file)
            if match:
                annotated_ids.add(match.group(1))
                
    print(f"    -> Trovati {len(annotated_ids)} casi ASOCA con annotazioni.\n")

    # --- 2. PREPARAZIONE CARTELLA DI OUTPUT ---
    # Creiamo una cartella specifica per ASOCA Normal estratto
    extracted_dir = output_yaml_path
    
    all_samples = []

    # --- 3. SCANSIONE DATI ASOCA (Normal e Diseased) ---
    # Assumiamo che data_path sia la radice che contiene le sottocartelle
    # oppure eseguiamo un ciclo su entrambi i percorsi.
    
    # Esempio: paths_to_scan = [".../ASOCA/Normal/CTCA", ".../ASOCA/Diseased/CTCA"]
    # Se ricevi un solo data_path, puoi gestirlo così:
    paths_to_scan = [
        os.path.join(data_path, "Normal", "CTCA"),
        os.path.join(data_path, "Diseased", "CTCA")
    ]

    for current_path in paths_to_scan:
        if not os.path.exists(current_path):
            print(f"[!] AVVISO: Cartella non trovata, salto: {current_path}")
            continue

        print(f"[*] Filtro e copia dei file .nrrd da: {current_path}")
        
        for item in os.listdir(current_path):
            if item.endswith('.nrrd'):
                
                # MODIFICA REGEX: Ora accetta sia 'Normal_...' che 'Diseased_...'
                match = re.search('((Normal|Diseased)_\\d+)', item)
                
                if not match:
                    continue
                    
                patient_id = match.group(1)
                
                # Controllo incrociato con il set creato al punto 1
                if patient_id in annotated_ids:
                    # Evitiamo duplicati nel caso un ID appaia due volte per errore
                    if patient_id not in all_samples:
                        all_samples.append(patient_id)
                        print(f"    [+] Aggiunto: {patient_id}.nrrd")

    if not all_samples:
        print("[!] ATTENZIONE: Nessun file .nrrd trovato o nessuna corrispondenza con le annotazioni.")
        return

    # --- 4. CREAZIONE SPLIT ---
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

    # --- 5. SALVATAGGIO ---
    with open(f"{output_yaml_path}/asoca_config.yaml", 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)
    
    print(f"\n[+] Completato! File YAML creato: {output_yaml_path}")
    print(f"    -> Split: {len(config_data['ASOCA']['split']['train'])} Train | "
          f"{len(config_data['ASOCA']['split']['val'])} Val | "
          f"{len(config_data['ASOCA']['split']['test'])} Test")

# --- ESECUZIONE ---
data_path = data_root + "/Datasets/ASOCA"
percorso_annotazioni_normal = data_root + "/OstiumAnnotations_ASOCA/Normal_ijk_ras_json"
percorso_annotazioni_diseased = data_root + "/OstiumAnnotations_ASOCA/Diseased_ijk_ras_json"
percorso_output_yaml = output_root + "/hcatostialocalization/data/dataset/ASOCA"
percorso_dati_normal = data_root + "/Datasets/ASOCA/Normal/CTCA"
percorso_dati_diseased = data_root + "/Datasets/ASOCA/Diseased/CTCA"
prepare_asoca_normal_dataset(data_path, percorso_annotazioni_normal, percorso_annotazioni_diseased, percorso_output_yaml, data_root)
