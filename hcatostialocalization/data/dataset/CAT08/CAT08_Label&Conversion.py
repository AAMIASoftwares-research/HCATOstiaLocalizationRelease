import os
import sys
import json
from pathlib import Path
import SimpleITK as sitk

# ==============================================================================
# 0. RISOLUZIONE DEL PERCORSO DI PROGETTO (Per poter importare il config)
# ==============================================================================
current_dir = os.path.abspath(os.getcwd())
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)

# Importiamo l'oggetto config magico creato dal vostro team!
from hcatostialocalization.config.config import config

# ==============================================================================
# 1. CONFIGURAZIONE PERCORSI DINAMICI (Via machine.yaml)
# ==============================================================================
# Invece del path fisso, prendiamo il data_root specifico per il PC che sta eseguendo il codice
DATA_ROOT = Path(config.active_paths.data_root)

# Cartella di INPUT (dove ci sono le sottocartelle dataset00, dataset01...)
BASE_DIR = DATA_ROOT / "Datasets" / "CAT08"

# Cartelle di OUTPUT (Dove andranno i file convertiti)
OUTPUT_IMG_DIR = BASE_DIR / "Extracted_Images_CAT08"
OUTPUT_JSON_DIR = DATA_ROOT / "OstiumAnnotations_CAT08" / "JSON_Completi"

# Creiamo le cartelle se non esistono
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"[*] Inizio Standardizzazione CAT08...")
    print(f"[*] Cartella Data Root (Dinamica): {DATA_ROOT}")
    print(f"[*] Cartella Input: {BASE_DIR}")
    
    dataset_folders = sorted([f for f in BASE_DIR.glob("dataset*") if f.is_dir()])
    
    if not dataset_folders:
        print("[!] ERRORE: Nessuna cartella 'datasetXX' trovata in Input.")
        print("    Verifica che il data_root nel tuo machine.yaml sia corretto!")
        return

    print(f"[*] Trovati {len(dataset_folders)} pazienti. Inizio elaborazione...\n")

    for folder in dataset_folders:
        patient_id = folder.name
        mhd_files = list(folder.glob("*.mhd"))
        
        if not mhd_files:
            print(f"  [!] Salto {patient_id}: nessun file .mhd trovato.")
            continue
            
        mhd_path = mhd_files[0]
        nifti_path = OUTPUT_IMG_DIR / f"{patient_id}_volume.nii.gz"
        json_path = OUTPUT_JSON_DIR / f"{patient_id}_labels.json"
        
        # --- FASE 1: CONVERSIONE IMMAGINE IN RAS (.nii.gz) ---
        print(f" -> Elaboro {patient_id}...")
        img = sitk.ReadImage(str(mhd_path))
        img_ras = sitk.DICOMOrient(img, 'RAS')
        img_ras = sitk.Cast(img_ras, sitk.sitkInt16)
        sitk.WriteImage(img_ras, str(nifti_path), useCompression=True)
        
        # --- FASE 2: ESTRAZIONE COORDINATE E CREAZIONE JSON ---
        patient_labels = {
            "id_immagine": patient_id,
            "spacing": list(img_ras.GetSpacing()),
            "orientation": list(img_ras.GetDirection()),
            "origin": list(img_ras.GetOrigin()),
            "ostia": {}, 
            "comment": "Coordinate convertite in RAS e IJK",
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
                                # 1. Coordinate originali grezze (Spazio LPS)
                                lps_coords = [float(c) for c in lines[0].strip().split()[:3]]
                                
                                # 2. Trasformazione matematica in spazio RAS
                                ras_coords = [-lps_coords[0], -lps_coords[1], lps_coords[2]]
                                
                                # 3. Estrazione Voxel (IJK) sull'immagine appena girata!
                                voxel_idx = img_ras.TransformPhysicalPointToIndex(lps_coords)
                                
                                patient_labels["ostia"][side] = {
                                    "coordinate_ras": ras_coords,
                                    "coordinate_ijk": list(voxel_idx),
                                    "world_mm_original_lps": lps_coords, 
                                    "source_file": filename
                                }
                                break 
                            except Exception as e:
                                print(f"    [!] Errore lettura coordinate in {txt_path}: {e}")

        # Salvataggio JSON definitivo
        if patient_labels["ostia"]:
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(patient_labels, jf, indent=4)

    print("\n[SUCCESS] Conversione e allineamento completati per tutto il dataset CAT08!")

if __name__ == "__main__":
    main()