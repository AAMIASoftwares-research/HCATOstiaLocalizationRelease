import os
import sys
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import psutil
import random

# ==============================================================================
# 0. CONFIGURAZIONE PERCORSI 
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
# 1. CLASSE DATASET MULTI-CLASSE (3 CANALI GAUSSIANI)
# ==============================================================================
class Dataset_CAT08_Ostia_Multi(LabeledDataset):
    labels: dict[int, str] = {
        0: "background",
        1: "RCA_ostium",
        2: "LCA_ostium",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma: float = HeatmapParams.sigma

    def _init_data_pool(self) -> list[tuple[str, str]]:
        # Coppie (immagine, label) esplicite, definite nello split YAML
        # (cat08_config.yaml, percorso in config/dataset.yaml -> dataset.CAT08.yaml_path)
        return self._data_pool_from_split_yaml("CAT08")

    def _get_patient_id(self, current_patient: tuple[str, str]) -> str:
        pid = os.path.basename(current_patient[0]).split('.')[0]  # imageXX.mhd -> imageXX
        return f"{self.split}-CAT08-Multi-{pid}"

    # I volumi CAT08 sono salvati con un offset di +1024 (valori in [0, 4096]) invece che in HU.
    # Un volume in HU contiene sempre aria/polmone (molto sotto -200 HU): se non c'e' nessun
    # valore sotto HU_SHIFT_THRESHOLD, il volume e' shiftato e va riportato in HU.
    HU_OFFSET = 1024.0
    HU_SHIFT_THRESHOLD = -200.0
    _hu_shift_reported: set = set()

    def _load_data(self, image_label_f: tuple[str, str], new_spacing: float|tuple|None = None) -> dict:
        # Immagine caricata in RAS (1, 1, R, A, S) con MONAI; target [3, R, A, S] costruito
        # dalle coordinate RAS world del JSON (vedi LabeledDataset._build_ostia_target).
        sample = self._load_ostia_data(image_label_f, new_spacing=new_spacing)

        img = sample['image']
        if float(img.min()) >= self.HU_SHIFT_THRESHOLD:
            # torch.tensor con lo stesso dtype: evita promozioni implicite a float32 (train = float16)
            img = img - torch.tensor(self.HU_OFFSET, dtype=img.dtype, device=img.device)
            sample['image'] = img
            path = image_label_f[0]
            if path not in self._hu_shift_reported:
                self._hu_shift_reported.add(path)
                msg = f"[CAT08] {os.path.basename(path)}: nessun valore sotto {self.HU_SHIFT_THRESHOLD:.0f} -> shift di -{self.HU_OFFSET:.0f} (HU ricostruiti, min={float(img.min()):.0f})"
                if float(img.min()) >= self.HU_SHIFT_THRESHOLD:
                    msg += " [ATTENZIONE: anche dopo lo shift nessun valore sotto la soglia, volume anomalo]"
                # print(msg) # do not send useless messages, used only for debugging
        return sample

    def __len__(self):
        return len(self.data_pool)

# ==============================================================================
# 2. LOGICA DI TEST CASUALE (Visualizzazione 5 Pazienti)
# ==============================================================================
def run_random_test():
    print(f"[*] Monitoraggio RAM: {psutil.virtual_memory().percent}%")
    ds = Dataset_CAT08_Ostia_Multi(split='val', nsdim=3)
    
    if not ds.data_pool: 
        print("[!] Dataset vuoto. Hai eseguito la conversione?")
        return

    n_samples = min(5, len(ds.data_pool))
    campioni_scelti = random.sample(ds.data_pool, n_samples)
    print(f"\n[*] --- ESTRAZIONE CASUALE DI {n_samples} PAZIENTI ---")

    for img_f, json_f in campioni_scelti:
        patient_id = ds._get_patient_id((img_f, json_f))
        print(f"-> Controllo: {patient_id}")


        sample = ds._load_data((img_f, json_f))
        if float(sample['ostia_coords_R'][0]) < 0 or float(sample['ostia_coords_L'][0]) < 0:
            print('[!] Ostio R o L mancante nella label. Salto.')
            continue
        # Indice della slice assiale = asse S (k) della griglia RAS caricata
        ostium_z_R = int(round(float(sample["ostia_coords_R"][2])))
        ostium_z_L = int(round(float(sample["ostia_coords_L"][2])))
        image_tensor = sample['image'].squeeze().numpy()  
        label_tensor = sample['label'].numpy()            
        
        img_slice_R = image_tensor[:, :, ostium_z_R].T
        heat_slice_R = label_tensor[1, :, :, ostium_z_R].T 

        img_slice_L = image_tensor[:, :, ostium_z_L].T
        heat_slice_L = label_tensor[2, :, :, ostium_z_L].T 

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Paziente (NUOVO NIFTI): {patient_id}", fontsize=14, fontweight='bold')
        
        axes[0].imshow(img_slice_R, cmap='gray', origin='lower')
        axes[0].imshow(np.ma.masked_where(heat_slice_R < 0.05, heat_slice_R), cmap='Greens', alpha=0.6, origin='lower')
        axes[0].set_title(f"Ostio DESTRO (Slice S: {ostium_z_R})")
        axes[0].axis('off')

        axes[1].imshow(img_slice_L, cmap='gray', origin='lower')
        axes[1].imshow(np.ma.masked_where(heat_slice_L < 0.05, heat_slice_L), cmap='Reds', alpha=0.6, origin='lower')
        axes[1].set_title(f"Ostio SINISTRO (Slice S: {ostium_z_L})")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_random_test()
