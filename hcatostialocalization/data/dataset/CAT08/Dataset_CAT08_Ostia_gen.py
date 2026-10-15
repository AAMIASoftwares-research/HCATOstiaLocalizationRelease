import os
import sys
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import psutil
import gc
import random

# ==============================================================================
# 0. CONFIGURAZIONE PERCORSI (TRUCCHETTO VS CODE)
# ==============================================================================
# Questo blocco assicura che Python trovi la cartella 'hcatostialocalization'
current_dir = os.path.abspath(os.getcwd())
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)

from hcatostialocalization.config.config import config
from hcatostialocalization.data.dataset.Base_Classes_ProvaMarco import LabeledDataset
from hcatostialocalization.data.dataset.Heatmap_Config import HeatmapParams

# ==============================================================================
# 1. CLASSE DATASET UFFICIALE (Per Addestramento SwinUNETR)
# ==============================================================================
class Dataset_CAT08_Ostia(LabeledDataset):
    """
    Classe Dataset per CAT08.
    Gestisce il caricamento dei volumi NIfTI e la generazione di heatmap 3D.
    """
    labels: dict[int, str] = {
        0: "background",
        1: "RCA_ostium",
        2: "LCA_ostium",
    }

    def _init_data_pool(self) -> list[tuple[str, str]]:
        c = config.dataset.CAT08
        yaml_path = c.yaml_path 
        img_folder = c.image_path
        json_folder = c.json_path
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"[!] Config YAML non trovato in: {yaml_path}")

        with open(yaml_path, 'r') as f:
            split_dict = yaml.safe_load(f)['CAT08']['split']
        
        patient_ids = split_dict.get(self.split, [])
        data_pool = []
        
        for pid in patient_ids:
            im_f = os.path.join(img_folder, f"{pid}_volume.nii.gz")
            lb_f = os.path.join(json_folder, f"{pid}_labels.json")
            
            if os.path.exists(im_f) and os.path.exists(lb_f):
                data_pool.append((im_f, lb_f))
                
        print(f"[INFO] Dataset CAT08 - Split '{self.split}': {len(data_pool)} pazienti pronti.")
        return data_pool

    def _get_patient_id(self, current_patient: tuple[str, str]) -> str:
        pid = os.path.basename(current_patient[0]).split('_')[0]
        return f"{self.split}-CAT08-{pid}"

    def _load_data(self, image_label_f: tuple[str, str], new_spacing: float|tuple|None = None) -> dict:
        """Caricamento standard (usato dalla rete neurale durante il training)."""
        img_f, json_f = image_label_f
        dict_img = self._load_volume_resize(img_f, which='image', new_spacing=new_spacing)
        img_tensor = dict_img['tensor']
        label_tensor = torch.zeros_like(img_tensor, dtype=torch.float)

        if os.path.exists(json_f):
            with open(json_f, 'r') as f:
                json_data = json.load(f)
            
            ostia_block = json_data.get("ostia", {})
            sigma = HeatmapParams.sigma if hasattr(HeatmapParams, 'sigma') else 2.0
            
            for side in ["R", "L"]: 
                side_data = ostia_block.get(side)
                if side_data and "coordinate_ijk" in side_data:
                    coords = side_data["coordinate_ijk"] 
                    # Coordinate IJK -> PyTorch (K, J, I)
                    i, j, k = int(round(coords[0])), int(round(coords[1])), int(round(coords[2]))
                    self._draw_3d_gaussian(label_tensor[0, 0], center=(k, j, i), sigma=sigma)

        if self.split == 'train':
            img_tensor = img_tensor.to(torch.float16)
            label_tensor = label_tensor.to(torch.float16)
            
        return {
            'image': img_tensor, 'label': label_tensor, 
            'spacing': dict_img['spacing'], 'id': self._get_patient_id(image_label_f)
        }

    def _draw_3d_gaussian(self, heatmap_tensor: torch.Tensor, center: tuple[float, float, float], sigma: float):
        z_c, y_c, x_c = center
        D, H, W = heatmap_tensor.shape
        radius = int(3 * sigma)
        z0, z1 = max(0, int(z_c - radius)), min(D, int(z_c + radius) + 1)
        y0, y1 = max(0, int(y_c - radius)), min(H, int(y_c + radius) + 1)
        x0, x1 = max(0, int(x_c - radius)), min(W, int(x_c + radius) + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1: return
        
        z_grid = torch.arange(z0, z1, dtype=torch.float)
        y_grid = torch.arange(y0, y1, dtype=torch.float)
        x_grid = torch.arange(x0, x1, dtype=torch.float)
        grid_z, grid_y, grid_x = torch.meshgrid(z_grid, y_grid, x_grid, indexing='ij')
        dist_sq = (grid_z - z_c)**2 + (grid_y - y_c)**2 + (grid_x - x_c)**2
        gaussian = torch.exp(-dist_sq / (2 * sigma**2))
        heatmap_tensor[z0:z1, y0:y1, x0:x1] = torch.maximum(heatmap_tensor[z0:z1, y0:y1, x0:x1], gaussian)
    def __len__(self):
        # Ritorna il numero di pazienti caricati nello split corrente
        return len(self.data_pool)
# ==============================================================================
# 2. LOGICA DI TEST CASUALE (Anti-Crash & Lazy Loading)
# ==============================================================================
def run_random_test():
    print(f"[*] Monitoraggio RAM: {psutil.virtual_memory().percent}%")
    
    # Inizializziamo il dataset per leggere il data_pool (senza caricare volumi)
    ds = Dataset_CAT08_Ostia(split='val', nsdim=3)
    
    if not ds.data_pool:
        print("[!] Errore: data_pool vuoto. Controlla i percorsi.")
        return

    # Scegliamo un paziente a caso
    img_f, json_f = random.choice(ds.data_pool)
    patient_id = ds._get_patient_id((img_f, json_f))
    
    print(f"[*] --- ESTRAZIONE CASUALE: {patient_id} ---")

    # Leggiamo il JSON per trovare la slice Z dell'ostio
    with open(json_f, 'r') as f:
        json_data = json.load(f)
    
    try:
        # Proviamo a prendere l'ostio destro (R) per il plot
        coords = json_data["ostia"]["R"]["coordinate_ijk"]
        ostium_z = int(round(coords[2]))
        y_c, x_c = int(round(coords[1])), int(round(coords[0]))
    except KeyError:
        print(f"[!] Coordinate non trovate per {patient_id}. Riprova l'esecuzione.")
        return

    # CARICAMENTO SICURO (Lazy Loading della singola fetta)
    print(f"[*] Caricamento slice {ostium_z}...")
    nii_img = nib.load(img_f, mmap=True)
    # np.asarray + .T per avere l'orientamento corretto e troncare mmap
    image_slice = np.asarray(nii_img.dataobj[..., ostium_z], dtype=np.float32).T
    
    # Pulizia immediata
    del nii_img
    gc.collect()

    # Generazione Heatmap 2D per il controllo visivo
    H, W = image_slice.shape
    heatmap_slice = np.zeros((H, W), dtype=np.float32)
    sigma = 2.0
    radius = int(3 * sigma)
    y0, y1 = max(0, int(y_c - radius)), min(H, int(y_c + radius) + 1)
    x0, x1 = max(0, int(x_c - radius)), min(W, int(x_c + radius) + 1)
    y_grid = np.arange(y0, y1, dtype=np.float32)
    x_grid = np.arange(x0, x1, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y_grid, x_grid, indexing='ij')
    dist_sq = (grid_y - y_c)**2 + (grid_x - x_c)**2
    gaussian = np.exp(-dist_sq / (2 * sigma**2))
    heatmap_slice[y0:y1, x0:x1] = np.maximum(heatmap_slice[y0:y1, x0:x1], gaussian)

    # VISUALIZZAZIONE
    print(f"[*] Apertura grafico... (Pixel range: {image_slice.min():.1f} to {image_slice.max():.1f})")
    plt.figure(figsize=(12, 6))
    
    # Sottoplot 1: TAC Originale
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title(f"TAC {patient_id} | Slice {ostium_z}")
    plt.axis('off')

    # Sottoplot 2: Overlay Heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(image_slice, cmap='gray')
    heatmap_masked = np.ma.masked_where(heatmap_slice < 0.05, heatmap_slice)
    plt.imshow(heatmap_masked, cmap='jet', alpha=0.6) 
    plt.title("Controllo Heatmap (Ostio R)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. PUNTO DI INGRESSO
# ==============================================================================
if __name__ == "__main__":
    # Esegue il test casuale sicuro
    run_random_test()