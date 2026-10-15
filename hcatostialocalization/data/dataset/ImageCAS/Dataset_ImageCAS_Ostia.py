import os
import sys
import json
import torch
import yaml
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Scopriamo dove VS Code ha fatto partire questo notebook
current_dir = os.path.abspath(os.getcwd())
print(f"[*] Punto di partenza di VS Code: {current_dir}")

# Lo script "sale" nelle cartelle finché non trova quella che contiene il tuo pacchetto
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        # Trovato! Lo inseriamo al primo posto tra i percorsi di Python
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        print(f"[+] BINGO! Cartella di progetto collegata: {current_dir}")
        break
    current_dir = os.path.dirname(current_dir)
else:
    print("[!] Errore: Non riesco a trovare la cartella del progetto.")
from hcatostialocalization.config.config import config

from hcatostialocalization.data.dataset.Base_Classes_ProvaMarco import LabeledDataset
from hcatostialocalization.data.dataset.Heatmap_Config import HeatmapParams

class Dataset_ImageCAS_Ostia(LabeledDataset):

    # Definiamo le etichette per il dataset: 0 per lo sfondo, 1 per l'ostio
    labels: dict[int, str] = {
        0: "background",
        1: "RCA_ostium",
        2: "LCA_ostium",
    }

    def _init_data_pool(self) -> list[tuple[str, str]]:
        # Peschiamo i percorsi direttamente dal nostro config
        yaml_path = config.dataset.ImageCAS.yaml_path 
        img_folder = config.dataset.ImageCAS.image_path
        json_folder = config.dataset.ImageCAS.json_path
        
        with open(yaml_path, 'r') as f:
            split_dict = yaml.safe_load(f)['ImageCAS']['split']
        
        patient_ids = split_dict[self.split]
        
        # NOVITÀ: Leggiamo TUTTI i nomi dei file presenti nella cartella JSON una sola volta
        all_json_files = []
        if os.path.exists(json_folder):
            all_json_files = os.listdir(json_folder)
        
        data_pool = []
        for pid in patient_ids:
            im_f = os.path.join(img_folder, f"{pid}.nii.gz")
            
            # Creiamo un percorso di fallback (se non trova nulla, prova col nome base)
            lb_f = os.path.join(json_folder, f"{pid}.json")
            
            # Ricerca intelligente del file JSON
            for filename in all_json_files:
                # Controlla che sia un file json (ignorando maiuscole/minuscole)
                if filename.lower().endswith('.json'):
                    # TRUCCO: Per evitare che cercando il paziente "13" ci trovi per sbaglio 
                    # il paziente "131_final", ci assicuriamo che dopo l'ID ci sia un punto, 
                    # un underscore o uno spazio!
                    if filename.startswith(f"{pid}.") or filename.startswith(f"{pid}_") or filename.startswith(f"{pid} "):
                        lb_f = os.path.join(json_folder, filename)
                        break # Trovato! Interrompiamo la ricerca per questo paziente
                        
            data_pool.append((im_f, lb_f))
            
        return data_pool

    def _get_patient_id(self, current_patient: tuple[str, str]) -> str:
        # Estrae l'ID dal nome del file
        id = os.path.basename(current_patient[0]).split('.')[0].strip()
        return f"{self.split}-ImageCAS-{id}"

    def _load_data(self, image_label_f: tuple[str, str], new_spacing: float|tuple|None = None) -> dict:
        img_f, json_f = image_label_f
        
        # 1. Carica il volume 3D usando la funzione del motore base
        dict_img = self._load_volume_resize(img_f, which='image', new_spacing=new_spacing)
        img_tensor = dict_img['tensor']
        
        # 2. Prepara un tensore vuoto per la label, con le stesse dimensioni dell'immagine
        label_tensor = torch.zeros_like(img_tensor, dtype=torch.float)
        
        # 3. Estrai le coordinate dal JSON e disegna la Heatmap
        if os.path.exists(json_f):
            with open(json_f, 'r') as f:
                json_data = json.load(f)
            
            ostia_block = json_data.get("ostia", {})
            sigma = HeatmapParams.sigma if hasattr(HeatmapParams, 'sigma') else 2.0 
            
            # Cicla sui lati R e L definiti nel tuo JSON
            for side in ["R", "L"]: 
                side_data = ostia_block.get(side)
                
                if side_data is not None:
                    coords_ijk = side_data.get("coordinate_ijk") 
                    
                    if coords_ijk is not None:
                        # Estrae X, Y, Z come indici interi
                        i, j, k = int(coords_ijk[0]), int(coords_ijk[1]), int(coords_ijk[2])
                        
                        # Disegna la gaussiana passando Z, Y, X (per via del formato tensore di PyTorch)
                        self._draw_3d_gaussian(label_tensor[0, 0], center=(k, j, i), sigma=sigma)

        # 4. Converti il formato dei dati per ottimizzare la memoria in fase di training
        if self.split == 'train':
            dict_img['tensor'] = dict_img['tensor'].to(torch.float16)
            label_tensor = label_tensor.to(torch.float16)
            
        return {
            'image': dict_img['tensor'], 
            'label': label_tensor, 
            'spacing': dict_img['spacing'], 
            'spacing_orig': dict_img['spacing_orig']
        }

    def _draw_3d_gaussian(self, heatmap_tensor: torch.Tensor, center: tuple[float, float, float], sigma: float):
        """Disegna una heatmap gaussiana 3D ottimizzata in una bounding box attorno al centro."""
        z_c, y_c, x_c = center
        D, H, W = heatmap_tensor.shape
        
        # Raggio della bounding box per il calcolo della gaussiana
        radius = int(3 * sigma)
        
        # Limiti della bounding box per non uscire dai bordi dell'immagine
        z0, z1 = max(0, int(z_c - radius)), min(D, int(z_c + radius) + 1)
        y0, y1 = max(0, int(y_c - radius)), min(H, int(y_c + radius) + 1)
        x0, x1 = max(0, int(x_c - radius)), min(W, int(x_c + radius) + 1)
        
        # Se il punto generato dovesse essere fuori dal volume dell'immagine, ignora
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            return

        # Crea griglie 3D di coordinate locali
        z_grid = torch.arange(z0, z1, dtype=torch.float)
        y_grid = torch.arange(y0, y1, dtype=torch.float)
        x_grid = torch.arange(x0, x1, dtype=torch.float)
        
        grid_z, grid_y, grid_x = torch.meshgrid(z_grid, y_grid, x_grid, indexing='ij')
        
        # Calcola la distanza quadratica e applica la formula Gaussiana
        dist_sq = (grid_z - z_c)**2 + (grid_y - y_c)**2 + (grid_x - x_c)**2
        gaussian = torch.exp(-dist_sq / (2 * sigma**2))
        
        # Applica la gaussiana alla heatmap originale, tenendo il valore massimo per sovrapposizioni
        heatmap_tensor[z0:z1, y0:y1, x0:x1] = torch.maximum(
            heatmap_tensor[z0:z1, y0:y1, x0:x1], 
            gaussian
        )
    def __len__(self):
        # Ritorna il numero di pazienti caricati nello split corrente
        return len(self.data_pool)
# --- TEST DI VISUALIZZAZIONE ---
if __name__ == "__main__":
    print("[*] Inizializzazione test visivo...")
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 1. Inizializziamo il dataset in modalità 'val' per avere l'immagine intera
    dataset = Dataset_ImageCAS_Ostia(
        split='val',
        nsdim=3
    )

    # 2. Peschiamo il primo paziente disponibile
    print("[*] Caricamento del volume 3D. Potrebbe volerci qualche secondo...")
    out = dataset.get()
    
    # In modalità val, l'output è un dizionario
    patient_id = out['id']
    img_tensor = out['image'][0, 0] # Rimuoviamo le dimensioni Batch e Canale -> (Z, Y, X)
    lab_tensor = out['label'][0, 0] # -> (Z, Y, X)

    print(f"[+] Dati caricati per: {patient_id}")
    print(f"    Shape Immagine: {img_tensor.shape}")
    print(f"    Shape Label: {lab_tensor.shape}")

    # 3. Troviamo la slice con l'ostio
    # Cerchiamo il punto con il valore massimo (il picco della nostra gaussiana)
    if lab_tensor.max() <= 0:
        print("[!] Nessuna heatmap trovata in questo paziente! (Il tensore è vuoto)")
    else:
        # Trova l'indice 1D del valore massimo e lo converte in coordinate 3D (Z, Y, X)
        max_idx = torch.argmax(lab_tensor)
        z_center, y_center, x_center = np.unravel_index(max_idx.item(), lab_tensor.shape)
        
        print(f"[+] Ostio rilevato al picco gaussiano -> Slice Z: {z_center}, Y: {y_center}, X: {x_center}")

        # 4. Estraiamo la fetta 2D per il plot
        img_slice = img_tensor[z_center, :, :].numpy()
        lab_slice = lab_tensor[z_center, :, :].numpy()

        # 5. Creiamo il Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Verifica Heatmap Ostio - {patient_id} (Slice Z={z_center})", fontsize=16)

        # Plot 1: Solo la TAC
        # Usiamo vmin e vmax per limitare la scala dei grigi ai valori Hounsfield tipici dei tessuti/vasi
        axes[0].imshow(img_slice, cmap='gray', vmin=-500, vmax=800)
        axes[0].set_title("Immagine Originale")
        axes[0].axis('off')

        # Plot 2: TAC + Heatmap
        axes[1].imshow(img_slice, cmap='gray', vmin=-500, vmax=800)
        
        # Mascheriamo i valori della label vicini allo zero, altrimenti copriranno tutta l'immagine di blu
        lab_masked = np.ma.masked_where(lab_slice < 0.05, lab_slice)
        
        # Sovrapponiamo la heatmap (in rosso/giallo, semi-trasparente)
        im = axes[1].imshow(lab_masked, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title("Immagine con Heatmap Sovrapposta")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()