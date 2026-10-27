import os
import sys
import json
import torch
from torch.utils.data import ConcatDataset
import yaml
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

from hcatostialocalization.data.dataset.ASOCA.Multi_Dataset_ASOCA import Dataset_ASOCA_Ostia_Multi
from hcatostialocalization.data.dataset.CAT08.Multi_Dataset_CAT08 import Dataset_CAT08_Ostia_Multi
from hcatostialocalization.data.dataset.ImageCAS.Multi_Dataset_ImageCAS import Dataset_ImageCAS_Ostia_Multi

# ---> IMPORTIAMO LA TUA CLASSE DALLE BASE CLASSES! <---
from hcatostialocalization.data.dataset.Base_Classes_Dataset import LabeledDatasetsCollection

# ==============================================================================
# 1. PATCH METODO GETITEM DEI SINGOLI DATASET
# ==============================================================================
def patched_getitem_multi(self, idx):
    image_label_f = self.data_pool[idx]
    sample = self._load_data(image_label_f, new_spacing=self.new_spacing)
    if "id" not in sample:
        sample["id"] = self._get_patient_id(image_label_f)
    # Converti spacing da tuple a tensor (3,) per garantire che il collate di
    # PyTorch produca (batch_size, 3) invece di (3, batch_size).
    if isinstance(sample.get('spacing'), (tuple, list)):
        sample['spacing'] = torch.tensor(sample['spacing'], dtype=torch.float32)
    return sample

Dataset_ASOCA_Ostia_Multi.__getitem__ = patched_getitem_multi
Dataset_CAT08_Ostia_Multi.__getitem__ = patched_getitem_multi
Dataset_ImageCAS_Ostia_Multi.__getitem__ = patched_getitem_multi

# ==============================================================================
# 2. PATCH PER LA TUA "LabeledDatasetsCollection"
# ==============================================================================
def patched_collection_getitem(self, index):
    """
    Insegniamo alla tua classe a restituire i DIZIONARI per il nuovo Trainer,
    mantenendo però intatto il tuo algoritmo probabilistico (draws_proportion)!
    """
    if self.split == "train":
        # IL TUO ALGORITMO: Sceglie il dataset in base alle percentuali 33-33-34
        dataset_idx = self.rng.choice(len(self.datasets), p=self.draws_proportion)
        
        # Sceglie un paziente a caso da quel dataset
        random_idx = self.rng.integers(0, len(self.datasets[dataset_idx].data_pool))
        
        # Restituisce il dizionario usando il getitem!
        return self.datasets[dataset_idx][random_idx]
    else:
        # Per validation e test usiamo il counter come avevi pensato tu
        data = self._dataset_to_draw_from[self._val_test_counter]
        random_idx = self.rng.integers(0, len(data.data_pool))
        self._val_test_counter = (self._val_test_counter + 1) % len(self)
        return data[random_idx]

# Applichiamo la modifica alla tua classe
LabeledDatasetsCollection.__getitem__ = patched_collection_getitem

# ==============================================================================
# 3. DATASET UNIFICATO MULTI-CLASSE
# ==============================================================================
class UnifiedOstiaDatasetMulti:
    def __init__(self, split='train', nsdim=3, transform=None, new_spacing=None):
        self.split = split
        print(f"[*] Inizializzazione Dataset Unificato MULTI | Split: {split} | nsdim: {nsdim}")
        
        self.asoca = Dataset_ASOCA_Ostia_Multi(split=split, nsdim=nsdim, transform=transform, new_spacing=new_spacing)
        self.cat08 = Dataset_CAT08_Ostia_Multi(split=split, nsdim=nsdim, transform=transform, new_spacing=new_spacing)
        self.imagecas = Dataset_ImageCAS_Ostia_Multi(split=split, nsdim=nsdim, transform=transform, new_spacing=new_spacing)
        
        # ----------------------------------------------------------------------
        # INTEGRAZIONE DELLA TUA CLASSE!
        # ----------------------------------------------------------------------
        if split == 'train':
            self.combined_dataset = LabeledDatasetsCollection(
                datasets=[self.asoca, self.cat08, self.imagecas],
                split=split,
                nsdim=nsdim,
                minibatch_size=1,
                draws_proportion=[0.33, 0.33, 0.34]  # <-- Le tue proporzioni!
            )
            print(f"[+] Applicato campionamento PROPORZIONALE [0.33, 0.33, 0.34] via LabeledDatasetsCollection")
        else:
            self.combined_dataset = ConcatDataset([self.asoca, self.cat08, self.imagecas])
            print(f"[+] Applicato concatenamento STANDARD per {split}")
            
        print(f"    - ASOCA: {len(self.asoca)} campioni")
        print(f"    - CAT08: {len(self.cat08)} campioni")
        print(f"    - ImageCAS: {len(self.imagecas)} campioni")
        print(f"    - TOTALE (Somma base): {len(self.asoca) + len(self.cat08) + len(self.imagecas)} campioni")

    def __getitem__(self, idx):
        return self.combined_dataset[idx]

    def __len__(self):
        return len(self.combined_dataset)

    def set_sigma(self, sigma: float):
        self.asoca.sigma = sigma
        self.cat08.sigma = sigma
        self.imagecas.sigma = sigma

    def get_loader(self, batch_size=1, shuffle=True):
        return torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle if self.split == 'train' else False
        )

# ==============================================================================
# 4. TEST DI VISUALIZZAZIONE UNIFICATO
# ==============================================================================
if __name__ == "__main__":
    def visualizza_campione(sample, titolo_split):
        img = sample['image'].squeeze().cpu().numpy() 
        lab = sample['label'].cpu().numpy()           
        
        max_R = lab[1].max()
        max_L = lab[2].max()
        
        if max_R < 0.05 and max_L < 0.05:
            print(f"[!] {titolo_split} - Paziente {sample.get('id', 'N/A')}: Nessun ostio trovato!")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Visualizzazione {titolo_split} | Paziente: {sample.get('id', 'N/A')}", fontsize=16)
        
        if max_R >= 0.05:
            z_R = np.unravel_index(np.argmax(lab[1]), lab[1].shape)[2]  # asse S (RAS)
            axes[0].imshow(img[:, :, z_R].T, cmap='gray', vmin=-500, vmax=800, origin='lower')
            axes[0].imshow(np.ma.masked_where(lab[1, :, :, z_R].T < 0.05, lab[1, :, :, z_R].T), cmap='Greens', alpha=0.6, origin='lower')
            axes[0].set_title(f"Ostio DESTRO (Verde) - Slice S: {z_R}")
        else:
            axes[0].text(0.5, 0.5, 'Ostio Destro Assente', ha='center', va='center', fontsize=12)
        axes[0].axis('off')
        
        if max_L >= 0.05:
            z_L = np.unravel_index(np.argmax(lab[2]), lab[2].shape)[2]  # asse S (RAS)
            axes[1].imshow(img[:, :, z_L].T, cmap='gray', vmin=-500, vmax=800, origin='lower')
            axes[1].imshow(np.ma.masked_where(lab[2, :, :, z_L].T < 0.05, lab[2, :, :, z_L].T), cmap='Reds', alpha=0.6, origin='lower')
            axes[1].set_title(f"Ostio SINISTRO (Rosso) - Slice S: {z_L}")
        else:
            axes[1].text(0.5, 0.5, 'Ostio Sinistro Assente', ha='center', va='center', fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    dataset_train = UnifiedOstiaDatasetMulti(split='train')
    if len(dataset_train) > 0:
        visualizza_campione(dataset_train[0], "TRAIN")

