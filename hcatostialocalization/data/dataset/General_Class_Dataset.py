import os
import torch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
# Importazione delle classi dataset dai file forniti
# Assicurati che i file .py siano nella stessa cartella o nel PYTHONPATH
from ASOCA.Dataset_ASOCA_Ostia import Dataset_ASOCA_Ostia
from CAT08.Dataset_CAT08_Ostia_gen import Dataset_CAT08_Ostia
# Per il notebook, dovrai aver esportato la classe in un file .py o importarla correttamente
from ImageCAS.Dataset_ImageCAS_Ostia import Dataset_ImageCAS_Ostia

##Salta attributo che dà errore in fase di importazione (non è necessario per il funzionamento del dataset unificato)
#Verrà riempito dopo con i dati corretti durante l'inizializzazione del dataset unificato
def patched_getitem(self, idx):
    """
    Sostituisce il __getitem__ originale per caricare i dati 
    direttamente senza passare per self.loaded.
    """
    # Prende la coppia (immagine, label) per l'indice richiesto
    image_label_f = self.data_pool[idx]

    # Carica i tensori usando il metodo già presente nelle tue classi
    sample = self._load_data(image_label_f, new_spacing=self.new_spacing)

    # Aggiunge l'id paziente, che _load_data da solo non aggiunge
    sample["id"] = self._get_patient_id(image_label_f)

    return sample


# Applica la patch alle classi importate
Dataset_ASOCA_Ostia.__getitem__ = patched_getitem
Dataset_CAT08_Ostia.__getitem__ = patched_getitem
Dataset_ImageCAS_Ostia.__getitem__ = patched_getitem

class UnifiedOstiaDataset:
    """
    Gestore per creare l'unione dei tre dataset mantenendo gli split.
    """
    def __init__(self, split='train', nsdim=3, transform=None, new_spacing=None):
        self.split = split
      
        print(f"[*] Inizializzazione Dataset Unificato | Split: {split} | nsdim: {nsdim}")
        
        # Inizializziamo i tre dataset individuali per lo split richiesto
        # Ognuno caricherà i propri ID dal rispettivo file YAML
        self.asoca = Dataset_ASOCA_Ostia(split=split, nsdim=nsdim, transform=transform, new_spacing=new_spacing)
        self.cat08 = Dataset_CAT08_Ostia(split=split, nsdim=nsdim, transform=transform, new_spacing=new_spacing)
        self.imagecas = Dataset_ImageCAS_Ostia(split=split, nsdim=nsdim, transform=transform, new_spacing=new_spacing)
        
        # Uniamo i dataset utilizzando ConcatDataset di PyTorch
        self.combined_dataset = ConcatDataset([self.asoca, self.cat08, self.imagecas])
        
        print(f"[+] Dataset unificato creato con successo.")
        print(f"    - ASOCA: {len(self.asoca)} campioni")
        print(f"    - CAT08: {len(self.cat08)} campioni")
        print(f"    - ImageCAS: {len(self.imagecas)} campioni")
        print(f"    - TOTALE: {len(self.combined_dataset)} campioni")

    def __getitem__(self, idx):
        return self.combined_dataset[idx]

    def __len__(self):
        return len(self.combined_dataset)

    def get_loader(self, batch_size=1, shuffle=True):
        """Ritorna un DataLoader per il dataset combinato."""
        return torch.utils.data.DataLoader(
            self.combined_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle if self.split == 'train' else False
        )

# Esempio di utilizzo:
if __name__ == "__main__":
    # Creazione dataset
    dataset_val = UnifiedOstiaDataset(split='val')
    dataset_train = UnifiedOstiaDataset(split='train')
    dataset_test = UnifiedOstiaDataset(split='test')

    # Caricamento del primo campione da train
    sample_train = dataset_train[0]
    img = sample_train['image'].squeeze().cpu().numpy() # [Z, Y, X]
    lab = sample_train['label'].squeeze().cpu().numpy() # [Z, Y, X]
    
    # Identifica la slice Z con il picco della heatmap (l'ostio)
    z_idx = np.argmax(np.max(lab, axis=(1, 2)))
    
    print(f"Visualizzazione Paziente: {sample_train.get('id', 'N/A')} - Slice Z: {z_idx}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img[z_idx], cmap='gray')
    plt.title("Immagine TAC")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img[z_idx], cmap='gray')
    # Overlay della heatmap (solo valori significativi)
    lab_m = np.ma.masked_where(lab[z_idx] < 0.1, lab[z_idx])
    plt.imshow(lab_m, cmap='jet', alpha=0.5)
    plt.title("Localizzazione Ostio")
    plt.show()

  # Caricamento del primo campione da val
    sample_val = dataset_val[0]
    img = sample_val['image'].squeeze().cpu().numpy() # [Z, Y, X]
    lab = sample_val['label'].squeeze().cpu().numpy() # [Z, Y, X]
    
    # Identifica la slice Z con il picco della heatmap (l'ostio)
    z_idx = np.argmax(np.max(lab, axis=(1, 2)))
    
    print(f"Visualizzazione Paziente: {sample_val.get('id', 'N/A')} - Slice Z: {z_idx}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img[z_idx], cmap='gray')
    plt.title("Immagine TAC")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img[z_idx], cmap='gray')
    # Overlay della heatmap (solo valori significativi)
    lab_m = np.ma.masked_where(lab[z_idx] < 0.1, lab[z_idx])
    plt.imshow(lab_m, cmap='jet', alpha=0.5)
    plt.title("Localizzazione Ostio")
    plt.show()

    # Caricamento del primo campione da test
    sample_test = dataset_test[0]
    img = sample_test['image'].squeeze().cpu().numpy() # [Z, Y, X]
    lab = sample_test['label'].squeeze().cpu().numpy() # [Z, Y, X]
    
    # Identifica la slice Z con il picco della heatmap (l'ostio)
    z_idx = np.argmax(np.max(lab, axis=(1, 2)))
    
    print(f"Visualizzazione Paziente: {sample_test.get('id', 'N/A')} - Slice Z: {z_idx}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img[z_idx], cmap='gray')
    plt.title("Immagine TAC")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img[z_idx], cmap='gray')
    # Overlay della heatmap (solo valori significativi)
    lab_m = np.ma.masked_where(lab[z_idx] < 0.1, lab[z_idx])
    plt.imshow(lab_m, cmap='jet', alpha=0.5)
    plt.title("Localizzazione Ostio")
    plt.show()