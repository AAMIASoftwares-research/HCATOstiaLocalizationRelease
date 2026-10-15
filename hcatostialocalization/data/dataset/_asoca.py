# PyTorch Datasets for ASOCA dataset
#esegui comando da terminale con seguente scrittura: python -m hcatostialocalization.data.dataset._asoca
import os
import sys
from pathlib import Path
# Individua la cartella 'HCATOstialLocalization' (salendo di 3 livelli dal file attuale)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print(root_path)
# La aggiunge al path se non è già presente
if root_path not in sys.path:
    sys.path.append(root_path)
from hcatostialocalization.config import config
from __base_classes import LabeledDataset


class DatasetASOCA(LabeledDataset):

    labels: dict[int, str] = {
        0: 'background',
        1: 'coronary artery',
    }

    def _init_data_pool(self) -> list[tuple[str, str]]:
        data_ids_pool: list[str] = config.dataset.ASOCA.split[self.split] ##Carica tutti gli id e l'indirizzo a cui sono salvati

        data_pool = []
        for id in data_ids_pool:
            # image
            if self.split == 'test':
                # patient_id is already the full path of the image
                pass
            elif self.split in ['train', 'val']:
                f = os.path.join(
                    config.dataset.ASOCA.path_normal_image ##Questi sono solo indirizzi e ci deve essere un modo per sostituirli in maniera efficace
                        if 'Normal' in id 
                        else config.dataset.ASOCA.path_diseased_image,
                    id + '.nrrd'
                )
            else:
                raise ValueError(f'Got split = {self.split} when it should be in [\'train\', \'val\', \'test\']')
            im_f = f
            # label
            if self.split == 'test':
                lb_f = None
            else:
                f = os.path.join(
                    config.dataset.ASOCA.path_normal_label
                        if 'Normal' in id 
                        else config.dataset.ASOCA.path_diseased_label,
                    id + '.nrrd'
                )
            lb_f = f
            data_pool.append((im_f, lb_f))
        return data_pool
    
    def _get_patient_id(self, current_patient: tuple[str, str]) -> str:
        id = os.path.basename(current_patient[0]).split('.')[0].strip()
        return f"{self.split}-ASOCA-{id}"
    
    
if __name__ == "__main__":
    print("Test")
    import matplotlib.pyplot as plt

    dataset = DatasetASOCA(
        split='train',
        nsdim=3,
        sample_sdim=128,
        minibatch_size=32,
        change_after_n_sampling=80,
        new_spacing=0.5,
        proportion_with_labels=0.4,
        is_2d3s=False
    )

    out = dataset.get()
    try:
        # train batches
        img, lab = out
    except:
        # val / test
        img = out['image']
        lab = out['label']

    print("img, lab", img.shape, lab.shape)
    quit("Success")
    
    if len(img.shape) == 4:
        c = int(img.shape[1]/2)
        img = img[0,c]
        try:
            lab = lab[0,0]
        except:
            pass
    else:
        s = int(img.shape[2]/2)
        img = img[0,0,s,...]
        try:
            lab = lab[0,0,s,...]
        except:
            pass

    plt.imshow(img, vmin=-1000, vmax=1000, cmap='gray')
    try:
        plt.imshow(lab, cmap='Reds', alpha=0.3)
    except:
        pass
    plt.title(f"{img.shape}, {(lab != 0).sum().item()}")
    plt.axis('off')
    plt.show()