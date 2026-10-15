# PyTorch Datasets for ImageCAS
import os

from hcatsegment.configs import config
from .__base_classes import LabeledDataset

IMAGE_SUFFIX = ".img.nii.gz"
LABEL_SUFFIX = ".label.nii.gz"

class DatasetImageCAS(LabeledDataset):

    labels: dict[int, str] = {
        0: 'background',
        1: 'coronary artery',
    }
    
    def _init_data_pool(self) -> list[tuple[str, str]]:
        folder = config.dataset.ImageCAS.path
        ids_min_max: list[int] = config.dataset.ImageCAS.split[self.split]

        image_ids = [
            f.split('.')[0].strip() 
            for f in os.listdir(folder) 
            if f.endswith(IMAGE_SUFFIX)
        ]
        label_ids = [
            f.split('.')[0].strip() 
            for f in os.listdir(folder) 
            if f.endswith(LABEL_SUFFIX)
        ]
        # intersection of two sets:
        # keep elements that appear in both sets
        valid_patient_ids_list = list(set(image_ids) & set(label_ids))
        valid_patient_ids_list.sort()
        data_pool = []
        for id in valid_patient_ids_list:
            if int(id) < ids_min_max[0] or int(id) > ids_min_max[1]:
                continue 
            im_f = os.path.join(folder, f"{id}"+IMAGE_SUFFIX)
            lb_f = os.path.join(folder, f"{id}"+LABEL_SUFFIX)
            data_pool.append((im_f, lb_f))
        return data_pool
    
    def _get_patient_id(self, current_patient: tuple[str, str]) -> str:
        id = os.path.basename(current_patient[0]).split('.')[0].strip()
        return f"{self.split}-ImageCAS-{id}"
    
    
if __name__ == "__main__":
    print("Test")
    import matplotlib.pyplot as plt

    dataset = DatasetImageCAS(
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