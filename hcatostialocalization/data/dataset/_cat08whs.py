import os

import numpy, torch

from hcatsegment.configs import config
from .__base_classes import LabeledDataset

class DatasetCAT08WHS(LabeledDataset):

    labels: dict[int, str] = {
        0:   'background',
        205: 'left ventricular myocardium',
        420: 'left atrial blood cavity',
        500: 'left ventricular blood cavity',
        550: 'right atrial blood cavity',
        600: 'right ventricular blood cavity',
        820: 'aorta',
        822: 'aorta',
        850: 'pulmonary artery',
    }

    def _init_data_pool(self) -> list[tuple[str, tuple[str, str]]]:
        data_ids_pool: list[str] = config.dataset.CAT08WHS.split[self.split]

        data_pool = []
        for id in data_ids_pool:
            # image
            f: str = os.path.normpath(config.dataset.CAT08WHS.path_images+id)
            im_f = f
            # label
            lb_f: str = os.path.normpath(config.dataset.CAT08WHS.path+id).replace('.mhd', '_segby_all29.nii.gz')
            data_pool.append((im_f, lb_f))
        return data_pool

    def _get_patient_id(self, current_patient: tuple[str, str]|tuple[str, tuple[str, str]]) -> str:
        id = os.path.splitext(os.path.basename(current_patient[0]))[0].strip()
        return f"{self.split}-CAT08WHS-{id}"

    
    def _load_data(self, image_label_f, new_spacing = None) -> dict:
        out = super()._load_data(image_label_f, new_spacing=new_spacing)
        out['image'] = out['image'] - torch.tensor(1024.0, dtype=out['image'].dtype, device=out['image'].device)  # CAT08 data are stored as values in [0, 4096] -> go back to HU scale, avoid implicit type promotions to float32
        return out
    
if __name__ == "__main__":
    print("Test")
    import sys
    from pprint import pprint

    import matplotlib.pyplot as plt

    dataset = DatasetCAT08WHS(
        split='train',
        nsdim=3,
        sample_sdim=128,
        minibatch_size=32,
        change_after_n_sampling=80,
        new_spacing=0.5,
        proportion_with_labels=0.4,
        is_2d3s=False
    )
    print("Dataset initialized")

    out = dataset.get()
    try:
        # train batches
        img, lab = out
    except:
        # val / test
        img = out['image']
        lab = out['label']

    print("img, lab, max(lab)", img.shape, lab.shape, lab.max().item())
    
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

    classes_colors = [
        (0, 'black'),
        (1, 'red', 'Reds'),
        (2, 'green', 'Greens'),
        (3, 'blue', 'Blues'),
        (4, 'yellow', 'YlOrRd'),
        (5, 'purple', 'Purples'),
        (6, 'cyan', 'BuPu'),
        (7, 'orange', 'Oranges'),
        (8, 'pink', 'pink'),
        (9, 'brown', 'brown'),
    ]
    
    if 0:
        # view one output
        plt.imshow(img, vmin=-1000, vmax=1000, cmap='gray')
        try:
            for i in range(1, int(lab.max().item())+1):
                plt.contour((lab==i).cpu(), colors=['r'], alpha=0.3)
        except:
            pass
        plt.title(f"{img.shape}, {(lab != 0).sum().item()}")
    else:
        # view a whole slice
        s = int(dataset.loaded['image'].shape[2]//1.5)
        plt.imshow(dataset.loaded['image'][0,0,s,...], vmin=-1000, vmax=1000, cmap='gray')
        try:
            unique_label_values = numpy.unique(dataset.loaded['label'].cpu())
            print("unique_label_values", unique_label_values)
            for i, lx in enumerate(unique_label_values):
                if lx == 0:
                    continue
                # plt.imshow(dataset.loaded['label'][0,0,s,...]==i, cmap=classes_colors[i][2], alpha=0.6)
                plt.contour((dataset.loaded['label'][0,0,s,...]==lx).cpu(), colors=[classes_colors[i][1]])
        except:
            print("No label to show, error occurred")
        plt.title(f"{dataset.__class__.__name__} {dataset.split} id={dataset._get_patient_id(dataset.current_patient)}\nshape={dataset.loaded['image'].shape} labels={numpy.unique(dataset.loaded['label'].cpu())}")
    
    
    plt.axis('off')
    if len(sys.argv) > 1:
        print("Saving to", sys.argv[1])
        plt.savefig(sys.argv[1], bbox_inches='tight', dpi=300)
    else:
        plt.show()
