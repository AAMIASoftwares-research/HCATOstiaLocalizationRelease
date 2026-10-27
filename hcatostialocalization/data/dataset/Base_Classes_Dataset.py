# PyTorch Datasets Base class for 3D datasets
from typing import Literal, Callable
import os
import errno
import json
from abc import ABC, abstractmethod
import numpy as np
import torch
from monai.transforms import Spacingd
import logging as logger
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ==============================================================================
# 0. PATH CONFIGURATION 
# ==============================================================================
current_dir = os.path.abspath(os.getcwd())
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)

# Import only 3D transformations and Heatmap configuration
from hcatostialocalization.data.transform.transformations_3d_PROVA import augment_3d
from hcatostialocalization.data.dataset.Heatmap_Config import HeatmapParams
from hcatostialocalization.io.io import asoca_raw_point_to_ras, asoca_header_from_json, load_image_ras

# ==============================================================================
# 1. BASE CLASS FOR INDIVIDUAL DATASETS (e.g. ImageCAS, ASOCA, CAT08)
# ==============================================================================

class LabeledDataset(torch.utils.data.Dataset, ABC):

    labels: dict[int, str] = {} # Mapping index -> name

    def __init__(
            self, 
            split: Literal['train', 'val', 'test'], 
            nsdim: Literal[2, 3], 
            sample_sdim: int|tuple = 128,
            minibatch_size: int = 1,
            change_after_n_sampling: int = 64, 
            new_spacing: float|tuple|None = None,
            label_transform_fn: Callable|None = None,
            train_sampling_strategy: Literal['random', 'empty-balanced', 'class-balanced'] = 'random',
            train_proportion_with_labels: float = 0.75,
            **kwargs
        ):

        super().__init__()
        self.nsdim = int(nsdim) 
        if self.nsdim != 3:
            raise ValueError("This class is optimized only for 3D data (nsdim=3).")
            
        self.sample_sdim = sample_sdim if isinstance(sample_sdim, tuple) else tuple([sample_sdim]*self.nsdim) 
        self.sample_sdim = tuple([int(d) for d in self.sample_sdim])
        self._sample_sdim_3d = self.sample_sdim 
        
        self.minibatch_size = int(max(minibatch_size, 1))
        self.split = split
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"'split' must be 'train', 'val' or 'test', received: {self.split}") 
            
        self.new_spacing = new_spacing 
        self._label_transform_fn = label_transform_fn
        self.rng: np.random.Generator = np.random.default_rng(seed=42)
        self._kwargs = kwargs if kwargs else {}
        
        # Data Initialization
        self.data_pool: list[tuple[str, str]] = self._init_data_pool() 

        if self.split == "train":
            self.change_after_n_sampling = int(change_after_n_sampling)
            self.counter = 0
            self.current_patient: tuple[str, str] = None
            self.loaded: dict = None    
            self.next() 
            self.train_sampling_strategy = train_sampling_strategy
            self.train_proportion_with_labels = np.clip(train_proportion_with_labels, 0.0, 1.0)
        else:
            self.current_patient = self.data_pool[0]

        if not self.labels or 'background' not in self.labels.values():
            raise ValueError("'labels' must be defined and contain the 'background' mapping.")
            
        logger.info(f"Loaded {self.__class__.__name__}")
        
    @abstractmethod
    def _init_data_pool(self) -> list[tuple[str, str]]:
        raise NotImplementedError()
    
    @abstractmethod
    def _get_patient_id(self, current_patient: tuple[str, str]) -> str:
        raise NotImplementedError()

    def _transform_label_upon_load(self, label_tensor: torch.Tensor) -> torch.Tensor:
        if self._label_transform_fn is not None:
            label_tensor = self._label_transform_fn(label_tensor)
        return label_tensor

    def choose_random_from_datapool(self, n: int = 1):
        n = max(1, int(n))
        if n == 1:
            idx = self.rng.integers(low=0, high=len(self.data_pool))
            return self.data_pool[idx]
        else:
            idxs = self.rng.integers(low=0, high=len(self.data_pool), size=n)
            return [self.data_pool[i] for i in idxs]

    def next(self) -> None:
        self.current_patient = self.choose_random_from_datapool(n=1)

    def _data_pool_from_split_yaml(self, dataset_name: str) -> list[tuple[str, str]]:
        """
        Data pool (image_path, label_path) of self.split, read from the split YAML of the
        dataset (config/dataset.yaml -> dataset.<dataset_name>.yaml_path), whose entries are
        explicit pairs:
            <dataset_name>:
              split:
                train:
                - image: <path>
                  label: <path>
        Relative paths are resolved against config.active_paths.data_root.
        Cases whose image or label does not exist are excluded and listed.
        """
        from hcatostialocalization.config.config import config
        import yaml

        yaml_path = config.dataset[dataset_name].yaml_path
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"[!] Split YAML not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            entries = (yaml.safe_load(f)[dataset_name]['split'] or {}).get(self.split) or []

        data_root = str(config.active_paths.data_root)
        resolve = lambda p: p if os.path.isabs(p) else os.path.join(data_root, p)

        data_pool, missing = [], []
        for entry in entries:
            im_f, lb_f = resolve(str(entry['image'])), resolve(str(entry['label']))
            problems = [what for what, p in (("image", im_f), ("label", lb_f)) if not os.path.exists(p)]
            if problems:
                missing.append(f"{entry['label']} (missing {' + '.join(problems)})")
            else:
                data_pool.append((im_f, lb_f))

        print(f"[INFO] {dataset_name} - split '{self.split}': {len(data_pool)}/{len(entries)} cases ready ({yaml_path})")
        for m in missing:
            print(f"[!] {dataset_name} '{self.split}' excluded: {m}")
        return data_pool

    # --------------------------------------------------------------------------
    # VOLUME LOADING (MONAI, RAS orientation)
    # --------------------------------------------------------------------------
    # All volumes are loaded with MONAI and reoriented to RAS.
    # Layout of returned tensor: (1, 1, R, A, S), i.e. axes (i, j, k) with
    #   i -> Right, j -> Anterior, k -> Superior (the last axis indexes axial slices).
    # Spacing and ostia coordinates follow the same order (R, A, S).
    # 'affine' is the 4x4 IJK -> RAS world (mm) matrix of the returned grid:
    # it is the only reference to use to convert world <-> voxel coordinates.

    def _load_volume(self, f) -> dict:
        """
        Reads the file from disk with MONAI (LoadImage + EnsureChannelFirst + Orientation RAS)
        via io.load_image_ras, which falls back to SimpleITK if the MONAI reader fails
        (e.g. some .nrrd), producing the exact same RAS grid and affine.
        """
        try:
            monai_img = load_image_ras(f)  # MetaTensor (1, R, A, S)
        except Exception as e:
            raise RuntimeError(f"Error loading volume {f}: {type(e).__name__}: {e}")
        spacing_orig = tuple(float(s) for s in monai_img.pixdim)  # (sR, sA, sS)
        return {"image": monai_img, "spacing_orig": spacing_orig}

    # --------------------------------------------------------------------------
    # CACHE OF RESAMPLED CT VOLUMES (optional: config.active_paths.cache_dir)
    # --------------------------------------------------------------------------
    # <cache_dir>/ct_ras_resampled_<sR>_<sA>_<sS>_mm/<original path with '/'->'+', ' '->'_'>.npy|.json
    # .npy = resampled RAS volume (R, A, S) in float16 (training casts it to float16 anyway);
    # .json = affine IJK -> RAS (float64), shape, spacing, spacing_orig, source file.
    # Resampled images only: labels (gaussians) depend on sigma and are reconstructed on each load.
    # Any cache read/write error falls back to loading from the original file.
    _cache_write_disabled: bool = False  # per-process: True after disk full / quota / read-only / permission error

    @staticmethod
    def _cache_paths(f, spacing: tuple) -> tuple[str, str] | None:
        """(npy, json) of the cache for file f resampled to 'spacing', None if cache is not configured"""
        try:
            from hcatostialocalization.config.config import config
            cache_dir = config.active_paths.get("cache_dir")
        except Exception:
            return None
        if not cache_dir:
            return None
        folder = os.path.join(str(cache_dir), "ct_ras_resampled_" + "_".join(f"{float(s):g}" for s in spacing) + "_mm")
        name = os.path.abspath(str(f)).replace("/", "+").replace(" ", "_")
        return os.path.join(folder, name + ".npy"), os.path.join(folder, name + ".json")

    @staticmethod
    def _read_cache(npy_f: str, json_f: str) -> dict | None:
        """Reads a cache entry; None (with warning) if missing, corrupted or inconsistent"""
        if not (os.path.exists(npy_f) and os.path.exists(json_f)):
            return None
        try:
            with open(json_f, 'r') as fh:
                meta = json.load(fh)
            arr = np.load(npy_f)
            if list(arr.shape) != list(meta["shape"]) or np.asarray(meta["affine"]).shape != (4, 4):
                raise ValueError(f"shape {arr.shape} vs {meta['shape']}, affine {np.asarray(meta['affine']).shape}")
            return {
                'tensor': torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0),  # (1, 1, R, A, S)
                'spacing': tuple(float(s) for s in meta["spacing"]),
                'spacing_orig': tuple(float(s) for s in meta["spacing_orig"]),
                'affine': torch.tensor(meta["affine"], dtype=torch.float64),
            }
        except Exception as e:
            logger.warning(f"Unreadable cache, loading from original file ({npy_f}): {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _write_cache(npy_f: str, json_f: str, source_f, out: dict) -> None:
        """Writes a cache entry atomically (tmp + os.replace, .json last); never blocking"""
        if LabeledDataset._cache_write_disabled:
            return
        tmp_npy, tmp_json = f"{npy_f}.tmp-{os.getpid()}.npy", f"{json_f}.tmp-{os.getpid()}"
        try:
            os.makedirs(os.path.dirname(npy_f), exist_ok=True)
            arr = out['tensor'][0, 0].numpy().astype(np.float16)
            np.save(tmp_npy, arr)
            os.replace(tmp_npy, npy_f)
            meta = {
                "source": os.path.abspath(str(source_f)),
                "shape": list(arr.shape),
                "affine": out['affine'].tolist(),
                "spacing": list(out['spacing']),
                "spacing_orig": list(out['spacing_orig']),
            }
            with open(tmp_json, 'w') as fh:
                json.dump(meta, fh)
            os.replace(tmp_json, json_f)
        except Exception as e:
            # Disk full / quota / read-only / permissions: pointless to retry for each case
            if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES):
                LabeledDataset._cache_write_disabled = True
            logger.warning(f"Writing cache failed ({npy_f}): {type(e).__name__}: {e}. "
                           f"Volume used without cache"
                           + (", this process will no longer write to cache." if LabeledDataset._cache_write_disabled else "."))
            for t in (tmp_npy, tmp_json):
                try:
                    os.remove(t)
                except OSError:
                    pass

    def _load_volume_resize(self, f, which: Literal['image', 'label'], new_spacing: float|tuple|None = None) -> dict:
        """
        Loads the volume in RAS and, if requested, resamples it with monai Spacingd.
        Resampled images use cache if config.active_paths.cache_dir is defined.
        """
        cache = None
        if new_spacing is not None:
            if isinstance(new_spacing, (int, float)):
                spacing = tuple([float(new_spacing)] * 3)
            else:
                spacing = tuple(float(s) for s in new_spacing)
            if which == 'image':
                cache = self._cache_paths(f, spacing)
                if cache is not None:
                    cached = self._read_cache(*cache)
                    if cached is not None:
                        return cached

        data = self._load_volume(f)
        monai_img = data['image']
        spacing_orig = data['spacing_orig']

        if new_spacing is None:
            spacing = spacing_orig
        else:
            resampler = Spacingd(
                keys=["image"],
                pixdim=spacing,
                mode="bilinear" if which == 'image' else "nearest",
                padding_mode="border",
                dtype=torch.float32,  # default float64: doubles resampling RAM
            )
            monai_img = resampler({"image": monai_img})["image"]

        affine = monai_img.affine.to(torch.float64).clone()  # IJK -> RAS of the final grid
        img = monai_img.as_tensor().float().unsqueeze(0)       # (1, 1, R, A, S)
        del monai_img

        out = {
            'tensor': img,
            'spacing': spacing,
            'spacing_orig': spacing_orig,
            'affine': affine,
        }
        if cache is not None:
            self._write_cache(*cache, f, out)
        return out

    # --------------------------------------------------------------------------
    # OSTIA LABELS (JSON -> 3-channel Gaussian heatmaps)
    # --------------------------------------------------------------------------
    @staticmethod
    def _read_ostia_ras(json_f: str) -> dict[str, np.ndarray]:
        """
        Reads the ostia JSON and returns {side: RAS world coordinates (mm)}.

        - Standard format '.ostium.json' (blueprint): 'pos' -> side -> 'ras'.
        - Old format ('ostia'):
          - if 'coordinate_ras' is present and not empty (ImageCAS, CAT08), it is used as is;
          - otherwise (ASOCA, 'coordinate_ras' empty) the raw ASOCA point in
            'world_mm_original_lps' is converted via io.asoca_raw_point_to_ras, using
            the header copied into the JSON ('origin', 'spacing', 'orientation'). The raw point
            is NEITHER LPS NOR RAS: see the docstring of that function.
        'coordinate_ijk' is NOT used: it refers to the file's native grid
        (neither reoriented nor resampled) and is truncated to integer.
        """
        with open(json_f, 'r') as fh:
            data = json.load(fh)
        out = {}
        if "pos" in data:
            for side in ["R", "L"]:
                ras = (data["pos"].get(side) or {}).get("ras") or []
                if len(ras) == 3:
                    out[side] = np.asarray(ras, dtype=np.float64)
            return out

        ostia_block = data.get("ostia", {})
        for side in ["R", "L"]:
            side_data = ostia_block.get(side) or {}
            ras = side_data.get("coordinate_ras") or []
            if len(ras) == 3:
                out[side] = np.asarray(ras, dtype=np.float64)
                continue
            raw = side_data.get("world_mm_original_lps") or []
            header = asoca_header_from_json(data)
            if len(raw) == 3 and header is not None:
                out[side], _ = asoca_raw_point_to_ras(raw, *header)
        return out

    @staticmethod
    def _ras_to_ijk(ras: np.ndarray, affine: torch.Tensor) -> np.ndarray:
        """RAS world (mm) -> continuous voxel index (i, j, k) on the grid described by 'affine'"""
        ras_to_ijk = np.linalg.inv(np.asarray(affine, dtype=np.float64))
        return (ras_to_ijk @ np.append(ras, 1.0))[:3]

    def _build_ostia_target(self, json_f: str, dict_img: dict) -> dict:
        """
        Constructs multi-channel target [3, R, A, S] and continuous GT coordinates.
        Channel 0 = Background, Channel 1 = Right Ostium (R), Channel 2 = Left Ostium (L).
        GT coordinates are continuous voxel indices (i, j, k) in the RAS grid
        (optionally resampled) of the image; [-1, -1, -1] if the ostium is absent.
        """
        shape = tuple(dict_img['tensor'].shape[-3:])
        label_tensor = torch.zeros((3, *shape), dtype=torch.float)

        _absent = torch.tensor([-1., -1., -1.], dtype=torch.float32)
        coords = {"R": _absent.clone(), "L": _absent.clone()}

        if os.path.exists(json_f):
            for side, ras in self._read_ostia_ras(json_f).items():
                ijk = self._ras_to_ijk(ras, dict_img['affine'])
                if not all(-0.5 <= c <= s - 0.5 for c, s in zip(ijk, shape)):
                    logger.warning(f"Ostium {side} outside volume ({json_f}): ijk={ijk.round(2).tolist()}, shape={shape}")
                class_idx = 1 if side == "R" else 2
                self._draw_3d_gaussian(label_tensor[class_idx], center=tuple(ijk), sigma=getattr(self, "sigma", HeatmapParams.sigma))
                coords[side] = torch.tensor(ijk, dtype=torch.float32)

            label_tensor[0] = 1.0 - torch.max(label_tensor[1], label_tensor[2])

        return {'label': label_tensor, 'ostia_coords_R': coords["R"], 'ostia_coords_L': coords["L"]}

    def _load_ostia_data(self, image_label_f: tuple[str, str], new_spacing: float|tuple|None = None) -> dict:
        """_load_data shared by ostia datasets (RAS image + Gaussian target)"""
        img_f, json_f = image_label_f
        dict_img = self._load_volume_resize(img_f, which='image', new_spacing=new_spacing)
        target = self._build_ostia_target(json_f, dict_img)

        img_tensor = dict_img['tensor']
        label_tensor = target['label']
        if self.split == 'train':
            img_tensor = img_tensor.to(torch.float16)
            label_tensor = label_tensor.to(torch.float16)

        return {
            'image':          img_tensor,
            'label':          label_tensor,
            'spacing':        dict_img['spacing'],
            'spacing_orig':   dict_img['spacing_orig'],
            'affine':         dict_img['affine'],
            'ostia_coords_R': target['ostia_coords_R'],
            'ostia_coords_L': target['ostia_coords_L'],
            'id':             self._get_patient_id(image_label_f),
        }

    @staticmethod
    def _draw_3d_gaussian(heatmap_tensor: torch.Tensor, center: tuple[float, float, float], sigma: float):
        """Draws (with torch.maximum) an isotropic Gaussian of std 'sigma' voxels, truncated at 3 sigma"""
        c0, c1, c2 = center
        D0, D1, D2 = heatmap_tensor.shape

        # When sigma reaches zero the target degenerates to a single-voxel annotation.
        if sigma <= 0.0:
            i0, i1, i2 = int(round(c0)), int(round(c1)), int(round(c2))
            if 0 <= i0 < D0 and 0 <= i1 < D1 and 0 <= i2 < D2:
                heatmap_tensor[i0, i1, i2] = 1.0
            return

        radius = int(3 * sigma)
        a0, b0 = max(0, int(np.floor(c0 - radius))), min(D0, int(np.ceil(c0 + radius)) + 1)
        a1, b1 = max(0, int(np.floor(c1 - radius))), min(D1, int(np.ceil(c1 + radius)) + 1)
        a2, b2 = max(0, int(np.floor(c2 - radius))), min(D2, int(np.ceil(c2 + radius)) + 1)
        if a0 >= b0 or a1 >= b1 or a2 >= b2:
            return

        g0, g1, g2 = torch.meshgrid(
            torch.arange(a0, b0, dtype=torch.float),
            torch.arange(a1, b1, dtype=torch.float),
            torch.arange(a2, b2, dtype=torch.float),
            indexing='ij',
        )
        dist_sq = (g0 - c0)**2 + (g1 - c1)**2 + (g2 - c2)**2
        gaussian = torch.exp(-dist_sq / (2 * sigma**2))
        heatmap_tensor[a0:b0, a1:b1, a2:b2] = torch.maximum(heatmap_tensor[a0:b0, a1:b1, a2:b2], gaussian)
    
    def _load_data(self, image_label_f: tuple[str, str], new_spacing: float|tuple|None = None) -> dict:
        """Streamlined version for compatibility. IN YOUR DATASETS YOU WILL OVERWRITE THIS METHOD!"""
        dict_img = self._load_volume_resize(image_label_f[0], which='image', new_spacing=new_spacing)
        
        reduce_volume = False
        if self.split == 'train':
            max_memory_GB = 1.0 
            shape_axial = dict_img['tensor'].shape[-2:] 
            max_slices = max(
                int((max_memory_GB * 1024**3) / (4 * shape_axial[0] * shape_axial[1])), 
                self._sample_sdim_3d[0], 
                32
            )
            if max_slices < dict_img['tensor'].shape[2]:
                start_slice = self.rng.integers(low=0, high=dict_img['tensor'].shape[2] - max_slices + 1)
                end_slice = start_slice + max_slices
                dict_img['tensor'] = dict_img['tensor'][:, :, start_slice:end_slice, ...]
                reduce_volume = True

        try:
            dict_lab = self._load_volume_resize(image_label_f[1], which='label', new_spacing=new_spacing)
            if reduce_volume:
                dict_lab['tensor'] = dict_lab['tensor'][:, :, start_slice:end_slice, ...]
        except Exception:
            dict_lab = {'tensor': None}

        if self.split == 'train':
            dict_img['tensor'] = dict_img['tensor'].to(torch.float32)
            if dict_lab['tensor'] is not None:
                dict_lab['tensor'] = dict_lab['tensor'].to(torch.float32) 
                
        return {
            'image': dict_img['tensor'], 
            'label': dict_lab['tensor'], 
            'spacing': dict_img['spacing'], 
            'spacing_orig': dict_img['spacing_orig']
        }

    def _load_data_and_transform_label(self, image_label_f: tuple[str, str], new_spacing: float|tuple|None = None) -> dict:
        out = self._load_data(image_label_f, new_spacing)
        if out['label'] is not None:
            out['label'] = self._transform_label_upon_load(out['label'])
        return out

    def __getitem__(self, index):
        return self.get()
    
    def get(self, minibatch_size: int|None = None):
        """Dispatches request based on split (Train vs Val/Test)"""
        if minibatch_size is None:
            minibatch_size = self.minibatch_size
        minibatch_size = int(max(minibatch_size, 1))
        
        if self.split == "train":
            return self._get_train(minibatch_size)
        else:
            return self._get_val_test()

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        sshape = shape[2:] 
        
        if any([k < s for k, s in zip(sshape, self._sample_sdim_3d)]):
            pads = [max(0, s-k) for k, s in zip(sshape, self._sample_sdim_3d)]
            padding_config = [[p//2, p-p//2] for p in pads[::-1]]
            padding_config = [x for xs in padding_config for x in xs]
            
            try:
                return torch.nn.functional.pad(tensor, padding_config, mode='reflect')
            except:
                return torch.nn.functional.pad(tensor, padding_config, mode='constant', value=0.0)
        return tensor

    def _get_train(self, minibatch_size: int):
        """Extracts 3D patches from the image loaded in memory"""
        self.loaded['image'] = self._pad_tensor(self.loaded['image'])
        if self.loaded['label'] is not None:
            self.loaded['label'] = self._pad_tensor(self.loaded['label'])

        ch_im = self.loaded['image'].shape[1]
        ch_lb = self.loaded['label'].shape[1] if self.loaded['label'] is not None else 1
        
        out_size_img = (minibatch_size, ch_im, *self.sample_sdim)
        out_size_lab = (minibatch_size, ch_lb, *self.sample_sdim)
        
        img_minibatch = torch.zeros(out_size_img, dtype=torch.float32)
        lab_minibatch = torch.zeros(out_size_lab, dtype=torch.float32)

        for b in range(minibatch_size):
            force_foreground = self.rng.random() < self.train_proportion_with_labels
            
            valid_patch = False
            attempts = 0
            
            while not valid_patch and attempts < 10:
                attempts += 1
                
                low = [
                    self.rng.integers(low=0, high=self.loaded['image'].shape[i+2] - self._sample_sdim_3d[i] + 1)
                    for i in range(3)
                ]
                
                if force_foreground and self.loaded['label'] is not None:
                    coords = (self.loaded['label'][0, 0] > HeatmapParams.min_threshold).nonzero()
                    if len(coords) > 0:
                        center = coords[self.rng.integers(0, len(coords))]
                        low = []
                        for i in range(3):
                            dim_len = self.loaded['image'].shape[i+2]
                            patch_size = self._sample_sdim_3d[i]
                            c = center[i].item()
                            l = max(0, min(c - patch_size // 2, dim_len - patch_size))
                            low.append(l)

                high = [l + s for l, s in zip(low, self._sample_sdim_3d)]
                s_idx = [slice(l, h) for l, h in zip(low, high)]
                
                img_patch = self.loaded['image'][(slice(None), slice(None), *s_idx)]
                lab_patch = self.loaded['label'][(slice(None), slice(None), *s_idx)]
                
                # Correct and safe data augmentation
                img_patch, lab_patch = augment_3d(img_patch, lab_patch)
                
                img_minibatch[b:b+1] = img_patch
                lab_minibatch[b:b+1] = lab_patch
                valid_patch = True 

        self.counter += minibatch_size
        if self.counter >= self.change_after_n_sampling:
            self.counter = 0
            self.next() 

        return (img_minibatch, lab_minibatch)

    def _get_val_test(self) -> dict:
        """Returns the entire 3D volume without cropping"""
        data = self._load_data_and_transform_label(self.current_patient, self.new_spacing)
        out_id = self._get_patient_id(self.current_patient)
        
        position = self.data_pool.index(self.current_patient)
        next_position = (position + 1) % len(self.data_pool)
        self.current_patient = self.data_pool[next_position]
        
        data['id'] = out_id
        return data


# ==============================================================================
# 2. COLLECTION CLASS (Mix of Datasets: ImageCAS + ASOCA + CAT08)
# ==============================================================================

class LabeledDatasetsCollection(torch.utils.data.Dataset):
    """Allows training the model by sampling from multiple Datasets simultaneously"""
    
    def __init__(
            self, 
            datasets: list[LabeledDataset],
            split: Literal['train', 'val', 'test'], 
            nsdim: Literal[2, 3] = 3, 
            sample_sdim: int|tuple = 128,
            minibatch_size: int = 1,
            draws_proportion: list[float] | None = None,
        ):    
            super().__init__()
            self.datasets = datasets
            self.split = split
            self.minibatch_size = int(max(minibatch_size, 1))
            self.rng = np.random.default_rng(seed=42)
            
            self.datasets_lenghts = [len(d.data_pool) for d in self.datasets]
            
            # Handling of proportions passed as input
            if draws_proportion is None:
                draws_proportion = [1.0/len(self.datasets) for _ in self.datasets]
            _t = sum(draws_proportion)
            # Normalize probabilities (for mathematical guarantee that they sum to 1.0)
            self.draws_proportion = [p/_t for p in draws_proportion]

            if self.split in ['val', 'test']:
                self._val_test_counter = 0
                self._dataset_to_draw_from = [] 
                for d in self.datasets:
                    self._dataset_to_draw_from.extend([d] * len(d.data_pool))

    def __len__(self):
        return sum(self.datasets_lenghts)

    def __getitem__(self, index):
        return self.get()

    def get(self, minibatch_size: int|None = None):
        if minibatch_size is None:
            minibatch_size = self.minibatch_size
            
        if self.split == "train":
            img_list, lab_list = [], []
            for _ in range(minibatch_size):
                # The "core" of Mix: Random draw based on draws_proportion!
                dataset_idx = self.rng.choice(len(self.datasets), p=self.draws_proportion)
                im, lb = self.datasets[dataset_idx].get(minibatch_size=1)
                
                img_list.append(im)
                lab_list.append(lb)
                
            return torch.cat(img_list, dim=0), torch.cat(lab_list, dim=0)
        else:
            data = self._dataset_to_draw_from[self._val_test_counter].get()
            self._val_test_counter = (self._val_test_counter + 1) % len(self)
            return data


# ==============================================================================
# PRACTICAL USAGE EXAMPLE (To use in your main training script)
# ==============================================================================
if __name__ == "__main__":
    # When you create the "Train.py" script, you will do exactly this:
    
    """
    # 1. Initialize the three datasets individually (assuming you created .py files for each)
    from hcatostialocalization.data.dataset.Dataset_ImageCAS_Ostia import Dataset_ImageCAS_Ostia
    from hcatostialocalization.data.dataset.Dataset_ASOCA_Ostia import Dataset_ASOCA_Ostia
    from hcatostialocalization.data.dataset.Dataset_CAT08_Ostia import Dataset_CAT08_Ostia
    
    dataset_cas = Dataset_ImageCAS_Ostia(split='train', nsdim=3, sample_sdim=64)
    dataset_asoca = Dataset_ASOCA_Ostia(split='train', nsdim=3, sample_sdim=64)
    dataset_cat08 = Dataset_CAT08_Ostia(split='train', nsdim=3, sample_sdim=64)
    
    # 2. Combine them into the Collection with requested proportions: 0.4, 0.3, 0.3
    dataset_misto = LabeledDatasetsCollection(
        datasets=[dataset_cas, dataset_asoca, dataset_cat08],
        split='train',
        nsdim=3,
        minibatch_size=4, # Example: take 4 hearts at a time to train the network
        draws_proportion=[0.4, 0.3, 0.3] 
    )
    
    # 3. Put everything in PyTorch DataLoader
    from torch.utils.data import DataLoader
    
    # batch_size=None because we handle batches inside the get() class
    dataloader = DataLoader(dataset_misto, batch_size=None, num_workers=0) 
    
    for batch_img, batch_lab in dataloader:
        print(f"Mixed image batch! Shape: {batch_img.shape}")
        # Neural network training here...
        break
    """
    print("[*] Base_Classes_Dataset.py compiled successfully!")
