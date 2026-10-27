import os
import sys
import glob
import random
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
from omegaconf import OmegaConf
from monai.inferers import sliding_window_inference
import mlflow
import math


# ==============================================================================
# 0. PROJECT ROOT RESOLUTION
# ==============================================================================
current_dir = os.path.abspath(os.getcwd())
root_path = None
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        root_path = current_dir
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
        break
    current_dir = os.path.dirname(current_dir)

if root_path:
    dataset_folder = os.path.join(root_path, "hcatostialocalization", "data", "dataset")
    if dataset_folder not in sys.path:
        sys.path.insert(0, dataset_folder)
else:
    root_path = os.path.abspath(os.getcwd())


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root_path, path))


from hcatostialocalization.config.config import config
from hcatostialocalization.layers.preprocessing import PreprocessorCT
from hcatostialocalization.model.model_factory import get_model
from hcatostialocalization.loss.loss_factory import get_loss
from hcatostialocalization.logger.logger import ExperimentLogger
from hcatostialocalization.data.dataset.General_Class_Dataset_Multi import UnifiedOstiaDatasetMulti
from hcatostialocalization.data.transform.augment_3d_ostia import augment_3d_ostia

try:
    from hcatostialocalization.train.schedulers import SchedulerLR
except ImportError:
    print("[!] Warning: schedulers.py not found. Custom warmup scheduler will be unavailable.")

from hcatostialocalization.train.sigma_scheduler import SigmaScheduler


# ==============================================================================
# 1. CONFIGURATION UTILITIES
# ==============================================================================
DEFAULT_SPACING_ISO = 0.5  # mm, used when `train.spacing_iso` is absent from the experiment config


def get_active_experiment_config(cfg):
    """Extracts and merges the active experiment sub-config from the global config."""
    active_experiment = cfg.active_experiment
    if active_experiment not in cfg.experiments:
        raise ValueError(f"Unknown active experiment: {active_experiment}")
    experiment_cfg = cfg.experiments[active_experiment]
    return OmegaConf.merge(
        OmegaConf.create({"active_experiment": active_experiment}),
        experiment_cfg,
        OmegaConf.create({"logger": cfg.logger}),
        OmegaConf.create({"resume": cfg.get("resume", {})}),
    )


def _spacing_iso_arg(value):
    """argparse type for --spacing-iso: a positive float (mm)."""
    try:
        spacing = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--spacing-iso must be a number, got '{value}'")
    if spacing <= 0:
        raise argparse.ArgumentTypeError("--spacing-iso must be > 0")
    return spacing


def parse_args():
    parser = argparse.ArgumentParser(description="Train HCAT ostia localization model.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-auto",
        action="store_true",
        help="Resume from the latest last_model.pth under the configured checkpoint folder.",
    )
    resume_group.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume from a specific checkpoint path.",
    )
    parser.add_argument(
        "--spacing-iso",
        type=_spacing_iso_arg,
        required=True,
        help="Isotropic training spacing in mm (> 0), e.g. 0.75. Overrides `train.spacing_iso`.",
    )
    return parser.parse_args()


def apply_cli_overrides(experiment_config, args):
    """Merges command-line resume flags into the experiment configuration."""
    if args.resume_auto:
        return OmegaConf.merge(
            experiment_config,
            OmegaConf.create({
                "resume": {
                    "enabled": True,
                    "checkpoint_path": None,
                    "auto_latest": True,
                }
            }),
        )

    if args.resume_checkpoint:
        return OmegaConf.merge(
            experiment_config,
            OmegaConf.create({
                "resume": {
                    "enabled": True,
                    "checkpoint_path": args.resume_checkpoint,
                    "auto_latest": False,
                }
            }),
        )

    return experiment_config


# ==============================================================================
# 2. DATA LOADING AND FIXED VISUALIZATION SAMPLE SELECTION
# ==============================================================================
def get_unified_dataloaders(cfg):
    """
    Constructs DataLoaders for the unified 3-class ostium dataset
    (ASOCA + CAT08 + ImageCAS).
    """
    # Isotropic training spacing (mm): optional `train.spacing_iso` in the experiment config.
    spacing_iso = float(cfg.get("train", {}).get("spacing_iso", DEFAULT_SPACING_ISO))
    if spacing_iso <= 0:
        raise ValueError(f"train.spacing_iso must be > 0, got {spacing_iso}")

    print(f"\n[*] Building unified dataset (3-class multi-source, spacing {spacing_iso} mm isotropic)...")
    train_dataset = UnifiedOstiaDatasetMulti(split='train', nsdim=3, new_spacing=spacing_iso)
    val_dataset   = UnifiedOstiaDatasetMulti(split='val',   nsdim=3, new_spacing=spacing_iso)

    workers_train = cfg.get("train", {}).get("num_workers", 0)

    persistent = workers_train > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("train", {}).get("batch_size", 1),
        shuffle=True,
        num_workers=workers_train,
        pin_memory=True,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers_train,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader


def select_fixed_viz_samples(train_dataset, val_dataset, patch_shape, seed: int = 42):
    """
    Select 1 volume per source dataset from both train and val splits for
    qualitative per-epoch monitoring.

    Pre-extracts crops (one per ostium side) around each GT for both splits,
    so every epoch only requires a cheap direct forward pass on the crops.

    Args:
        train_dataset: UnifiedOstiaDatasetMulti instance (split='train').
        val_dataset:   UnifiedOstiaDatasetMulti instance (split='val').
        patch_shape:   (pz, py, px) crop size matching the model's roi_size.
        seed:          Random seed for reproducibility.

    Returns:
        Tuple[list, list]: (train_samples, val_samples) — each a list of 3 dicts
        (one per source dataset: ASOCA, CAT08, ImageCAS), each containing:
            'sample_id', 'dataset_name', 'spacing',
            'image_R'      (1, pz, py, px),  'gt_in_crop_R' [z, y, x],
            'image_L'      (1, pz, py, px),  'gt_in_crop_L' [z, y, x].
    """
    import random as _rng_mod
    rng = _rng_mod.Random(seed)
    pz, py, px = patch_shape

    def _ranges(dataset):
        a = len(dataset.asoca)
        c = len(dataset.cat08)
        return {
            'ASOCA':    list(range(0, a)),
            'CAT08':    list(range(a, a + c)),
            'ImageCAS': list(range(a + c, len(dataset))),
        }

    def _normalise_img(sample):
        img = sample['image'].float()
        while img.dim() > 4:
            img = img.squeeze(0)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        return img  # (1, D, H, W)

    def _crop_around(img, coord, pz, py, px):
        """Return (crop, gt_in_crop) — crop is (1, pz, py, px)."""
        D, H, W = img.shape[-3:]
        gz, gy, gx = float(coord[0]), float(coord[1]), float(coord[2])
        z0 = max(0, min(int(round(gz)) - pz // 2, max(0, D - pz)))
        y0 = max(0, min(int(round(gy)) - py // 2, max(0, H - py)))
        x0 = max(0, min(int(round(gx)) - px // 2, max(0, W - px)))
        crop = img[:, z0:z0+pz, y0:y0+py, x0:x0+px]
        pad = (0, px - crop.shape[3], 0, py - crop.shape[2], 0, pz - crop.shape[1])
        if any(p > 0 for p in pad):
            crop = torch.nn.functional.pad(crop, pad)
        return crop, torch.tensor([gz - z0, gy - y0, gx - x0], dtype=torch.float32)

    # --- FIX START: SMART SEARCH ---
    def _pick_one(dataset, ds_name):
        """Iterates the dataset until a sample from the requested dataset with valid ostia is found."""
        all_indices = list(range(len(dataset)))
        rng.shuffle(all_indices)  # Shuffle for random selection
        
        for idx in all_indices:
            sample = dataset[idx]
            sample_id = sample.get('id', '')
            
            # 1. Check if the dataset name is contained in the sample ID
            if ds_name.lower() not in sample_id.lower():
                continue
                
            # 2. Check that the coordinates are valid
            if float(sample['ostia_coords_R'][0]) < 0:
                continue
            if float(sample['ostia_coords_L'][0]) < 0:
                continue
                
            return sample
        return None

    def _build_train_sample(dataset, ds_name):
        s = _pick_one(dataset, ds_name)
        if s is None:
            return None
        img = _normalise_img(s)
        crop_R, gt_R = _crop_around(img, s['ostia_coords_R'], pz, py, px)
        crop_L, gt_L = _crop_around(img, s['ostia_coords_L'], pz, py, px)
        return {
            'sample_id':    s.get('id', 'unknown'),
            'dataset_name': ds_name,
            'spacing':      tuple(float(v) for v in s['spacing']),
            'image_R':      crop_R,
            'gt_in_crop_R': gt_R,
            'image_L':      crop_L,
            'gt_in_crop_L': gt_L,
        }

    def _build_val_sample(dataset, ds_name):
        s = _pick_one(dataset, ds_name)
        if s is None:
            return None
        img = _normalise_img(s)
        return {
            'sample_id':    s.get('id', 'unknown'),
            'dataset_name': ds_name,
            'spacing':      tuple(float(v) for v in s['spacing']),
            'image_full':   img,
            'gt_R_zyx':     s['ostia_coords_R'].float(),
            'gt_L_zyx':     s['ostia_coords_L'].float(),
        }

    print("[*] Selecting fixed visualization samples...")
    train_samples, val_samples = [], []
    
    for ds_name in ['ASOCA', 'CAT08', 'ImageCAS']:
        t = _build_train_sample(train_dataset, ds_name)
        if t is not None:
            train_samples.append(t)
            print(f"    [train] {ds_name}: {t['sample_id']}")
            
        v = _build_val_sample(val_dataset, ds_name)
        if v is not None:
            val_samples.append(v)
            print(f"    [val]   {ds_name}: {v['sample_id']}")
    # --- FIX END ---

    return train_samples, val_samples


# ==============================================================================
# 3. PATCH SAMPLING
# ==============================================================================
def fix_tensor_shape(tensor, is_target=False):
    """Ensures the tensor is exactly 5D (B, C, D, H, W), collapsing any spurious inner dims."""
    if tensor.dim() > 5:
        B = tensor.shape[0]
        return tensor.reshape(B, tensor.shape[1], *tensor.shape[-3:])
    return tensor


def probabilistic_crop(images, targets, patch_shape, probs=(0.32, 0.34, 0.34), is_training=True):
    """
    Extracts a 3D patch from each volume using a class-aware sampling strategy.

    During training, each patch is drawn according to a probability distribution
    over three sampling modes:
        - Background (p_bg):      Random crop anywhere in the volume.
        - Right ostium (p_r):     Crop guaranteed to contain the right ostium.
        - Left ostium (p_l):      Crop guaranteed to contain the left ostium.

    This balanced sampling addresses the severe spatial imbalance of coronary
    ostia, which occupy only a small fraction of the full CT volume.

    During validation, the crop is centered deterministically on the peak of the
    combined ostium ground-truth heatmap.

    Args:
        images (Tensor):        Input volumes, shape (B, C, D, H, W).
        targets (Tensor):       Heatmap targets, shape (B, 3, D, H, W).
        patch_shape (tuple):    Desired patch size (pz, py, px).
        probs (tuple):          Sampling probabilities (p_bg, p_r, p_l).
        is_training (bool):     If True, applies probabilistic sampling.

    Returns:
        Tuple[Tensor, Tensor]:  Cropped image and target patches.
    """
    b, c, d, h, w = images.shape
    pz, py, px = patch_shape
    p_bg, p_r, p_l = probs

    cropped_imgs = torch.zeros((b, images.shape[1], pz, py, px), device=images.device, dtype=images.dtype)
    cropped_tgts = torch.zeros((b, targets.shape[1], pz, py, px), device=targets.device, dtype=targets.dtype)

    for i in range(b):
        if is_training:
            rand_val = random.random()
            if rand_val < p_bg:
                target_class = 0
            elif rand_val < p_bg + p_r:
                target_class = 1
            else:
                target_class = 2

            ostium_found = False
            if target_class in [1, 2]:
                if targets[i, target_class].max() > 0.05:
                    max_idx = torch.argmax(targets[i, target_class])
                    z_center = (max_idx // (h * w)).item()
                    y_center = ((max_idx % (h * w)) // w).item()
                    x_center = (max_idx % w).item()
                    ostium_found = True

            if ostium_found:
                z_min = max(0, z_center - pz + 1)
                z_max = min(d - pz, z_center)
                z0 = random.randint(z_min, z_max) if z_max >= z_min else max(0, z_center - pz // 2)

                y_min = max(0, y_center - py + 1)
                y_max = min(h - py, y_center)
                y0 = random.randint(y_min, y_max) if y_max >= y_min else max(0, y_center - py // 2)

                x_min = max(0, x_center - px + 1)
                x_max = min(w - px, x_center)
                x0 = random.randint(x_min, x_max) if x_max >= x_min else max(0, x_center - px // 2)
            else:
                z0 = random.randint(0, max(0, d - pz))
                y0 = random.randint(0, max(0, h - py))
                x0 = random.randint(0, max(0, w - px))
        else:
            # Deterministic crop centered on the strongest heatmap response.
            ostia_combined = torch.max(targets[i, 1], targets[i, 2])
            if ostia_combined.max() > 0.05:
                max_idx = torch.argmax(ostia_combined)
                z_c = (max_idx // (h * w)).item()
                y_c = ((max_idx % (h * w)) // w).item()
                x_c = (max_idx % w).item()
            else:
                z_c, y_c, x_c = d // 2, h // 2, w // 2

            z0 = max(0, min(z_c - pz // 2, d - pz))
            y0 = max(0, min(y_c - py // 2, h - py))
            x0 = max(0, min(x_c - px // 2, w - px))

        # At coarse training spacings a volume can be smaller than the patch along some axis (e.g. few
        # slices): use its whole extent there and pad at the far end, as infer/utils.py does (constant =
        # volume minimum, i.e. air). The padded targets are background (channel 0 = 1, ostia channels = 0).
        if d <= pz: z0 = 0
        if h <= py: y0 = 0
        if w <= px: x0 = 0
        img_crop = images[i, :, z0:z0+pz, y0:y0+py, x0:x0+px]
        tgt_crop = targets[i, :, z0:z0+pz, y0:y0+py, x0:x0+px]
        if img_crop.shape[1:] != (pz, py, px):
            cropped_imgs[i] = images[i].min()
            cropped_tgts[i, 0] = 1.0
            cropped_tgts[i, 1:] = 0.0
            cz, cy, cx = img_crop.shape[1:]
            cropped_imgs[i, :, :cz, :cy, :cx] = img_crop
            cropped_tgts[i, :, :cz, :cy, :cx] = tgt_crop
        else:
            cropped_imgs[i] = img_crop
            cropped_tgts[i] = tgt_crop

    return cropped_imgs, cropped_tgts


# ==============================================================================
# 4. VALIDATION METRIC: Local Centroid Extraction
# ==============================================================================
def extract_local_centroid(heatmap, window_size=11):
    """
    Finds the maximum peak of the heatmap and computes the centroid
    only within a small local window for sub-pixel precision.
    """
    max_idx = torch.argmax(heatmap)
    D, H, W = heatmap.shape
    
    z_peak = (max_idx // (H * W)).item()
    y_peak = ((max_idx % (H * W)) // W).item()
    x_peak = (max_idx % W).item()
    
    half_w = window_size // 2
    z_min, z_max = max(0, z_peak - half_w), min(D, z_peak + half_w + 1)
    y_min, y_max = max(0, y_peak - half_w), min(H, y_peak + half_w + 1)
    x_min, x_max = max(0, x_peak - half_w), min(W, x_peak + half_w + 1)
    
    local_window = heatmap[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Subtract the local window minimum to zero out the baseline noise
    # and ensure non-negative weights well-shaped around the peak.
    local_window = local_window - local_window.min()
    
    sum_weights = torch.sum(local_window) + 1e-6
    
    z_coords = torch.arange(local_window.shape[0], device=heatmap.device, dtype=torch.float32)
    y_coords = torch.arange(local_window.shape[1], device=heatmap.device, dtype=torch.float32)
    x_coords = torch.arange(local_window.shape[2], device=heatmap.device, dtype=torch.float32)
    
    grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    c_z = torch.sum(local_window * grid_z) / sum_weights
    c_y = torch.sum(local_window * grid_y) / sum_weights
    c_x = torch.sum(local_window * grid_x) / sum_weights
    
    final_z = z_min + c_z.item()
    final_y = y_min + c_y.item()
    final_x = x_min + c_x.item()
    
    return torch.tensor([final_z, final_y, final_x], dtype=torch.float32)

# ==============================================================================
# 5. OPTIMIZER AND SCHEDULER FACTORIES  (section 5 of 7)
# ==============================================================================
def get_scheduler(optimizer, cfg, total_steps=None):
    """
    Instantiates the learning rate scheduler specified in the experiment config.

    Supported schedulers:
        - none / off / disabled:    No scheduling; constant learning rate.
        - custom_warmup:            Linear warmup followed by cosine decay.
        - step:                     Step decay by gamma every step_size epochs.
        - cosine:                   Cosine annealing to a minimum learning rate.
        - reduce_on_plateau:        Reduce on validation loss plateau.
    """
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "none").lower()

    if scheduler_name in ["none", "off", "disabled"]:
        return None

    if scheduler_name == "custom_warmup":
        return SchedulerLR(
            lr_ref=cfg.optimizer.get("learning_rate", 1e-4),
            max_training_steps=total_steps,  # Computed from total epochs × steps per epoch.
            warmup_steps=scheduler_cfg.get("warmup_steps", 1000)
        )
    
    if scheduler_name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.get("t_0", 10),       # Epochs before the first restart
            T_mult=scheduler_cfg.get("t_mult", 2),  # Multiplier to lengthen subsequent cycles
            eta_min=scheduler_cfg.get("min_lr", 0.0)
        )

    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.5),
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=scheduler_cfg.get("min_lr", 0.0),
        )
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get("gamma", 0.5),
            patience=scheduler_cfg.get("patience", 5),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def get_optimizer(model, cfg):
    """Instantiates the optimizer specified in the experiment config."""
    optimizer_cfg = cfg.get("optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "adamw").lower()
    learning_rate = optimizer_cfg.get("learning_rate", 1e-4)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate,
                               momentum=optimizer_cfg.get("momentum", 0.9),
                               weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_current_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


# DSNT curriculum for asymmetric ostia weighting:
# start balanced-ish (R/L = 0.35/0.65), then shift more weight to the left ostium
# from the configured epoch onward. Applying it every epoch also keeps resume correct.

def apply_dsnt_weight_schedule(criterion, cfg, epoch):
    """Update DSNT R/L loss weights from config, if a switch epoch is defined."""
    if not hasattr(criterion, "weight_R") or not hasattr(criterion, "weight_L"):
        return None

    loss_cfg = cfg.get("loss", {})
    switch_epoch = loss_cfg.get("weight_switch_epoch", None)
    if switch_epoch is None:
        return (criterion.weight_R, criterion.weight_L)

    if epoch >= int(switch_epoch):
        target_R = float(loss_cfg.get("weight_R_after_switch", criterion.weight_R))
        target_L = float(loss_cfg.get("weight_L_after_switch", criterion.weight_L))
    else:
        target_R = float(loss_cfg.get("weight_R", criterion.weight_R))
        target_L = float(loss_cfg.get("weight_L", criterion.weight_L))

    if abs(criterion.weight_R - target_R) > 1e-12 or abs(criterion.weight_L - target_L) > 1e-12:
        print(f"[*] DSNT loss weights: R={target_R:.2f}, L={target_L:.2f} (epoch {epoch})")
        criterion.weight_R = target_R
        criterion.weight_L = target_L

    return (criterion.weight_R, criterion.weight_L)


def get_batch_sample_id(batch):
    sample_id = batch.get('id', batch.get('sample_id', ['unknown']))
    if isinstance(sample_id, (list, tuple)):
        sample_id = sample_id[0]
    if isinstance(sample_id, torch.Tensor):
        sample_id = sample_id.item()
    return str(sample_id)


def infer_dataset_name(sample_id: str) -> str:
    for name in ("ASOCA", "CAT08", "ImageCAS"):
        if name.lower() in sample_id.lower():
            return name
    return "unknown"

def apply_threshold_weight_schedule(criterion, cfg, err_R, err_L, threshold_mm=2.0, focus_weight=0.75):
    """
    Starts with the base weights defined in the config. If the error of one ostium drops
    below threshold_mm while the other does not, shifts weight toward the one with the larger error.
    """
    if not hasattr(criterion, "weight_R") or not hasattr(criterion, "weight_L"):
        return None

    loss_cfg = cfg.get("loss", {})
    # 1. Retrieve base weights from the yaml file
    base_R = float(loss_cfg.get("weight_R", 0.5))
    base_L = float(loss_cfg.get("weight_L", 0.5))

    # 2. Initial condition (first validation not yet performed)
    if err_R == 0.0 and err_L == 0.0:
        target_R, target_L = base_R, base_L
    else:
        # 3. Threshold logic
        if err_R < threshold_mm and err_L >= threshold_mm:
            # Right ostium is below the threshold (e.g. 1.5mm), left is still struggling.
            # Shift weight toward the left.
            target_R = 1.0 - focus_weight  # e.g. 0.15
            target_L = focus_weight        # e.g. 0.85
        elif err_L < threshold_mm and err_R >= threshold_mm:
            # Left ostium is below the threshold, shift weight toward the right.
            target_R = focus_weight        # e.g. 0.85
            target_L = 1.0 - focus_weight  # e.g. 0.15
        else:
            # If both are below (or above) the threshold, use the default weights
            target_R, target_L = base_R, base_L

    # 4. Apply and log the change only if there was a variation
    if abs(criterion.weight_R - target_R) > 1e-12 or abs(criterion.weight_L - target_L) > 1e-12:
        print(f"[*] Loss weight change: R={target_R:.2f}, L={target_L:.2f} (Err R: {err_R:.1f}mm, Err L: {err_L:.1f}mm)")
        criterion.weight_R = target_R
        criterion.weight_L = target_L

    return (criterion.weight_R, criterion.weight_L)

def apply_sinh_weight_schedule(
    criterion, epoch, err_R, err_L, max_epochs,
    a_min=0.1, a_max=1.5, max_weight=0.75, smoothing=0.5,
):
    """
    Updates loss function weights continuously and with damping:
    - Hard limits: proportions confined to the interval [1 - max_weight, max_weight] (e.g. [0.25, 0.75]).
    - Intermediate step: applies (w_current + w_target) / 2 to avoid abrupt transitions.
    """
    if not hasattr(criterion, "weight_R") or not hasattr(criterion, "weight_L"):
        return None
    if not 0.5 <= max_weight < 1.0:
        raise ValueError("max_weight must be in the interval [0.5, 1.0)")

    min_weight = 1.0 - max_weight

    # 1. Initial condition: balanced start at 0.5 / 0.5
    if (err_R == 0.0 and err_L == 0.0) or epoch <= 1:
        target_R, target_L = 0.5, 0.5
        criterion.weight_R = target_R
        criterion.weight_L = target_L
        return (target_R, target_L)

    # 2. Parameter A (Curriculum): linear progression from a_min to a_max
    progress = (epoch - 1) / max(1, max_epochs - 1)
    progress = max(0.0, min(1.0, progress))
    A = a_min + progress * (a_max - a_min)

    # 3. Asymptotic target computation via sinh and sigmoid
    mean_err = (err_R + err_L) / 2.0
    diff_L = err_L - mean_err
    diff_L = max(-50.0, min(50.0, diff_L))
    
    x = A * math.sinh(diff_L)
    x = max(-100.0, min(100.0, x))
    ideal_L = 1.0 / (1.0 + math.exp(-x))

    # 4. Clamp target between min_weight and max_weight
    ideal_L = max(min_weight, min(max_weight, ideal_L))

    # 5. Intermediate step: (w_current + w_target) / 2 via smoothing=0.5
    curr_L = float(criterion.weight_L)
    applied_L = (1.0 - smoothing) * curr_L + smoothing * ideal_L
    applied_L = max(min_weight, min(max_weight, applied_L))
    applied_R = 1.0 - applied_L

    # 6. Update only if the change is significant
    if abs(criterion.weight_R - applied_R) > 1e-4 or abs(criterion.weight_L - applied_L) > 1e-4:
        print(
            f"[*] Loss weight change (sinh smooth): R={applied_R:.4f}, L={applied_L:.4f} "
            f"| Ideal Target L={ideal_L:.4f} | A={A:.3f} | (Err R: {err_R:.1f}mm, Err L: {err_L:.1f}mm)"
        )
        criterion.weight_R = applied_R
        criterion.weight_L = applied_L

    return (criterion.weight_R, criterion.weight_L)

# ==============================================================================
# 6. CHECKPOINT RESUME UTILITIES
# ==============================================================================
def resolve_resume_checkpoint(cfg):
    """
    Resolves the checkpoint path to resume from, based on the experiment config.

    Returns None if resume is disabled. Raises FileNotFoundError if the
    specified or latest checkpoint cannot be found.
    """
    resume_cfg = cfg.get("resume", {})
    if not resume_cfg.get("enabled", False):
        return None
    checkpoint_path = resume_cfg.get("checkpoint_path", None)
    if checkpoint_path:
        checkpoint_path = resolve_project_path(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    if resume_cfg.get("auto_latest", False):
        checkpoint_root = resolve_project_path(cfg.logger.get("save_dir", "./checkpoints"))
        # Only this run's folders (<timestamp>_<run_name>): never another spacing's checkpoint.
        pattern = os.path.join(checkpoint_root, cfg.model.name, f"*_{cfg.logger.run_name}", "last_model.pth")
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found matching pattern: {pattern}")
        return max(candidates, key=os.path.getmtime)
    raise ValueError("Resume is enabled, but neither checkpoint_path nor auto_latest is configured.")


def load_training_state(checkpoint_path, model, optimizer, scheduler, device):
    """
    Restores model weights, optimizer state, and scheduler state from a checkpoint.

    The custom SchedulerLR does not implement state_dict / load_state_dict;
    any AttributeError during scheduler restoration is silently ignored.

    Returns:
        Tuple: (start_epoch, best_val_loss, resume_dir, mlflow_run_id,
                epoch_history, training_state)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except AttributeError:
            pass  # Custom scheduler does not support state persistence.

    start_epoch    = int(checkpoint["epoch"]) + 1
    best_val_loss  = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
    resume_dir     = checkpoint.get("save_dir", os.path.dirname(checkpoint_path))
    mlflow_run_id  = checkpoint.get("mlflow_run_id", None)
    epoch_history  = checkpoint.get("epoch_history", [])
    training_state = checkpoint.get("training_state", {})

    return start_epoch, best_val_loss, resume_dir, mlflow_run_id, epoch_history, training_state


# ==============================================================================
# 7. TRAINING ENGINE
# ==============================================================================
def main():
    args = parse_args()

    print("=" * 60)
    print("HCAT OSTIA LOCALIZATION — TRAINING (MULTI-CLASS)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    experiment_config = get_active_experiment_config(config)
    experiment_config = apply_cli_overrides(experiment_config, args)
    print(f"[*] Active experiment: {experiment_config.active_experiment}")

    # Separate checkpoint folder / MLflow run for every training spacing.
    OmegaConf.update(experiment_config, "train.spacing_iso", args.spacing_iso, force_add=True)
    run_name = f"{experiment_config.active_experiment}_ablation_training_spacing_{args.spacing_iso:.2f}mm"
    OmegaConf.update(experiment_config, "logger.run_name", run_name, force_add=True)
    print(f"[*] Ablation run: {run_name} (training spacing: {args.spacing_iso} mm isotropic)")

    if experiment_config.resume.get("enabled", False):
        if experiment_config.resume.get("auto_latest", False):
            print("[*] Resume mode: latest checkpoint")
        else:
            print(f"[*] Resume mode: {experiment_config.resume.get('checkpoint_path')}")

    # Build dataloaders before initializing the scheduler, as total_steps
    # depends on the number of training batches.
    train_loader, val_loader = get_unified_dataloaders(experiment_config)
    num_epochs        = experiment_config.train.epochs
    total_train_steps = num_epochs * len(train_loader)

    sigma_sched = SigmaScheduler(
        sigma_start=experiment_config.train.get("sigma_start", 2.5),
        sigma_min=0.0,
        reduction_factor=experiment_config.train.get("sigma_reduction_factor", 0.75),
        min_relative_improvement=experiment_config.train.get("sigma_min_improvement", 0.05),
        patience=experiment_config.train.get("sigma_patience", 2),
        warmup_epochs=experiment_config.train.get("sigma_warmup_epochs", 5),
    )

    preprocessor = PreprocessorCT()
    model     = get_model(experiment_config).to(device).float()
    criterion = get_loss(experiment_config)
    optimizer = get_optimizer(model, experiment_config)
    scheduler = get_scheduler(optimizer, experiment_config, total_train_steps)

    use_mixed_precision = device.type == "cuda"
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=use_mixed_precision)

    start_epoch = 1
    resume_checkpoint = resolve_resume_checkpoint(experiment_config)
    if resume_checkpoint:
        print(f"[*] Resuming from: {resume_checkpoint}")
        start_epoch, best_val_loss, resume_dir, mlflow_run_id, epoch_history, training_state = load_training_state(
            resume_checkpoint, model, optimizer, scheduler, device
        )
        # --- RESUME FIX START ---
        # Unlock the config to allow adding new variables
        OmegaConf.set_struct(experiment_config, False)
        
        # Safely add only the text data
        experiment_config.logger.resume_dir = resume_dir
        experiment_config.logger.mlflow_run_id = mlflow_run_id
        # --- FIX END ---
    else:
        best_val_loss = float("inf")
        training_state = {}

    logger = ExperimentLogger(experiment_config)
    logger.best_val_loss = best_val_loss

    # --- ADD THIS LINE BELOW ---
    if resume_checkpoint:
        logger.epoch_history = epoch_history

    # =================================================================
    # --- NEW: MLFLOW ARTIFACT AND TAG LOGGING ---
    # =================================================================
    if mlflow.active_run():
        print("[*] Logging parameters, tags and script to MLflow...")
        
        # 1. Add tags for smart filtering in the UI
        mlflow.set_tags({
            "Model": experiment_config.model.get("name", "Unknown"),
            "Scheduler": experiment_config.scheduler.get("name", "Unknown"),
            "Loss": experiment_config.loss.get("name", "Unknown"),
            "FOV_Size": str(experiment_config.model.get("img_size", "Unknown"))
        })

        mlflow.set_tags({
            "ablation": "training_spacing",
            "training_spacing_mm": f"{args.spacing_iso:.2f}",
        })

        # 2. Log the current training script
        mlflow.log_artifact(os.path.abspath(__file__), artifact_path="scripts_backup")
        
        # 3. Log the configuration file
        config_path = resolve_project_path("hcatostialocalization/config/model.yaml")
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="config_backup")
    # =================================================================

    max_train_steps  = experiment_config.train.get("max_train_steps", None)
    save_checkpoints = experiment_config.train.get("save_checkpoints", True)
    gradient_accumulation_steps = max(int(experiment_config.train.get("gradient_accumulation_steps", 1)), 1)
    gradient_clip_norm          = experiment_config.train.get("gradient_clip_norm", None)
    best_warmup_epochs          = int(experiment_config.train.get("best_warmup_epochs", 0))

    patch_z, patch_y, patch_x = [int(s) for s in experiment_config.model.img_size]
    patch_shape = (patch_z, patch_y, patch_x)
    crop_probs  = experiment_config.train.get("crop_probs", [0.32, 0.34, 0.34])
    # Data augmentation on training patches (see data/transform/augment_3d_ostia.py).
    # Optional train.augmentation block in the YAML: 'enabled' + kwargs of augment_3d_ostia.
    augmentation_kwargs = experiment_config.train.get("augmentation", None)
    augmentation_kwargs = OmegaConf.to_container(augmentation_kwargs, resolve=True) if augmentation_kwargs is not None else {}
    augmentation_enabled = bool(augmentation_kwargs.pop("enabled", True))
    print(f"[*] Data augmentation: {'ON' if augmentation_enabled else 'OFF'} {augmentation_kwargs}")
    outlier_threshold_mm = float(experiment_config.train.get("outlier_threshold_mm", 15.0))

    # Select and register fixed visualization samples.
    fixed_train_viz, fixed_val_viz = select_fixed_viz_samples(
        train_loader.dataset, val_loader.dataset, patch_shape
    )
    logger.init_fixed_viz(fixed_train_viz, fixed_val_viz)

    val_every_n_epochs = int(experiment_config.train.get("val_every_n_epochs", 1))
    sinh_plateau_cfg = experiment_config.loss.get("sinh_plateau", {})
    sinh_plateau_enabled = bool(sinh_plateau_cfg.get("enabled", False))
    plateau_patience = int(sinh_plateau_cfg.get("patience", 3))
    plateau_min_delta_mm = float(sinh_plateau_cfg.get("min_delta_mm", 0.1))
    plateau_warmup_epochs = int(sinh_plateau_cfg.get("warmup_epochs", 0))
    sinh_activation_delay_epochs = int(sinh_plateau_cfg.get("activation_delay_epochs", 5))
    sinh_a_min = float(sinh_plateau_cfg.get("a_min", 0.1))
    sinh_a_max = float(sinh_plateau_cfg.get("a_max", 1.5))
    sinh_max_weight = float(sinh_plateau_cfg.get("max_weight", 0.75))
    sinh_smoothing = float(sinh_plateau_cfg.get("smoothing", 0.5))

    plateau_best_error = float(training_state.get("plateau_best_error", float("inf")))
    plateau_counter = int(training_state.get("plateau_counter", 0))
    sinh_active = bool(training_state.get("sinh_active", False))
    sinh_activation_epoch = training_state.get("sinh_activation_epoch", None)
    if sinh_activation_epoch is not None:
        sinh_activation_epoch = int(sinh_activation_epoch)

    prev_dist_R = float(training_state.get("prev_dist_R", 0.0))
    prev_dist_L = float(training_state.get("prev_dist_L", 0.0))
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'-' * 20} EPOCH {epoch}/{num_epochs} {'-' * 20}")
        epoch_start_time = time.perf_counter()
        run_validation = (epoch % val_every_n_epochs == 0) or (epoch == num_epochs)
        sinh_used_this_epoch = (
            sinh_plateau_enabled
            and sinh_active
            and sinh_activation_epoch is not None
            and epoch >= sinh_activation_epoch
        )
        if sinh_used_this_epoch:
            sinh_epoch = epoch - sinh_activation_epoch + 1
            sinh_max_epochs = max(1, num_epochs - sinh_activation_epoch + 1)
            dsnt_weights = apply_sinh_weight_schedule(
                criterion=criterion,
                epoch=sinh_epoch,
                err_R=prev_dist_R,
                err_L=prev_dist_L,
                max_epochs=sinh_max_epochs,
                a_min=sinh_a_min,
                a_max=sinh_a_max,
                max_weight=sinh_max_weight,
                smoothing=sinh_smoothing,
            )
        else:
            dsnt_weights = apply_dsnt_weight_schedule(
                criterion, experiment_config, epoch
            )

        # ------------------------------------------------------------------
        # Training phase
        # ------------------------------------------------------------------
        model.train()
        train_loss_accum, train_steps = 0.0, 0
        train_pbar       = tqdm(train_loader, desc="Training")
        train_start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for batch in train_pbar:
            # here batch is a whole volume together with label, not a cropped sample (see General_Class_Dataset_Multi, patched_getitem_multi())
            global_step = (epoch - 1) * len(train_loader) + train_steps

            # Update learning rate at each step for the custom warmup scheduler.
            if scheduler is not None and experiment_config.scheduler.name.lower() == "custom_warmup":
                scheduler.step_update_optimizer_lr(global_step, optimizer)

            images  = fix_tensor_shape(batch['image'].to(device).float())
            targets = fix_tensor_shape(batch['label'].to(device).float(), is_target=True)
            images, targets = probabilistic_crop(images, targets, patch_shape,
                                                 probs=crop_probs, is_training=True)
            if augmentation_enabled:
                images, targets = augment_3d_ostia(images, targets, **augmentation_kwargs)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                 enabled=use_mixed_precision):
                images            = preprocessor(images)
                preds             = model(images)
                loss              = criterion(preds, targets)
                loss_for_backward = loss / gradient_accumulation_steps

            scaler.scale(loss_for_backward).backward()

            train_loss_accum += loss.item()
            train_steps      += 1

            should_update    = train_steps % gradient_accumulation_steps == 0
            reached_limit    = max_train_steps is not None and train_steps >= max_train_steps
            is_last_batch    = train_steps == len(train_loader)

            if should_update or reached_limit or is_last_batch:
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            current_lr = get_current_learning_rate(optimizer)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

            if max_train_steps is not None and train_steps >= max_train_steps:
                break

        train_time_sec = time.perf_counter() - train_start_time

        # ------------------------------------------------------------------
        # Validation phase — Euclidean Distance (mm) on Full Volume
        # ------------------------------------------------------------------
        val_start_time = time.perf_counter()
        
        final_activation = experiment_config.model.get("final_activation", "softmax").lower()
        loss_name        = experiment_config.get("loss", {}).get("name", "").lower()
        apply_sigmoid    = final_activation in ["none", "identity"] and loss_name != "dsnt_jsd"

        avg_dist_R, avg_dist_L, avg_val_loss = 0.0, 0.0, float('inf')
        med_dist_R, med_dist_L, med_val_loss = 0.0, 0.0, float('inf')
        p75_dist_R, p75_dist_L, p75_val_loss = 0.0, 0.0, float('inf')

        if run_validation:
            model.eval()
            val_pbar = tqdm(val_loader, desc="Validation (Sliding Window)")
            
            dist_R_accum = 0.0
            dist_L_accum = 0.0
            valid_samples = 0
            all_dists_R = []
            all_dists_L = []
            
            with torch.inference_mode():
                for batch in val_pbar:
                    images = fix_tensor_shape(batch['image'].to(device).float())
                    
                    # Extract ground-truth coordinates and spacing
                    gt_R = batch['ostia_coords_R'][0] # [z, y, x]
                    gt_L = batch['ostia_coords_L'][0] # [z, y, x]
                    spacing = batch['spacing'][0].flatten()  # [sp_z, sp_y, sp_x] — ensures shape (3,)
                    
                    # Sliding Window inference on the volume
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_mixed_precision):
                        images = preprocessor(images)
                        preds = sliding_window_inference(
                            inputs=images,
                            roi_size=patch_shape,
                            sw_batch_size=4, # Increase to 8 or reduce to 2 depending on VRAM
                            predictor=model,
                            overlap=0.5,
                            mode="gaussian"
                        )
                    
                    if apply_sigmoid:
                        preds = torch.sigmoid(preds)
                        
                    heatmap_R = preds[0, 1] 
                    heatmap_L = preds[0, 2]
                    
                    # Local centroid extraction
                    pred_coords_R = extract_local_centroid(heatmap_R)
                    pred_coords_L = extract_local_centroid(heatmap_L)
                    
                    # Compute Euclidean distance in millimetres
                    diff_R = pred_coords_R.cpu() - gt_R.cpu()
                    diff_L = pred_coords_L.cpu() - gt_L.cpu()
                    dist_R_mm = torch.sqrt(torch.sum((diff_R * spacing)**2)).item()
                    dist_L_mm = torch.sqrt(torch.sum((diff_L * spacing)**2)).item()
                    
                    # --- OUTLIER TRACKING ---
                    sample_id = get_batch_sample_id(batch)
                    dataset_name = infer_dataset_name(sample_id)
                    if dist_R_mm > outlier_threshold_mm or dist_L_mm > outlier_threshold_mm:
                        print(
                            f"\n[OUTLIER] Patient ID: {sample_id} | "
                            f"Error R: {dist_R_mm:.1f} mm | Error L: {dist_L_mm:.1f} mm"
                        )
                    if dist_R_mm > outlier_threshold_mm:
                        logger.log_outlier(
                            epoch=epoch,
                            sample_id=sample_id,
                            dataset_name=dataset_name,
                            side="R",
                            image_volume=images[0, 0],
                            gt_zyx=gt_R,
                            pred_zyx=pred_coords_R,
                            distance_mm=dist_R_mm,
                            spacing=spacing,
                            threshold_mm=outlier_threshold_mm,
                        )
                    if dist_L_mm > outlier_threshold_mm:
                        logger.log_outlier(
                            epoch=epoch,
                            sample_id=sample_id,
                            dataset_name=dataset_name,
                            side="L",
                            image_volume=images[0, 0],
                            gt_zyx=gt_L,
                            pred_zyx=pred_coords_L,
                            distance_mm=dist_L_mm,
                            spacing=spacing,
                            threshold_mm=outlier_threshold_mm,
                        )

                    dist_R_accum += dist_R_mm
                    dist_L_accum += dist_L_mm
                    all_dists_R.append(dist_R_mm)
                    all_dists_L.append(dist_L_mm)
                    valid_samples += 1
                    
                    val_pbar.set_postfix({"Dist_R (mm)": f"{dist_R_mm:.2f}", "Dist_L (mm)": f"{dist_L_mm:.2f}"})

            # Means, Medians and 75th Percentile
            avg_dist_R = dist_R_accum / max(valid_samples, 1)
            avg_dist_L = dist_L_accum / max(valid_samples, 1)
            avg_val_loss = (avg_dist_R + avg_dist_L) / 2.0  # Global mean error

            med_dist_R = float(np.median(all_dists_R)) if all_dists_R else 0.0
            med_dist_L = float(np.median(all_dists_L)) if all_dists_L else 0.0
            med_val_loss = (med_dist_R + med_dist_L) / 2.0  # Global median error

            p75_dist_R = float(np.percentile(all_dists_R, 75)) if all_dists_R else 0.0
            p75_dist_L = float(np.percentile(all_dists_L, 75)) if all_dists_L else 0.0
            p75_val_loss = (p75_dist_R + p75_dist_L) / 2.0  # Global 75th-percentile error

            # Dynamic R/L weights and the plateau monitor the 75th percentile:
            prev_dist_R = p75_dist_R
            prev_dist_L = p75_dist_L

            if sinh_plateau_enabled and not sinh_active:
                improved = p75_val_loss <= plateau_best_error - plateau_min_delta_mm
                if improved:
                    plateau_best_error = p75_val_loss
                    plateau_counter = 0
                elif epoch >= plateau_warmup_epochs:
                    plateau_counter += 1

                if epoch >= plateau_warmup_epochs and plateau_counter >= plateau_patience:
                    sinh_active = True
                    sinh_activation_epoch = epoch + sinh_activation_delay_epochs
                    print(
                        f"[*] Plateau detected at epoch {epoch} (p75={p75_val_loss:.2f}mm): "
                        f"sinh will be activated from epoch {sinh_activation_epoch}."
                    )

        val_time_sec = time.perf_counter() - val_start_time

        # ------------------------------------------------------------------
        # Fixed visualization (for images and GIFs on MLflow)
        # ------------------------------------------------------------------
        model.eval()
        with torch.inference_mode():
            # Train crops
            for s in fixed_train_viz:
                for img_key, pred_key, ch in [('image_R', 'pred_R', 1), ('image_L', 'pred_L', 2)]:
                    img_in = s[img_key].to(device).float().unsqueeze(0)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_mixed_precision):
                        img_in = preprocessor(img_in)
                        out = model(img_in)
                    
                    pred_ch = out[0, ch].float().cpu()
                    if loss_name == "dsnt_jsd" or apply_sigmoid:
                        pred_ch = torch.sigmoid(pred_ch)
                    s[pred_key] = pred_ch

            # Val full volumes
            for s in fixed_val_viz:
                img_in = s['image_full'].to(device).float().unsqueeze(0)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_mixed_precision):
                    img_in = preprocessor(img_in)
                    out = sliding_window_inference(
                        inputs=img_in, roi_size=patch_shape, sw_batch_size=4,
                        predictor=model, overlap=0.5, buffer_steps=4, buffer_dim=-1,
                        mode="gaussian"
                    )
                
                for pred_key, ch in [('pred_R_full', 1), ('pred_L_full', 2)]:
                    pred_ch = out[0, ch].float().cpu()
                    if loss_name == "dsnt_jsd" or apply_sigmoid:
                        pred_ch = torch.sigmoid(pred_ch)
                    s[pred_key] = pred_ch

        logger.log_fixed_train_visualizations(epoch, sigma_sched.sigma)
        logger.log_fixed_val_visualizations(epoch, sigma_sched.sigma)

        # ------------------------------------------------------------------
        # Epoch summary
        # ------------------------------------------------------------------
        avg_train = train_loss_accum / train_steps
        epoch_time_sec = time.perf_counter() - epoch_start_time

        timing_metrics = {
            "epoch_time_sec": epoch_time_sec,
            "train_time_sec": train_time_sec,
            "val_time_sec": val_time_sec,
            "seconds_per_train_step": train_time_sec / train_steps,
            "sinh_active": int(sinh_used_this_epoch),
            "plateau_counter": plateau_counter,
        }
        if dsnt_weights is not None:
            timing_metrics["dsnt_weight_R"] = dsnt_weights[0]
            timing_metrics["dsnt_weight_L"] = dsnt_weights[1]

        print(f"\n[Epoch {epoch}] Train loss: {avg_train:.4f}")
        if dsnt_weights is not None:
            print(f"          DSNT weights R/L: {dsnt_weights[0]:.2f}/{dsnt_weights[1]:.2f}")
        if run_validation:
            print(
                f"          Val Median: R={med_dist_R:.2f}mm, L={med_dist_L:.2f}mm | Global Med: {med_val_loss:.2f}mm\n"
                f"          Val p75:    R={p75_dist_R:.2f}mm, L={p75_dist_L:.2f}mm | Global p75: {p75_val_loss:.2f}mm\n"
                f"          Val Mean:   R={avg_dist_R:.2f}mm, L={avg_dist_L:.2f}mm | Global Mean: {avg_val_loss:.2f}mm"
            )
            
            # --- NEW: LOG METRICS TO MLFLOW ---
            mlflow.log_metric("Val_Error_Global_Median_mm", med_val_loss, step=epoch)
            mlflow.log_metric("Val_Distance_Right_Median_mm", med_dist_R, step=epoch)
            mlflow.log_metric("Val_Distance_Left_Median_mm", med_dist_L, step=epoch)

            mlflow.log_metric("Val_Error_Global_p75_mm", p75_val_loss, step=epoch)
            mlflow.log_metric("Val_Distance_Right_p75_mm", p75_dist_R, step=epoch)
            mlflow.log_metric("Val_Distance_Left_p75_mm", p75_dist_L, step=epoch)

            mlflow.log_metric("Val_Error_Global_mm", avg_val_loss, step=epoch)
            mlflow.log_metric("Val_Distance_Right_mm", avg_dist_R, step=epoch)
            mlflow.log_metric("Val_Distance_Left_mm", avg_dist_L, step=epoch)

            mlflow.log_metric("Plateau_Best_Global_p75_mm", plateau_best_error, step=epoch)
            if sinh_activation_epoch is not None:
                mlflow.log_metric("Sinh_Activation_Epoch", sinh_activation_epoch, step=epoch)

            timing_metrics.update({
                "val_dist_R_median": med_dist_R,
                "val_dist_L_median": med_dist_L,
                "val_global_median": med_val_loss,
                "val_dist_R_p75": p75_dist_R,
                "val_dist_L_p75": p75_dist_L,
                "val_global_p75": p75_val_loss,
                "val_dist_R_mean": avg_dist_R,
                "val_dist_L_mean": avg_dist_L,
                "val_global_mean": avg_val_loss,
            })
            # --- END NEW ---
            
        print(f"          Time: {epoch_time_sec:.1f}s | train {train_time_sec:.1f}s")

        logger.log_epoch_metrics(
            epoch, avg_train,
            get_current_learning_rate(optimizer),
            timing_metrics={**timing_metrics, "sigma": sigma_sched.sigma},
            # Also pass val loss if the logger supports it
        )

        if scheduler is not None and experiment_config.scheduler.name.lower() != "custom_warmup":
            if experiment_config.scheduler.name.lower() == "reduce_on_plateau":
                scheduler.step(p75_val_loss)
            else:
                scheduler.step()

        # Curriculum sigma annealing based on the ERROR IN MILLIMETRES!
        # The Gaussian narrows only if the distance improves.
        if run_validation:
            new_sigma = sigma_sched.step(p75_val_loss)
            train_loader.dataset.set_sigma(new_sigma)
            # Sincronizza il sigma al criterion se si usa DSNTJSDLoss
            if hasattr(criterion, 'sigma_jsd'):
                criterion.sigma_jsd = new_sigma
            print(f"           Sigma: {new_sigma:.4f}")

        if save_checkpoints:
            # During warmup best_model.pth is locked: we pass float('inf')
            # so is_best=False and last_model.pth is still updated.
            # val_loss_for_best is based on the 75th percentile to save the truly best model.
            in_warmup = best_warmup_epochs > 0 and epoch <= best_warmup_epochs
            val_loss_for_best = float('inf') if in_warmup else p75_val_loss
            if in_warmup:
                print(f"    [best locked] Warmup active: best_model.pth locked until epoch {best_warmup_epochs}.")
            elif best_warmup_epochs > 0 and epoch == best_warmup_epochs + 1:
                print(f"    [best unlocked] Warmup finished: best_model.pth is now updatable.")
            plateau_state = {
                "plateau_best_error": plateau_best_error,
                "plateau_counter": plateau_counter,
                "sinh_active": sinh_active,
                "sinh_activation_epoch": sinh_activation_epoch,
                "prev_dist_R": prev_dist_R,
                "prev_dist_L": prev_dist_L,
            }
            logger.save_checkpoint(
                model, optimizer, epoch, val_loss_for_best, scheduler,
                training_state=plateau_state,
            )

    logger.save_loss_plot()
    logger.save_validation_plot()
    logger.save_prediction_gifs()
    logger.close()
    print("\nTRAINING COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()

