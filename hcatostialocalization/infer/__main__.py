import warnings
warnings.filterwarnings("ignore", message=".*setupterm.*")

import os, sys, shlex, time
from typing import List
from dataclasses import dataclass

import argparse
import inquirer
from inquirer import themes

import json
from termcolor import colored
import pandas

import numpy
import torch

import nibabel
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Orientation,
    Spacingd,
    EnsureTyped,
    Invertd,
    SaveImage
)
from monai.networks.utils import one_hot
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.spatial import KDTree
from scipy.ndimage import label as scipy_ndimage_label_fn
from skimage.segmentation import clear_border

import matplotlib.pyplot as plt

from hcatostialocalization.model.hub.hub import models_hub_pretrained
from hcatostialocalization.infer.thresholds import (
    BUCKETS, THRESHOLDS_MM, classify_pair, bucket_summary,
)
from hcatostialocalization.infer.utils import (
    predict_minibatches_inplace,
    get_location_from_probability_one_channel
)
from hcatostialocalization.io.io import (
    save_blueprint_json,
    save_slicer_json,
    compute_image_metadata,
    extract_simplified_ras
)





def _collect_input_paths(items: List[str], are_labels:bool = False) -> List[str]:
    """Given a list of input arguments, expand any text-file lists and
    return a normalized, deduplicated list of absolute file paths.

    - If an item is a path to a .txt/.list/.lst file, read it (one path per line).
    - Skip empty lines and lines starting with '#'.
    - Expand ~ and make absolute paths.
    """
    collected = []
    for it in items:
        if it is None:
            continue
        it = os.path.expanduser(str(it))
        # If this item is a readable text file, parse its lines
        if os.path.isfile(it) and it.lower().endswith(('.txt', '.list', '.lst')):
            try:
                with open(it, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if not are_labels:
                            # image files must be valid paths
                            if not line or line.startswith('#'):
                                continue
                            p = os.path.abspath(os.path.expanduser(line))
                            collected.append(p)
                        else:
                            # labels can be also empty lines,
                            # meaning that no label is present for the corresponding image
                            if line.startswith('#'):
                                continue
                            if len(line)==0:
                                collected.append('')
                            else:
                                p = os.path.abspath(os.path.expanduser(line))
                                collected.append(p)
            except Exception:
                # if file can't be read, treat it as a single path below
                raise RuntimeError(f"Could not read input list file: {it}")
        elif len(it) == 0 and are_labels:
            collected.append('')
        else:
            collected.append(os.path.abspath(it))

    # deduplicate while preserving order
    # fix: deduplication does not work as it cause inconsistencies between images and labels
    # ordering in case of missing labels. Also, missing labels are all counted as just one empty label file.
    #
    #seen = set()
    #out = []
    #for p in collected:
    #    if p not in seen:
    #        seen.add(p)
    #        out.append(p)
    out = collected
    return out

def _list_none_to_nan(_list: list|tuple) -> list:
    if not isinstance(_list, (list, tuple)):
        raise TypeError(f'arg must be of type list or tuple, got {type(_list)}')
    def convert(x):
        if x is None:
            return numpy.nan
        elif isinstance(x, (list,tuple)):
            return _list_none_to_nan(x)
        else:
            return x
    out = map(convert, _list)
    out = list(out) if isinstance(out, list) else tuple(list(out))
    return out

def _process_none_lists(_list: list) -> numpy.ndarray:
    """
    Per-class metrics list (one row per image) -> float array (N, C), None -> nan.
    Rows that are None or shorter than the longest row are padded with nan, so that
    a ragged list (e.g. an image that failed to load) never breaks the conversion.
    """
    rows = [list(r) if isinstance(r, (list, tuple, numpy.ndarray)) else [r] for r in _list]
    if len(rows) == 0:
        return numpy.array([], dtype=float)
    width = max(len(r) for r in rows)
    rows = [
        [numpy.nan if v is None else float(v) for v in r] + [numpy.nan] * (width - len(r))
        for r in rows
    ]
    return numpy.array(rows, dtype=float)

def _set_all_metrics_to_none(metrics_list: list[list], pattern: list = [None]) -> None:
    i = 0
    for m in metrics_list:
        value = pattern[i]
        m.append(value)
        i+=1
        i = i % len(pattern)

def _round_dict_if_float(dict_: dict, significant_digits: int = 6) -> dict:
    if isinstance(dict_, dict):
        return {
            k: (
                round(v, significant_digits)
                if isinstance(v, float)
                else list(map(
                    lambda _x: (round(_x, significant_digits) if isinstance(_x, float) else _x),
                    v
                )) if isinstance(v, list)
                else v
            )
            for k, v in dict_.items()
        }
    else:
        raise TypeError("Input must be a dict.")

def onehot_encode_indexed_segmentation_mask(mask: torch.Tensor|numpy.ndarray, n_classes: int|None = None):
    # equivalent to monai's onehot
    """
    # Args
    - mask: a tensor of shape (B, 1, ...spatial...) where the channel dimension (axis 1)
            encodes classes as index numbers (such as 0, 1, 2, ...).
    - n_classes: number of classes to one-hot encode. If none provided, it will be auto-detected from the
                 maximum value in the mask, assuming numeration starts from 0 included (dangerous).
    # Returns
    - mask_onehot: a tensor of shape (B, n_classes, ...spatial...) dtype uint8
    """
    if n_classes is None:
        n_classes = mask.max().item() if isinstance(mask, torch.Tensor) else mask.max()
        n_classes += 1
    out_shape = list(mask.shape)
    out_shape[1] = n_classes
    out_shape = tuple(out_shape)
    out = torch.zeros(out_shape, dtype=torch.uint8, device=mask.device) if isinstance(mask, torch.Tensor) else numpy.zeros(shape=out_shape, dtype=numpy.uint8)
    for c in range(n_classes):
        out[:,c:c+1,...] = (mask == c).to(torch.uint8) if isinstance(mask, torch.Tensor) else (mask == c).astype(numpy.uint8)
    return out

def onehot_decode_sparse_segmentation_mask(mask: torch.Tensor|numpy.ndarray):
    """
    # Args
    - mask: a tensor of shape (B, n_classes, ...spatial...) where the channel dimension (axis 1)
            encodes classes as one-hot encoded values. Pixels must not have more than one hot channel.
            Pixels with no hot channels (all zeros) will automatically be placed as class 0 (usually background).
    # Returns
    - mask_indexed: a tensor of shape (B, 1, ...spatial...) of dtype uint8 or uint16
    """
    # check input is correctly onehot encoded
    _res = numpy.sum(mask, axis=1)
    if (_res > 1).any():
        raise ValueError("mask should be one-hot encoded, but here some pixels/voxels correspond to more than one class (multi-class multi-label is not supported).")

    n_classes = mask.shape[1]
    if n_classes < 256:
        _dtype = torch.uint8 if isinstance(mask, torch.Tensor) else numpy.uint8
    else:
        _dtype = torch.uint16 if isinstance(mask, torch.Tensor) else numpy.uint16
    out_shape = list(mask.shape)
    out_shape[1] = 1
    out_shape = tuple(out_shape)
    out = torch.zeros(out_shape, dtype=_dtype, device=mask.device) if isinstance(mask, torch.Tensor) else numpy.zeros(shape=out_shape, dtype=_dtype)
    for c in range(1, n_classes):
        o_ = c*(mask[:,c:c+1,...] == 1)
        out += o_.to(_dtype) if isinstance(mask, torch.Tensor) else o_.astype(_dtype)
    return out

def _get_metrics_statistics(arr):
        arr = numpy.array(arr)
        # If arr is 1D, reshape to (N,); if 2D, keep as (N, C)
        if arr.ndim == 1:
            axis = 0
        elif arr.ndim == 2:
            axis = 0
        else:
            raise ValueError("Input array must be 1D or 2D.")
        def safe_stat(fn, arr, axis):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
                    warnings.filterwarnings("ignore", message="Mean of empty slice")
                    result = fn(arr, axis=axis)
                # If result is array, convert to float or list
                if isinstance(result, numpy.ndarray):
                    result = result.tolist()
                else:
                    result = float(result)
                # If all nan, set to None
                if (isinstance(result, float) and numpy.isnan(result)) or \
                   (isinstance(result, list) and all(numpy.isnan(x) for x in result)):
                    return None
                return result
            except Exception:
                return None
        metrics = {
            "min": safe_stat(numpy.nanmin, arr, axis),
            "p05": safe_stat(lambda a, axis: numpy.nanpercentile(a, 5, axis=axis), arr, axis),
            "p25": safe_stat(lambda a, axis: numpy.nanpercentile(a, 25, axis=axis), arr, axis),
            "median": safe_stat(numpy.nanmedian, arr, axis),
            "p75": safe_stat(lambda a, axis: numpy.nanpercentile(a, 75, axis=axis), arr, axis),
            "p95": safe_stat(lambda a, axis: numpy.nanpercentile(a, 95, axis=axis), arr, axis),
            "max": safe_stat(numpy.nanmax, arr, axis),
            "mean": safe_stat(numpy.nanmean, arr, axis),
            "std": safe_stat(numpy.nanstd, arr, axis),
        }
        return metrics



def sanitize_spatial_params(patch_size, overlap, spatial_dims):
    """
    Sanitizes and standardizes sliding window parameters into multi-dimensional tuples.

    Args:
        patch_size (int|float|list|tuple): The target window crop size.
        overlap (int|float|list|tuple): The window step overlap.
        spatial_dims (int): Number of spatial dimensions (e.g., tensor.ndim - 2).

    Returns:
        tuple: (sanitized_patch_size, sanitized_overlap) as exact dimensional tuples.
    """
    def _ensure_tuple(value, name, dtype):
        if isinstance(value, (int, float)):
            return tuple([dtype(value)] * spatial_dims)
        elif isinstance(value, (tuple, list)):
            if len(value) != spatial_dims:
                raise RuntimeError(
                    f"{name} ({len(value)}) must be of same length as "
                    f"number of spatial dimensions: ({spatial_dims})"
                )
            return tuple([dtype(v) for v in value])
        else:
            raise TypeError(f"{name} must be a scalar number or a sequence, got {type(value)}")

    # 1. Standardize patch size to an integer voxel tuple
    clean_patch = _ensure_tuple(patch_size, "patch_size", int)

    # 2. Standardize overlap to a float tuple
    clean_overlap = _ensure_tuple(overlap, "overlap", int)

    return clean_patch, clean_overlap

# Some volumes (e.g. CAT08) are stored with a +1024 offset (values in [0, 4096]) instead of HU.
# A volume in HU always contains air/lung well below -200: if no value is below HU_SHIFT_THRESHOLD,
# the volume is shifted and must be brought back to HU. Same constants as data/dataset/CAT08/Multi_Dataset_CAT08.py
HU_OFFSET = 1024.0
HU_SHIFT_THRESHOLD = -200.0

def load_native_ras_pair(im_path, lab_path):
    """
    STAGE 1: Loads an image/label pair, enforces RAS orientation, and casts to PyTorch tensors.
    Returns tensors in their exact native physical spacing.

    Returns:
        image (MetaTensor): 5D shape (1, C, R, A, S), Float32
        label (dict|None): ostia RAS coordinates in mm, read from any format supported by
            io.extract_simplified_ras (standard .ostium.json, old ostia JSON, 3D Slicer .mrk.json)
            {
                R: {
                    coordinate_ras: list[3,float],
                },
                L: ...
            }
        base_transforms: MONAI transform pipeline
    """
    base_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        EnsureTyped(keys=["image"], dtype=(torch.float32))
    ])

    data_dict = {"image": im_path}
    try:
        # load and process image
        processed_data = base_transforms(data_dict)

        # Add the 5D batch dimension
        image = processed_data["image"][None]

        # HU safeguard: no value under HU_SHIFT_THRESHOLD -> volume stored with +1024 offset, go back to HU
        if float(image.min()) >= HU_SHIFT_THRESHOLD:
            image = image - torch.tensor(HU_OFFSET, dtype=image.dtype, device=image.device)
            # print(f"[HU] {os.path.basename(str(im_path))}: no value below {HU_SHIFT_THRESHOLD:.0f}, shifted by -{HU_OFFSET:.0f} (new min={float(image.min()):.0f})")

        # Load label dict
        label = None
        if lab_path and len(lab_path) > 0 and os.path.exists(lab_path):
            simple_pos = extract_simplified_ras(lab_path, image_path=im_path)
            if simple_pos is None or any(len(simple_pos[side]['ras']) != 3 for side in ('R', 'L')):
                raise ValueError(f"could not read both R and L ostium RAS coordinates from label file {lab_path}")
            label = {side: {'coordinate_ras': [float(c) for c in simple_pos[side]['ras']]} for side in ('R', 'L')}

        return image, label, base_transforms

    except Exception as e:
        raise RuntimeError(f"MONAI Stage 1 (Load/RAS) failed for image {im_path}. Error:\n{e}")

def resample_for_model(image, target_spacing=None):
    """
    STAGE 2: Resamples 5D native tensors to the target spacing required by the model.

    Args:
        image (MetaTensor): 5D native image
        target_spacing (tuple|float|None): The target voxel size.

    Returns:
        resampled_image (MetaTensor): 5D shape (1, C, X, Y, Z)
        resample_transforms (Compose): The transform pipeline object (crucial for Invertd later!)
    """
    # If no target spacing is provided, skip resampling entirely
    if target_spacing is None:
        return image, Compose([])

    # Sanitize inputs
    if isinstance(target_spacing, (int, float)):
        target_spacing = tuple([float(target_spacing)] * 3)
    if isinstance(target_spacing, (tuple, list)):
        target_spacing = tuple([float(_) for _ in target_spacing])

    resample_transforms = Compose([
        Spacingd(
            keys=["image"],
            pixdim=target_spacing,
            mode=("bilinear"),
            allow_missing_keys=True
        )
    ])

    # Strip the 5D batch dimension [0] to pack them back into a 4D MONAI dictionary
    data_dict = {"image": image[0]}
    processed_data = resample_transforms(data_dict)
    # Re-add the 5D batch dimension
    resampled_image = processed_data["image"][None]
    return resampled_image, resample_transforms


def invert_resampling(input_image, tensors: dict, resample_transforms) -> dict:
    """
    Inverts the resampling strictly for the model outputs.

    Args:
        input_image: The 5D resampled MetaTensor fed to the model.
                     (Used as the master history blueprint).
        tensors: dict[str, MetaTensor] of 5D model outputs (logits, probabilities, ...) at the resampled spacing.
                 The key "image" is reserved.
        resample_transforms: The resample Compose pipeline.

    Returns:
        dict with the same keys as `tensors`, each 5D tensor returned to native Stage 1 spacing.
    """
    # 0. SHORT CIRCUIT FOR EMPTY TRANSFORMS
    # if the Compose pipeline is completely empty, the image was never resampled.
    # Therefore, the model outputs are already in native space.
    if resample_transforms is None or (hasattr(resample_transforms, "transforms") and len(resample_transforms.transforms) == 0):
        return dict(tensors)
    if "image" in tensors:
        raise ValueError("invert_resampling: key 'image' is reserved for the input image.")
    keys = list(tensors.keys())

    # 1. + 2. Strip the 5D batch dimension -> (C, R, A, S) and setup dynamic inversion dictionary
    # CRITICAL: "image" must be present so Invertd can read its specific history
    inversion_dict = {"image": input_image[0]}
    inversion_dict.update({k: v[0] for k, v in tensors.items()})

    # 3. REWIND ONLY THE RESAMPLING
    invert_transform = Invertd(
        keys=keys,
        transform=resample_transforms,
        orig_keys="image", # CRITICAL: Tells MONAI to apply "image"'s history to the outputs!
        nearest_interp=[False] * len(keys),
        to_tensor=True
    )

    native_space_dict = invert_transform(inversion_dict)

    # 4. Extract and restore the 5D batch dimension -> (1, C, R, A, S)
    return {k: native_space_dict[k][None] for k in keys}


def _class_display_name(model, c: int) -> str:
    """Underscore-prefixed class name suffix (e.g. '_right_ostium'), or '' if unavailable."""
    if hasattr(model, 'infer_classes') and \
            isinstance(model.infer_classes, (list, dict)) and \
            c < len(model.infer_classes) and \
            isinstance(model.infer_classes[c], str) and \
            len(model.infer_classes[c]) > 0:
        return '_' + "_".join(model.infer_classes[c].lower().split())
    return ''


def _make_nifti_saver(out_dir: str) -> SaveImage:
    return SaveImage(
        output_dir=out_dir,
        output_postfix=f"",
        output_ext=".nii.gz",
        resample=False,
        print_log=False,
        data_root_dir="",
        separate_folder=False
    )



# ####################################### #
# MODEL INPUT SPACE <-> NATIVE RAS SPACE  #
# ####################################### #
#
# Every model sees the image in its own "model space": resampled to its spacing, reoriented to its orientation
# ("ras" or "sitk") and preprocessed. Everything that comes out of a model is brought back to the native RAS
# space of the loaded image (image_metatensor_ras) with the same two functions, to_model_space / from_model_space,
# so the full-volume prediction and the cascade patches share exactly the same geometry code.

# SimpleITK exposes arrays as (Z, Y, X): transpose the three spatial dims of a 5D tensor (self-inverse)
SITK_PERMUTE = (0, 1, 4, 3, 2)

# output channels of the ostia models (index 0 = background), same as the rows of the predicted coordinates
RIGHT_OSTIUM_CHANNEL = 1
LEFT_OSTIUM_CHANNEL = 2


@dataclass
class InferenceSettings:
    target_spacing_mm_ras: tuple[float]|float|None
    patch_size: tuple[int]|int|None
    overlap: tuple[int]|int|None
    orientation: str  # "ras" | "sitk"


def resolve_inference_settings(model, target_spacing_mm_ras, patch_size, overlap, orientation, cli_wins: bool) -> InferenceSettings:
    """
    Merge the CLI arguments with the model's infer_* attributes.
    - cli_wins=True (first model): a CLI value that is not None wins, else the model attribute.
    - cli_wins=False (cascade refinement models): the model attribute wins whenever the model declares it
      (even as None, e.g. infer_target_spacing_mm_ras = None means "native spacing"), else the CLI value.
    Orientation falls back to "ras" if neither is set.
    """
    def _pick(cli_value, attr):
        if cli_wins:
            return cli_value if cli_value is not None else getattr(model, attr, None)
        return getattr(model, attr) if hasattr(model, attr) else cli_value
    orientation_ = _pick(orientation, 'infer_orientation')
    orientation_ = "ras" if orientation_ is None else str(orientation_).lower()
    if orientation_ not in ("ras", "sitk"):
        raise ValueError(f"orientation (passed as argument or defined inside of model as model.infer_orientation) must be 'ras' or 'sitk'. Got: {orientation_}")
    return InferenceSettings(
        target_spacing_mm_ras=_pick(target_spacing_mm_ras, 'infer_target_spacing_mm_ras'),
        patch_size=_pick(patch_size, 'infer_patch_size'),
        overlap=_pick(overlap, 'infer_overlap'),
        orientation=orientation_,
    )


@dataclass
class ModelSpace:
    """
    What is needed to bring a tensor computed in a model's input space back to native RAS space.
    - resampled_image: 5D (1, 1, R, A, S) RAS MetaTensor at the model spacing. Geometry (meta) donor, and the
      transform history read by Invertd.
    - resample_transform: the Compose used for resampling (empty Compose = no resampling).
    - orientation: "ras" | "sitk".
    - sitk_donor_meta: 4D (C, X, Y, Z) MetaTensor in the original disk layout, meta donor of the backward pass ("sitk" only).
    """
    resampled_image: MetaTensor
    resample_transform: Compose
    orientation: str
    sitk_donor_meta: MetaTensor|None = None


def _ras_to_model_orientation(x_ras: MetaTensor, orientation: str) -> tuple[MetaTensor, MetaTensor|None]:
    """5D RAS MetaTensor (1, C, R, A, S) -> (5D tensor in the model axis order, sitk meta donor or None)."""
    match orientation:
        case "sitk":
            # Strip the batch dimension (1, C, R, A, S) -> (C, R, A, S)
            x_4d = x_ras[0]
            # Read the raw orientation the image had on disk before MONAI forced RAS
            original_affine = x_4d.meta["original_affine"]
            orig_axcodes_str = "".join(nibabel.orientations.aff2axcodes(original_affine))  # e.g., "LPS", "RAI", etc.
            # Revert the spatial axes/flips to match the original disk layout (X, Y, Z)
            x_xyz_4d = Orientation(axcodes=orig_axcodes_str, labels=None)(x_4d)
            # Restore the batch dimension -> (1, C, X, Y, Z), then SimpleITK (Z, Y, X) axis order
            x_model = x_xyz_4d.unsqueeze(0).permute(SITK_PERMUTE).contiguous()
            return x_model, x_xyz_4d
        case "ras":
            # (1, C, R, A, S) already ok
            return x_ras, None
        case _:
            raise ValueError(f"orientation must be 'ras' or 'sitk'. Got: {orientation}")


def _model_orientation_to_ras(x_model: torch.Tensor, space: ModelSpace) -> MetaTensor:
    """Inverse of _ras_to_model_orientation: 5D tensor in model axis order -> 5D RAS MetaTensor at the model spacing."""
    match space.orientation:
        case "sitk":
            # Undo the SITK (Z, Y, X) --> (X, Y, Z) permutation, strip batch dimension -> (C, X, Y, Z)
            x_xyz_4d = MetaTensor(x_model.permute(SITK_PERMUTE).contiguous()[0])
            # Re-wrap as MetaTensor with the metadata of the donor
            x_xyz_4d.copy_meta_from(space.sitk_donor_meta)
            # Apply MONAI's Orientation transform to force strict RAS space, restore batch dim -> (1, C, R, A, S)
            return Orientation(axcodes="RAS", labels=None)(x_xyz_4d).unsqueeze(0)
        case "ras":
            # (1, C, R, A, S) already ok, only the geometry (affine) of the resampled image is needed
            x_ras = MetaTensor(x_model)
            x_ras.copy_meta_from(space.resampled_image)
            return x_ras
        case _:
            raise ValueError(f"orientation must be 'ras' or 'sitk'. Got: {space.orientation}")


def to_model_space(image_metatensor_ras: MetaTensor, target_spacing, orientation: str) -> tuple[MetaTensor, ModelSpace]:
    """
    Native 5D RAS image -> (5D image in model space, not yet preprocessed nor moved to device; ModelSpace).
    """
    resampled_image, resample_transform = resample_for_model(image_metatensor_ras, target_spacing=target_spacing)
    x_model, sitk_donor_meta = _ras_to_model_orientation(resampled_image, orientation)
    return x_model, ModelSpace(resampled_image, resample_transform, orientation, sitk_donor_meta)


def from_model_space(tensors: dict, space: ModelSpace) -> dict:
    """dict of 5D tensors in model space (same spatial shape as the model-space image) -> dict of native RAS MetaTensors."""
    tensors_ras = {k: _model_orientation_to_ras(v, space) for k, v in tensors.items()}
    return invert_resampling(space.resampled_image, tensors_ras, space.resample_transform)


def _cached_model_space(cache: dict, image_metatensor_ras: MetaTensor, settings: InferenceSettings) -> tuple[MetaTensor, ModelSpace]:
    # the model-space image depends only on (spacing, orientation): models sharing them share one resampling
    key = (repr(settings.target_spacing_mm_ras), settings.orientation)
    if key not in cache:
        cache[key] = to_model_space(image_metatensor_ras, settings.target_spacing_mm_ras, settings.orientation)
    return cache[key]


def _prepare_model_input(x_model: torch.Tensor, model, device: torch.device) -> torch.Tensor:
    """Move to device, drop the MONAI metadata, apply the model preprocessing (if any)."""
    x = x_model.to(device)
    x = x.as_tensor() if isinstance(x, MetaTensor) else x
    if getattr(model, 'infer_preprocessing_fn', None) is not None:
        try:
            x = model.infer_preprocessing_fn(x)
        except:
            pass
    return x


def _run_sliding_window(model, x: torch.Tensor, patch_size: tuple[int], overlap: tuple[int], device: torch.device, use_monai: bool) -> torch.Tensor:
    """Sliding window inference -> raw logits (1, C, *spatial), float32, on cpu."""
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        if use_monai:
            logits = sliding_window_inference(
                inputs=x,
                roi_size=patch_size,
                sw_batch_size=2,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )
        else:
            logits = predict_minibatches_inplace(
                model=model,
                images=x,
                patch_size=patch_size,
                overlap=overlap,
                minibatch_size=2,
                ensure_sdims_shape_multiple_of=16,
                activation = None,
                augment = False,
                output_logits_dtype = torch.float32,
                model_features_to_save=[],
                use_monai=use_monai,
                verbose=False,
                storage_device_output=torch.device("cpu"),
                storage_device_features=torch.device("cpu") # Store large outputs on CPU to save VRAM
            )['prediction']
    return logits.to(device='cpu', dtype=torch.float32)


def predict_full_volume(model, x_model: MetaTensor, space: ModelSpace, settings: InferenceSettings, device: torch.device, use_monai: bool) -> MetaTensor:
    """Sliding window over the whole image -> logits in native RAS space, MetaTensor (1, C, R, A, S), float32, cpu."""
    x = _prepare_model_input(x_model, model, device)
    # sanitize overlap and patch size
    active_overlap = 0 if settings.overlap is None else settings.overlap
    active_patch_size = x.shape[2:] if settings.patch_size is None else settings.patch_size
    patch_size_clean, overlap_clean = sanitize_spatial_params(active_patch_size, active_overlap, 3)
    logits = _run_sliding_window(model, x, patch_size_clean, overlap_clean, device, use_monai)
    del x
    torch.cuda.empty_cache()
    return from_model_space({"logits": logits}, space)["logits"]


def refine_ostium_logits(
    model,
    x_model: MetaTensor,
    space: ModelSpace,
    settings: InferenceSettings,
    device: torch.device,
    use_monai: bool,
    logits_metatensor_ras: MetaTensor,
    channel: int,
    center_ras_mm: torch.Tensor,
    logit_offset: str = 'match-peak',
) -> bool:
    """
    Cascade step: predict ONE patch (the model patch size) centred on `center_ras_mm`, and superimpose the logits
    of `channel` onto `logits_metatensor_ras` (native RAS, modified in place) inside the patch footprint.
    Other channels are left untouched.

    logit_offset:
    - 'none': the patch logits overwrite the previous ones as they are.
    - 'match-peak': the patch logits are shifted by a constant so that their maximum equals the maximum of the
      previous logits inside the same footprint. The spatial distribution inside the patch is unchanged, but the
      patch cannot lose (or win) against the rest of the volume only because the two models have different logit scales.

    Returns False if the patch footprint is empty in native space (nothing superimposed), else True.
    """
    # patch size of the model
    if settings.patch_size is None:
        raise ValueError("cascade refinement models must declare infer_patch_size (or --patch-size must be given)")
    patch_size_clean, _ = sanitize_spatial_params(settings.patch_size, 0, 3)
    model_shape = tuple(x_model.shape[2:])

    # 1. centre, RAS mm -> voxel index of the resampled RAS image
    affine = numpy.asarray(space.resampled_image.affine.cpu().numpy(), dtype=numpy.float64).reshape(4, 4)
    center_homo = numpy.append(numpy.asarray(center_ras_mm, dtype=numpy.float64).reshape(3), 1.0)
    center_ijk = numpy.rint((numpy.linalg.inv(affine) @ center_homo)[:3]).astype(int)
    ras_shape = tuple(space.resampled_image.shape[2:])
    center_ijk = numpy.clip(center_ijk, 0, numpy.array(ras_shape) - 1)

    # 2. voxel index in model space: push a one-voxel marker through the same reorientation as the image
    marker = torch.zeros((1, 1) + ras_shape, dtype=torch.uint8)
    marker[0, 0, center_ijk[0], center_ijk[1], center_ijk[2]] = 1
    marker = MetaTensor(marker)
    marker.copy_meta_from(space.resampled_image)
    marker_model, _ = _ras_to_model_orientation(marker, space.orientation)
    marker_model = marker_model.as_tensor() if isinstance(marker_model, MetaTensor) else marker_model
    center_model = numpy.unravel_index(int(torch.flatten(marker_model).nonzero()[0, 0]), model_shape)
    del marker, marker_model

    # 3. patch box in model space, centred on the marker, shifted inside the image at the borders
    box = []
    for c, p, s in zip(center_model, patch_size_clean, model_shape):
        start = min(max(int(c) - p // 2, 0), max(s - p, 0))
        box.append(slice(start, min(start + p, s)))
    box = (slice(None), slice(None)) + tuple(box)

    # 4. predict the patch (preprocessing on the whole image, exactly as in the full-volume pass)
    x = _prepare_model_input(x_model, model, device)
    patch = x[box].clone()
    del x
    patch_logits = _run_sliding_window(model, patch, patch_size_clean, (0, 0, 0), device, use_monai)
    del patch
    torch.cuda.empty_cache()

    # 5. model-space canvas: channel 0 = patch logits of `channel`, channel 1 = patch footprint
    canvas = torch.zeros((1, 2) + model_shape, dtype=torch.float32)
    canvas[0, 0][box[2:]] = patch_logits[0, channel]
    canvas[0, 1][box[2:]] = 1.0
    del patch_logits
    canvas_native = from_model_space({"canvas": canvas}, space)["canvas"]
    canvas_native = canvas_native.as_tensor() if isinstance(canvas_native, MetaTensor) else canvas_native
    del canvas

    # 6. superimpose in native space. With resampling the footprint is interpolated: voxels at least half covered
    #    by the patch are replaced, and dividing by the footprint undoes the blending with the zeros around the patch.
    footprint = canvas_native[0, 1]
    region = footprint >= 0.5
    if not bool(region.any()):
        return False
    new_logits = canvas_native[0, 0][region] / footprint[region]
    target = logits_metatensor_ras.as_tensor()[0, channel]  # view: writes go into logits_metatensor_ras
    if logit_offset == 'match-peak':
        new_logits = new_logits + (target[region].max() - new_logits.max())
    elif logit_offset != 'none':
        raise ValueError(f"logit_offset must be 'none' or 'match-peak'. Got: {logit_offset}")
    target[region] = new_logits.to(target.dtype)
    return True


def locate_ostia(logits_metatensor_ras: MetaTensor, window_size: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ostia positions from native RAS logits (1, C, R, A, S).
    Returns (ijk, ras): float32 tensors of shape (2, 3), rows = [right ostium, left ostium], sub-voxel precision.
    """
    ijk = torch.vstack([
        get_location_from_probability_one_channel(logits_metatensor_ras[0, RIGHT_OSTIUM_CHANNEL], window_size=window_size),
        get_location_from_probability_one_channel(logits_metatensor_ras[0, LEFT_OSTIUM_CHANNEL], window_size=window_size),
    ])
    # Convert from IJK to RAS
    _affine = logits_metatensor_ras.affine.cpu().numpy()
    _ijk_homo = numpy.hstack((ijk.cpu().numpy(), numpy.ones((2, 1))))
    _ras_homo = (_affine @ _ijk_homo.T).T
    ras = torch.tensor(_ras_homo[:, :3], dtype=torch.float32, device='cpu')
    return ijk, ras


def _check_model(model, nickname: str) -> None:
    if not hasattr(model, 'infer_classes') or not isinstance(model.infer_classes, (list, dict)):
        raise AttributeError(f"{nickname}: model must have attribute 'infer_classes' and it must be a list of strings or a dict[int: str]")
    if not hasattr(model, 'infer_prediction_fn'):
        raise AttributeError(f"{nickname}: model must have a callable attribute infer_prediction_fn (usually torch.nn.Sigmoid or torch.nn.Softmax)")














def main(
    images: List[str]|str,
    labels: List[str]|str,
    model_nickname: str|list[str],
    weights_path: str|list[str],
    output_dir: str,
    model_channel_filters: list[int]|None=None,
    try_gpu: bool = True,
    job_id: str = '',
    artifacts: list[str] = ['none'],
    metrics: list[str] = ['all'],
    use_monai: bool=False,
    target_spacing_mm_ras: tuple[float]|float|None = None,
    patch_size: tuple[int]|int|None = None,
    overlap: tuple[int]|int|None = None,
    orientation: str|None = None, # "ras" or "sitk"; None = use the model's infer_orientation, else "ras"
    cascade_logit_offset: str = 'match-peak', # 'none' | 'match-peak', see refine_ostium_logits
    keep_models_on_gpu: bool = False,
):

    # ##### #
    # SETUP #
    # ##### #

    # start
    start_datetime = time.strftime("%Y.%m.%d - %H:%M:%S")
    print()
    print(f"Starting at {start_datetime}")

    # device
    if (not try_gpu) or (not torch.cuda.is_available()):
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cuda")

    # model and weights
    #
    # The model returned by the hub is an nn.Module (forward -> logits, same spatial shape as the input) with extra
    # attributes, all prefixed "infer_" (see model/hub/hub.py):
    # Required:
    # - infer_classes: list[str] | dict[int, str], class names, index 0 = background, one per output channel
    # - infer_prediction_fn: callable logits -> probabilities (torch.nn.Softmax / torch.nn.Sigmoid instance or custom function)
    # Optional (an attribute that is None is treated as absent):
    # - infer_preprocessing_fn: callable, image tensor in HU -> processed tensor of same shape
    # - infer_classification_threshold: float | list[float], used only with a torch.nn.Sigmoid prediction_fn
    # - infer_target_spacing_mm_ras, infer_patch_size, infer_overlap, infer_orientation ("ras" | "sitk"):
    #   defaults for the same-named arguments. An argument given by the user (CLI) ALWAYS wins; the model attribute
    #   is used only when the argument was not given (orientation falls back to "ras" if neither is set).
    #
    # Cascade (1 to 3 models):
    # - 1 model: the model predicts the full volume (sliding window).
    # - 2 models: the first model predicts the full volume; the second model then predicts one patch (its own patch
    #   size) centred on the left ostium predicted by the first model, then one centred on the predicted right ostium.
    #   Each patch overwrites, in logits, only the channel of the ostium it is centred on.
    # - 3 models: as with 2, but the second model refines the left ostium and the third model the right ostium.
    # Logits are merged in native RAS space and turned into probabilities only after the merge.
    # CLI spacing / patch size / overlap / orientation apply to the FIRST model. Refinement models use their own
    # infer_* attributes, and fall back to the CLI values only for attributes they do not declare.
    model_nicknames = [model_nickname] if isinstance(model_nickname, str) else list(model_nickname)
    if not (1 <= len(model_nicknames) <= 3):
        raise ValueError(f"Between 1 and 3 models can be given, got {len(model_nicknames)}: {model_nicknames}")
    weights_paths = [weights_path] if isinstance(weights_path, str) else list(weights_path)
    if len(weights_paths) == 1 and len(model_nicknames) > 1:
        if weights_paths[0] != 'best':
            raise ValueError("With more than one model, give either a single 'best' or one weights path per model.")
        weights_paths = weights_paths * len(model_nicknames)
    if len(weights_paths) != len(model_nicknames):
        raise ValueError(f"Got {len(weights_paths)} weights paths for {len(model_nicknames)} models: give one per model.")
    for w in weights_paths:
        if not (w == 'best' or (isinstance(w, str) and len(w)>0 and os.path.isfile(w))):
            raise ValueError(f"Provided weights path is invalid: {w}\nChoose a valid path, or use 'best'")
    is_cascade = len(model_nicknames) > 1
    models_tag = "+".join(model_nicknames) # used in output file names

    # Dynamic device placement: in a cascade on GPU, models wait on CPU and only the model in use sits on the GPU,
    # so the full-volume pass gets all the VRAM. Moving a model costs milliseconds, inference costs seconds.
    offload_models = is_cascade and torch_device.type == "cuda" and not keep_models_on_gpu
    loaded_models = {} # (nickname, weights) -> model: a model given twice is loaded once
    models = []
    for nick, w in zip(model_nicknames, weights_paths):
        if (nick, w) not in loaded_models:
            m = models_hub_pretrained[nick](load_weights=w)
            _check_model(m, nick)
            m.eval()
            m = m.to(torch.device("cpu") if offload_models else torch_device)
            loaded_models[(nick, w)] = m
            print(f"Loaded {nick} on device {'cpu (moved to ' + str(torch_device) + ' when in use)' if offload_models else torch_device} with weights from {w}")
        models.append(loaded_models[(nick, w)])
    model = models[0]

    _active_model = [None]
    def activate(m):
        if offload_models and _active_model[0] is not m:
            if _active_model[0] is not None:
                _active_model[0].to(torch.device("cpu"))
                torch.cuda.empty_cache()
            m.to(torch_device)
            _active_model[0] = m
        return m

    # inference parameters: for the first model the user-given argument always wins, else the model attribute
    settings = [resolve_inference_settings(model, target_spacing_mm_ras, patch_size, overlap, orientation, cli_wins=True)]
    for m in models[1:]:
        settings.append(resolve_inference_settings(m, target_spacing_mm_ras, patch_size, overlap, orientation, cli_wins=False))
    target_spacing_mm_ras = settings[0].target_spacing_mm_ras
    patch_size = settings[0].patch_size
    overlap = settings[0].overlap
    orientation = settings[0].orientation

    # cascade stages: (model index, name, channel superimposed, row of the first-model coordinates used as centre)
    refinement_stages = []
    if len(models) == 2:
        refinement_stages = [(1, "left ostium", LEFT_OSTIUM_CHANNEL, 1), (1, "right ostium", RIGHT_OSTIUM_CHANNEL, 0)]
    elif len(models) == 3:
        refinement_stages = [(1, "left ostium", LEFT_OSTIUM_CHANNEL, 1), (2, "right ostium", RIGHT_OSTIUM_CHANNEL, 0)]
    for k, m in enumerate(models[1:], start=1):
        if list(m.infer_classes) != list(model.infer_classes):
            raise ValueError(f"Cascade models must share the same classes. {model_nicknames[0]}: {model.infer_classes}, {model_nicknames[k]}: {m.infer_classes}")
        if settings[k].patch_size is None:
            raise ValueError(f"{model_nicknames[k]}: cascade refinement models must declare infer_patch_size (or --patch-size must be given)")
        _fn_id = lambda f: (type(f), getattr(f, '__name__', None))
        if _fn_id(m.infer_prediction_fn) != _fn_id(model.infer_prediction_fn):
            print(colored(f"Warning: {model_nicknames[k]} has a different infer_prediction_fn than {model_nicknames[0]}. Merged logits are turned into probabilities with the one of {model_nicknames[0]}.", (200, 120, 20)))
    if cascade_logit_offset not in ('none', 'match-peak'):
        raise ValueError(f"cascade_logit_offset must be 'none' or 'match-peak'. Got: {cascade_logit_offset}")
    cascade_description = [
        {
            "stage": "full volume" if k == 0 else f"{refinement_stages[k-1][1]} patch" if len(models) == 3 else "left and right ostium patches",
            "model nickname": model_nicknames[k],
            "weights path": weights_paths[k],
            "inference spacing (mm)": settings[k].target_spacing_mm_ras,
            "inference patch size (px)": settings[k].patch_size,
            "inference overlap (px)": settings[k].overlap if k == 0 else None,
            "orientation": settings[k].orientation,
        }
        for k in range(len(models))
    ]
    if is_cascade:
        print("Cascade:")
        for d in cascade_description:
            print(f"  - {d['stage']}: {d['model nickname']} (spacing {d['inference spacing (mm)']}, patch {d['inference patch size (px)']}, orientation {d['orientation']})")
        print(f"  logit offset of refinement patches: {cascade_logit_offset}")

    # input: expand input arguments into a normalized list of file paths
    image_paths = _collect_input_paths(images if isinstance(images, (list, tuple)) else [images])
    if not image_paths:
        raise ValueError("No input paths resolved from -i/--input argument(s)")

    # labels: expand input arguments into a normalized list of file paths
    #         If no label for an image, save `` to preserve the input-label alignment
    label_paths = _collect_input_paths(labels if isinstance(labels, (list, tuple)) else [labels], are_labels=True)
    if all(len(l) == 0 for l in label_paths):
        # -l/--label not provided (default ''): unlabelled images, one empty label per image
        label_paths = [''] * len(image_paths)
    if len(label_paths) != len(image_paths):
        raise ValueError(f"Number of label paths ({len(label_paths)}) does not match number of input image paths ({len(image_paths)}).")

    # output paths
    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)

    # log file
    # what if i'm almost done with the computations and the results
    # and the program gets interrupted at the last moment?
    # Save results on the go in a simpler way in a log file:
    logfile = os.path.join(output_path, f"_test_log {start_datetime}.txt")

    # job id
    if len(job_id) > 0:
        job_id_file = os.path.join(output_path, f"_job_id__{job_id}")
        with open(job_id_file, 'wt') as f:
            f.write(f'{job_id}')

    # image, label pairs
    im_lab_paths_pairs: list[tuple[str, str]] = [(im, lab) for im, lab in zip(image_paths, label_paths)]

    # CLI
    print(f"Found {len(image_paths)} input image(s) for inference.")
    print(f"Found {len([l for l in label_paths if len(l)>0])} input label(s) for modelling and comparison.")
    print("The program will now execute disregarding errors.")

    # #################### #
    # Main Inference Cycle #
    # #################### #

    errors: list[tuple[str, str, str]] = [] # list elements are (image_path, label_path, error)

    try:
        num_foreground_classes = len(model.infer_classes) - 1
    except:
        num_foreground_classes = int(input("How many classes should the selected model have (background excluded)? [int]"))
        num_foreground_classes = max(1, num_foreground_classes)

    # metrics accumulators #

    distance_list, distance_per_class_list = [], []
    peak_confidence_list, peak_confidence_per_class_list = [], []
    outlier_list, outlier_per_class_list = [], []
    bucket_list, bucket_per_class_list = [], []   # accuracy buckets, see infer/thresholds.py

    list_of_metrics_lists = [
        # used to set all metrics to none in one go - shortens and simplifies code
        distance_list, distance_per_class_list,
        peak_confidence_list, peak_confidence_per_class_list,
        outlier_list, outlier_per_class_list,
        bucket_list, bucket_per_class_list,
    ]

    # artifacts saving #
    markers_out_dir = os.path.join(output_path, 'markers')
    markups_3d_slicer_out_dir = os.path.join(output_path, 'markups-3d-slicer')
    segmentation_mask_out_dir = os.path.join(output_path, 'segmentation-mask')
    probability_tensor_out_dir = os.path.join(output_path, 'probability-tensor')
    logits_tensor_out_dir = os.path.join(output_path, 'logits-tensor')
    probability_nifti_out_dir = os.path.join(output_path, 'probability-nifti')
    logits_nifti_out_dir = os.path.join(output_path, 'logits-nifti')
    nifti_saver = _make_nifti_saver(segmentation_mask_out_dir)
    probability_nifti_saver = _make_nifti_saver(probability_nifti_out_dir)
    logits_nifti_saver = _make_nifti_saver(logits_nifti_out_dir)


    # ############### #
    # MIAN LOOP START #
    # ############### #

    t0 = time.time()
    _progress_bar_width = 40

    for i, (im_path, lab_path) in enumerate(im_lab_paths_pairs):
        print("[" + "#"*int((_progress_bar_width*(i+1))/len(im_lab_paths_pairs)) + "."*(_progress_bar_width-(int((_progress_bar_width*(i+1))/len(im_lab_paths_pairs)))) + f"] {(100*(i+1))/len(im_lab_paths_pairs): 3.0f}% ({i+1}/{len(im_lab_paths_pairs)})", end='\r')

        # ############ #
        # DATA LOADING #
        # ############ #

        # Open Image and Label
        try:
            image_metatensor_ras, label_json_dict, base_transforms = load_native_ras_pair(im_path, lab_path)
            # image_metatensor_ras: (1, 1, R, A, S) - torch.float32 - not resampled
            # (
            #   contain <name>.meta, which contains infos about tensors:
            #   ['sizeof_hdr', 'extents', 'session_error', 'dim_info', 'dim', 'intent_p1', 'intent_p2', 'intent_p3', 'intent_code',
            #    'datatype', 'bitpix', 'slice_start', 'pixdim', 'vox_offset', 'scl_slope', 'scl_inter', 'slice_end', 'slice_code',
            #    'xyzt_units', 'cal_max', 'cal_min', 'slice_duration', 'toffset', 'glmax', 'glmin', 'qform_code', 'sform_code',
            #    'quatern_b', 'quatern_c', 'quatern_d', 'qoffset_x', 'qoffset_y', 'qoffset_z', 'srow_x', 'srow_y', 'srow_z',
            #   affine, original_affine, 'as_closest_canonical', spatial_shape, space, original_channel_dim, 'filename_or_obj']
            # )label (dict|None):
            #
            # label_json_dict:
            # {
            #     R: {
            #         coordinate_ras: list[3,float],
            #     },
            #     L: ...
            # }
            if label_json_dict:
                label_coordinates_ras_tensor = torch.tensor(
                    [
                        list(label_json_dict['R']['coordinate_ras']),
                        list(label_json_dict['L']['coordinate_ras'])
                    ],
                    dtype=torch.float32,
                    device='cpu'
                )

        except Exception as e:
            _e = (im_path, lab_path, f"Could not open image file. {e}")
            errors.append(_e)
            with open(logfile, '+at') as f:
                f.write(f"{_e}\n")
            # per-class rows must have the same length as everywhere else (foreground classes only)
            _set_all_metrics_to_none(list_of_metrics_lists, pattern=[None, [None]*num_foreground_classes])
            continue


        # ############ #
        # SEGMENTATION #
        # ############ #

        # model-space images (resampled + reoriented), shared by the models with the same spacing and orientation
        model_space_cache = {}

        # stage 1: first model, full volume -> logits in native RAS space (1, C, R, A, S)
        x_model, space = _cached_model_space(model_space_cache, image_metatensor_ras, settings[0])
        logits_metatensor_ras = predict_full_volume(activate(models[0]), x_model, space, settings[0], torch_device, use_monai)

        # stage 2 (cascade only): refine each ostium on one patch centred on the first-model prediction,
        # superimposing the patch logits onto the full-volume logits
        if len(refinement_stages) > 0:
            if logits_metatensor_ras.is_inference():
                logits_metatensor_ras = logits_metatensor_ras.clone()
            _, coarse_coordinates_ras_tensor = locate_ostia(logits_metatensor_ras)
            for k, stage_name, channel, row in refinement_stages:
                x_model, space = _cached_model_space(model_space_cache, image_metatensor_ras, settings[k])
                superimposed = refine_ostium_logits(
                    activate(models[k]), x_model, space, settings[k], torch_device, use_monai,
                    logits_metatensor_ras=logits_metatensor_ras,
                    channel=channel,
                    center_ras_mm=coarse_coordinates_ras_tensor[row],
                    logit_offset=cascade_logit_offset,
                )
                if not superimposed:
                    print(colored(f"\nWarning: {stage_name} patch of {model_nicknames[k]} fell outside the image for {im_path}, kept the full-volume prediction.", (200, 120, 20)))
        del model_space_cache, x_model, space

        # ################## #
        # ARTIFACTS CREATION #
        # ################## #

        # RAS PREDICTED LOGITS AND PROBABILITY #

        # logits -> probabilities only now, in native RAS space, after all the logits have been merged
        prediction_metatensor_ras = MetaTensor(model.infer_prediction_fn(logits_metatensor_ras.as_tensor()))
        prediction_metatensor_ras.copy_meta_from(logits_metatensor_ras)


        # ###################################
        # now i have three MetaTensors on cpu:
        #
        # - image_metatensor_ras
        # - prediction_metatensor_ras
        # - logits_metatensor_ras
        # ###################################

        # ONEHOT ENCODED RAS METATENSORS #

        n_classes = prediction_metatensor_ras.shape[1]
        # Create predicted segmentation mask
        if isinstance(model.infer_prediction_fn, torch.nn.Softmax):
            # Argmax along the channel dimension (dim=1) -> shape: (B, 1, X, Y, Z)
            argmax_mask = torch.argmax(prediction_metatensor_ras, dim=1, keepdim=True)
            # Convert to one-hot. MONAI's one_hot is highly optimized for medical
            # (B, 1, ...) shapes, unlike PyTorch's native F.one_hot which expects the channel last.
            # -> shape: (B, n_classes, X, Y, Z)
            prediction_metatensor_ras_onehot = one_hot(argmax_mask, num_classes=n_classes)

        elif isinstance(model.infer_prediction_fn, torch.nn.Sigmoid):
            classification_threshold = [0.5]*prediction_metatensor_ras.shape[1]
            if getattr(model, 'infer_classification_threshold', None) is not None and isinstance(model.infer_classification_threshold, (float, list)):
                if isinstance(model.infer_classification_threshold, float):
                    # Apply the same float to all classes
                    classification_threshold = [model.infer_classification_threshold] * n_classes
                elif isinstance(model.infer_classification_threshold, list):
                    if len(model.infer_classification_threshold) == n_classes:
                        # Perfect match
                        classification_threshold = model.infer_classification_threshold
                    elif len(model.infer_classification_threshold) == n_classes - 1:
                        # Common edge case: config only defined thresholds for foreground classes.
                        # We prepend 0.5 for the background (index 0).
                        classification_threshold = [0.5] + model.infer_classification_threshold
                    else:
                        print(f"Warning: model.infer_classification_threshold length ({len(model.infer_classification_threshold)}) "
                              f"does not match channel count ({n_classes}). Falling back to 0.5.")
            # Convert list to tensor on the same device as the predictions
            thresh_tensor = torch.tensor(
                classification_threshold,
                device=prediction_metatensor_ras.device,
                dtype=prediction_metatensor_ras.dtype
            )
            # Reshape from (C,) to (1, C, 1, 1, 1) so it broadcasts correctly across the spatial voxels
            thresh_tensor = thresh_tensor.view(1, -1, 1, 1, 1)
            # Threshold probabilities (keeps data as MetaTensor, casts to uint8)
            prediction_metatensor_ras_onehot = (prediction_metatensor_ras >= thresh_tensor).to(torch.uint8)
            # Find voxels that do not belong to ANY class (sum across channels == 0)
            unassigned_voxels = (prediction_metatensor_ras_onehot.sum(dim=1, keepdim=True) == 0).to(torch.uint8)
            # Add them blank voxels into the background channel (index 0)
            # Using slice 0:1 keeps the dimension intact for perfect broadcasting
            prediction_metatensor_ras_onehot[:, 0:1, ...] += unassigned_voxels

        else:
            # Argmax along the channel dimension (dim=1) -> shape: (B, 1, X, Y, Z)
            argmax_mask = torch.argmax(prediction_metatensor_ras, dim=1, keepdim=True)
            # Convert to one-hot. MONAI's one_hot is highly optimized for medical
            # (B, 1, ...) shapes, unlike PyTorch's native F.one_hot which expects the channel last.
            # -> shape: (B, n_classes, X, Y, Z)
            prediction_metatensor_ras_onehot = one_hot(argmax_mask, num_classes=n_classes)
        # Ensure final mask is strictly uint8 to save RAM
        prediction_metatensor_ras_onehot = prediction_metatensor_ras_onehot.to(torch.uint8)
        prediction_tensor_ras_onehot = prediction_metatensor_ras_onehot.as_tensor().float()

        # #######################################################################
        # now i also have:
        # prediction_metatensor_ras_onehot: MetaTensors
        # (plus their pure pytorch versions used for overall metrics aggregation)
        # both of shape (1, C, R, A, S) and dtype torch.uint8
        # #######################################################################


        # CONVERSION: PROBABILITY TENSOR TO MARKERS LOCATION
        # ##################################################
        # - prediction_metatensor_ras
        # - logits_metatensor_ras
        _location_window_size = 11 # default 11 is good
        prediction_coordinates_ijk_tensor, prediction_coordinates_ras_tensor = locate_ostia(logits_metatensor_ras, window_size=_location_window_size)
        # converted to long after calculations, so we can have sub-pixel precision
        prediction_coordinates_ijk_tensor = prediction_coordinates_ijk_tensor.long()
        # now i have:
        # - prediction_coordinates_ijk_tensor torch.Tensor (long) of shape (2, 3) [(left, right), (i, j, k)]
        # - prediction_coordinates_ras_tensor torch.Tensor (float32) of shape (2, 3) [(left, right), (r, a, s)]
        # - (from before) label_coordinates_ras_tensor torch.Tensor of shape (2, 3) [(left, right), (r, a, s)]







        # ############## #
        # SAMPLE METRICS #
        # ############## #

        # Metrics: single image

        if (label_json_dict is not None) and not ('none' in metrics):
            if 'all' in metrics or 'distance' in metrics:
                # compute the euclidean distance between prediction and label in RAS space.
                # dim=1 computes the distance across the (r, a, s) coordinates for each row.
                # the resulting tensor will have shape (2,) -> [distance_left, distance_right]
                distances_tensor = torch.linalg.norm(
                    prediction_coordinates_ras_tensor - label_coordinates_ras_tensor,
                    dim=1
                )
                distance_left = distances_tensor[0].item()
                distance_right = distances_tensor[1].item()
                distance_mean = distances_tensor.mean().item()
                # append to the lists
                distance_list.append(distance_mean)
                distance_per_class_list.append([distance_left, distance_right])
            else:
                _set_all_metrics_to_none(
                    [
                        distance_list,
                        distance_per_class_list,
                    ],
                    pattern = [
                        None,
                        [None] * num_foreground_classes
                    ]
                )
            if 'all' in metrics or 'peak-confidence' in metrics:
                pass
                # Peak Confidence: sample the probability exactly at the predicted coordinate
                conf_right = prediction_metatensor_ras[0, 1, prediction_coordinates_ijk_tensor[0, 0], prediction_coordinates_ijk_tensor[0, 1], prediction_coordinates_ijk_tensor[0, 2]].item()
                conf_left = prediction_metatensor_ras[0, 2, prediction_coordinates_ijk_tensor[1, 0], prediction_coordinates_ijk_tensor[1, 1], prediction_coordinates_ijk_tensor[1, 2]].item()
                peak_confidence_list.append(0.5*(conf_right+conf_left))
                peak_confidence_per_class_list.append([conf_right, conf_left])
            else:
                _set_all_metrics_to_none(
                    [
                        peak_confidence_list,
                        peak_confidence_per_class_list,
                    ],
                    pattern = [
                        None,
                        [None] * num_foreground_classes
                    ]
                )
            if 'all' in metrics or 'outliers' in metrics:
                # thresholds and buckets are defined in infer/thresholds.py
                # rows of the coordinates tensors are [right ostium, left ostium]
                distances_tensor = torch.linalg.norm(
                    prediction_coordinates_ras_tensor - label_coordinates_ras_tensor,
                    dim=1
                )
                distance_r = distances_tensor[0].item()
                distance_l = distances_tensor[1].item()
                bucket_r, bucket_l, bucket_patient = classify_pair(distance_r, distance_l)
                # "outlier" is kept for backward compatibility: it means "failed" (above the failure threshold)
                is_failed_r = bucket_r == "failed"
                is_failed_l = bucket_l == "failed"
                outlier_list.append(int(is_failed_r or is_failed_l))
                outlier_per_class_list.append([int(is_failed_r), int(is_failed_l)])
                bucket_list.append(bucket_patient)
                bucket_per_class_list.append([bucket_r, bucket_l])
            else:
                _set_all_metrics_to_none(
                    [
                        outlier_list,
                        outlier_per_class_list,
                        bucket_list,
                        bucket_per_class_list,
                    ],
                    pattern = [
                        None,
                        [None] * num_foreground_classes
                    ]
                )
        else:
            # set all metrics to none accordingly
            _set_all_metrics_to_none(
                list_of_metrics_lists,
                pattern = [
                    None,
                    [None] * num_foreground_classes
                ]
            )

        # write metrics to temporary log file on the go
        # a single-row dataframe containing the latest metrics for this image
        def _fmt_value(v):
            try:
                if isinstance(v, numpy.ndarray):
                    v = v.tolist()
            except Exception:
                pass
            if isinstance(v, (list, tuple)):
                return str(v)
            return v
        row = {
            "image": im_path,
            "label": lab_path,
            "distance": distance_list[-1] if len(distance_list) > 0 else None,
            "distance_per_class": distance_per_class_list[-1] if len(distance_per_class_list) > 0 else None,
            "peak_confidence": peak_confidence_list[-1] if len(peak_confidence_list) > 0 else None,
            "peak_confidence_per_class": peak_confidence_per_class_list[-1] if len(peak_confidence_per_class_list) > 0 else None,
            "outlier": outlier_list[-1] if len(outlier_list) > 0 else None,
            "outlier_per_class": outlier_per_class_list[-1] if len(outlier_per_class_list) > 0 else None,
            "bucket": bucket_list[-1] if len(bucket_list) > 0 else None,
            "bucket_per_class": bucket_per_class_list[-1] if len(bucket_per_class_list) > 0 else None,
        }
        row = {k: _fmt_value(v) for k, v in row.items()}
        df = pandas.DataFrame([row])
        # convert dataframe to a wide string so all fields are visible when appended to the logfile
        with pandas.option_context('display.max_colwidth', None, 'display.width', 3000, 'display.max_columns', None):
            df_str = df.to_string(index=False)
        # append the pretty-printed dataframe row to the running logfile
        with open(logfile, 'a') as f:
            f.write(df_str + '\n')





        # ##################### #
        # SAVE SAMPLE ARTIFACTS #
        # ##################### #

        # Names for saving
        _image_name_ext = os.path.basename(im_path)
        _name, _ext = os.path.splitext(_image_name_ext)
        if _name.endswith('.nii'):
            _name = _name[:-4]
            _ext = '.nii'+_ext
        unique_name = f"case_{int(i):06d}_{_name}"


        # MARKERS & 3D SLICER MARKUPS (Using custom IO module)
        if ('all' in artifacts or 'marker' in artifacts or 'markup-3d-slicer' in artifacts):
            try:
                # 1. Build the simple RAS dictionary expected by compute_image_metadata
                # prediction_coordinates_ras_tensor shape is [2, 3] -> [Right, Left]
                r_ras = prediction_coordinates_ras_tensor[0].tolist()
                l_ras = prediction_coordinates_ras_tensor[1].tolist()
                simple_pos_dict = {
                    "R": {"ras": r_ras},
                    "L": {"ras": l_ras}
                }

                # 2. Compute full SITK/MONAI mappings and affine matrices
                # (If this fails, the outer except block catches it)
                full_pos_dict, affine_ijk_to_ras, affine_ras_to_ijk = compute_image_metadata(
                    image_path=im_path,
                    simple_pos_dict=simple_pos_dict
                )

                # 3. Save Ostium Blueprint (.ostium.json)
                if ('all' in artifacts or 'marker' in artifacts):
                    try:
                        os.makedirs(os.path.abspath(markers_out_dir), exist_ok=True)
                        blueprint_filepath = os.path.join(markers_out_dir, f"{unique_name}.ostium.json")
                        save_blueprint_json(
                            filepath=blueprint_filepath,
                            image_id=_image_name_ext,
                            image_filename=_image_name_ext,
                            pos_dict=full_pos_dict,
                            affine_ijk_to_ras=affine_ijk_to_ras,
                            affine_ras_to_ijk=affine_ras_to_ijk,
                            comment=f"Generated by HCATOstiaLocalization inference utility. Model: {models_tag}"
                        )
                    except Exception as e:
                        print(colored(f"Could not save ostium blueprint {blueprint_filepath}, error:\n{e}", (200, 20, 50)))

                # 4. Save 3D Slicer Markup (.mrk.json)
                if ('all' in artifacts or 'markup-3d-slicer' in artifacts):
                    try:
                        os.makedirs(os.path.abspath(markups_3d_slicer_out_dir), exist_ok=True)
                        slicer_filepath = os.path.join(markups_3d_slicer_out_dir, f"{unique_name}.mrk.json")

                        save_slicer_json(
                            filepath=slicer_filepath,
                            pos_dict=full_pos_dict
                        )
                    except Exception as e:
                        print(colored(f"Could not save 3D Slicer marker {slicer_filepath}, error:\n{e}", (200, 20, 50)))

            except Exception as e:
                # Catch-all for metadata computation or unexpected structural errors
                print(colored(f"Could not compute image metadata or export JSON/Slicer artifacts for {im_path}. Error:\n{e}", (200, 20, 50)))

        # - segmentation mask
        if ('all' in artifacts or 'segmentation-mask' in artifacts):
            os.makedirs(segmentation_mask_out_dir, exist_ok=True)
            is_mutually_exclusive = (prediction_metatensor_ras_onehot.sum(dim=1) == 1).all().item()
            if is_mutually_exclusive:
                # Save as 1 combined file
                # shape must be (C, R, A, S) with no batch dimension
                indexed_mask = torch.argmax(prediction_metatensor_ras_onehot, dim=1, keepdim=True).to(torch.uint8)[0]
                indexed_mask.meta["filename_or_obj"] = f"{unique_name}_segmentation_{models_tag}.ext"
                try:
                    nifti_saver(indexed_mask)
                except Exception as e:
                    print(colored(f"Could not save segmentation mask {indexed_mask.meta['filename_or_obj']}, error:\n{e}", (200, 20, 50)))
            else:
                # Save separately
                for c in range(1, prediction_metatensor_ras_onehot.shape[1]):
                    # Extract just this class channel -> Shape (1, R, A, S)
                    class_mask = prediction_metatensor_ras_onehot[0, c:c+1, ...].to(torch.uint8)
                    class_display_name = _class_display_name(model, c)
                    class_mask.meta["filename_or_obj"] = f"{unique_name}_segmentation_mask_class{c:03d}{class_display_name}.ext"
                    try:
                        nifti_saver(class_mask)
                    except Exception as e:
                        print(colored(f"Could not save segmentation mask {class_mask.meta['filename_or_obj']}, error:\n{e}", (200, 20, 50)))

        # - probability-tensor
        if ('all' in artifacts or 'probability-tensor' in artifacts):
            os.makedirs(probability_tensor_out_dir, exist_ok=True)
            # Save as 1 combined file
            # shape must be (1, C, R, A, S) with batch dimension
            prob_tensor = prediction_metatensor_ras[0:1].detach().to(torch.float16).cpu()
            # OPTION 1: Save as PyTorch .pt file
            out_path_pt = os.path.join(probability_tensor_out_dir, f"{unique_name}_probability.pt")
            try:
                torch.save(prob_tensor.as_tensor(), out_path_pt)
            except Exception as e:
                print(colored(f"Could not save torch probability tensor {out_path_pt}, error:\n{e}", (200, 20, 50)))
            # OPTION 2:  Numpy .npy
            # out_path_npy = os.path.join(probability_tensor_out_dir, f"{unique_name}_probabilities_{model_nickname}.npy")
            # numpy.save(out_path_npy, prob_tensor.numpy())

        # - logits-tensor
        if ('all' in artifacts or 'logits-tensor' in artifacts):
            os.makedirs(logits_tensor_out_dir, exist_ok=True)
            # Save as 1 combined file
            # shape must be (1, C, R, A, S) with batch dimension
            logits_tensor = logits_metatensor_ras[0:1].detach().to(torch.float16).cpu()
            out_path_pt_logits = os.path.join(logits_tensor_out_dir, f"{unique_name}_logits.pt")
            try:
                torch.save(logits_tensor.as_tensor(), out_path_pt_logits)
            except Exception as e:
                print(colored(f"Could not save torch logits tensor {out_path_pt_logits}, error:\n{e}", (200, 20, 50)))

        # - probability-nifti (Slicer-loadable, per-class, correct affine/orientation)
        if ('all' in artifacts or 'probability-nifti' in artifacts):
            os.makedirs(probability_nifti_out_dir, exist_ok=True)
            for c in range(1, prediction_metatensor_ras.shape[1]):
                class_display_name = _class_display_name(model, c)
                prob_class_volume = prediction_metatensor_ras[0, c:c+1, ...].detach().to(torch.float32).cpu()
                prob_class_volume.meta["filename_or_obj"] = f"{unique_name}_probability_class{c:03d}{class_display_name}.ext"
                try:
                    probability_nifti_saver(prob_class_volume)
                except Exception as e:
                    print(colored(f"Could not save probability NIfTI {prob_class_volume.meta['filename_or_obj']}, error:\n{e}", (200, 20, 50)))

        # - logits-nifti (Slicer-loadable, per-class, correct affine/orientation)
        if ('all' in artifacts or 'logits-nifti' in artifacts):
            os.makedirs(logits_nifti_out_dir, exist_ok=True)
            for c in range(1, logits_metatensor_ras.shape[1]):
                class_display_name = _class_display_name(model, c)
                logits_class_volume = logits_metatensor_ras[0, c:c+1, ...].detach().to(torch.float32).cpu()
                logits_class_volume.meta["filename_or_obj"] = f"{unique_name}_logits_class{c:03d}{class_display_name}.ext"
                try:
                    logits_nifti_saver(logits_class_volume)
                except Exception as e:
                    print(colored(f"Could not save logits NIfTI {logits_class_volume.meta['filename_or_obj']}, error:\n{e}", (200, 20, 50)))


        # #################################### #
        # DYNAMIC METRICS REPORTING (PER-STEP) #
        # #################################### #

        raw_metrics_2d = {
            "distance": _process_none_lists(distance_per_class_list),
            "peak_confidence": _process_none_lists(peak_confidence_per_class_list),
            "outlier": _process_none_lists(outlier_per_class_list),
        }
        metrics_dict = {}
        for name, arr_2d in raw_metrics_2d.items():
            if arr_2d.size == 0:
                continue # Failsafe in case no valid images were processed yet
            # Per-class statistics
            metrics_dict[f"{name}_per_class"] = _round_dict_if_float(_get_metrics_statistics(arr_2d))
            # Overall statistics (Average across classes first -> 1D array of per-image scores)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                arr_1d = numpy.nanmean(arr_2d, axis=1)
            metrics_dict[name] = _round_dict_if_float(_get_metrics_statistics(arr_1d))

        # Accuracy buckets (optimal / good / acceptable / failed), per ostium and per patient (worse of the two)
        _bucket_pairs = [p for p in bucket_per_class_list if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(_bucket_pairs) > 0:
            metrics_dict["buckets"] = bucket_summary(
                [p[0] for p in _bucket_pairs],
                [p[1] for p in _bucket_pairs],
                bucket_list,
            )

        # Reconstructs the exact CLI command safely
        full_cli_command = f"{sys.executable} {shlex.join(sys.argv)}"

        # Build JSON Output Structure
        metrics_to_write = []
        metrics_to_write.append('Overall Metrics')
        metrics_to_write.append(
            {
                "cli": full_cli_command,
                "date and time": start_datetime,
                "model nickname": models_tag,
                "weights path": weights_paths if is_cascade else weights_paths[0],
                "torch device": str(torch_device),
                "inference spacing (mm)": target_spacing_mm_ras,
                "inference patch size (px)": patch_size,
                "inference overlap (px)": overlap,
                **({"cascade": cascade_description, "cascade logit offset": cascade_logit_offset} if is_cascade else {}),
                "total time (s)": round(time.time() - t0, 2),
                "images processed": len(distance_list),       # Tracks current progress
                "total images": len(im_lab_paths_pairs),      # Expected total
                "Number of errors": len(errors),
                **metrics_dict,
            }
        )

        metrics_to_write.append('Per Image Metrics')

        # Map the python lists for the per-image JSON rows
        list_mapping = {
            "distance": (distance_list, distance_per_class_list),
            "peak_confidence": (peak_confidence_list, peak_confidence_per_class_list),
            "outlier": (outlier_list, outlier_per_class_list),
            "bucket": (bucket_list, bucket_per_class_list),
        }

        # Populate Per-image metrics
        for idx, (im_path, lab_path) in enumerate(im_lab_paths_pairs):
            row = {
                "image": im_path,
                "label": lab_path,
            }
            for name, (overall_l, per_class_l) in list_mapping.items():
                val = overall_l[idx] if idx < len(overall_l) else None
                per_class_val = per_class_l[idx] if idx < len(per_class_l) else None
                if isinstance(per_class_val, numpy.ndarray):
                    per_class_val = per_class_val.tolist()
                row[name] = round(val, 6) if isinstance(val, float) else val
                row[f"{name}_per_class"] = list(map(
                    lambda x: round(x, 6) if isinstance(x, float) else x,
                    per_class_val
                )) if isinstance(per_class_val, (list, tuple)) else per_class_val
            metrics_to_write.append(row)

        metrics_to_write.append('Errors')
        metrics_to_write.append({"errors": errors})

        # --- SAFE ATOMIC FILE OVERWRITE ---
        import shutil # Ensure this is imported at the top of your script

        metrics_file = os.path.join(output_path, f"metrics__{models_tag}.json")
        hidden_backup = os.path.join(output_path, f".metrics__{models_tag}.json.bak")

        try:
            # 1. Create a hidden backup of the previous step's file
            if os.path.exists(metrics_file):
                shutil.copy2(metrics_file, hidden_backup)

            # 2. Write the new file directly (using json.dump is safer/cleaner than f.write)
            with open(metrics_file, 'w') as f:
                json.dump(metrics_to_write, f, indent=4)

            # 3. Remove the hidden backup on success
            if os.path.exists(hidden_backup):
                os.remove(hidden_backup)

        except Exception as e:
            # If the script crashes during the write, restore the backup
            if os.path.exists(hidden_backup):
                shutil.move(hidden_backup, metrics_file)
            print(colored(f"\nWarning: Could not update intermediate metrics file. Error: {e}", (200, 20, 50)))

    # ############# #
    # FINAL CLI LOG #
    # ############# #

    print(' '*80)
    print(colored("Inference Complete! Results saved in:", (167, 201, 87)))
    print(output_path)
    print(colored("Final Metrics:", (167, 201, 87)))
    print(metrics_file)
    print()
    _final_buckets = metrics_dict.get("buckets") if 'metrics_dict' in locals() else None
    if _final_buckets is not None:
        print(colored("Accuracy buckets (count, % of images with a label):", (167, 201, 87)))
        print("thresholds (mm)  " + "  ".join(
            f"{name}: R {vals['R']:.2f} / L {vals['L']:.2f}" for name, vals in THRESHOLDS_MM.items()
        ))
        print(f"{'bucket':12s} " + " ".join(f"{col:>16s}" for col in ("Right", "Left", "Patient")))
        for b in BUCKETS:
            cells = []
            for col in ("R", "L", "patient"):
                c, p = _final_buckets[col]["counts"][b], _final_buckets[col]["pct"][b]
                cells.append(f"{c:5d} ({p:6.2f}%)" if p is not None else f"{'-':>16s}")
            print(f"{b:12s} " + " ".join(f"{cell:>16s}" for cell in cells))
        print()
    if len(errors) > 0:
        print(colored("ERRORS", (200, 20, 50)))
        print(f"Some errors ({len(errors)}) were encountered during the execution of the program.")
        print("Please check the metrics file to see which images presented some errors and try again.")

# done main()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='HCATOstiaLocalization Inference Utility',
        usage='python -m hcatostialocalization.infer [options]',
        description='Perform inference on 3D medical images using a trained HCATOstiaLocalization model and generate Ostium points markers (json and 3D Slicer format).',
        epilog='(Experimental)'
    )

    parser.add_argument('--slurm-id', type=str, required=False, default='', help='Id of the slurm job - can be useful for tracking and closing jobs manually.')

    parser.add_argument('-i', '--input', nargs='+', type=str, required=True, help='Path(s) to the input 3D medical image file(s) (e.g., NRRD, nii, nii.gz) or one text file containing a list of image paths (new line-separated). Provide one or more paths after -i, or a single path to a .txt list file.')

    parser.add_argument('-l', '--label', nargs='+', type=str, required=False, default='', help='Path(s) to the ground truth label file corresponding to the input image(s). Optional, used for evaluation if provided.')

    parser.add_argument('-m', '--model', nargs='+', type=str, required=False, default=['choose'], choices=['choose'] + list(models_hub_pretrained.keys()), help=f'Nickname(s) of the model(s) to use, 1 to 3 (cascade). 1 model: the model predicts the full volume. 2 models: the first model predicts the full volume, then the second model predicts one patch (its own patch size) centred on the predicted left ostium and one centred on the predicted right ostium; each patch overwrites the logits of the ostium it is centred on. 3 models: as with 2, but the second model refines the left ostium and the third model the right ostium. Logits are merged first, then turned into probabilities. If \'choose\' (alone), an interactive menu will let you choose one model. Default \'choose\'.')

    parser.add_argument('--show-models', action='store_true', required=False, help=f'Displays the list of pretrained models from which to choose from. Then exits.')

    parser.add_argument('-w', '--weights', nargs='+', type=str, required=False, default=['best'], help=f'Path(s) to the model weights file(s) (.pth), one per model given with -m (or a single \'best\' for all). If not provided, default best weights for each model will be used.')

    parser.add_argument('--cascade-logit-offset', type=str, required=False, default='match-peak', choices=['none', 'match-peak'], help='Cascade only. \'none\': refinement patch logits overwrite the full-volume logits as they are. \'match-peak\': refinement patch logits are first shifted by a constant so that their maximum equals the maximum of the full-volume logits under the patch, so that different logit scales of different models cannot move the peak outside (or keep it inside) the patch. Default \'match-peak\'.')

    parser.add_argument('--keep-models-on-gpu', action='store_true', required=False, help='Cascade on GPU only. By default, models wait on CPU and only the model in use is moved to the GPU. With this flag all models stay on the GPU.')

    parser.add_argument('-s', '--spacing', type=str, required=False, default='', help=f'The spacing images should be resampled to (see model specifications). Usually could be "[0.5, 0.5, 0.5]". Expressed in mm, in RAS coordinates, or also as a single float for isotropic spacing. In a cascade, it applies to the first model only; refinement models use their own settings and fall back to this value only if they do not declare one (same for --patch-size, --overlap, --orientation).')

    parser.add_argument('--patch-size', type=str, required=False, default='', help=f'The patch size for the sliding window inference. Usually could be "[128, 128, 128]" down to "[64, 64, 64]". If 2d model, must have only two elements. Also as a single float for isotropic size.')

    parser.add_argument('--overlap', type=str, required=False, default='', help=f'The overlap between patches, expressed in number of pixels. Usually could be "[64, 64, 64]" down to "[0, 0, 0]" (no overlap). If 2d model, must have only two elements. Also as a single float for isotropic overlap.')

    parser.add_argument('--orientation', type=str, required=False, default=None, choices=['ras', 'sitk'], help=f'The orientation of the image for inference. If model was trained on RAS oriented image, use "ras", else, use "sitk" to specify that the model was trained on SimpleITK-opened images. If not specified, the orientation declared by the model is used if it declares one, else "ras".')

    parser.add_argument('--monai', action='store_true', required=False, help='Use MONAIs sliding_window_inference instead of our custom inference engine.')

    parser.add_argument('--gpu', action='store_true', required=False, help='Enable GPU acceleration if available. If present, flag is true; otherwise false.')

    parser.add_argument('-a', '--artifacts', type=str, nargs='+', choices=['none', 'marker', 'markup-3d-slicer', 'probability-tensor', 'logits-tensor', 'probability-nifti', 'logits-nifti', 'segmentation-mask', 'all'], default=['none'], required=False, help='Which artifacts should the program produce (default \'none\': only the metrics are saved, the tensors and NIfTI volumes are heavy). \'probability-tensor\'/\'logits-tensor\' save a combined torch .pt file of the (post-activation) probability / (pre-activation) logits tensor. \'probability-nifti\'/\'logits-nifti\' save the same two tensors instead as per-class NIfTI volumes (correct affine/orientation), loadable directly in 3D Slicer for visual inspection. \'all\' produces every artifact. If \'none\' is specified (default), no artifacts other than the segmentation metrics will be saved.')

    parser.add_argument('--metrics', type=str, nargs='+', choices=['none', 'distance', 'peak-confidence', 'outliers', 'all'], default=['all'], required=False, help='Which metrics should be computed. Metrics are saved in the specified output directory. If \'none\' is specified, no metrics other than test runtime statistics will be saved.')

    parser.add_argument('-o', '--output', type=str, required=False, default=f'{os.getcwd()}', help='Directory to save the inference results, including segmentation masks, metrics, and renderings. generated file names will take from the name of the provided image(s). If not provided, current working directory.')


    args = parser.parse_args()

    # ######### #
    # Fast args #
    # ######### #

    if args.show_models:
        print("List of pretrained models:")
        print(
            json.dumps(
                list(models_hub_pretrained.keys()),
                indent=0
            ).replace(',', '').replace('[', '').replace(']', '').replace('"', '')
        )
        sys.exit(0)

    # ########## #
    # Clean args #
    # ########## #

    # - model
    if 'choose' in args.model and len(args.model) != 1:
        parser.error("'choose' cannot be combined with other models in -m/--model.")
    if len(args.model) > 3:
        parser.error(f"-m/--model accepts at most 3 models (cascade), got {len(args.model)}.")
    if args.model == ['choose']:
        print()
        questions = [
            inquirer.List(
                name='model',
                message='Please select a segmentation model',
                choices=list(models_hub_pretrained.keys()),
                carousel=True
            ),
        ]
        answers = inquirer.prompt(questions, theme=themes.BlueComposure())
        args.model = [answers['model']]
        print(f"Chosen: {args.model[0]}. To specify the segmentation model as an argument, use the -m or --model args.")

    # - image respacing
    spacing_mm_ras = None
    if args.spacing:
        _spacing = json.loads(args.spacing)
        if isinstance(_spacing, (int, float)):
            _spacing = [float(_spacing)]*3
        spacing_mm_ras = tuple(float(_) for _ in _spacing)
        if len(spacing_mm_ras) != 3:
            raise ValueError("provided spacing must be either a single float number, or a list of three numbers similar to '[0.5, 0.4, 0.3]' ")

    # - patch size
    patch_size = None
    if args.patch_size:
        try:
            _patch_size = json.loads(args.patch_size)
            # Strict float rejection
            if isinstance(_patch_size, float) or (isinstance(_patch_size, (list, tuple)) and any(isinstance(x, float) for x in _patch_size)):
                raise ValueError("provided patch size must contain integers only, not floats.")
            # If it's a single integer, keep it (sanitize_spatial_params will broadcast it later based on model dims)
            if isinstance(_patch_size, int):
                patch_size = _patch_size
            else:
                patch_size = tuple(int(_) for _ in _patch_size)
        except Exception as e:
            raise ValueError(f"--patch-size: error occurred while parsing this (it should be a single integer or a list like '[128, 128, 128]').\n {e}")

    # - overlap
    overlap = None
    if args.overlap:
        try:
            _overlap = json.loads(args.overlap)
            # Strict float rejection
            if isinstance(_overlap, float) or (isinstance(_overlap, (tuple, list)) and any(isinstance(x, float) for x in _overlap)):
                raise ValueError("provided overlap must contain integers only, not floats.")
            if isinstance(_overlap, int):
                overlap = _overlap
            else:
                overlap = tuple(int(_) for _ in _overlap)
        except Exception as e:
            raise ValueError(f"--overlap: error occurred while parsing this (it should be a single integer or a list like '[64, 64, 64]').\n {e}")

    # - orientation
    orientation = args.orientation

    # - artifacts
    if 'none' in args.artifacts and len(args.artifacts) != 1:
        print("[ERROR] if 'none' is specified in argument -a/--artifacts, nothing else must be included.")
    if 'all' in args.artifacts and len(args.artifacts) != 1:
        print("[ERROR] if 'all' is specified in argument -a/-artifacts, nothing else must be included.")

    # - metrics
    if 'none' in args.metrics and len(args.metrics) != 1:
        print("[ERROR] if 'none' is specified in argument --metrics, nothing else must be included.")
    if 'all' in args.metrics and len(args.metrics) != 1:
        print("[ERROR] if 'all' is specified in argument --metrics, nothing else must be included.")


    # ### #
    # Run #
    # ### #

    main(
        images = args.input,
        labels = args.label,
        model_nickname = args.model,
        weights_path = args.weights,
        try_gpu = args.gpu,
        patch_size=patch_size,
        overlap=overlap,
        orientation=orientation,
        artifacts = args.artifacts,
        metrics = args.metrics,
        output_dir = args.output,
        target_spacing_mm_ras=spacing_mm_ras,
        use_monai=args.monai,
        job_id=args.slurm_id,
        cascade_logit_offset=args.cascade_logit_offset,
        keep_models_on_gpu=args.keep_models_on_gpu,
    )




