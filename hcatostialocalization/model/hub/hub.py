import os
import torch

from hcatostialocalization.config.config import config
from hcatostialocalization.layers.preprocessing import PreprocessorCT

from hcatostialocalization.model.hub._safe_globals import safe_checkpoint_globals
torch.serialization.add_safe_globals(safe_checkpoint_globals)


# HCATOstiaLOcalization - models ready to serve
#
# Attributes that infer/__main__.py reads from the returned model:
#   infer_classes                  REQUIRED. list[str]|dict[int,str], index 0 = background. Sets the number of
#                            foreground classes and the class names in the saved file names.
#   infer_prediction_fn            REQUIRED. callable logits -> probabilities. If it is a torch.nn.Softmax /
#                            torch.nn.Sigmoid instance it also decides how the segmentation mask is built.
#   infer_orientation              optional, "ras" | "sitk". Default for the --orientation CLI argument (fallback "ras").
#   infer_preprocessing_fn         optional. callable applied to the image (in HU) right before the model. Errors
#                            inside it are silently ignored by the program. Leave the attribute unset (do not
#                            set None) for models trained on raw HU.
#   infer_classification_threshold optional, float | list[float]. Only used if infer_prediction_fn is a torch.nn.Sigmoid.
# Attributes below are defaults for the same-named CLI arguments (--spacing, --patch-size, --overlap). For the first
# model a CLI value wins; cascade refinement models always use their own (see infer/__main__.py).
# All carry the infer_ prefix so they never clash with attributes of the wrapped nn.Module: a
# plain `model.patch_size` would overwrite SwinUNETR's internal patch_size (=2) and break its input-size check.
#   infer_target_spacing_mm_ras    (R, A, S) spacing in mm the model was trained at, None = native spacing (no respacing).
#   infer_patch_size     sliding-window patch size in voxels (the img_size the model was built with).
#   infer_overlap        suggested sliding-window overlap in voxels (50% of the patch, as in training validation).


def _spatial_softmax(x: torch.Tensor) -> torch.Tensor:
    # trained with dsnt loss that has spatial softmax inside: one distribution over all voxels of each channel
    return torch.softmax(x.flatten(start_dim=2), dim=-1).view(x.shape)

def _load_swinunetr_v2(load_weights: str) -> torch.nn.Module:
    # SwinUNETR v2, 1 input channel, 3 output channels (background, right ostium, left ostium), 128^3 patches
    from monai.networks.nets import SwinUNETR
    checkpoint = torch.load(load_weights, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    swin_kwargs = {
        "in_channels": 1,
        "out_channels": 3,
        "feature_size": 24,
        "use_checkpoint": False,
        "use_v2": True
    }
    try:
        model = SwinUNETR(img_size=[128,128,128], **swin_kwargs)
    except TypeError:
        model = SwinUNETR(**swin_kwargs)

    model.load_state_dict(state_dict)
    return model


def SwinUNETRv2_DSNT(load_weights:str = 'best') -> torch.nn.Module:
    # SwinUNETR v2 trained with images loaded and oriented to RAS, resampled to 0.5 mm isotropic,
    # HU clipped to [-200, 800] and rescaled to [-1, 1] (PreprocessorCT), DSNT loss (spatial softmax prediction).
    if load_weights == 'best':
        load_weights = os.path.join(
            config.active_paths.root_repo,
            "checkpoints/SwinUNETRv2_DSNT.pth",
        )
    model = _load_swinunetr_v2(load_weights)

    model.infer_classes = ['background', 'right ostium', 'left ostium']
    model.infer_prediction_fn = _spatial_softmax
    model.infer_orientation = "ras"
    model.infer_preprocessing_fn = PreprocessorCT(min_hu=-200, max_hu=800)
    # model.infer_classification_threshold = 0.5  # NOT SET: only used when infer_prediction_fn is torch.nn.Sigmoid
    model.infer_target_spacing_mm_ras = (0.5, 0.5, 0.5)
    model.infer_patch_size = (128, 128, 128)
    model.infer_overlap = (64, 64, 64)
    return model


def ResNet50_DSNT(load_weights:str = 'best') -> torch.nn.Module:
    # 3D ResNet50 encoder (MONAI) + Conv3d heatmap head + trilinear upsampling (config/model.yaml: resnet50_dsnt).
    # Trained with images loaded and oriented to RAS, resampled to 0.5 mm isotropic,
    # HU clipped to [-200, 800] and rescaled to [-1, 1] (PreprocessorCT), DSNT loss (spatial softmax prediction).
    from hcatostialocalization.model.model_factory import ResNetHeatmap
    if load_weights == 'best':
        load_weights = os.path.join(
            config.active_paths.root_repo,
            "checkpoints/ResNet50_DSNT.pth",
        )
    checkpoint = torch.load(load_weights, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model = ResNetHeatmap(model_type="resnet50", in_channels=1, out_channels=3, decoder_channels=64)
    model.load_state_dict(state_dict)

    model.infer_classes = ['background', 'right ostium', 'left ostium']
    model.infer_prediction_fn = _spatial_softmax
    model.infer_orientation = "ras"
    model.infer_preprocessing_fn = PreprocessorCT(min_hu=-200, max_hu=800)
    # model.infer_classification_threshold = 0.5  # NOT SET: only used when infer_prediction_fn is torch.nn.Sigmoid
    model.infer_target_spacing_mm_ras = (0.5, 0.5, 0.5)
    model.infer_patch_size = (128, 128, 128)
    model.infer_overlap = (64, 64, 64)
    return model


def ResNet101_DSNT(load_weights:str = 'best') -> torch.nn.Module:
    # 3D ResNet101 encoder (MONAI) + Conv3d heatmap head + trilinear upsampling (config/model.yaml: resnet101_dsnt).
    # Trained with images loaded and oriented to RAS, resampled to 0.5 mm isotropic,
    # HU clipped to [-200, 800] and rescaled to [-1, 1] (PreprocessorCT), DSNT loss (spatial softmax prediction).
    from hcatostialocalization.model.model_factory import ResNetHeatmap
    if load_weights == 'best':
        load_weights = os.path.join(
            config.active_paths.root_repo,
            "checkpoints/ResNet101_DSNT.pth",
        )
    checkpoint = torch.load(load_weights, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model = ResNetHeatmap(model_type="resnet101", in_channels=1, out_channels=3, decoder_channels=128)
    model.load_state_dict(state_dict)

    model.infer_classes = ['background', 'right ostium', 'left ostium']
    model.infer_prediction_fn = _spatial_softmax
    model.infer_orientation = "ras"
    model.infer_preprocessing_fn = PreprocessorCT(min_hu=-200, max_hu=800)
    # model.infer_classification_threshold = 0.5  # NOT SET: only used when infer_prediction_fn is torch.nn.Sigmoid
    model.infer_target_spacing_mm_ras = (0.5, 0.5, 0.5)
    model.infer_patch_size = (128, 128, 128)
    model.infer_overlap = (64, 64, 64)
    return model


def nnUNetResidualUNet_DSNT(load_weights:str = 'best') -> torch.nn.Module:
    # nnU-Net v2 residual encoder U-Net (dynamic_network_architectures), same arguments as model_factory.get_model
    # (config/model.yaml: nnunet_residualunet_dsnt).
    # Trained with images loaded and oriented to RAS, resampled to 0.5 mm isotropic,
    # HU clipped to [-200, 800] and rescaled to [-1, 1] (PreprocessorCT), DSNT loss (spatial softmax prediction).
    from functools import partial
    from dynamic_network_architectures.architectures.unet import ResidualUNet
    if load_weights == 'best':
        load_weights = os.path.join(
            config.active_paths.root_repo,
            "checkpoints/nnUNetResidualUNet_DSNT.pth",
        )
    checkpoint = torch.load(load_weights, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model = ResidualUNet(
        input_channels=1,
        n_stages=5,
        features_per_stage=(32, 64, 128, 256, 320),
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 5,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=(2, 2, 3, 2, 2),
        num_classes=3,
        n_conv_per_stage_decoder=(2, 2, 2, 2),
        conv_bias=True,
        norm_op=partial(torch.nn.InstanceNorm3d, eps=1e-03),
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )
    model.load_state_dict(state_dict)

    model.infer_classes = ['background', 'right ostium', 'left ostium']
    model.infer_prediction_fn = _spatial_softmax
    model.infer_orientation = "ras"
    model.infer_preprocessing_fn = PreprocessorCT(min_hu=-200, max_hu=800)
    # model.infer_classification_threshold = 0.5  # NOT SET: only used when infer_prediction_fn is torch.nn.Sigmoid
    model.infer_target_spacing_mm_ras = (0.5, 0.5, 0.5)
    model.infer_patch_size = (128, 128, 128)
    model.infer_overlap = (64, 64, 64)
    return model


def SwinUNETRv2_DSNT_ablation_training_spacing_1p0mm(load_weights:str = 'best') -> torch.nn.Module:
    # SwinUNETR v2 trained (and, by default, run) at 1.0 mm isotropic instead of 0.5 mm
    if load_weights == 'best':
        load_weights = os.path.join(
            config.active_paths.root_repo,
            "checkpoints/SwinUNETRv2_DSNT_ablation_training_spacing_1.0mm.pth",
        )
    model = SwinUNETRv2_DSNT(load_weights)
    model.infer_target_spacing_mm_ras = (1.0, 1.0, 1.0)
    return model


# ####################### #
# IMPORTABLE DICTIONARIES #
# ####################### #

models_hub_pretrained = {
    'SwinUNETRv2_DSNT': SwinUNETRv2_DSNT,
    'ResNet50_DSNT': ResNet50_DSNT,
    'ResNet101_DSNT': ResNet101_DSNT,
    'nnUNetResidualUNet_DSNT': nnUNetResidualUNet_DSNT,

    # trained and run at a coarser isotropic spacing
    'SwinUNETRv2_DSNT_ablation_training_spacing_1.0mm': SwinUNETRv2_DSNT_ablation_training_spacing_1p0mm,
}
