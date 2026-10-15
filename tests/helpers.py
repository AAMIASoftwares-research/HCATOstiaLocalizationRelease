"""Helpers of the smoke tests: random-weight checkpoints and a synthetic CT volume."""
import numpy
import torch

# The released models (hub nicknames); the last one is SwinUNETRv2 run at 1.0 mm.
SWIN_1MM = "SwinUNETRv2_DSNT_ablation_training_spacing_1.0mm"
RELEASED_MODELS = ["SwinUNETRv2_DSNT", "ResNet50_DSNT", "ResNet101_DSNT", "nnUNetResidualUNet_DSNT", SWIN_1MM]
EXPECTED_SPACING_MM = {SWIN_1MM: 1.0}      # every other model: 0.5


def build_architecture(nickname: str) -> torch.nn.Module:
    """Same architecture (and arguments) as the corresponding function of model/hub/hub.py, random weights."""
    if nickname in ("SwinUNETRv2_DSNT", SWIN_1MM):
        from monai.networks.nets import SwinUNETR
        kwargs = dict(in_channels=1, out_channels=3, feature_size=24, use_checkpoint=False, use_v2=True)
        try:
            return SwinUNETR(img_size=[128, 128, 128], **kwargs)
        except TypeError:
            return SwinUNETR(**kwargs)
    if nickname in ("ResNet50_DSNT", "ResNet101_DSNT"):
        from hcatostialocalization.model.model_factory import ResNetHeatmap
        if nickname == "ResNet50_DSNT":
            return ResNetHeatmap(model_type="resnet50", in_channels=1, out_channels=3, decoder_channels=64)
        return ResNetHeatmap(model_type="resnet101", in_channels=1, out_channels=3, decoder_channels=128)
    if nickname == "nnUNetResidualUNet_DSNT":
        from functools import partial
        from dynamic_network_architectures.architectures.unet import ResidualUNet
        return ResidualUNet(
            input_channels=1, n_stages=5, features_per_stage=(32, 64, 128, 256, 320),
            conv_op=torch.nn.Conv3d, kernel_sizes=[[3, 3, 3]] * 5,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=(2, 2, 3, 2, 2), num_classes=3, n_conv_per_stage_decoder=(2, 2, 2, 2),
            conv_bias=True, norm_op=partial(torch.nn.InstanceNorm3d, eps=1e-03),
            nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        )
    raise ValueError(nickname)


def write_random_checkpoint(nickname: str, path) -> None:
    """Saves a random-weight checkpoint in the format of the released ones ({'model_state_dict': ...})."""
    torch.manual_seed(0)
    torch.save({"model_state_dict": build_architecture(nickname).state_dict()}, str(path))


def write_phantom_ct(path, size: int = 64, spacing_mm: float = 1.0) -> None:
    """Synthetic 'CT' NIfTI (HU values, a bright sphere in a soft-tissue box) that is NOT a real image:
    it only lets the whole inference pipeline run end to end."""
    import SimpleITK as sitk
    rng = numpy.random.default_rng(0)
    zz, yy, xx = numpy.mgrid[:size, :size, :size]
    arr = rng.normal(40.0, 15.0, size=(size, size, size)).astype(numpy.float32)   # soft tissue
    arr[:, :6, :] = -1000.0                                 # some air, as in a real CT (keeps the HU check of the loader quiet)
    arr[(zz - size / 2) ** 2 + (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size / 5) ** 2] = 350.0  # contrast "aorta"
    img = sitk.GetImageFromArray(arr)                      # array order (z, y, x)
    img.SetSpacing((spacing_mm,) * 3)
    img.SetOrigin((-10.0, -20.0, 30.0))
    sitk.WriteImage(img, str(path))
