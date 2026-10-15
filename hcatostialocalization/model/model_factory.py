import os
import sys
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
# Importiamo sia resnet50 che resnet101 da MONAI
from monai.networks.nets import SwinUNETR, DynUNet, resnet50, resnet101

# MedNeXt: installed as a real package (pip name "mednextv1", importable as "nnunet_mednext"),
# no vendored copy in this repo. See the "mednext" branch below for the install command.
try:
    from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1
except ImportError:
    # Non installato: placeholder per evitare crash all'avvio, l'errore vero arriva al momento
    # di selezionare model.name == "mednext" (vedi messaggio nel branch corrispondente).
    create_mednext_v1 = None

try:
    from dynamic_network_architectures.architectures.unet import ResidualUNet
except ImportError:
    ResidualUNet = None

# ==============================================================================
# 0. PROJECT ROOT RESOLUTION
# ==============================================================================
current_dir = os.path.abspath(os.getcwd())
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "hcatostialocalization")):
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)

from hcatostialocalization.config.config import config

# ==============================================================================
# 1. ARCHITECTURE UTILITIES
# ==============================================================================
def get_kernels_strides(img_size, filters=(16, 32, 64, 128, 256)):
    # img_size is unused: this is a fixed 5-stage schema, not nnU-Net's real data-fingerprint
    # self-configuration. filters[-1] is the bottleneck channel count.
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    return kernels, strides, list(filters)


class ResNetHeatmap(nn.Module):
    """
    Wrapper generico per encoder 3D ResNet (50 o 101) con testa di regressione per Heatmap.
    """
    def __init__(self, model_type="resnet50", in_channels=1, out_channels=3, decoder_channels=64):
        super().__init__()
        norm_groups = 8 if decoder_channels % 8 == 0 else 1
        
        # Scegliamo dinamicamente l'encoder richiesto dallo YAML
        if model_type == "resnet101":
            self.encoder = resnet101(
                spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels, feed_forward=False
            )
        else:
            self.encoder = resnet50(
                spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels, feed_forward=False
            )
            
        self.heatmap_head = nn.Sequential(
            nn.Conv3d(2048, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        input_shape = x.shape[-3:]
        encoder = self.encoder

        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.act(x)
        x = encoder.maxpool(x)
        x = encoder.layer1(x)
        x = encoder.layer2(x)
        x = encoder.layer3(x)
        x = encoder.layer4(x)

        x = self.heatmap_head(x)
        return F.interpolate(x, size=input_shape, mode="trilinear", align_corners=False)


# ==============================================================================
# 2. MODEL FACTORY
# ==============================================================================
def get_model(cfg):
    model_name = cfg.model.name.lower()
    img_size = cfg.model.img_size
    in_channels = cfg.model.in_channels
    out_channels = cfg.model.out_channels
    final_activation = cfg.model.get("final_activation", "softmax").lower()

    if model_name == "swin_unetr":
        use_v2_flag = cfg.model.get("use_v2", False)
        print(f"[*] Initializing SwinUNETR (use_v2={use_v2_flag}) | in={in_channels}ch, out={out_channels}ch, patch={img_size}")
        swin_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "feature_size": cfg.model.get('feature_size', 48),
            "use_checkpoint": True,
        }
        if use_v2_flag:
            swin_kwargs["use_v2"] = True
        try:
            model = SwinUNETR(img_size=img_size, **swin_kwargs)
        except TypeError:
            model = SwinUNETR(**swin_kwargs)

    elif model_name == "dynunet":
        # Plain (non-residual) MONAI U-Net with a fixed, hand-picked kernel/stride/filter schema.
        # NOT nnU-Net's real self-configuring method and NOT the same architecture as
        # "nnunet_residualunet" below: no residual blocks, no data-fingerprint-driven config.
        filters = list(cfg.model.get("filters", [16, 32, 64, 128, 256]))
        print(f"[*] Initializing DynUNet (plain U-Net, fixed schema) | in={in_channels}ch, out={out_channels}ch, "
              f"patch={img_size}, filters={filters} (bottleneck={filters[-1]})")
        kernels, strides, filters = get_kernels_strides(img_size, filters=filters)
        model = DynUNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernels, strides=strides, upsample_kernel_size=strides[1:],
            filters=filters, norm_name="instance", deep_supervision=False,
        )

    elif model_name in ["nnunet_residualunet", "nnunet_residual_unet", "residualunet"]:
        if ResidualUNet is None:
            raise ImportError(
                "Impossibile caricare nnU-Net ResidualUNet. "
                "Installa le dipendenze con 'pip install nnunetv2'."
            )

        from omegaconf import OmegaConf

        n_stages = int(cfg.model.get("n_stages", 5))
        features_per_stage = tuple(OmegaConf.to_container(cfg.model.features_per_stage))
        kernel_sizes = [list(k) for k in OmegaConf.to_container(cfg.model.kernel_sizes)]
        strides = [list(s) for s in OmegaConf.to_container(cfg.model.strides)]
        n_blocks_per_stage = tuple(OmegaConf.to_container(cfg.model.n_blocks_per_stage))
        n_conv_per_stage_decoder = tuple(OmegaConf.to_container(cfg.model.n_conv_per_stage_decoder))

        print(
            "[*] Initializing nnU-Net ResidualUNet "
            f"| in={in_channels}ch, out={out_channels}ch, patch={img_size}, "
            f"features={features_per_stage}"
        )
        model = ResidualUNet(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=torch.nn.Conv3d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=out_channels,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=partial(torch.nn.InstanceNorm3d, eps=1e-03),
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
        )

    elif model_name in ["resnet50", "resnet50_3d", "resnet101"]:
        print(f"[*] Initializing {model_name.upper()} 3D Heatmap | in={in_channels}ch, out={out_channels}ch, patch={img_size}")
        model = ResNetHeatmap(
            model_type=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_channels=cfg.model.get("decoder_channels", 64),
        )

    elif model_name in ("mednext", "mednext_b", "mednext_m"):  # "mednext": size from architecture_type (S in mednext_s_dsnt); "mednext_b" / "mednext_m": always B / M
        # Install (not vendored in this repo, only the architecture is used, no nnU-Net training/CLI code):
        #   pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
        # (pip package "mednextv1", importable as "nnunet_mednext"; official repo also documents
        # cloning it and running 'pip install -e .' from inside, same result.)
        if create_mednext_v1 is None:
            raise ImportError(
                "Impossibile caricare MedNeXt (nnunet_mednext non installato). "
                "Installa con: pip install git+https://github.com/MIC-DKFZ/MedNeXt.git"
            )

        arch_type = {"mednext_b": "B", "mednext_m": "M"}.get(model_name) or cfg.model.get("architecture_type", "b").upper()  # 'S' | 'B' | 'M' | 'L'
        k_size = cfg.model.get("kernel_size", 3)
        print(f"[*] Initializing MedNeXt | Type={arch_type}, Kernel={k_size}x{k_size}x{k_size} | in={in_channels}ch, out={out_channels}ch")

        # create_mednext_v1 already bakes in the paper's per-size n_channels/exp_r/block_counts
        # (S/B/M/L) — no need to hardcode them here as the old vendored call used to.
        model = create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=out_channels,
            model_id=arch_type,
            kernel_size=k_size,
            deep_supervision=False,
        )

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: 'swin_unetr', "
            "'dynunet', 'nnunet_residualunet', 'resnet50', 'resnet101', 'mednext', 'mednext_b', 'mednext_m'."
        )

    # Gestione attivazione finale
    if final_activation == "sigmoid":
        return nn.Sequential(model, nn.Sigmoid())
    elif final_activation == "softmax":
        return nn.Sequential(model, nn.Softmax(dim=1))
    elif final_activation in ["none", "identity"]:
        return model

    raise ValueError(f"Unsupported final activation '{final_activation}'.")
