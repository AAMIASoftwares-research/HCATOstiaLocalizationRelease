"""
Count the parameters (in millions) of every model used in the benchmark.

Models are built with the exact same factory and YAML experiment configs used
for training (config/model.yaml), so the numbers match the trained networks.
Only the network is instantiated: no data, no checkpoint, no GPU needed.

Run from the repository root:
    python -m hcatostialocalization.model.hub.explore_models
    python -m hcatostialocalization.model.hub.explore_models --experiments swin_unetr_v2_dsnt mednext_s_dsnt mednext_b_dsnt mednext_m_dsnt
"""
import argparse

import torch.nn as nn
from omegaconf import OmegaConf

from hcatostialocalization.config.config import config
from hcatostialocalization.model.dsnt_wrapper import DSNTModelWrapper
from hcatostialocalization.model.model_factory import get_model

# (label used in the paper table, experiment name in config/model.yaml)
BENCHMARK_MODELS = [
    ("SwinUNETRv2", "swin_unetr_v2_dsnt"),
    ("ResNet50", "resnet50_dsnt"),
    ("ResNet101", "resnet101_dsnt"),
    ("nnResUNet", "nnunet_residualunet_dsnt"),
    ("MedNeXt-S", "mednext_s_dsnt"),
    ("MedNeXt-B", "mednext_b_dsnt"),
    ("MedNeXt-M", "mednext_m_dsnt"),
]


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build(experiment: str) -> nn.Module:
    if experiment not in config.experiments:
        raise ValueError(f"Unknown experiment '{experiment}' in config/model.yaml")
    cfg = OmegaConf.create({"model": config.experiments[experiment].model})
    return get_model(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment names from config/model.yaml (default: the five benchmark models).",
    )
    args = parser.parse_args()

    if args.experiments is None:
        models = BENCHMARK_MODELS
    else:
        models = [(name, name) for name in args.experiments]

    rows = []
    for label, experiment in models:
        model = build(experiment)
        # The DSNT head is parameter-free, so wrapping does not change the count;
        # checked explicitly so the table is about the backbone + heads as trained.
        wrapped = DSNTModelWrapper(model)
        total, trainable = count_parameters(wrapped)
        rows.append((label, experiment, total, trainable))

    print("\n" + "=" * 78)
    print(f"{'Model':<14}{'Experiment':<28}{'Total (M)':>12}{'Trainable (M)':>16}")
    print("-" * 78)
    for label, experiment, total, trainable in rows:
        print(f"{label:<14}{experiment:<28}{total / 1e6:>12.2f}{trainable / 1e6:>16.2f}")
    print("=" * 78)

    print("\nLaTeX cells (Parameters column, total, 1 decimal):")
    for label, _, total, _ in rows:
        print(f"    {label} & ... & ... & {total / 1e6:.1f}M \\\\")


if __name__ == "__main__":
    main()
