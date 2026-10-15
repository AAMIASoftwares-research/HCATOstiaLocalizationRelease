"""Every released architecture is built through the hub from a random-weight checkpoint (this also proves that the
hub constructs exactly the architecture the checkpoints are made for) and run on a small volume."""
import pytest
import torch

from helpers import EXPECTED_SPACING_MM, RELEASED_MODELS, SWIN_1MM, write_random_checkpoint

SLOW = {"SwinUNETRv2_DSNT", "ResNet101_DSNT", "nnUNetResidualUNet_DSNT", SWIN_1MM}
PARAMS = [pytest.param(m, marks=pytest.mark.slow) if m in SLOW else m for m in RELEASED_MODELS]


@pytest.mark.parametrize("nickname", PARAMS)
def test_hub_model_forward(nickname, tmp_path):
    from hcatostialocalization.model.hub.hub import models_hub_pretrained
    ckpt = tmp_path / f"{nickname}.pth"
    write_random_checkpoint(nickname, ckpt)
    model = models_hub_pretrained[nickname](load_weights=str(ckpt)).eval()

    assert model.infer_classes == ["background", "right ostium", "left ostium"]
    assert tuple(model.infer_patch_size) == (128, 128, 128)
    s = EXPECTED_SPACING_MM.get(nickname, 0.5)
    assert tuple(model.infer_target_spacing_mm_ras) == (s, s, s)

    x = torch.zeros(1, 1, 64, 64, 64)
    with torch.no_grad():
        logits = model(x)
        probs = model.infer_prediction_fn(logits)
    assert tuple(logits.shape) == (1, 3, 64, 64, 64)
    # spatial softmax: each channel is one probability distribution over all voxels
    assert torch.allclose(probs.flatten(2).sum(-1), torch.ones(1, 3), atol=1e-4)
