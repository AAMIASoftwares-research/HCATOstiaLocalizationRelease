import torch


def _gaussian_logits(shape, center, sigma=1.5):
    grids = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij")
    r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return -r2 / (2 * sigma ** 2)


def test_location_from_heatmap_is_subvoxel():
    from hcatostialocalization.infer.utils import get_location_from_probability_one_channel
    from hcatostialocalization.model.hub.hub import _spatial_softmax
    center = (30.3, 40.6, 50.2)
    logits = _gaussian_logits((64, 64, 64), center)[None, None]        # (1, 1, D, H, W)
    probs = _spatial_softmax(logits)[0, 0]
    loc = get_location_from_probability_one_channel(probs)
    assert loc.shape == (3,)
    assert torch.allclose(loc, torch.tensor(center), atol=0.5), loc


def test_ct_preprocessing_range():
    from hcatostialocalization.layers.preprocessing import PreprocessorCT
    x = torch.tensor([-1000.0, -200.0, 300.0, 800.0, 2000.0]).view(1, 1, 1, 1, 5)
    y = PreprocessorCT(min_hu=-200, max_hu=800)(x).flatten()
    assert torch.allclose(y, torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]))
