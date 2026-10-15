import torch
from torchvision.transforms.v2 import functional

from numpy.random import default_rng
from hcatostialocalization.config import config

# Augment 2D CT slices

_rng = default_rng(42)

def hflip(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _rng.binomial(1, 0.5):
        img = functional.hflip(img)
        lab = functional.hflip(lab)
    return img, lab

def vflip(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _rng.binomial(1, 0.5):
        img = functional.vflip(img)
        lab = functional.vflip(lab)
    return img, lab

def rotate(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = _rng.integers(low=0, high=4)
    img = torch.rot90(img, k, dims=[-2, -1])
    lab = torch.rot90(lab, k, dims=[-2, -1])
    return img, lab

def noise_normal(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    std = abs(_rng.normal(0, 40))
    noise = torch.normal(mean=0.0, std=std, size=img.shape, device=img.device)
    img += noise
    return img, lab

def intensity_shift(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    brightness = _rng.normal(0.0, 5.0)
    img += brightness
    return img, lab

def zoom(img: torch.Tensor, lab: torch.Tensor, min_zoom: float = 0.9, max_zoom: float = 1.1) -> tuple[torch.Tensor, torch.Tensor]:
    # zoom < 1.0: zoom out (image gets smaller, black borders)
    # zoom > 1.0: zoom in (image gets bigger, crop)
    assert 0.0 < min_zoom <= max_zoom
    shape_orig = img.shape
    # - zoom
    zoom_factor = _rng.uniform(min_zoom, max_zoom)
    h, w = img.shape[-2], img.shape[-1]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    if h != new_h:
        img = functional.resize(img, [new_h, new_w], antialias=True, interpolation=functional.InterpolationMode.BILINEAR)
        lab = functional.resize(lab.float(), [new_h, new_w], antialias=False, interpolation=functional.InterpolationMode.NEAREST).long()
    # - crop or pad to original size
    if new_w < w:
        # new image has less pixels than original, use zero padding
        diff = w - new_w
        # Padding for the left, top, right and bottom borders respectively
        p0 = _rng.choice([0, 1])
        p1 = _rng.choice([0, 1])
        padding = [
            0 if p0 else diff,
            0 if p1 else diff,
            diff if p0 else 0,
            diff if p1 else 0
        ]
        img = functional.pad(img, padding, padding_mode='reflect') 
        lab = functional.pad(lab, padding, padding_mode='reflect')
    elif new_w > w:
        # new image has more pixels than original, crop at random position
        start_k1 = _rng.integers(low=0, high=new_h-h)
        end_k1 = start_k1 + h
        start_k2 = _rng.integers(low=0, high=new_w-w)
        end_k2 = start_k2 + w
        img = img[..., start_k1:end_k1, start_k2:end_k2]
        lab = lab[..., start_k1:end_k1, start_k2:end_k2]
    else:
        pass
    assert img.shape == shape_orig
    return img, lab


# ############ #
# ALL TOGETHER #
# ############ #

def augment_2d(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # geomety
    img, lab = hflip(img, lab)
    img, lab = vflip(img, lab)
    img, lab = rotate(img, lab)
    img, lab = zoom(img, lab, min_zoom=0.8, max_zoom=1.2)
    # value
    img, lab = noise_normal(img, lab)
    img, lab = intensity_shift(img, lab)
    return img, lab




if __name__ == "__main__":
    _s = (32, 1, 128, 128)
    image = torch.randn(_s, dtype=torch.float32)
    label = torch.clamp(torch.randn(_s), 0, 1).long()
    for i in range(500):
        im_, la_ = augment_2d(image, label)
        print(i, ":", image.shape, im_.shape, "\n", label.shape, la_.shape, "\n")
