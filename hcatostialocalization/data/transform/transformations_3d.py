import torch
from torchvision.transforms.v2 import functional

from numpy.random import default_rng
from hcatostialocalization.config import config

# Augment 3D CT volumes of shape (B, C, K1, K2, K3)
# (for CT/CCTA images, C is 1 usually)

_rng = default_rng(42)

def k1flip(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _rng.binomial(1, 0.5):
        # hflip flips on the last image dimension
        img = img.transpose(2, 4)
        lab = lab.transpose(2, 4)
        img = functional.hflip(img)
        lab = functional.hflip(lab)
        img = img.transpose(2, 4)
        lab = lab.transpose(2, 4)
    return img, lab

def k2flip(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _rng.binomial(1, 0.5):
        # hflip flips on the last image dimension
        img = img.transpose(3, 4)
        lab = lab.transpose(3, 4)
        img = functional.hflip(img)
        lab = functional.hflip(lab)
        img = img.transpose(3, 4)
        lab = lab.transpose(3, 4)
    return img, lab

def k3flip(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _rng.binomial(1, 0.5):
        # hflip flips on the last image dimension
        img = functional.hflip(img)
        lab = functional.hflip(lab)
    return img, lab

def rotate(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # if sample has not same dimension for each spatial dimension, do not rotate
    if img.shape[2] != img.shape[3] or img.shape[2] != img.shape[4]:
        return img, lab
    # K1, K2
    k = _rng.integers(low=0, high=4)
    img = torch.rot90(img, k, dims=[2, 3])
    lab = torch.rot90(lab, k, dims=[2, 3])
    # K2, K3
    k = _rng.integers(low=0, high=4)
    img = torch.rot90(img, k, dims=[3, 4])
    lab = torch.rot90(lab, k, dims=[3, 4])
    # K1, K3
    k = _rng.integers(low=0, high=4)
    img = torch.rot90(img, k, dims=[2, 4])
    lab = torch.rot90(lab, k, dims=[2, 4])
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
    k1, k2, k3 = img.shape[2], img.shape[3], img.shape[4]
    new_k1, new_k2, new_k3 = int(k1 * zoom_factor), int(k2 * zoom_factor), int(k3 * zoom_factor) 
    target_shape = list(shape_orig)
    target_shape[2], target_shape[3], target_shape[4]=  new_k1, new_k2, new_k3
    target_shape = tuple(target_shape)
    if k1 != new_k1:
        img = torch.nn.functional.interpolate(img, size=(new_k1, new_k2, new_k3), mode="trilinear", align_corners=False)
        lab = torch.nn.functional.interpolate(lab.float(), size=(new_k1, new_k2, new_k3), mode="nearest").long()
    # - crop or pad to original size
    if new_k1 < k1:
        # new image has less pixels than original, use min padding
        diff1 = k1 - new_k1
        diff2 = k2 - new_k2
        diff3 = k3 - new_k3 
        def pad_im_lab(img, lab, padding, dim: int):
            # padding will happen only on the last dimension,
            # (padding patterns below were created as such, 
            # left/right yes, top/bottom are zero, and py pytorch convention
            # pad works on 2d images)
            # so we have to move the desired dimension to the last one,
            # pad, then move it back.
            # - padding must be a list of int like this:
            #   [left, top=0, right, bottom=0]
            if dim<0 or not isinstance(dim, int):
                raise ValueError('dim must be >=0 of type (int)')
            # put dimension of interest in last position
            img = img.transpose(dim, 4)
            lab = lab.transpose(dim, 4)
            # pad
            img = functional.pad(img, padding, padding_mode='reflect')
            lab = functional.pad(lab, padding, padding_mode='reflect')
            # put it back in its place
            img = img.transpose(dim, 4)
            lab = lab.transpose(dim, 4)
            return img, lab
        p0 = _rng.choice([0, 1])
        padding1 = [0 if p0 else diff1, 0, diff1 if p0 else 0, 0]
        padding2 = [0 if p0 else diff2, 0, diff2 if p0 else 0, 0]
        padding3 = [0 if p0 else diff3, 0, diff3 if p0 else 0, 0]
        img, lab = pad_im_lab(img, lab, padding1, dim=2)
        img, lab = pad_im_lab(img, lab, padding2, dim=3)
        img, lab = pad_im_lab(img, lab, padding3, dim=4)
    elif new_k1 > k1:
        # new image has more pixels than original, crop at random position
        start_k1 = _rng.integers(low=0, high=new_k1-k1)
        end_k1 = start_k1 + k1
        start_k2 = _rng.integers(low=0, high=new_k2-k2)
        end_k2 = start_k2 + k2
        start_k3 = _rng.integers(low=0, high=new_k3-k3)
        end_k3 = start_k3 + k3
        img = img[..., start_k1:end_k1, start_k2:end_k2, start_k3:end_k3]
        lab = lab[..., start_k1:end_k1, start_k2:end_k2, start_k3:end_k3]
    else:
        pass
    try:
        assert img.shape == shape_orig
    except Exception as e:
        raise ValueError(f'Shapes do not match: original: {shape_orig}, after zoom: {img.shape}')
    return img, lab


# ############ #
# ALL TOGETHER #
# ############ #

def augment_3d(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # geomety
    img, lab = k1flip(img, lab)
    img, lab = k2flip(img, lab)
    img, lab = k3flip(img, lab)
    img, lab = rotate(img, lab)
    img, lab = zoom(img, lab, min_zoom=0.8, max_zoom=1.2)
    # value
    img, lab = noise_normal(img, lab)
    img, lab = intensity_shift(img, lab)
    return img, lab




if __name__ == "__main__":
    _s = (16, 1, 128, 128, 128)
    image = torch.randn(_s, dtype=torch.float32)
    label = torch.clamp(torch.randn(_s), 0, 1).long()
    for i in range(50):
        im_, la_ = augment_3d(image, label)
        print(i, ":", image.shape, im_.shape, "\n", label.shape, la_.shape, "\n")
