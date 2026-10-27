# 3D Data augmentation for coronary ostia localization.
#
# Designed to be called in the trainer RIGHT AFTER probabilistic_crop, on the training
# patch (before PreprocessorCT, so the image is still in HU):
#   img: (B, 1, R, A, S) float  — HU
#   lab: (B, 3, R, A, S) float  — heatmap [background, ostium R, ostium L]
# Runs on the patch device (GPU), with independent random parameters for each sample
# of the batch, drawn from the global torch generator (hence seedable by the trainer).
#
# Design choices:
# - NO flips and NO 90° rotations: volumes are all in RAS orientation, and standard
#   anatomical position is useful information to distinguish the right ostium
#   from the left. Labels always follow anatomical structure (no R <-> L swap).
# - Small rotation (±15° per axis) + zoom + soft elastic deformation are
#   composed into ONE sampling grid -> a single interpolation.
# - Ostia heatmaps are deformed with the same grid as the image; the background
#   channel is recomputed as 1 - max(R, L).
# - Gaussian noise and intensity shift act only on the image.

import math
import torch
import torch.nn.functional as F


def _rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """angles (B, 3) in radians -> rotation matrices (B, 3, 3)"""
    c0, c1, c2 = torch.cos(angles).unbind(-1)
    s0, s1, s2 = torch.sin(angles).unbind(-1)
    one, zero = torch.ones_like(c0), torch.zeros_like(c0)
    r0 = torch.stack([one, zero, zero, zero, c0, -s0, zero, s0, c0], dim=-1).view(-1, 3, 3)
    r1 = torch.stack([c1, zero, s1, zero, one, zero, -s1, zero, c1], dim=-1).view(-1, 3, 3)
    r2 = torch.stack([c2, -s2, zero, s2, c2, zero, zero, zero, one], dim=-1).view(-1, 3, 3)
    return r2 @ r1 @ r0


def spatial_augment(
        img: torch.Tensor,
        lab: torch.Tensor,
        max_rotation_deg: float = 15.0,
        zoom_range: tuple[float, float] = (0.9, 1.1),
        elastic_max_voxels: float = 2.0,
        elastic_grid_points: int = 5,
        p_affine: float = 0.8,
        p_elastic: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotation (±max_rotation_deg on each axis) + isotropic zoom + soft elastic
    deformation, around patch center.

    Elastic deformation: uniform random displacement field in [-elastic_max_voxels,
    +elastic_max_voxels] on a coarse grid of elastic_grid_points^3 nodes,
    trilinearly interpolated on the patch. With 5 nodes on 128 voxels (~32 voxels between nodes)
    and 2 voxels max amplitude, the field is smooth and cannot fold on itself.
    """
    B = img.shape[0]
    spatial = tuple(img.shape[2:])
    dev = img.device

    do_affine = torch.rand(B, device=dev) < p_affine
    do_elastic = torch.rand(B, device=dev) < p_elastic
    if not (do_affine.any() or (elastic_max_voxels > 0 and do_elastic.any())):
        return img, lab

    # F.affine_grid / grid_sample: grid's last dimension is (x, y, z),
    # with x -> last tensor axis (S), y -> A, z -> R.
    half = torch.tensor(spatial[::-1], dtype=torch.float32, device=dev) / 2.0  # (3,) in order (x, y, z)

    # Linear part in voxels: samples input at q = Rot @ p / zoom (p relative to center).
    angles = (torch.rand(B, 3, device=dev) * 2 - 1) * math.radians(max_rotation_deg)
    angles = angles * do_affine[:, None]
    zoom = zoom_range[0] + torch.rand(B, device=dev) * (zoom_range[1] - zoom_range[0])
    zoom = torch.where(do_affine, zoom, torch.ones_like(zoom))
    lin = _rotation_matrices(angles) / zoom[:, None, None]
    # Normalized coordinates [-1, 1]: theta = diag(1/half) @ lin @ diag(half)
    # ("true" rotation even if the patch were not cubic)
    lin = lin * half[None, None, :] / half[None, :, None]
    theta = torch.cat([lin, torch.zeros(B, 3, 1, device=dev)], dim=2)
    grid = F.affine_grid(theta, size=(B, 1, *spatial), align_corners=False)  # (B, R, A, S, 3)

    if elastic_max_voxels > 0 and do_elastic.any():
        g = int(elastic_grid_points)
        coarse = (torch.rand(B, 3, g, g, g, device=dev) * 2 - 1) * elastic_max_voxels
        coarse = coarse * do_elastic[:, None, None, None, None]
        disp = F.interpolate(coarse, size=spatial, mode="trilinear", align_corners=True)  # voxels, channels (x, y, z)
        grid = grid + disp.permute(0, 2, 3, 4, 1) / half  # voxels -> normalized coordinates

    grid = grid.to(img.dtype)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    ostia = F.grid_sample(lab[:, 1:].to(img.dtype), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    background = 1.0 - ostia.max(dim=1, keepdim=True).values
    lab = torch.cat([background, ostia], dim=1).to(lab.dtype)
    return img, lab


def intensity_augment(
        img: torch.Tensor,
        noise_std_hu: float = 40.0,
        shift_std_hu: float = 5.0,
        p_noise: float = 0.5,
        p_shift: float = 0.5,
    ) -> torch.Tensor:
    """
    Gaussian noise with std ~ |N(0, noise_std_hu)| HU and global shift ~ N(0, shift_std_hu) HU
    (same values as old augment_3d), each applied with its probability.
    """
    B = img.shape[0]
    dev = img.device
    per_sample = (B,) + (1,) * (img.dim() - 1)

    noise_std = torch.randn(B, device=dev).abs() * noise_std_hu * (torch.rand(B, device=dev) < p_noise)
    img = img + torch.randn_like(img) * noise_std.view(per_sample).to(img.dtype)

    shift = torch.randn(B, device=dev) * shift_std_hu * (torch.rand(B, device=dev) < p_shift)
    img = img + shift.view(per_sample).to(img.dtype)
    return img


@torch.no_grad()
def augment_3d_ostia(
        img: torch.Tensor,
        lab: torch.Tensor,
        max_rotation_deg: float = 15.0,
        zoom_range: tuple[float, float] = (0.9, 1.1),
        elastic_max_voxels: float = 2.0,
        elastic_grid_points: int = 5,
        noise_std_hu: float = 40.0,
        shift_std_hu: float = 5.0,
        p_affine: float = 0.8,
        p_elastic: float = 0.3,
        p_noise: float = 0.5,
        p_shift: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Master function: first geometric transformations, then intensity ones"""
    img, lab = spatial_augment(
        img, lab,
        max_rotation_deg=max_rotation_deg,
        zoom_range=tuple(zoom_range),
        elastic_max_voxels=elastic_max_voxels,
        elastic_grid_points=elastic_grid_points,
        p_affine=p_affine,
        p_elastic=p_elastic,
    )
    img = intensity_augment(
        img,
        noise_std_hu=noise_std_hu,
        shift_std_hu=shift_std_hu,
        p_noise=p_noise,
        p_shift=p_shift,
    )
    return img, lab


