# Data augmentation 3D per la localizzazione degli osti coronarici.
#
# Pensata per essere chiamata nel trainer SUBITO DOPO probabilistic_crop, sulla patch
# di training (prima di PreprocessorCT, quindi l'immagine è ancora in HU):
#   img: (B, 1, R, A, S) float  — HU
#   lab: (B, 3, R, A, S) float  — heatmap [sfondo, ostio R, ostio L]
# Gira sul device della patch (GPU), con parametri casuali indipendenti per ogni campione
# del batch, estratti dal generatore globale di torch (quindi seedabile dal trainer).
#
# Scelte di design:
# - NESSUN flip e NESSUNA rotazione di 90°: i volumi sono tutti in orientamento RAS, e la
#   posizione anatomica standard è un'informazione utile per distinguere l'ostio destro
#   dal sinistro. Le label seguono sempre la struttura anatomica (nessuno scambio R <-> L).
# - Rotazione piccola (±15° per asse) + zoom + deformazione elastica morbida vengono
#   composte in UNA sola griglia di campionamento -> una sola interpolazione.
# - Le heatmap degli osti vengono deformate con la stessa griglia dell'immagine; il canale
#   di sfondo viene ricalcolato come 1 - max(R, L).
# - Rumore gaussiano e shift di intensità agiscono solo sull'immagine.

import math
import torch
import torch.nn.functional as F


def _rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """angles (B, 3) in radianti -> matrici di rotazione (B, 3, 3)"""
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
    Rotazione (±max_rotation_deg su ciascun asse) + zoom isotropo + deformazione elastica
    morbida, attorno al centro della patch.

    Deformazione elastica: campo di spostamento casuale uniforme in [-elastic_max_voxels,
    +elastic_max_voxels] su una griglia grossolana di elastic_grid_points^3 nodi,
    interpolato trilinearmente sulla patch. Con 5 nodi su 128 voxel (~32 voxel tra nodi)
    e 2 voxel di ampiezza massima il campo è liscio e non può ripiegarsi su se stesso.
    """
    B = img.shape[0]
    spatial = tuple(img.shape[2:])
    dev = img.device

    do_affine = torch.rand(B, device=dev) < p_affine
    do_elastic = torch.rand(B, device=dev) < p_elastic
    if not (do_affine.any() or (elastic_max_voxels > 0 and do_elastic.any())):
        return img, lab

    # F.affine_grid / grid_sample: l'ultima dimensione della griglia è (x, y, z),
    # con x -> ultimo asse del tensore (S), y -> A, z -> R.
    half = torch.tensor(spatial[::-1], dtype=torch.float32, device=dev) / 2.0  # (3,) in ordine (x, y, z)

    # Parte lineare in voxel: campiona l'input in q = Rot @ p / zoom (p relativo al centro).
    angles = (torch.rand(B, 3, device=dev) * 2 - 1) * math.radians(max_rotation_deg)
    angles = angles * do_affine[:, None]
    zoom = zoom_range[0] + torch.rand(B, device=dev) * (zoom_range[1] - zoom_range[0])
    zoom = torch.where(do_affine, zoom, torch.ones_like(zoom))
    lin = _rotation_matrices(angles) / zoom[:, None, None]
    # Coordinate normalizzate [-1, 1]: theta = diag(1/half) @ lin @ diag(half)
    # (rotazione "vera" anche se la patch non fosse cubica)
    lin = lin * half[None, None, :] / half[None, :, None]
    theta = torch.cat([lin, torch.zeros(B, 3, 1, device=dev)], dim=2)
    grid = F.affine_grid(theta, size=(B, 1, *spatial), align_corners=False)  # (B, R, A, S, 3)

    if elastic_max_voxels > 0 and do_elastic.any():
        g = int(elastic_grid_points)
        coarse = (torch.rand(B, 3, g, g, g, device=dev) * 2 - 1) * elastic_max_voxels
        coarse = coarse * do_elastic[:, None, None, None, None]
        disp = F.interpolate(coarse, size=spatial, mode="trilinear", align_corners=True)  # voxel, canali (x, y, z)
        grid = grid + disp.permute(0, 2, 3, 4, 1) / half  # voxel -> coordinate normalizzate

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
    Rumore gaussiano con std ~ |N(0, noise_std_hu)| HU e shift globale ~ N(0, shift_std_hu) HU
    (stessi valori della vecchia augment_3d), ciascuno applicato con la sua probabilità.
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
    """Master function: prima le trasformazioni geometriche, poi quelle di intensità"""
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


