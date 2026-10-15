import torch
import torch.nn.functional as F
from numpy.random import default_rng

# Inizializziamo il generatore di numeri casuali
_rng = default_rng(42)

# ==========================================
# 1. TRASFORMAZIONI GEOMETRICHE (Spatial)
# ==========================================

def k1flip(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ribalta l'asse Z (Profondità)"""
    if _rng.binomial(1, 0.5):
        img = torch.flip(img, dims=[2])
        lab = torch.flip(lab, dims=[2])
    return img, lab

def k2flip(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ribalta l'asse Y (Altezza)"""
    if _rng.binomial(1, 0.5):
        img = torch.flip(img, dims=[3])
        lab = torch.flip(lab, dims=[3])
    return img, lab

def k3flip(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ribalta l'asse X (Larghezza — flip Left/Right anatomico).

    IMPORTANTE: un flip sull'asse X scambia fisicamente la posizione
    dell'ostio destro con quella del sinistro. Dobbiamo quindi scambiare
    i canali 1 (destro) e 2 (sinistro) nella label DOPO il flip geometrico,
    altrimenti il modello riceve supervision contradittoria.
    """
    if _rng.binomial(1, 0.5):
        img = torch.flip(img, dims=[4])
        lab = torch.flip(lab, dims=[4])
        # Scambia canale 1 (R) ↔ canale 2 (L) dopo il flip L/R
        lab = lab.clone()
        lab[:, 1], lab[:, 2] = lab[:, 2].clone(), lab[:, 1].clone()
    return img, lab

def rotate(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ruota casualmente di 90 gradi sui 3 piani ortogonali"""
    # Se le dimensioni non sono cubiche, evitiamo la rotazione per non distorcere
    if img.shape[2] != img.shape[3] or img.shape[2] != img.shape[4]:
        return img, lab
    
    # Piano Z-Y
    k = _rng.integers(low=0, high=4)
    img = torch.rot90(img, k, dims=[2, 3])
    lab = torch.rot90(lab, k, dims=[2, 3])
    # Piano Y-X
    k = _rng.integers(low=0, high=4)
    img = torch.rot90(img, k, dims=[3, 4])
    lab = torch.rot90(lab, k, dims=[3, 4])
    # Piano Z-X
    k = _rng.integers(low=0, high=4)
    img = torch.rot90(img, k, dims=[2, 4])
    lab = torch.rot90(lab, k, dims=[2, 4])
    
    return img, lab

def zoom(img: torch.Tensor, lab: torch.Tensor, min_zoom: float = 0.9, max_zoom: float = 1.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Effettua uno zoom in / zoom out mantenendo intatta la Heatmap"""
    shape_orig = img.shape
    zoom_factor = _rng.uniform(min_zoom, max_zoom)
    
    k1, k2, k3 = img.shape[2], img.shape[3], img.shape[4]
    new_k1, new_k2, new_k3 = int(k1 * zoom_factor), int(k2 * zoom_factor), int(k3 * zoom_factor) 
    
    # Applicazione dello zoom (ridimensionamento)
    if k1 != new_k1:
        img = F.interpolate(img, size=(new_k1, new_k2, new_k3), mode="trilinear", align_corners=False)
        # [MODIFICA FONDAMENTALE] Anche la label (heatmap) viene interpolata in float in modo continuo!
        lab = F.interpolate(lab, size=(new_k1, new_k2, new_k3), mode="trilinear", align_corners=False)
        
    # Crop o Pad per tornare alla dimensione originale (patch size)
    if new_k1 < k1:
        # Immagine più piccola -> facciamo padding ai bordi
        diff1, diff2, diff3 = k1 - new_k1, k2 - new_k2, k3 - new_k3
        p0 = _rng.choice([0, 1])
        
        pad_z = (0 if p0 else diff3, diff3 if p0 else 0)
        pad_y = (0 if p0 else diff2, diff2 if p0 else 0)
        pad_x = (0 if p0 else diff1, diff1 if p0 else 0)
        
        # In PyTorch il pad si passa dalla dimensione finale a quella iniziale (X, Y, Z)
        padding = pad_z + pad_y + pad_x 
        
        img = F.pad(img, padding, mode='reflect')
        lab = F.pad(lab, padding, mode='reflect')
        
    elif new_k1 > k1:
        # Immagine più grande -> ritagliamo
        start_k1 = _rng.integers(low=0, high=new_k1-k1)
        start_k2 = _rng.integers(low=0, high=new_k2-k2)
        start_k3 = _rng.integers(low=0, high=new_k3-k3)
        
        img = img[..., start_k1:start_k1+k1, start_k2:start_k2+k2, start_k3:start_k3+k3]
        lab = lab[..., start_k1:start_k1+k1, start_k2:start_k2+k2, start_k3:start_k3+k3]

    return img, lab

# ==========================================
# 2. TRASFORMAZIONI DI INTENSITÀ (Pixel)
# ==========================================

def noise_normal(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggiunge rumore casuale alla TAC per renderla più 'realistica' o meno pulita"""
    std = abs(_rng.normal(0, 40))
    noise = torch.normal(mean=0.0, std=std, size=img.shape, device=img.device)
    img += noise
    # NOTA: La label (la heatmap) ovviamente NON viene toccata!
    return img, lab

def intensity_shift(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sposta la luminosità generale della TAC"""
    brightness = _rng.normal(0.0, 5.0)
    img += brightness
    return img, lab

# ==========================================
# MASTER FUNCTION (Richiamata dal dataset)
# ==========================================

def augment_3d(img: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Applico le geometrie
    img, lab = k1flip(img, lab)
    img, lab = k2flip(img, lab)
    img, lab = k3flip(img, lab)
    img, lab = rotate(img, lab)
    img, lab = zoom(img, lab, min_zoom=0.8, max_zoom=1.2)
    
    # Applico i colori/pixel
    img, lab = noise_normal(img, lab)
    img, lab = intensity_shift(img, lab)
    
    return img, lab