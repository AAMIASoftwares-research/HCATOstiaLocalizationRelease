import torch
import torch.nn as nn
import torch.nn.functional as F

from hcatostialocalization.model.dsnt_layer import DSNT3D

# ==============================================================================
# 1. LOSS FUNCTION DEFINITIONS
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss multiclasse per target continui (Heatmap Gaussiane).
    Calcola l'overlapping geometrico tra predizioni e ground truth.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_probs, target):
        # pred_probs: (B, C, D, H, W) post-softmax o post-sigmoid
        # target: (B, C, D, H, W)
        
        # Flattiamo le dimensioni spaziali (D, H, W) mantenendo Batch e Canali
        B, C = pred_probs.shape[0], pred_probs.shape[1]
        pred_flat = pred_probs.view(B, C, -1)
        target_flat = target.view(B, C, -1)

        intersection = torch.sum(pred_flat * target_flat, dim=-1)
        denominator = torch.sum(pred_flat * pred_flat, dim=-1) + torch.sum(target_flat * target_flat, dim=-1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        
        # Consideriamo solo i canali dell'ostio (1 e 2), lo sfondo (canale 0) viene escluso dal calcolo
        # in modo che l'overlapping sia guidato chirurgicamente dai punti di interesse.
        return 1.0 - dice[:, 1:].mean()


class WeightedMSELoss(nn.Module):
    def __init__(self, foreground_weight=10.0):
        super().__init__()
        self.fg_weight = foreground_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        base_loss = self.mse(pred, target)
        foreground_mask = target[:, 1:].sum(dim=1, keepdim=True)
        weight_map = 1.0 + (foreground_mask * self.fg_weight)
        weighted_loss = base_loss * weight_map
        return weighted_loss.mean()


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_logits, target_probs):
        return self.ce(pred_logits, target_probs)


class HybridSegLoss(nn.Module):
    """
    Loss Ibrida Avanzata: Cross-Entropy (CE) + Weighted BCE + Soft Dice Loss.
    """
    def __init__(self, ce_weight=0.5, bce_weight=0.5, dice_weight=0.0, foreground_weight=100.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.fg_weight = foreground_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred_logits, target):
        # 1. Termine CE: Competizione globale tra canali (richiede i Logit raw)
        loss_ce = self.ce(pred_logits, target)

        # 2. Termine Weighted BCE: Supervisione binaria voxel-per-voxel
        bce_base = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        foreground_mask = target[:, 1:].sum(dim=1, keepdim=True)
        weight_map = 1.0 + (foreground_mask * self.fg_weight)
        loss_bce = (bce_base * weight_map).mean()

        # 3. Termine DICE: Richiede le probabilità (applichiamo Softmax sull'asse dei canali)
        pred_probs = F.softmax(pred_logits, dim=1)
        loss_dice = self.dice(pred_probs, target)

        # Combinazione convessa finale dei tre termini
        return (self.ce_weight * loss_ce) + (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


class DSNTJSDLoss(nn.Module):
    """
    Loss DSNT (Differentiable Spatial to Numerical Transforms) + Jensen-Shannon Divergence.

    Formula dal paper:
        L(Ĥ, y) = w_R · L_R  +  w_L · L_L

    Per ciascun canale (R o L):
        L_ch = L_E + λ_JSD · L_D + λ_bg · L_BCE_bg

    Dove:
        L_E     = distanza Euclidea in spazio normalizzato [-1,1] tra coordinate
                  predette (via DSNT) e coordinate GT.
        L_D     = JSD tra distribuzione predetta e gaussiana target — forza la
                  heatmap a restare compatta invece di spalmarsi sul volume.
        L_BCE_bg= BCE tra i logit e la gaussiana binarizzata — supervisione
                  diretta per-voxel per sopprimere il rumore di sfondo che la
                  JSD da sola non riesce a eliminare (gradiente troppo debole
                  per singolo voxel su 128³ volume).
        w_R     = 0.35  (pesi asimmetrici: L più difficile anatomicamente)
        w_L     = 0.65

    Parametri chiave aggiornabili dall'esterno (SigmaScheduler):
        criterion.sigma_jsd_R = new_sigma
        criterion.sigma_jsd_L = new_sigma

    Richiede final_activation: none (logit raw dal modello).
    Crop senza ostio (heatmap GT < min_heatmap_sum) vengono mascherati e
    non contribuiscono alla loss.

    Args:
        lambda_jsd:      Peso del termine JSD. Default: 1.0.
        sigma_jsd_R:     Sigma iniziale (voxel) della gaussiana target per il canale R.
        sigma_jsd_L:     Sigma iniziale (voxel) della gaussiana target per il canale L.
                         Più grande di R: variabilità anatomica LMCA maggiore.
        lambda_bg:       Peso del termine BCE background. Default: 0.5.
        temperature:     Temperatura della spatial softmax. < 1.0 → distribuzione
                         più nitida, meno rumore di fondo. Default: 0.3.
        weight_R:        Peso del canale R nella loss totale. Default: 0.35.
        weight_L:        Peso del canale L nella loss totale. Default: 0.65.
        min_heatmap_sum: Soglia sotto cui il canale è considerato assente. Default: 0.001.
    """

    def __init__(
        self,
        lambda_jsd: float = 1.0,
        sigma_jsd_R: float = 2.5,
        sigma_jsd_L: float = 4.0,
        lambda_bg: float = 0.5,
        temperature: float = 0.3,
        weight_R: float = 0.35,
        weight_L: float = 0.65,
        min_heatmap_sum: float = 0.001,
    ):
        super().__init__()
        self.lambda_jsd      = lambda_jsd
        self.sigma_jsd_R     = sigma_jsd_R   # aggiornato via criterion.sigma_jsd_R = new_sigma
        self.sigma_jsd_L     = sigma_jsd_L   # aggiornato via criterion.sigma_jsd_L = new_sigma
        self.lambda_bg       = lambda_bg
        self.temperature     = temperature
        self.weight_R        = weight_R
        self.weight_L        = weight_L
        self.min_heatmap_sum = min_heatmap_sum
        self.dsnt_layer      = DSNT3D(normalized_coords=True)

    # Proprietà di compatibilità: criterion.sigma_jsd aggiorna entrambi i canali
    @property
    def sigma_jsd(self) -> float:
        return (self.sigma_jsd_R + self.sigma_jsd_L) / 2.0

    @sigma_jsd.setter
    def sigma_jsd(self, value: float):
        """Consente al SigmaScheduler di aggiornare entrambi i canali con un'unica riga."""
        self.sigma_jsd_R = value
        self.sigma_jsd_L = value

    # ------------------------------------------------------------------
    # Primitive differenziabili
    # ------------------------------------------------------------------

    def _spatial_softmax(self, logits_c: torch.Tensor) -> torch.Tensor:
        """Softmax spaziale con temperatura.

        temperature < 1.0 → distribuzione più peakata (meno rumore di fondo).
        temperature = 1.0 → softmax standard.
        """
        B = logits_c.shape[0]
        flat = logits_c.view(B, -1) / self.temperature
        return F.softmax(flat, dim=-1).view_as(logits_c)

    def _make_gaussian_target(
        self,
        gt_coords_norm: torch.Tensor,   # (B, 3) in [-1, +1]
        D: int, H: int, W: int,
        device: torch.device,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Genera una gaussiana 3D centrata su gt_coords_norm.
        Il sigma viene convertito da voxel a spazio normalizzato [-1,+1].
        Output:
          - gauss_unnorm: (B, D, H, W) con picco a 1.0 (per BCE).
          - gauss_norm:   (B, D, H, W) distribuzione prob. somma 1 (per JSD).
        """
        B = gt_coords_norm.shape[0]
        sz = sigma * 2.0 / max(D - 1, 1)
        sy = sigma * 2.0 / max(H - 1, 1)
        sx = sigma * 2.0 / max(W - 1, 1)

        z_g = torch.linspace(-1.0, 1.0, D, device=device)
        y_g = torch.linspace(-1.0, 1.0, H, device=device)
        x_g = torch.linspace(-1.0, 1.0, W, device=device)

        gz = gt_coords_norm[:, 0].view(B, 1, 1, 1)
        gy = gt_coords_norm[:, 1].view(B, 1, 1, 1)
        gx = gt_coords_norm[:, 2].view(B, 1, 1, 1)

        dist_sq = ((z_g.view(1, D, 1, 1) - gz) / sz) ** 2 \
                + ((y_g.view(1, 1, H, 1) - gy) / sy) ** 2 \
                + ((x_g.view(1, 1, 1, W) - gx) / sx) ** 2

        gauss_unnorm = torch.exp(-0.5 * dist_sq)
        gauss_flat = gauss_unnorm.view(B, -1)
        gauss_norm = gauss_flat / (gauss_flat.sum(dim=-1, keepdim=True) + 1e-8)
        return gauss_unnorm.view(B, D, H, W), gauss_norm.view(B, D, H, W)

    @staticmethod
    def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Jensen-Shannon Divergence tra due distribuzioni piatte (B, N).
        JSD(P||Q) = 0.5·KL(P||M) + 0.5·KL(Q||M),  M = (P+Q)/2.
        Simmetrica, limitata in [0, ln2]. Output: (B,).
        """
        m = 0.5 * (p + q)
        kl_pm = (p * torch.log((p + eps) / (m + eps))).sum(dim=-1)
        kl_qm = (q * torch.log((q + eps) / (m + eps))).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)

    def _channel_loss(
        self,
        logits_c: torch.Tensor,   # (B, D, H, W) — logit del canale c
        target_c: torch.Tensor,   # (B, D, H, W) — heatmap GT del canale c
        sigma: float,             # sigma della gaussiana target per questo canale
    ) -> torch.Tensor:
        """
        Calcola L_E + λ_JSD·L_D + λ_bg·L_BCE per un singolo canale ostio.
        I campioni senza ostio nel crop corrente sono esclusi tramite maschera.
        """
        B, D, H, W = logits_c.shape
        device = logits_c.device

        # Maschera: ostio presente se la somma dell'heatmap GT supera la soglia
        heatmap_sum = target_c.view(B, -1).sum(dim=-1)           # (B,)
        valid = heatmap_sum > self.min_heatmap_sum                # (B,) bool

        if not valid.any():
            return (logits_c * 0.0).sum()

        # 1. Distribuzione predetta via spatial softmax con temperatura
        prob_pred = self._spatial_softmax(logits_c)               # (B, D, H, W)

        # 2. Coordinate predette via DSNT (centro di massa della distribuzione)
        # Usiamo get_coords_from_probs per allineare il centro di massa alla probabilità
        # con temperatura T=0.3, in modo che JSD e L_E operino sulla stessa distribuzione.
        pred_coords = self.dsnt_layer.get_coords_from_probs(prob_pred.unsqueeze(1))
        pred_coords = pred_coords.squeeze(1)                      # (B, 3) in [-1,+1]

        # 3. Coordinate GT estratte dalla heatmap target normalizzata
        t_sum = heatmap_sum.clamp(min=1e-8).view(B, 1, 1, 1)
        prob_target = target_c / t_sum
        # ERRORE FIXATO: prob_target è già una probabilità!
        # Se usiamo self.dsnt_layer(...) applichiamo una SECONDA softmax, appiattendo la GT.
        # Usiamo get_coords_from_probs per calcolare l'aspettativa corretta.
        gt_coords = self.dsnt_layer.get_coords_from_probs(prob_target.unsqueeze(1))
        gt_coords = gt_coords.squeeze(1)                          # (B, 3)

        # 4. L_E: distanza Euclidea in spazio normalizzato
        diff = pred_coords - gt_coords
        l_e = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)         # (B,)

        # 5. Gaussiana target per JSD e BCE (con sigma specifico del canale)
        gauss_unnorm, gauss_norm = self._make_gaussian_target(gt_coords, D, H, W, device, sigma)

        # 6. L_D: Jensen-Shannon Divergence (richiede distribuzione probabilità)
        l_d = self._jsd(prob_pred.view(B, -1), gauss_norm.view(B, -1))  # (B,)

        # 7. L_BCE_bg: supervisione diretta per-voxel contro il rumore di sfondo.
        #    Usiamo la gaussiana non normalizzata (picco a 1.0) come target binario morbido.
        #    Questo fornisce il gradiente diretto che la JSD non riesce a dare
        #    ai singoli voxel di sfondo su un volume 128³.
        bce_flat = F.binary_cross_entropy_with_logits(
            logits_c.view(B, -1),
            gauss_unnorm.view(B, -1),
            reduction='none',
        ).mean(dim=-1)                                             # (B,)

        # 8. Combinazione e media solo sui campioni validi
        combined = (l_e + self.lambda_jsd * l_d + self.lambda_bg * bce_flat) * valid.float()
        n_valid  = valid.float().sum().clamp(min=1.0)
        return combined.sum() / n_valid

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, 3, D, H, W) — logit raw (final_activation: none).
            target:      (B, 3, D, H, W) — heatmap gaussiane multi-canale.
        Returns:
            Scalar loss.
        """
        loss_R = self._channel_loss(pred_logits[:, 1], target[:, 1], self.sigma_jsd_R)
        loss_L = self._channel_loss(pred_logits[:, 2], target[:, 2], self.sigma_jsd_L)
        # Pesi asimmetrici: L più pesante perché anatomicamente più difficile
        return self.weight_R * loss_R + self.weight_L * loss_L


# ==============================================================================
# 2. LOSS FACTORY
# ==============================================================================
def get_loss(config):
    loss_name = config.get("loss", {}).get("name", "weighted_mse").lower()

    if loss_name == "mse":
        print("[*] Loss: Standard MSE (requires final_activation: softmax)")
        return nn.MSELoss()

    elif loss_name == "weighted_mse":
        fg_weight = config.get("loss", {}).get("foreground_weight", 50.0)
        print(f"[*] Loss: Foreground-Weighted MSE (foreground_weight={fg_weight}, requires softmax)")
        return WeightedMSELoss(foreground_weight=fg_weight)

    elif loss_name in ["ce", "cross_entropy"]:
        print("[*] Loss: Soft Cross-Entropy (requires final_activation: none)")
        return SoftCrossEntropyLoss()

    elif loss_name == "hybrid_ce_bce":
        fg_weight = config.get("loss", {}).get("foreground_weight", 500.0)
        ce_w  = config.get("loss", {}).get("ce_weight",  0.33)
        bce_w = config.get("loss", {}).get("bce_weight", 0.33)
        dice_w = config.get("loss", {}).get("dice_weight", 0.34)
        print(f"[*] Loss: Hybrid (CE={ce_w:.2f} + BCE={bce_w:.2f} + DICE={dice_w:.2f}) | fg_weight={fg_weight}")
        return HybridSegLoss(
            ce_weight=ce_w,
            bce_weight=bce_w,
            dice_weight=dice_w,
            foreground_weight=fg_weight
        )

    elif loss_name == "dsnt_jsd":
        # --- Guardia di compatibilità ------------------------------------------
        # DSNTJSDLoss applica spatial-softmax sui logit RAW di ogni canale.
        # Se il modello applica già channel-softmax (final_activation: softmax)
        # o sigmoid, la distribuzione spaziale sarebbe calcolata su valori già
        # normalizzati in modo incompatibile → gradienti errati silenziosamente.
        final_act = config.get("model", {}).get("final_activation", "softmax").lower()
        if final_act not in ["none", "identity"]:
            raise ValueError(
                f"\n[CONFLITTO] DSNTJSDLoss richiede final_activation: none, "
                f"trovato: '{final_act}'.\n"
                "  La loss applica internamente spatial-softmax per canale: se il modello\n"
                "  normalizza già l'output, i gradienti saranno errati.\n"
                "  → Imposta  final_activation: none  nel blocco model: del config."
            )
        # -----------------------------------------------------------------------
        lambda_jsd  = config.get("loss", {}).get("lambda_jsd",  1.0)
        sigma_jsd_R = config.get("loss", {}).get("sigma_jsd_R", 2.5)
        sigma_jsd_L = config.get("loss", {}).get("sigma_jsd_L", 4.0)
        lambda_bg   = config.get("loss", {}).get("lambda_bg",   0.5)
        temperature = config.get("loss", {}).get("temperature", 0.3)
        weight_R    = config.get("loss", {}).get("weight_R",    0.35)
        weight_L    = config.get("loss", {}).get("weight_L",    0.65)
        min_sum     = config.get("loss", {}).get("min_heatmap_sum", 0.001)
        print(
            f"[*] Loss: DSNT + JSD | λ_jsd={lambda_jsd} | σ_R={sigma_jsd_R} σ_L={sigma_jsd_L} "
            f"| λ_bg={lambda_bg} | T={temperature} | w_R={weight_R} w_L={weight_L} "
            f"| final_activation: {final_act} ✓"
        )
        return DSNTJSDLoss(
            lambda_jsd=lambda_jsd,
            sigma_jsd_R=sigma_jsd_R,
            sigma_jsd_L=sigma_jsd_L,
            lambda_bg=lambda_bg,
            temperature=temperature,
            weight_R=weight_R,
            weight_L=weight_L,
            min_heatmap_sum=min_sum,
        )

    else:
        raise ValueError(
            f"Unsupported loss '{loss_name}'. "
            "Choose from: 'mse', 'weighted_mse', 'ce', 'hybrid_ce_bce', 'dsnt_jsd'."
        )