import torch
import torch.nn as nn
import torch.nn.functional as F

from hcatostialocalization.model.dsnt_layer import DSNT3D

# ==============================================================================
# 1. LOSS FUNCTION DEFINITIONS
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss for continuous targets (Gaussian Heatmaps).
    Computes geometric overlap between predictions and ground truth.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_probs, target):
        # pred_probs: (B, C, D, H, W) post-softmax or post-sigmoid
        # target: (B, C, D, H, W)
        
        # Flatten spatial dimensions (D, H, W) preserving Batch and Channels
        B, C = pred_probs.shape[0], pred_probs.shape[1]
        pred_flat = pred_probs.view(B, C, -1)
        target_flat = target.view(B, C, -1)

        intersection = torch.sum(pred_flat * target_flat, dim=-1)
        denominator = torch.sum(pred_flat * pred_flat, dim=-1) + torch.sum(target_flat * target_flat, dim=-1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        
        # Consider only ostium channels (1 and 2), background (channel 0) is excluded
        # so that overlap is surgically guided by points of interest.
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
    Advanced Hybrid Loss: Cross-Entropy (CE) + Weighted BCE + Soft Dice Loss.
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
        # 1. CE Term: Global competition between channels (requires raw Logits)
        loss_ce = self.ce(pred_logits, target)

        # 2. Weighted BCE Term: Voxel-per-voxel binary supervision
        bce_base = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        foreground_mask = target[:, 1:].sum(dim=1, keepdim=True)
        weight_map = 1.0 + (foreground_mask * self.fg_weight)
        loss_bce = (bce_base * weight_map).mean()

        # 3. DICE Term: Requires probabilities (apply Softmax on channel axis)
        pred_probs = F.softmax(pred_logits, dim=1)
        loss_dice = self.dice(pred_probs, target)

        # Final convex combination of the three terms
        return (self.ce_weight * loss_ce) + (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


class DSNTJSDLoss(nn.Module):
    """
    DSNT (Differentiable Spatial to Numerical Transforms) + Jensen-Shannon Divergence Loss.

    Formula from paper:
        L(Ĥ, y) = w_R · L_R  +  w_L · L_L

    For each channel (R or L):
        L_ch = L_E + λ_JSD · L_D + λ_bg · L_BCE_bg

    Where:
        L_E     = Euclidean distance in normalized space [-1,1] between predicted
                  coordinates (via DSNT) and GT coordinates.
        L_D     = JSD between predicted distribution and target Gaussian — forces
                  the heatmap to remain compact instead of spreading over the volume.
        L_BCE_bg= BCE between logits and binarized Gaussian — direct per-voxel
                  supervision to suppress background noise that JSD alone cannot
                  eliminate (gradient too weak per single voxel on a 128³ volume).
        w_R     = 0.35  (asymmetric weights: L anatomically more difficult)
        w_L     = 0.65

    Key parameters updatable externally (SigmaScheduler):
        criterion.sigma_jsd_R = new_sigma
        criterion.sigma_jsd_L = new_sigma

    Requires final_activation: none (raw logits from model).
    Crops without ostium (GT heatmap < min_heatmap_sum) are masked out and
    do not contribute to the loss.

    Args:
        lambda_jsd:      Weight of JSD term. Default: 1.0.
        sigma_jsd_R:     Initial sigma (voxels) of target Gaussian for channel R.
        sigma_jsd_L:     Initial sigma (voxels) of target Gaussian for channel L.
                         Larger than R: greater LMCA anatomical variability.
        lambda_bg:       Weight of background BCE term. Default: 0.5.
        temperature:     Spatial softmax temperature. < 1.0 → sharper distribution,
                         less background noise. Default: 0.3.
        weight_R:        Weight of channel R in total loss. Default: 0.35.
        weight_L:        Weight of channel L in total loss. Default: 0.65.
        min_heatmap_sum: Threshold below which channel is considered absent. Default: 0.001.
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
        self.sigma_jsd_R     = sigma_jsd_R   # updated via criterion.sigma_jsd_R = new_sigma
        self.sigma_jsd_L     = sigma_jsd_L   # updated via criterion.sigma_jsd_L = new_sigma
        self.lambda_bg       = lambda_bg
        self.temperature     = temperature
        self.weight_R        = weight_R
        self.weight_L        = weight_L
        self.min_heatmap_sum = min_heatmap_sum
        self.dsnt_layer      = DSNT3D(normalized_coords=True)

    # Compatibility property: criterion.sigma_jsd updates both channels
    @property
    def sigma_jsd(self) -> float:
        return (self.sigma_jsd_R + self.sigma_jsd_L) / 2.0

    @sigma_jsd.setter
    def sigma_jsd(self, value: float):
        """Allows SigmaScheduler to update both channels with a single line."""
        self.sigma_jsd_R = value
        self.sigma_jsd_L = value

    # ------------------------------------------------------------------
    # Differentiable primitives
    # ------------------------------------------------------------------

    def _spatial_softmax(self, logits_c: torch.Tensor) -> torch.Tensor:
        """Spatial softmax with temperature.

        temperature < 1.0 → more peaked distribution (less background noise).
        temperature = 1.0 → standard softmax.
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
        Generates a 3D Gaussian centered on gt_coords_norm.
        Sigma is converted from voxels to normalized space [-1,+1].
        Output:
          - gauss_unnorm: (B, D, H, W) with peak at 1.0 (for BCE).
          - gauss_norm:   (B, D, H, W) prob. distribution summing to 1 (for JSD).
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
        Jensen-Shannon Divergence between two flat distributions (B, N).
        JSD(P||Q) = 0.5·KL(P||M) + 0.5·KL(Q||M),  M = (P+Q)/2.
        Symmetric, bounded in [0, ln2]. Output: (B,).
        """
        m = 0.5 * (p + q)
        kl_pm = (p * torch.log((p + eps) / (m + eps))).sum(dim=-1)
        kl_qm = (q * torch.log((q + eps) / (m + eps))).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)

    def _channel_loss(
        self,
        logits_c: torch.Tensor,   # (B, D, H, W) — channel c logit
        target_c: torch.Tensor,   # (B, D, H, W) — channel c GT heatmap
        sigma: float,             # target Gaussian sigma for this channel
    ) -> torch.Tensor:
        """
        Computes L_E + λ_JSD·L_D + λ_bg·L_BCE for a single ostium channel.
        Samples without ostium in current crop are excluded via mask.
        """
        B, D, H, W = logits_c.shape
        device = logits_c.device

        # Mask: ostium present if sum of GT heatmap exceeds threshold
        heatmap_sum = target_c.view(B, -1).sum(dim=-1)           # (B,)
        valid = heatmap_sum > self.min_heatmap_sum                # (B,) bool

        if not valid.any():
            return (logits_c * 0.0).sum()

        # 1. Predicted distribution via spatial softmax with temperature
        prob_pred = self._spatial_softmax(logits_c)               # (B, D, H, W)

        # 2. Predicted coordinates via DSNT (center of mass of distribution)
        # We use get_coords_from_probs to align center of mass with temperature T=0.3
        # probability, so that JSD and L_E operate on the same distribution.
        pred_coords = self.dsnt_layer.get_coords_from_probs(prob_pred.unsqueeze(1))
        pred_coords = pred_coords.squeeze(1)                      # (B, 3) in [-1,+1]

        # 3. GT coordinates extracted from normalized target heatmap
        t_sum = heatmap_sum.clamp(min=1e-8).view(B, 1, 1, 1)
        prob_target = target_c / t_sum
        # prob_target is already a probability, if we use self.dsnt_layer(...) we apply a SECOND softmax, flattening GT.
        # We use get_coords_from_probs to calculate correct expectation.
        gt_coords = self.dsnt_layer.get_coords_from_probs(prob_target.unsqueeze(1))
        gt_coords = gt_coords.squeeze(1)                          # (B, 3)

        # 4. L_E: Euclidean distance in normalized space
        diff = pred_coords - gt_coords
        l_e = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)         # (B,)

        # 5. Target Gaussian for JSD and BCE (with channel-specific sigma)
        gauss_unnorm, gauss_norm = self._make_gaussian_target(gt_coords, D, H, W, device, sigma)

        # 6. L_D: Jensen-Shannon Divergence (requires probability distribution)
        l_d = self._jsd(prob_pred.view(B, -1), gauss_norm.view(B, -1))  # (B,)

        # 7. L_BCE_bg: direct per-voxel supervision against background noise.
        #    We use unnormalized Gaussian (peak at 1.0) as soft binary target.
        #    This provides direct gradient that JSD cannot provide
        #    to individual background voxels on a 128³ volume.
        bce_flat = F.binary_cross_entropy_with_logits(
            logits_c.view(B, -1),
            gauss_unnorm.view(B, -1),
            reduction='none',
        ).mean(dim=-1)                                             # (B,)

        # 8. Combination and mean only over valid samples
        combined = (l_e + self.lambda_jsd * l_d + self.lambda_bg * bce_flat) * valid.float()
        n_valid  = valid.float().sum().clamp(min=1.0)
        return combined.sum() / n_valid

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, 3, D, H, W) — raw logits (final_activation: none).
            target:      (B, 3, D, H, W) — multi-channel Gaussian heatmaps.
        Returns:
            Scalar loss.
        """
        loss_R = self._channel_loss(pred_logits[:, 1], target[:, 1], self.sigma_jsd_R)
        loss_L = self._channel_loss(pred_logits[:, 2], target[:, 2], self.sigma_jsd_L)
        # Asymmetric weights: L heavier because anatomically more difficult
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
        # --- Compatibility Guard ----------------------------------------------
        # DSNTJSDLoss applies spatial-softmax on RAW logits of each channel.
        # If the model already applies channel-softmax (final_activation: softmax)
        # or sigmoid, spatial distribution would be computed on values already
        # normalized incompatibly -> silently incorrect gradients.
        final_act = config.get("model", {}).get("final_activation", "softmax").lower()
        if final_act not in ["none", "identity"]:
            raise ValueError(
                f"\n[CONFLICT] DSNTJSDLoss requires final_activation: none, "
                f"found: '{final_act}'.\n"
                "  Loss applies spatial-softmax per channel internally: if the model\n"
                "  already normalizes output, gradients will be incorrect.\n"
                "  -> Set final_activation: none in model: block of config."
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