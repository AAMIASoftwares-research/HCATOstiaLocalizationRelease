import torch
import torch.nn as nn
import torch.nn.functional as F


class DSNT3D(nn.Module):
    """
    3D Differentiable Spatial to Numerical Transform (DSNT).

    Converts 3D heatmap logits into continuous (z, y, x) coordinates via a
    differentiable soft-argmax operation.  The transform is fully differentiable
    so that coordinate-level losses can back-propagate through the network.

    Pipeline:
        1.  Flatten the spatial dimensions (D, H, W) of each channel.
        2.  Apply a temperature-scaled spatial softmax to obtain a probability
            distribution over the volume.
        3.  Compute the expected (z, y, x) coordinate as the weighted sum of
            a coordinate meshgrid against the probabilities.

    Args:
        temperature (float):  Softmax temperature.  Lower values produce
            sharper distributions; ``1.0`` leaves the logits unscaled.
        normalized_coords (bool): If True, coordinates are generated in the
            range ``[-1.0, 1.0]`` (scale-invariant, ideal for loss functions).
            If False, outputs absolute voxel indices ``[0, D-1]``.

    Input:
        heatmaps — ``(B, C, D, H, W)`` raw logits (pre-softmax).

    Returns:
        coords — ``(B, C, 3)`` predicted coordinates in the chosen space.
        probs  — ``(B, C, D, H, W)`` normalised probability maps (useful for
            visualisation or optional regularisation like JSD).

    Reference:
        Nibali et al., "Numerical Coordinate Regression with Convolutional
        Neural Networks", 2018.  arXiv:1801.07372

        Gajowczyk et al., "Coronary Ostia Localization Using Residual U-Net
        with Heatmap Matching and 3D DSNT", MLMI 2022.
    """

    def __init__(self, temperature: float = 1.0, normalized_coords: bool = False):
        super().__init__()
        self.temperature = temperature
        self.normalized_coords = normalized_coords

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, heatmaps: torch.Tensor):
        B, C, D, H, W = heatmaps.shape

        # 1. Spatial softmax → probability distribution per channel
        #    Flatten (D, H, W) into a single dimension so softmax normalises
        #    over the entire volume for each (batch, channel) pair.
        flat = heatmaps.view(B, C, -1)                          # (B, C, D*H*W)
        probs_flat = F.softmax(flat / self.temperature, dim=-1)
        probs = probs_flat.view(B, C, D, H, W)

        # 2 & 3. Compute coordinates from probabilities
        coords = self.get_coords_from_probs(probs)

        return coords, probs

    def get_coords_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the expected (z, y, x) coordinates given a spatial probability
        distribution.

        Args:
            probs: ``(B, C, D, H, W)`` normalised probability maps.
        Returns:
            coords: ``(B, C, 3)`` predicted coordinates.
        """
        B, C, D, H, W = probs.shape

        # Voxel-index or normalized meshgrid
        grid_z, grid_y, grid_x = self._make_grid(D, H, W, probs.device)

        # Spatial expectation  →  (B, C, 3) coordinates
        # E[coord] = Σ probs * grid   (summed over D, H, W)
        exp_z = torch.sum(probs * grid_z, dim=(2, 3, 4))       # (B, C)
        exp_y = torch.sum(probs * grid_y, dim=(2, 3, 4))
        exp_x = torch.sum(probs * grid_x, dim=(2, 3, 4))

        return torch.stack([exp_z, exp_y, exp_x], dim=-1)      # (B, C, 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_grid(self, D: int, H: int, W: int, device: torch.device):
        """Return ``(grid_z, grid_y, grid_x)`` each of shape ``(D, H, W)``."""
        if self.normalized_coords:
            z = torch.linspace(-1.0, 1.0, D, device=device, dtype=torch.float32)
            y = torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32)
            x = torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32)
        else:
            z = torch.arange(D, dtype=torch.float32, device=device)
            y = torch.arange(H, dtype=torch.float32, device=device)
            x = torch.arange(W, dtype=torch.float32, device=device)
            
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        return grid_z, grid_y, grid_x
