import torch
import torch.nn as nn

from hcatostialocalization.model.dsnt_layer import DSNT3D


class DSNTModelWrapper(nn.Module):
    """
    Modular wrapper that adds a 3D DSNT coordinate-extraction head on top of
    any heatmap-producing backbone.

    When this wrapper is active the model returns a **dictionary**::

        {
            'heatmaps': Tensor (B, C, D, H, W),     # raw logits from the backbone
            'coords':   Tensor (B, K, 3),           # (z, y, x) in voxel indices
        }

    where ``K = len(dsnt_channels)`` (default 2: right and left ostium).

    When the wrapper is *not* used the backbone is called directly and its
    original output signature is preserved — zero impact on the existing
    pipeline.

    Args:
        base_model (nn.Module):  Any model that maps ``(B, C_in, D, H, W)``
            to ``(B, C_out, D, H, W)`` heatmap logits.
        dsnt_channels (tuple[int, ...]):  Indices of the output channels on
            which DSNT should extract coordinates.  Default ``(1, 2)``
            corresponds to right and left ostium (channel 0 = background).
        temperature (float):  Softmax temperature forwarded to
            :class:`DSNT3D`.
    """

    def __init__(
        self,
        base_model: nn.Module,
        dsnt_channels: tuple = (1, 2),
        temperature: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.dsnt_channels = list(dsnt_channels)
        self.dsnt = DSNT3D(temperature=temperature)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        heatmaps = self.base_model(x)               # (B, C, D, H, W)

        # Extract only the channels of interest for coordinate regression.
        selected = heatmaps[:, self.dsnt_channels]   # (B, K, D, H, W)
        coords, _probs = self.dsnt(selected)         # (B, K, 3)

        return {"heatmaps": heatmaps, "coords": coords}
