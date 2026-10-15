import numpy
import torch

from hcatostialocalization.logger.stdout import error

def assert_last_n_dimensions_multiple_of(last_n_dims: int, multiple_of: int, object_to_check: torch.Tensor | numpy.ndarray) -> None:
    """
    Check wether each of the `last_n_dims` (int) are a clean multiple of `multiple_of` (int).
    """
    interested_shape = object_to_check.shape[-last_n_dims:]
    for axis_dim in interested_shape:
        if axis_dim % multiple_of != 0:
            err_message = error(f"Object last {last_n_dims} dimensions should be multiple of {multiple_of}, but got {interested_shape}", out=True)
            raise ValueError(err_message)

class PreprocessorCT(torch.nn.Module):
    # Preprocessed a 2d or 3d CT image
    # No learnable parameters
    def __init__(self, min_hu: int=-200, max_hu: int=800, check_multiple_of: int | None = None):
        super().__init__()
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.check_multiple_of = check_multiple_of

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects a 2d or 3d image of shapes (B, 1 or 3, K1, K2 [, K3])
        """
        # Check shape
        if (len(x.shape) != 4 and len(x.shape) != 5) or x.shape[1] not in (1, 3):
            raise ValueError(error(f"Input tensor must be of shape (B, 1 or 3, K1, K2 [, K3]), got {x.shape}", True))
        if self.check_multiple_of is not None:
            assert_last_n_dimensions_multiple_of(len(x.shape)-2, self.check_multiple_of, x)
        # Clamping values
        x = torch.clamp(x, min=self.min_hu, max=self.max_hu)
        # Normalize to [-1; 1]
        x = 2*(x - self.min_hu) / (self.max_hu - self.min_hu) - 1

        return x
        

