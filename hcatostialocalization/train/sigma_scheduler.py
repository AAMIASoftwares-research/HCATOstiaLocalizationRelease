class SigmaScheduler:
    """
    Curriculum scheduler for the Gaussian annotation sigma.

    Progressively sharpens the heatmap target by reducing sigma as training
    loss improves. The schedule mirrors ReduceLROnPlateau: when the relative
    improvement in training loss exceeds `min_relative_improvement`, a patience
    countdown begins; after `patience` consecutive epochs the sigma is multiplied
    by `reduction_factor`. This repeats until `sigma_min` is reached, at which
    point the target degenerates to a single-voxel annotation.

    An initial warmup of `warmup_epochs` epochs keeps sigma fixed regardless of
    loss, allowing the network to first learn coarse localization before the
    targets begin to sharpen.

    Args:
        sigma_start            (float): Initial Gaussian sigma (voxels).
        sigma_min              (float): Minimum sigma; set to 0.0 for single-voxel target.
        reduction_factor       (float): Multiplicative factor applied at each reduction step.
        min_relative_improvement (float): Minimum relative loss improvement to trigger patience.
        patience               (int):   Epochs to wait after improvement before reducing sigma.
        warmup_epochs          (int):   Epochs during which sigma is held fixed unconditionally.
    """

    def __init__(
        self,
        sigma_start: float = 2.5,
        sigma_min: float = 0.0,
        reduction_factor: float = 0.75,
        min_relative_improvement: float = 0.05,
        patience: int = 2,
        warmup_epochs: int = 5,
    ):
        self.sigma              = sigma_start
        self.sigma_min          = sigma_min
        self.factor             = reduction_factor
        self.min_delta          = min_relative_improvement
        self.patience           = patience
        self.warmup_epochs      = warmup_epochs

        self._epoch             = 0
        self._ref_loss          = None   # Loss at the last sigma reduction (or first epoch).
        self._wait              = 0      # Patience counter.
        self._state             = "monitoring"  # "monitoring" | "waiting"

    def step(self, train_loss: float) -> float:
        """
        Update sigma based on the current epoch training loss.

        Must be called once per epoch. Returns the (possibly updated) sigma value.

        Args:
            train_loss (float): Mean training loss for the completed epoch.

        Returns:
            float: Current sigma after the update.
        """
        self._epoch += 1

        # Warmup phase: keep sigma fixed for the first warmup_epochs epochs.
        if self._epoch <= self.warmup_epochs:
            if self._ref_loss is None:
                self._ref_loss = train_loss
            return self.sigma

        # Initialize reference loss at the first active epoch.
        if self._ref_loss is None:
            self._ref_loss = train_loss
            return self.sigma

        # Already at minimum — nothing left to reduce.
        if self.sigma <= self.sigma_min:
            self.sigma = self.sigma_min
            return self.sigma

        relative_improvement = (self._ref_loss - train_loss) / (self._ref_loss + 1e-8)

        if self._state == "monitoring":
            if relative_improvement >= self.min_delta:
                # Sufficient improvement detected — begin patience countdown.
                self._state = "waiting"
                self._wait  = 1

        elif self._state == "waiting":
            self._wait += 1
            if self._wait >= self.patience:
                # Patience exhausted — reduce sigma and reset state.
                self.sigma     = max(self.sigma * self.factor, self.sigma_min)
                self._ref_loss = train_loss
                self._state    = "monitoring"
                self._wait     = 0

        return self.sigma

    def state_dict(self) -> dict:
        return {
            "sigma":   self.sigma,
            "epoch":   self._epoch,
            "ref_loss": self._ref_loss,
            "wait":    self._wait,
            "state":   self._state,
        }

    def load_state_dict(self, d: dict):
        self.sigma      = d["sigma"]
        self._epoch     = d["epoch"]
        self._ref_loss  = d["ref_loss"]
        self._wait      = d["wait"]
        self._state     = d["state"]
