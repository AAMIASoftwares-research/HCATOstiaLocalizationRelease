import numpy as np
import torch

class SchedulerLR():
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """
    def __init__(self, lr_ref, max_training_steps, warmup_steps=0):
        self.lr_ref = lr_ref
        self.lr = lr_ref
        self.max_training_steps = max_training_steps
        self.warmup_steps = warmup_steps
        
        # first steps internal state
        self._start_default_warmup_steps = 100
        self._start_warmpup_s = 0

    def _update(self, i: int):
        if i < self.warmup_steps and self.warmup_steps > 0:
            lr = self.lr_ref * (i + 1) / self.warmup_steps
        else:
            lr = self.lr_ref * 0.5 * (1 + np.cos(np.pi * (i - self.warmup_steps) / (self.max_training_steps - self.warmup_steps)))
        
        lr = max(lr, 10e-9)
        self.lr = lr

    def _update_lr_with_default_warmup_mutliplier(self):
        # no matter what the rule, when starting up,
        # also after resuming a training, the learning rate has
        # an internal warmup. This is very good for the optimizer.
        if self._start_warmpup_s < self._start_default_warmup_steps:
            self.lr *= max(0.0, min(1.0, self._start_warmpup_s / self._start_default_warmup_steps))
            self._start_warmpup_s += 1
        
    def _update_optimizer_lr(self, optimizer: torch.optim.Optimizer, lr: float):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def step_update_optimizer_lr(self, i: int, optimizer: torch.optim.Optimizer):
        self._update(i)
        self._update_lr_with_default_warmup_mutliplier()
        self._update_optimizer_lr(optimizer, self.lr)

    def __call__(self, *args, **kwds):
        return self.lr