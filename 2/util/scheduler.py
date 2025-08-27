import os
import math
import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineAnnealingLR(CosineAnnealingLR):
    def __init__(
        self,
        optimizer,
        T_max,
        warmup_iters,
        epoch_iters,
        warmup_type="linear",
        warmup_ratio=1e-6,
        eta_min=0,
        last_epoch=-1,
    ):
        self.T_max = T_max
        self.warmup_iters = warmup_iters * epoch_iters
        self.warmup_type = warmup_type
        self.eta_min = eta_min
        self.warmup_ratio = warmup_ratio
        self.epoch = (T_max - warmup_iters) * epoch_iters
        self.optimizer = optimizer
        self.regular_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        super(WarmupCosineAnnealingLR, self).__init__(
            optimizer, T_max=self.epoch, eta_min=eta_min, last_epoch=last_epoch
        )

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            if self.warmup_type == "constant":
                warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
            elif self.warmup_type == "linear":
                k = (1 - self.last_epoch / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
            elif self.warmup_type == "exp":
                k = self.warmup_ratio ** (1 - self.last_epoch / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in self.regular_lr]
            else:
                raise ValueError(f"Invalid warmup type: {self.warmup_type}")
            return warmup_lr
        else:
            lr = super(WarmupCosineAnnealingLR, self).get_lr()
        return lr
