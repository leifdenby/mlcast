"""Exponential moving average helpers for diffusion weights."""

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Track an exponential moving average of trainable module parameters.

    Parameters
    ----------
    module : nn.Module
        Module whose parameters should be tracked.
    decay : float, optional
        EMA decay factor. Default is ``0.999``.
    """

    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay ({decay}) must be in [0, 1).")
        self.module = module
        self.decay = decay
        self.shadow_params = [
            parameter.detach().clone() for parameter in module.parameters() if parameter.requires_grad
        ]
        self.backup_params: list[torch.Tensor] | None = None

    def update(self) -> None:
        """Update EMA shadow parameters from the current module parameters."""
        trainable_params = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        for shadow_param, parameter in zip(self.shadow_params, trainable_params, strict=True):
            shadow_param.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def apply(self) -> None:
        """Swap EMA parameters into the tracked module."""
        trainable_params = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        self.backup_params = [parameter.detach().clone() for parameter in trainable_params]
        for parameter, shadow_param in zip(trainable_params, self.shadow_params, strict=True):
            parameter.data.copy_(shadow_param.data)

    def restore(self) -> None:
        """Restore module parameters saved before :meth:`apply`."""
        if self.backup_params is None:
            raise RuntimeError("EMA restore() called before apply().")
        trainable_params = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        for parameter, backup_param in zip(trainable_params, self.backup_params, strict=True):
            parameter.data.copy_(backup_param.data)
        self.backup_params = None
