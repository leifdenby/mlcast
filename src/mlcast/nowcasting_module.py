"""Generic Lightning module for radar precipitation nowcasting.

Wraps an injected PyTorch :class:`nn.Module` (the network architecture) and
handles training, validation, and test steps including loss computation,
image logging, and optimizer configuration.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from mlcast.data.normalization import DENORMALIZATION_REGISTRY, NORMALIZATION_REGISTRY
from mlcast.losses import build_loss
from mlcast.visualization import log_images


class NowcastLightningModule(pl.LightningModule):
    """Generic PyTorch Lightning module for nowcasting.

    Wraps an injected PyTorch `nn.Module` (the network architecture) and
    handles training, validation, test steps, loss computation, ensemble
    generation, and TensorBoard logging.

    Parameters
    ----------
    network : torch.nn.Module
        The PyTorch network architecture to train.
    loss_class : type[torch.nn.Module] or str, optional
        Loss function class or its string name. Default is ``"mse"``.
    loss_params : dict or None, optional
        Keyword arguments for the loss constructor. Default is ``None``.
    masked_loss : bool, optional
        Whether to wrap the loss with :class:`MaskedLoss`. Default is ``False``.
    optimizer : Callable[..., torch.optim.Optimizer] or None, optional
        A callable (e.g., a ``functools.partial``) that takes network parameters
        and returns an instantiated optimizer. Default is ``None`` (Adam).
    lr_scheduler : Callable[..., torch.optim.lr_scheduler.LRScheduler] or None, optional
        A callable (e.g., a ``functools.partial``) that takes an optimizer
        and returns an instantiated learning rate scheduler. Default is ``None``.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        loss_class: type[torch.nn.Module] | str = "mse",
        loss_params: dict[str, Any] | None = None,
        masked_loss: bool = False,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__()
        # Explicitly save hyperparameters that are accessed later via self.hparams
        self.save_hyperparameters("loss_class", "loss_params", "masked_loss")

        self.network = network
        self.optimizer_factory = optimizer
        self.lr_scheduler_factory = lr_scheduler

        self.criterion = build_loss(
            loss_class=loss_class,
            loss_params=loss_params,
            masked_loss=masked_loss,
        )
        self.log_images_iterations = [50, 100, 200, 500, 750, 1000, 2000, 5000]

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch time channels height width"],
    ) -> Float[torch.Tensor, "batch forecast_steps out_channels height width"]:
        """Run the network forward pass.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Input tensor.

        Returns
        -------
        preds : Float[torch.Tensor, "batch forecast_steps out_channels height width"]
            Forecast tensor.
        """
        return self.network(x)

    def shared_step(self, batch: dict[str, torch.Tensor], split: str = "train") -> torch.Tensor:
        """Shared forward step for training, validation, and testing.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            A dictionary containing the batched input data. Must contain the
            key ``"data"`` and optionally ``"mask"`` if ``masked_loss`` is ``True``.
        split : str, optional
            The data split being processed (e.g., ``"train"``, ``"val"``, ``"test"``).
            Used for logging. Default is ``"train"``.
        Returns
        -------
        loss : torch.Tensor
            The computed loss for the batch.
        """
        past = batch["input"]
        future = batch["target"]

        preds = self(past).clamp(min=-1, max=1)

        if self.hparams["masked_loss"]:
            mask = batch["target_mask"]
            loss = self.criterion(preds, future, mask)
        else:
            loss = self.criterion(preds, future)

        if isinstance(loss, tuple):
            loss, log_dict = loss
            self.log_dict(
                log_dict, prog_bar=False, logger=True, on_step=(split == "train"), on_epoch=True, sync_dist=True
            )

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, on_step=(split == "train"), sync_dist=True)

        ensemble_size = getattr(self.network, "ensemble_size", 1)
        if ensemble_size > 1:
            ensemble_std = preds.std(dim=2).mean()
            self.log(f"{split}_ensemble_std", ensemble_std, on_epoch=True, sync_dist=True)

        if (
            split == "train"
            and self.logger is not None
            and getattr(self.logger, "experiment", None) is not None
            and (
                self.global_step in self.log_images_iterations or self.global_step % self.log_images_iterations[-1] == 0
            )
        ):
            log_images(
                past=past,
                future=future,
                preds=preds,
                logger=self.logger,  # type: ignore
                global_step=self.global_step,
                ensemble_size=ensemble_size,
                split=split,
            )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            A dictionary containing the batched input data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        loss : torch.Tensor
            The training loss.
        """
        return self.shared_step(batch, split="train")

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Execute a single validation step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            A dictionary containing the batched input data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        loss : torch.Tensor
            The validation loss.
        """
        return self.shared_step(batch, split="val")

    def test_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Execute a single test step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            A dictionary containing the batched input data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        loss : torch.Tensor
            The test loss.
        """
        return self.shared_step(batch, split="test")

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and optional learning rate scheduler.

        Returns
        -------
        config : dict of str to Any
            A dictionary containing the instantiated ``"optimizer"`` and
            optionally ``"lr_scheduler"`` configurations for PyTorch Lightning.
        """
        if self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(self.parameters())
        else:
            optimizer = torch.optim.Adam(self.parameters())

        if self.lr_scheduler_factory is not None:
            lr_scheduler = self.lr_scheduler_factory(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}
        else:
            return {"optimizer": optimizer}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "NowcastLightningModule":
        """Load a model from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to the saved PyTorch Lightning checkpoint (``.ckpt``) file.
        device : str, optional
            The device to map the model weights to (e.g., ``"cpu"`` or ``"cuda"``).
            Default is ``"cpu"``.

        Returns
        -------
        model : NowcastLightningModule
            The loaded PyTorch Lightning model instance.
        """
        return cls.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device(device),
            strict=True,
            weights_only=False,
        )

    def predict(
        self,
        past: torch.Tensor,
        standard_name: str = "rainfall_rate",
    ) -> np.ndarray[Any, Any]:
        """Generate precipitation forecasts from past radar observations.

        Input should be raw unnormalized values.

        Parameters
        ----------
        past : torch.Tensor
            Past radar frames as unnormalized values (e.g., mm/h or kg m-2 s-1), of shape ``(T, H, W)``.
        standard_name : str, optional
            The CF standard name defining the input/output domain for normalization lookup.
            Default is ``"rainfall_rate"``.

        Returns
        -------
        preds : np.ndarray
            Forecasted unnormalized values, of shape
            ``(ensemble_size, forecast_steps, H, W)``. The ensemble size and
            forecast horizon are determined by the configured network.
        """
        if len(past.shape) != 3:
            raise ValueError("Input must be of shape (T, H, W)")

        past_clean = np.nan_to_num(past.cpu().numpy())
        past_clean = past_clean[np.newaxis, :, np.newaxis, ...]

        norm_func = NORMALIZATION_REGISTRY[standard_name]
        norm_past = norm_func(past_clean)

        x = torch.from_numpy(norm_past)
        x = x.to(self.device)

        self.eval()
        with torch.no_grad():
            preds_tensor = self.network(x)

        preds_np: np.ndarray[Any, Any] = preds_tensor.cpu().numpy()

        denorm_func = DENORMALIZATION_REGISTRY[standard_name]
        preds_np = denorm_func(preds_np)

        preds_np = preds_np.squeeze(0)
        preds_np = np.swapaxes(preds_np, 0, 1)

        return preds_np
