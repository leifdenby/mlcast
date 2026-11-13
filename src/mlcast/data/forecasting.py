"""Forecasting dataset wrappers built on top of sequence datasets."""

from typing import TypedDict

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.utils.data import Dataset


class ForecastingSample(TypedDict, total=False):
    """Typed dictionary returned by :class:`ForecastingDataset`.

    Keys
    ----
    input : Float[Tensor, "input_steps channels height width"]
        Past frames fed to the forecasting model.
    target : Float[Tensor, "forecast_steps channels height width"]
        Future frames the forecasting model should predict.
    target_mask : Float[Tensor, "1 channels height width"]
        Per-cell validity mask collapsed over the whole sequence: a cell is 1
        only if it was finite at every step (inputs and targets), else 0. Its
        leading axis is a single step that the loss broadcasts over the forecast
        steps. Only present when ``return_mask=True``.
    """

    input: Float[Tensor, "input_steps channels height width"]
    target: Float[Tensor, "forecast_steps channels height width"]
    target_mask: Float[Tensor, "1 channels height width"]


class ForecastingDataset(Dataset):
    """Wrap a sequence dataset to produce forecasting samples.

    Parameters
    ----------
    base_sequence_dataset : Dataset
        Dataset returning normalized sequence tensors of shape
        ``(sequence_steps, channels, height, width)``.
    input_steps : int
        Number of past timesteps fed to the forecasting model.
    forecast_steps : int
        Number of future timesteps the forecasting model should predict.
    return_mask : bool, optional
        Whether to derive and return a target validity mask. Default is
        ``False``.
    """

    def __init__(
        self,
        base_sequence_dataset: Dataset,
        input_steps: int,
        forecast_steps: int,
        return_mask: bool = False,
    ) -> None:
        if input_steps < 1:
            raise ValueError(f"input_steps ({input_steps}) must be at least 1.")
        if forecast_steps < 1:
            raise ValueError(f"forecast_steps ({forecast_steps}) must be at least 1.")

        self.base_sequence_dataset = base_sequence_dataset
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.return_mask = return_mask

        sequence_steps = getattr(base_sequence_dataset, "sequence_steps", None)
        if sequence_steps is None:
            raise AttributeError("base_sequence_dataset must expose a 'sequence_steps' attribute.")
        if input_steps + forecast_steps != sequence_steps:
            raise ValueError(
                "ForecastingDataset requires input_steps + forecast_steps to equal sequence_steps; "
                f"got input_steps={input_steps}, forecast_steps={forecast_steps}, sequence_steps={sequence_steps}."
            )

    def __len__(self) -> int:
        """Return the number of available forecasting samples.

        Returns
        -------
        int
            Number of samples in the wrapped sequence dataset.
        """
        return len(self.base_sequence_dataset)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> ForecastingSample:
        """Return one forecasting sample derived from the wrapped sequence.

        Parameters
        ----------
        idx : int
            Index of the wrapped sequence sample.

        Returns
        -------
        ForecastingSample
            Dictionary containing ``input`` and ``target`` tensors, and
            ``target_mask`` when ``return_mask=True``.
        """
        sequence = self.base_sequence_dataset[idx]

        # Validity mask, collapsed over the whole sequence: a cell is scored only
        # if it is finite at EVERY step (inputs and targets). A temporal
        # discontinuity anywhere makes the forecast trajectory at that cell
        # ill-defined (and the temporal-consistency loss term meaningless), so we
        # mask it for the whole sequence. Kept as (1, C, H, W); the loss broadcasts
        # it over the forecast steps. Computed before NaNs are filled below.
        if self.return_mask:
            target_mask_t = (~torch.isnan(sequence).any(dim=0, keepdim=True)).to(dtype=torch.float32)

        sequence = torch.nan_to_num(sequence, nan=-1.0).to(dtype=torch.float32)
        input_t = sequence[: self.input_steps]
        target_t = sequence[self.input_steps :]

        sample = ForecastingSample(input=input_t, target=target_t)
        if self.return_mask:
            sample["target_mask"] = target_mask_t
        return sample
