"""Backward-compatible import shim for the forecasting Lightning module."""

from mlcast.modules.forecasting import OutputSpaceForecastingTaskModule

ForecastingTaskModule = OutputSpaceForecastingTaskModule
ForecastingModule = OutputSpaceForecastingTaskModule
NowcastLightningModule = OutputSpaceForecastingTaskModule

__all__ = [
    "ForecastingModule",
    "ForecastingTaskModule",
    "NowcastLightningModule",
    "OutputSpaceForecastingTaskModule",
]
