"""Backward-compatible import shim for the forecasting Lightning module."""

from mlcast.modules.forecasting import ForecastingTaskModule

ForecastingModule = ForecastingTaskModule
NowcastLightningModule = ForecastingTaskModule

__all__ = ["ForecastingModule", "ForecastingTaskModule", "NowcastLightningModule"]
