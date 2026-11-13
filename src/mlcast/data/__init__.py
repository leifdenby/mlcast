"""Data loading for mlcast.

The public classes are imported lazily (PEP 562) so that importing
:mod:`mlcast.data.source_data.sampling`, as the data-prep CLI commands do,
does not pull in torch and Lightning.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datamodules import ForecastingDataModule, ReconstructionDataModule
    from .forecasting import ForecastingDataset
    from .reconstruction import ReconstructionDataset
    from .sequence import (
        SourceDataIndexedSequenceDataset,
        SourceDataRandomSequenceDataset,
        SourceDataSequenceDatasetBase,
    )

__all__ = [
    "ForecastingDataModule",
    "ForecastingDataset",
    "ReconstructionDataModule",
    "ReconstructionDataset",
    "SourceDataIndexedSequenceDataset",
    "SourceDataRandomSequenceDataset",
    "SourceDataSequenceDatasetBase",
]

_LAZY_ATTRS = {
    "ForecastingDataModule": ".datamodules",
    "ReconstructionDataModule": ".datamodules",
    "ForecastingDataset": ".forecasting",
    "ReconstructionDataset": ".reconstruction",
    "SourceDataIndexedSequenceDataset": ".sequence",
    "SourceDataRandomSequenceDataset": ".sequence",
    "SourceDataSequenceDatasetBase": ".sequence",
}


def __getattr__(name: str) -> object:
    """Import the public classes on first access.

    Parameters
    ----------
    name : str
        Attribute name requested from the package.

    Returns
    -------
    object
        The lazily imported public class.

    Raises
    ------
    AttributeError
        If ``name`` is not a public attribute of the package.
    """
    if name in _LAZY_ATTRS:
        value = getattr(import_module(_LAZY_ATTRS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
