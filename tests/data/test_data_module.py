import functools
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from torch.utils.data import DataLoader, Dataset

from mlcast.data.source_data_datamodule import SourceDataDataModule, _validate_splits


class MockDataset(Dataset):
    """Minimal dataset mock that records how it was constructed.

    ``__len__`` returns a fixed size so that dataloader batch-count assertions
    work correctly; the test varies this via ``epoch_size``.
    """

    def __init__(
        self,
        zarr_path: str,
        subset: dict | None = None,
        augment: bool = False,
        epoch_size: int = 100,
        **kwargs,
    ) -> None:
        self.zarr_path = zarr_path
        self.subset = subset
        self.augment = augment
        self.epoch_size = epoch_size
        self.kwargs = kwargs

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        return {"data": idx}


def _mock_zarr(time_index: pd.DatetimeIndex) -> MagicMock:
    """Return a mock xr.Dataset with a given pandas DatetimeIndex for 'time'."""
    mock_ds = MagicMock()
    mock_ds.indexes = {"time": time_index}
    return mock_ds


def _make_time_index(n: int, start: str = "2016-01-01", freq: str = "10min") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


def test_validate_splits_ratio_mode() -> None:
    """Valid ratio-mode splits should not raise."""
    _validate_splits({"time": {"train": 0.7, "val": 0.15}})
    _validate_splits({"time": {"train": 0.7, "val": 0.15, "test": 0.15}})
    _validate_splits({"time": {"train": 0.7, "val": 0.15, "test": None}})


def test_validate_splits_datetime_mode() -> None:
    """Valid datetime-mode splits should not raise."""
    _validate_splits({
        "time": {
            "train": ("2016-01-01", "2021-12-31"),
            "val": ("2022-01-01", "2023-12-31"),
            "test": None,
        }
    })


def test_validate_splits_missing_train() -> None:
    with pytest.raises(ValueError, match="must contain 'train'"):
        _validate_splits({"time": {"val": 0.2}})


def test_validate_splits_ratio_exceeds_one() -> None:
    with pytest.raises(ValueError, match="sum to"):
        _validate_splits({"time": {"train": 0.8, "val": 0.3}})


def test_validate_splits_mixed_mode() -> None:
    with pytest.raises(ValueError, match="mix"):
        _validate_splits({"time": {"train": 0.7, "val": ("2022-01-01", "2023-12-31")}})


def test_validate_splits_datetime_missing_test() -> None:
    with pytest.raises(ValueError, match="must contain 'test'"):
        _validate_splits({
            "time": {
                "train": ("2016-01-01", "2021-12-31"),
                "val": ("2022-01-01", "2023-12-31"),
            }
        })


def test_validate_splits_unknown_coord() -> None:
    with pytest.raises(ValueError, match="Unknown coordinate"):
        _validate_splits({"space": {"train": 0.7, "val": 0.2}})


def test_data_module_ratio_splits() -> None:
    """DataModule ratio mode passes correct (start, end) ISO subsets to factory.

    100 timesteps, train=0.5, val=0.2:
      train_end = int(100 * 0.5) = 50
      val_end   = 50 + int(100 * 0.2) = 70
    """
    n = 100
    time_index = _make_time_index(n)
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr", foo="bar")

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={"time": {"train": 0.5, "val": 0.2}},
        batch_size=2,
    )

    with patch("mlcast.data.source_data_datamodule.xr.open_zarr", return_value=_mock_zarr(time_index)):
        dm.setup(stage="fit")

    assert dm.train_dataset.augment is True
    assert dm.train_dataset.kwargs["foo"] == "bar"

    # Verify the subset time ranges align with the expected split boundaries.
    train_start, train_end = dm.train_dataset.subset["time"]
    val_start, val_end = dm.val_dataset.subset["time"]
    test_start, test_end = dm.test_dataset.subset["time"]

    assert train_start == str(time_index[0])
    assert train_end == str(time_index[49])    # slice(0, 50) → index 49 inclusive
    assert val_start == str(time_index[50])
    assert val_end == str(time_index[69])      # slice(50, 70) → index 69 inclusive
    assert test_start == str(time_index[70])
    assert test_end == str(time_index[99])     # slice(70, 100) → index 99 inclusive

    assert dm.val_dataset.augment is False
    assert dm.test_dataset.augment is False

    train_dl = dm.train_dataloader()
    assert isinstance(train_dl, DataLoader)
    assert train_dl.batch_size == 2


def test_data_module_invalid_dataset() -> None:
    """Ensure DataModule raises if zarr_path is not accessible via the factory."""

    class _NoZarrPathFactory:
        def __call__(self, **kwargs) -> Dataset:
            return MagicMock(spec=Dataset)

    dm = SourceDataDataModule(dataset_factory=_NoZarrPathFactory())

    with pytest.raises((AttributeError, KeyError)):
        dm.setup()


def test_data_module_split_lengths_and_batches() -> None:
    """Dataloader batch counts are correct after ratio splitting.

    240 time steps, train=1/2, val=1/3:
      train_end = int(240 * 1/2)       = 120
      val_end   = 120 + int(240 * 1/3) = 200

    Each MockDataset defaults to epoch_size=100; we override via the partial so
    all three datasets return the same fixed size and we just check batches.
    """
    n_time = 240
    batch_size = 10
    time_index = _make_time_index(n_time)
    # epoch_size=10 → 1 batch each; we only care that the DL is created properly.
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr", epoch_size=10)

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={"time": {"train": 1 / 2, "val": 1 / 3}},
        batch_size=batch_size,
    )

    with patch("mlcast.data.source_data_datamodule.xr.open_zarr", return_value=_mock_zarr(time_index)):
        dm.setup()

    assert len(dm.train_dataloader()) == 1
    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 1


def test_data_module_datetime_splits() -> None:
    """DateTime-mode splits pass tuples verbatim to the factory."""
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr")

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={
            "time": {
                "train": ("2016-01-01", "2021-12-31"),
                "val": ("2022-01-01", "2023-12-31"),
                "test": None,
            }
        },
        batch_size=4,
    )

    dm.setup()

    assert dm.train_dataset.subset == {"time": ("2016-01-01", "2021-12-31")}
    assert dm.val_dataset.subset == {"time": ("2022-01-01", "2023-12-31")}
    assert dm.test_dataset is None
