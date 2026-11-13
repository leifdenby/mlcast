from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import xarray as xr
from torch.utils.data import Dataset

from mlcast.data.forecasting import ForecastingDataset
from mlcast.data.reconstruction import ReconstructionDataset
from mlcast.data.sequence import SourceDataIndexedSequenceDataset, SourceDataRandomSequenceDataset
from mlcast.data.source_data.sampling import ImportanceSelector, UniformSelector
from mlcast.data.source_data.sampling.sampling_index_spec import SamplingParameters, build_schema


class MockSequenceDataset(Dataset):
    def __init__(self, sequence_steps: int, num_samples: int = 2) -> None:
        self.sequence_steps = sequence_steps
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.arange(self.sequence_steps, dtype=torch.float32)[:, None, None, None].expand(-1, 1, 2, 2)


class FixedSequenceDataset(Dataset):
    """Sequence dataset returning one fixed tensor (e.g. containing NaNs)."""

    def __init__(self, sequence: torch.Tensor) -> None:
        self.sequence = sequence
        self.sequence_steps = sequence.shape[0]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequence.clone()


@pytest.fixture
def mock_csv(tmp_path: Path) -> str:
    """Create a temporary CSV file with coordinates."""
    df = pd.DataFrame(
        {
            "t": [0, 5, 10],
            "x": [10, 20, 30],
            "y": [10, 20, 30],
        }
    )
    csv_path = tmp_path / "coords.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_parquet(tmp_path: Path) -> str:
    """Create a temporary sampling index (build-sampling-index output) with a mean column."""
    meta = SamplingParameters(
        zarr_path="dummy.zarr",
        data_var="RR",
        time_var="time",
        start_date="2016-01-01",
        end_date="2016-12-31",
        time_step_minutes=5,
        time_depth=2,
        width=16,
        height=16,
        step_t=1,
        step_x=1,
        step_y=1,
        max_nan=0,
        wet_threshold=0.1,
        data_kind="rainrate",
        units="mm/h",
    )
    table = pa.table(
        {
            "t": pa.array([0, 5, 10], pa.int32()),
            "x": pa.array([10, 20, 30], pa.int32()),
            "y": pa.array([10, 20, 30], pa.int32()),
            "nan_count": pa.array([0, 0, 0], pa.int32()),
            "sum": pa.array([1.0, 2.0, 3.0], pa.float32()),
            "mean": pa.array([0.05, 2.0, 10.0], pa.float32()),
            "frac_wet": pa.array([0.0, 0.1, 0.5], pa.float32()),
        },
        schema=build_schema(meta),
    )
    parquet_path = tmp_path / "stats.parquet"
    pq.write_table(table, parquet_path)
    return str(parquet_path)


def test_indexed_sequence_dataset_parquet(fp_test_dataset: Path, mock_parquet: str) -> None:
    """Reads a sampling index; with no selector the full pool is used."""
    sequence_steps = 3
    ds = SourceDataIndexedSequenceDataset(
        zarr_path=str(fp_test_dataset),
        index_path=mock_parquet,
        standard_names=["rainfall_flux"],
        sequence_steps=sequence_steps,
        width=16,
        height=16,
    )

    assert len(ds) == 3  # all candidates, as-is
    sample = ds[0]
    assert sample.shape == (sequence_steps, 1, 16, 16)
    assert sample.dtype == torch.float32


def test_indexed_importance_selection_is_fixed_and_keeps_extremes(fp_test_dataset: Path, mock_parquet: str) -> None:
    """ImportanceSelector selects a fixed, reproducible subset that keeps extremes."""
    kwargs = dict(
        zarr_path=str(fp_test_dataset),
        index_path=mock_parquet,
        standard_names=["rainfall_flux"],
        sequence_steps=3,
        width=16,
        height=16,
        selector=ImportanceSelector(),
        selection_seed=0,
    )
    ds = SourceDataIndexedSequenceDataset(**kwargs)

    # a subset of the 3 candidates, with the wettest (t=10) always kept
    assert 1 <= len(ds) <= 3
    assert 10 in ds.candidates["t"].to_numpy()
    # reproducible: same seed -> identical kept set
    ds2 = SourceDataIndexedSequenceDataset(**kwargs)
    assert np.array_equal(ds.candidates["t"].to_numpy(), ds2.candidates["t"].to_numpy())


def test_indexed_importance_selector_requires_mean_column(fp_test_dataset: Path, mock_csv: str) -> None:
    """ImportanceSelector rejects a CSV index that has no mean column."""
    with pytest.raises(ValueError, match="mean"):
        SourceDataIndexedSequenceDataset(
            zarr_path=str(fp_test_dataset),
            index_path=mock_csv,
            standard_names=["rainfall_flux"],
            sequence_steps=3,
            width=16,
            height=16,
            selector=ImportanceSelector(),
        )


def test_indexed_uniform_selector_works_on_csv(fp_test_dataset: Path, mock_csv: str) -> None:
    """A non-importance selector (uniform) needs no mean column, so works on a CSV."""
    ds = SourceDataIndexedSequenceDataset(
        zarr_path=str(fp_test_dataset),
        index_path=mock_csv,
        standard_names=["rainfall_flux"],
        sequence_steps=3,
        width=16,
        height=16,
        selector=UniformSelector(keep_fraction=1.0),
    )
    assert len(ds) == 3  # keep_fraction=1.0 -> the whole (3-row) index


def test_indexed_sequence_dataset(fp_test_dataset: Path, mock_csv: str) -> None:
    """Indexed sequence dataset should output normalized sequence tensors."""
    sequence_steps = 3
    ds = SourceDataIndexedSequenceDataset(
        zarr_path=str(fp_test_dataset),
        index_path=mock_csv,
        standard_names=["rainfall_flux"],
        sequence_steps=sequence_steps,
        width=16,
        height=16,
    )

    assert len(ds) == 3
    sample = ds[0]
    assert sample.shape == (sequence_steps, 1, 16, 16)
    assert sample.dtype == torch.float32


def test_indexed_sequence_dataset_time_subset(fp_test_dataset: Path, mock_csv: str) -> None:
    """Subset keeps only index rows whose full window fits the subset, and
    rebases their absolute ``t`` onto the sliced (0-based) store."""
    zarr_ds = xr.open_zarr(str(fp_test_dataset))
    time_index = zarr_ds.indexes["time"]
    # mock_csv has t = [0, 5, 10]; subset [3, 21) with time_depth 3 drops t=0
    # (before the start) and keeps t=5, 10, rebased onto the slice to [2, 7].
    ds = SourceDataIndexedSequenceDataset(
        zarr_path=str(fp_test_dataset),
        index_path=mock_csv,
        standard_names=["rainfall_flux"],
        sequence_steps=3,
        time_depth=3,
        subset={"time": (str(time_index[3]), str(time_index[20]))},
    )
    assert len(ds) == 2
    assert sorted(ds.candidates["t"].tolist()) == [2, 7]


def test_indexed_sequence_dataset_sequence_steps_guard(fp_test_dataset: Path, mock_csv: str) -> None:
    """Instantiation with sequence_steps=0 should raise ValueError."""
    with pytest.raises(ValueError, match="sequence_steps"):
        SourceDataIndexedSequenceDataset(
            zarr_path=str(fp_test_dataset),
            index_path=mock_csv,
            standard_names=["rainfall_flux"],
            sequence_steps=0,
        )


def test_random_sequence_dataset(fp_test_dataset: Path) -> None:
    """Random sequence dataset should output normalized sequence tensors."""
    sequence_steps = 5
    ds = SourceDataRandomSequenceDataset(
        zarr_path=str(fp_test_dataset),
        standard_names=["rainfall_flux"],
        sequence_steps=sequence_steps,
        width=32,
        height=32,
        epoch_size=10,
    )

    assert len(ds) == 10
    sample = ds[0]
    assert sample.shape == (sequence_steps, 1, 32, 32)
    assert sample.dtype == torch.float32


def test_random_sequence_dataset_time_subset(fp_test_dataset: Path) -> None:
    """Subset should correctly slice the Zarr store."""
    zarr_ds = xr.open_zarr(str(fp_test_dataset))
    time_index = zarr_ds.indexes["time"]
    ds = SourceDataRandomSequenceDataset(
        zarr_path=str(fp_test_dataset),
        standard_names=["rainfall_flux"],
        sequence_steps=5,
        subset={"time": (str(time_index[0]), str(time_index[49]))},
        epoch_size=10,
    )

    assert ds.max_t == 50
    assert len(ds) == 10


def test_forecasting_dataset_splits_sequence_and_derives_mask() -> None:
    """ForecastingDataset should split one sequence into input and target tensors."""
    base_dataset = MockSequenceDataset(sequence_steps=6)
    dataset = ForecastingDataset(base_dataset, input_steps=2, forecast_steps=4, return_mask=True)

    sample = dataset[0]
    assert sample["input"].shape == (2, 1, 2, 2)
    assert sample["target"].shape == (4, 1, 2, 2)
    assert sample["target_mask"].shape == (1, 1, 2, 2)  # collapsed over the sequence
    assert torch.all(sample["target_mask"] == 1.0)


def test_target_mask_collapses_over_sequence() -> None:
    """A cell that is NaN at *any* step (input or target) must be masked across
    every forecast step — a temporal discontinuity makes the trajectory there
    ill-defined, so the cell is not scored anywhere in the sequence."""
    input_steps, forecast_steps = 3, 2
    steps = input_steps + forecast_steps
    sequence = torch.zeros((steps, 1, 8, 8), dtype=torch.float32)
    sequence[1, 0, 2, 3] = torch.nan  # NaN at one input step
    sequence[steps - 1, 0, 5, 5] = torch.nan  # NaN at the last forecast step
    dataset = ForecastingDataset(
        FixedSequenceDataset(sequence), input_steps=input_steps, forecast_steps=forecast_steps, return_mask=True
    )

    sample = dataset[0]
    mask = sample["target_mask"].numpy()

    assert mask.shape == (1, 1, 8, 8)  # collapsed over the sequence
    assert mask[0, 0, 2, 3] == 0  # NaN at an input step masks the cell
    assert mask[0, 0, 5, 5] == 0  # NaN at a forecast step masks the cell
    assert mask[0, 0, 0, 0] == 1  # a cell valid at every step stays valid
    assert not torch.isnan(sample["input"]).any()
    assert not torch.isnan(sample["target"]).any()


def test_reconstruction_dataset_creates_overlapping_windows() -> None:
    """ReconstructionDataset should expose all overlapping windows."""
    base_dataset = MockSequenceDataset(sequence_steps=5, num_samples=2)
    dataset = ReconstructionDataset(base_dataset, input_steps=3)

    assert len(dataset) == 6
    first_window = dataset[0]
    second_window = dataset[1]
    assert first_window.shape == (3, 1, 2, 2)
    assert torch.equal(first_window[:, 0, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(second_window[:, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
