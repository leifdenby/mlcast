"""PyTorch Lightning data module for spatio-temporal datasets.

Handles train/val/test splitting and DataLoader creation from an injected
dataset factory.
"""

from collections.abc import Callable
from typing import Any

import fiddle as fdl
import pytorch_lightning as pl
import torch
import xarray as xr
from loguru import logger
from torch.utils.data import DataLoader, Dataset


def _safe_collate(batch: list) -> Any:
    """Collate a list of samples into a batch without shared-memory pre-allocation.

    PyTorch's default collate, when called inside a DataLoader worker process,
    pre-allocates a shared-memory buffer via ``UntypedStorage._new_using_fd_cpu``
    and then calls ``resize_()`` on it.  On some Linux configurations (e.g.
    CINECA Leonardo with PyTorch 2.11+cu128) that ``resize_()`` raises
    ``RuntimeError: Trying to resize storage that is not resizable``.

    This collate avoids that code path entirely: for tensors it calls
    ``torch.stack`` directly (no shared-memory pre-allocation); for everything
    else it delegates to the standard collate helpers.

    ``None`` entries (produced by ``_build_sample`` for malformed/edge patches)
    are filtered out before stacking.  If the entire batch is ``None`` the
    function returns ``None`` and the caller is responsible for skipping it.
    """
    # Filter out None samples (e.g. edge patches with wrong spatial size).
    batch = [s for s in batch if s is not None]
    if not batch:
        return None

    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, dim=0)
    if isinstance(elem, dict):
        return {key: _safe_collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, list | tuple):
        transposed = list(zip(*batch, strict=False))
        return type(elem)(_safe_collate(samples) for samples in transposed)
    # scalars, strings, numpy arrays — fall back to default behaviour
    from torch.utils.data.dataloader import default_collate

    return default_collate(batch)


_SPLIT_NAMES = frozenset({"train", "val", "test"})
_SUPPORTED_COORDS = frozenset({"time"})


def _validate_splits(splits: dict[str, dict[str, Any]]) -> None:
    """Validate the nested ``splits`` dict passed to :class:`SourceDataDataModule`.

    The expected structure is ``{coord: {split_name: value, ...}, ...}`` where
    each coordinate maps split names to either ``float`` fractions (ratio mode)
    or ``(start, end)`` ISO datetime tuples (datetime mode).

    Parameters
    ----------
    splits : dict
        Mapping from coordinate name to a per-split sub-dict.  Currently only
        ``"time"`` is a recognised coordinate.

    Raises
    ------
    ValueError
        If the top-level keys contain unrecognised coordinate names, if any
        per-coordinate sub-dict is missing ``"train"`` or ``"val"``, if any
        split name inside a sub-dict is not in ``{"train", "val", "test"}``,
        if ratio values within a coordinate sub-dict sum above 1.0, if ratio
        mode and datetime mode are mixed within a single coordinate sub-dict,
        or if the ``"time"`` coordinate sub-dict uses datetime mode but
        ``"test"`` is absent.
    """
    if not splits:
        raise ValueError("splits must not be empty.")

    unknown_coords = set(splits) - _SUPPORTED_COORDS
    if unknown_coords:
        raise ValueError(
            f"Unknown coordinate(s) in splits: {sorted(unknown_coords)}. " f"Supported: {sorted(_SUPPORTED_COORDS)}."
        )

    for coord, coord_splits in splits.items():
        unknown_names = set(coord_splits) - _SPLIT_NAMES
        if unknown_names:
            raise ValueError(
                f"Unknown split name(s) in splits[{coord!r}]: {sorted(unknown_names)}. "
                f"Must be one of {sorted(_SPLIT_NAMES)}."
            )

        for required in ("train", "val"):
            if required not in coord_splits:
                raise ValueError(f"splits[{coord!r}] must contain '{required}'.")

        train_is_tuple = isinstance(coord_splits["train"], tuple)
        val_is_tuple = isinstance(coord_splits["val"], tuple)
        if train_is_tuple != val_is_tuple:
            raise ValueError(
                f"Cannot mix datetime tuples and float ratios in splits[{coord!r}]. "
                "'train' and 'val' must both be floats (ratio mode) or both be "
                "(start, end) tuples (datetime mode)."
            )

        is_datetime_mode = train_is_tuple

        if not is_datetime_mode:
            # Ratio mode
            ratio_sum = coord_splits["train"] + coord_splits["val"]
            test_val = coord_splits.get("test")
            if isinstance(test_val, float):
                ratio_sum += test_val
            if ratio_sum > 1.0 + 1e-9:
                raise ValueError(f"Split ratios in splits[{coord!r}] sum to {ratio_sum:.4f}, which exceeds 1.0.")
        else:
            # Datetime mode
            if "test" not in coord_splits:
                raise ValueError(
                    f"In datetime mode splits[{coord!r}] must contain 'test' "
                    "(set to a (start, end) tuple or None to skip the test split)."
                )
            test_val = coord_splits["test"]
            if test_val is not None and not isinstance(test_val, tuple):
                raise ValueError(
                    f"In datetime mode splits[{coord!r}]['test'] must be a "
                    f"(start, end) tuple or None, got {test_val!r}."
                )


class SourceDataDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for spatio-temporal datasets.

    Handles train/val/test splitting and DataLoader creation by utilizing
    an injected ``dataset_factory``.

    Parameters
    ----------
    dataset_factory : Callable[..., Dataset]
        A factory function (e.g., ``fdl.Partial``) that returns a Dataset
        instance.  It must accept ``subset`` and ``augment`` as keyword
        arguments.
    splits : dict of {str: dict}, optional
        Nested mapping ``{coord: {split_name: value, ...}, ...}`` that defines
        how to split each coordinate into ``"train"``, ``"val"``, and
        optionally ``"test"`` subsets.  Currently only ``"time"`` is a
        supported coordinate.  ``"train"`` and ``"val"`` are required; ``"test"``
        is optional and defaults to ``None`` (no test dataset) when absent.

        Two modes are supported per coordinate:

        **Ratio mode** — values are ``float`` fractions:

        - ``"test"`` may be a float, ``None``, or absent; when absent or
          ``None`` the test split receives the remainder (``1.0 - train - val``).
        - Splits are resolved chronologically in dict insertion order to
          consecutive coordinate segments.
        - Example: ``{"time": {"train": 0.70, "val": 0.15}}``

        **Datetime mode** — values are ``(start, end)`` ISO 8601 string pairs:

        - ``"test"`` must be present; set to ``None`` for no test split.
        - Splits may overlap or have gaps.
        - Example::

              {
                  "time": {
                      "train": ("2016-01-01", "2021-12-31"),
                      "val":   ("2022-01-01", "2023-12-31"),
                      "test":  ("2024-01-01", "2025-12-31"),
                  }
              }

        Default is ``{"time": {"train": 0.70, "val": 0.15}}``.
    use_safe_collate : bool, optional
        If ``True`` (default), use :func:`_safe_collate` instead of PyTorch's
        default collate function.  This avoids a ``RuntimeError`` on CINECA
        Leonardo with PyTorch 2.11+cu128 where ``resize_()`` on a
        shared-memory storage raises ``Trying to resize storage that is not
        resizable``.  Set to ``False`` to restore default collate behaviour.
    **dataloader_kwargs : Any
        Additional keyword arguments forwarded to ``DataLoader`` (e.g.,
        ``batch_size``, ``num_workers``, ``pin_memory``).
    """

    def __init__(
        self,
        dataset_factory: Callable[..., Dataset],
        splits: dict[str, dict[str, Any]] | None = None,
        use_safe_collate: bool = True,
        **dataloader_kwargs: Any,
    ) -> None:
        super().__init__()
        self.dataset_factory = dataset_factory
        self.splits = splits if splits is not None else {"time": {"train": 0.70, "val": 0.15}}
        self.use_safe_collate = use_safe_collate
        self.dataloader_kwargs = dataloader_kwargs
        _validate_splits(self.splits)

    def setup(self, stage: str | None = None) -> None:
        """Create train, validation, and test datasets.

        For each coordinate in ``splits``, resolves split boundaries and
        assembles a ``subset`` dict for each split name.  In **ratio mode**
        the zarr store is opened to read the coordinate values and each ratio
        is resolved to a ``(start, end)`` pair of actual coordinate values.
        In **datetime mode** the tuples are passed through verbatim.  The
        ``subset`` dict is then forwarded to ``dataset_factory``.

        Parameters
        ----------
        stage : str or None, optional
            Passed by Lightning (e.g. ``"fit"``, ``"test"``); unused here.
        """
        # Build per-split subset dicts by iterating over each coordinate.
        # All three splits start active ({}); a split is disabled (set to None)
        # only when an explicit None appears in the splits config (datetime mode).
        subset_per_split: dict[str, dict[str, Any] | None] = {
            "train": {},
            "val": {},
            "test": {},
        }

        for coord, coord_splits in self.splits.items():
            is_datetime_mode = isinstance(coord_splits["train"], tuple)

            if is_datetime_mode:
                coord_values_per_split: dict[str, tuple[str, str] | None] = {
                    "train": coord_splits["train"],
                    "val": coord_splits["val"],
                    "test": coord_splits.get("test"),
                }
            else:
                # Ratio mode: open the zarr to resolve ratios to coordinate values.
                zarr_path = (
                    getattr(self.dataset_factory, "zarr_path", None) or self.dataset_factory.keywords["zarr_path"]
                )
                storage_options = getattr(
                    self.dataset_factory, "storage_options", None
                ) or self.dataset_factory.keywords.get("storage_options")
                ds = xr.open_zarr(zarr_path, storage_options=storage_options)
                coord_vals = ds.indexes[coord]
                n = len(coord_vals)

                train_ratio = coord_splits["train"]
                val_ratio = coord_splits["val"]
                test_val = coord_splits.get("test")
                test_ratio = test_val if isinstance(test_val, float) else 1.0 - train_ratio - val_ratio

                total = train_ratio + val_ratio + (test_ratio if test_ratio > 0 else 0)
                if abs(total - 1.0) > 1e-6:
                    logger.warning(
                        "splits[{!r}] ratios sum to {:.4f} (expected 1.0). "
                        "Coverage may not span the full coordinate extent.",
                        coord,
                        total,
                    )

                logger.debug(
                    "Ratio mode splits[{!r}] — train: {:.4f}, val: {:.4f}, test: {:.4f}.",
                    coord,
                    train_ratio,
                    val_ratio,
                    test_ratio,
                )

                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)

                coord_values_per_split = {
                    "train": (str(coord_vals[0]), str(coord_vals[train_end - 1])),
                    "val": (str(coord_vals[train_end]), str(coord_vals[val_end - 1])),
                    "test": (str(coord_vals[val_end]), str(coord_vals[n - 1])),
                }

            for split_name, split_val in coord_values_per_split.items():
                if split_val is None:
                    # Explicit None means no dataset for this split.
                    subset_per_split[split_name] = None
                elif subset_per_split[split_name] is not None:
                    # Still active — populate the coord entry.
                    subset_per_split[split_name][coord] = split_val
                # else: a previous coord already disabled this split; leave it None.

        augment_flags = {"train": True, "val": False, "test": False}

        for split in ("train", "val", "test"):
            subset = subset_per_split[split]
            if subset is None:
                setattr(self, f"{split}_dataset", None)
            else:
                setattr(
                    self,
                    f"{split}_dataset",
                    self.dataset_factory(subset=subset, augment=augment_flags[split]),
                )

        logger.info("{}.setup() complete, containing:", self.__class__.__name__)
        for split in ("train", "val", "test"):
            dataset = getattr(self, f"{split}_dataset", None)
            if dataset is not None:
                subset = subset_per_split[split]
                logger.info(
                    "  {:5s}: {:>6d} samples, subset={}",
                    split,
                    len(dataset),
                    subset,
                )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        kwargs = dict(self.dataloader_kwargs)
        if self.use_safe_collate:
            kwargs.setdefault("collate_fn", _safe_collate)
        return DataLoader(self.train_dataset, shuffle=True, **kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        kwargs = dict(self.dataloader_kwargs)
        if self.use_safe_collate:
            kwargs.setdefault("collate_fn", _safe_collate)
        return DataLoader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        kwargs = dict(self.dataloader_kwargs)
        if self.use_safe_collate:
            kwargs.setdefault("collate_fn", _safe_collate)
        return DataLoader(self.test_dataset, shuffle=False, **kwargs)


def count_split_samples(cfg: fdl.Config) -> dict[str, Any]:
    """Return sample counts, zarr time extent, and split config for an experiment.

    Builds the data module from ``cfg.data``, opens the zarr store to read the
    full time coordinate, calls ``setup()``, and returns a summary dict.

    Parameters
    ----------
    cfg : fdl.Config
        A Fiddle config whose ``data`` attribute is a
        :class:`SourceDataDataModule` config (i.e. the top-level experiment
        config returned by ``training_experiment.as_buildable()``).

    Returns
    -------
    dict
        A dict with the following keys:

        ``"samples"`` : dict of {str: int}
            Mapping from split name (``"train"``, ``"val"``, ``"test"``) to
            the number of samples.  Splits whose dataset is ``None`` are
            omitted.
        ``"zarr_tmin"`` : str
            ISO string of the first timestep in the zarr store.
        ``"zarr_tmax"`` : str
            ISO string of the last timestep in the zarr store.
        ``"zarr_nsteps"`` : int
            Total number of timesteps in the zarr store.
        ``"splits"`` : dict
            The ``splits`` dict from the data module config (as built).
    """
    data_module: SourceDataDataModule = fdl.build(cfg.data)

    # Open zarr to read the full time coordinate before setup() filters it.
    zarr_path = (
        getattr(data_module.dataset_factory, "zarr_path", None) or data_module.dataset_factory.keywords["zarr_path"]
    )
    storage_options = getattr(
        data_module.dataset_factory, "storage_options", None
    ) or data_module.dataset_factory.keywords.get("storage_options")
    ds = xr.open_zarr(zarr_path, storage_options=storage_options)
    time_values = ds.indexes["time"]

    data_module.setup()
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        dataset = getattr(data_module, f"{split}_dataset", None)
        if dataset is not None:
            counts[split] = len(dataset)

    return {
        "samples": counts,
        "zarr_tmin": str(time_values[0]),
        "zarr_tmax": str(time_values[-1]),
        "zarr_nsteps": len(time_values),
        "splits": data_module.splits,
    }
