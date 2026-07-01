"""PyTorch Lightning data module for spatio-temporal datasets.

Handles train/val/test splitting and DataLoader creation from an injected
dataset factory.
"""

from collections.abc import Callable, Mapping
from typing import Any

import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from mlcast.data.splits import (
    compute_split_ranges_from_splitting_ratios,
    splitting_uses_fractions,
    splitting_uses_tuple_ranges,
    validate_splits,
)
from mlcast.sampling import Sampler


class SourceDataDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for spatio-temporal datasets.

    Handles train/val/test splitting and DataLoader creation by utilizing
    an injected ``dataset_factory``.

    Parameters
    ----------
    dataset_factory : Callable[..., Dataset]
        A factory function (e.g., ``fdl.Partial``) that returns a Dataset instance.
        It must accept ``subset``, ``augment``, and ``sampler`` keyword arguments.
    splits : dict of {str: dict}
        Nested mapping ``{coord: {split_name: value, ...}, ...}`` describing
        train/val/test subsets. Currently only the ``time`` coordinate is
        supported. Ratio mode uses float fractions, while datetime mode uses
        inclusive ``(start, end)`` ISO 8601 string tuples.
    train_sampler, eval_sampler : Sampler or None, optional
        Per-split candidate sampler passed to the factory (train vs val/test),
        like ``augment``. Default ``None`` uses the full index. Keep importance
        sampling on ``train_sampler`` so val/test stay representative.
    **dataloader_kwargs : Any
        Additional keyword arguments forwarded to ``DataLoader`` (e.g.,
        ``batch_size``, ``num_workers``, ``pin_memory``).
    """

    def __init__(
        self,
        dataset_factory: Callable[..., Dataset],
        splits: dict[str, dict[str, Any]],
        train_sampler: Sampler | None = None,
        eval_sampler: Sampler | None = None,
        **dataloader_kwargs: Any,
    ) -> None:
        super().__init__()
        self.dataset_factory = dataset_factory
        self.splits = splits
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.dataloader_kwargs = dataloader_kwargs
        validate_splits(self.splits)

    @staticmethod
    def _is_combined_dataset_factory(dataset_factory: Callable[..., Dataset]) -> bool:
        dataset_factories = getattr(dataset_factory, "dataset_factories", None)
        if isinstance(dataset_factories, Mapping):
            return True

        keywords = getattr(dataset_factory, "keywords", None)
        if isinstance(keywords, Mapping):
            return isinstance(keywords.get("dataset_factories"), Mapping)

        return False

    @staticmethod
    def _get_combined_dataset_factories(dataset_factory: Callable[..., Dataset]) -> Mapping[str, Any]:
        dataset_factories = getattr(dataset_factory, "dataset_factories", None)
        if isinstance(dataset_factories, Mapping):
            return dataset_factories

        keywords = getattr(dataset_factory, "keywords", None)
        if isinstance(keywords, Mapping):
            maybe_dataset_factories = keywords.get("dataset_factories")
            if isinstance(maybe_dataset_factories, Mapping):
                return maybe_dataset_factories

        raise TypeError("dataset_factory does not expose combined dataset factories")

    @staticmethod
    def _resolve_subset_per_split(
        dataset_factory: Callable[..., Dataset],
        splits: dict[str, dict[str, Any]],
        requested_splits: set[str],
    ) -> dict[str, dict[str, Any] | None]:
        subset_per_split: dict[str, dict[str, Any] | None] = {
            split_name: (
                {}
                if split_name in requested_splits
                and any(split_name in coord_splits for coord_splits in splits.values())
                else None
            )
            for split_name in ("train", "val", "test")
        }

        for coord, coord_splits in splits.items():
            if splitting_uses_tuple_ranges(coord_splits):
                # tuple-based splits are expected to present the start and end
                # of each split, and so are passed through directly as the
                # subset values for each split
                coord_values_per_split: dict[str, tuple[str, str] | None] = {
                    "train": coord_splits["train"],
                    "val": coord_splits["val"],
                    "test": coord_splits.get("test"),
                }
            elif splitting_uses_fractions(coord_splits):
                # for ratio-based splits, the splitting start-end range tuples
                # are constructed by breaking up the given coordinate in
                # successive segments (the succession is defined from the order
                # of the keys in the splits dict)
                coord_values_per_split = compute_split_ranges_from_splitting_ratios(
                    dataset_factory,
                    coord,
                    coord_splits,
                )
            else:
                raise NotImplementedError(f"Unsupported split mode for coordinate {coord!r}: {coord_splits!r}")

            for split_name, split_val in coord_values_per_split.items():
                if split_name not in requested_splits:
                    continue
                if split_val is None:
                    subset_per_split[split_name] = None
                elif subset_per_split[split_name] is not None:
                    subset_per_split[split_name][coord] = split_val

        return subset_per_split

    def _resolve_split_subsets(self, requested_splits: set[str]) -> dict[str, dict[str, Any] | None]:
        if not self._is_combined_dataset_factory(self.dataset_factory):
            return self._resolve_subset_per_split(self.dataset_factory, self.splits, requested_splits)

        dataset_factories = self._get_combined_dataset_factories(self.dataset_factory)
        missing_dataset_names = set(dataset_factories) - set(self.splits)
        extra_dataset_names = set(self.splits) - set(dataset_factories)
        if missing_dataset_names or extra_dataset_names:
            raise ValueError(
                "Combined dataset splits must use the same dataset names as the combined factory; "
                f"missing={sorted(missing_dataset_names)}, extra={sorted(extra_dataset_names)}"
            )

        per_dataset_subset_per_split = {}
        for dataset_name, child_factory in dataset_factories.items():
            per_dataset_subset_per_split[dataset_name] = self._resolve_subset_per_split(
                child_factory,
                self.splits[dataset_name],
                requested_splits,
            )

        subset_per_split: dict[str, dict[str, Any] | None] = {
            split_name: ({} if split_name in requested_splits else None) for split_name in ("train", "val", "test")
        }

        for split_name in ("train", "val", "test"):
            if split_name not in requested_splits:
                continue

            for dataset_name, dataset_subset_per_split in per_dataset_subset_per_split.items():
                split_subset = dataset_subset_per_split[split_name]
                if split_subset is None:
                    subset_per_split[split_name] = None
                    break
                subset_per_split[split_name][dataset_name] = split_subset

        return subset_per_split

    def setup(self, stage: str | None = None) -> None:
        """Create train, validation, and test datasets.

        Splits are assembled into per-dataset ``subset`` dictionaries.
        Datetime-mode splits are passed through unchanged, while ratio-mode
        splits are first resolved against the zarr coordinate values and then
        converted to inclusive coordinate ranges before dataset instantiation.
        Dataset construction depends on the requested Lightning stage:

        - ``"fit"`` builds train and validation datasets;
        - ``"validate"`` builds only the validation dataset;
        - ``"test"`` builds only the test dataset; and
        - ``None`` builds all configured datasets.

        Parameters
        ----------
        stage : str | None, optional
            Lightning stage hint controlling which datasets are constructed.

        Raises
        ------
        ValueError
            If ``stage`` is not one of ``None``, ``"fit"``, ``"validate"``,
            or ``"test"``.
        """
        if stage == "fit":
            requested_splits = {"train", "val"}
        elif stage == "validate":
            requested_splits = {"val"}
        elif stage == "test":
            requested_splits = {"test"}
        elif stage is None:
            requested_splits = {"train", "val", "test"}
        else:
            raise ValueError(f"Unsupported LightningDataModule setup stage: {stage!r}")

        augment_flags = {"train": True, "val": False, "test": False}
        sampler_flags = {"train": self.train_sampler, "val": self.eval_sampler, "test": self.eval_sampler}
        subset_per_split = self._resolve_split_subsets(requested_splits)
        for split in ("train", "val", "test"):
            subset = subset_per_split[split]
            if subset is None:
                setattr(self, f"{split}_dataset", None)
            else:
                dataset = self.dataset_factory(
                    subset=subset,
                    augment=augment_flags[split],
                    sampler=sampler_flags[split],
                )
                setattr(self, f"{split}_dataset", dataset)

        logger.info("{}.setup() complete, containing:", self.__class__.__name__)
        for split in ("train", "val", "test"):
            dataset = getattr(self, f"{split}_dataset", None)
            if dataset is not None:
                logger.info(
                    "  {:5s}: {:>6d} samples, subset={}",
                    split,
                    len(dataset),
                    subset_per_split[split],
                )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_kwargs)
