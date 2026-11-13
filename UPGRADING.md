# Upgrading

This guide is a best effort at listing the steps needed to upgrade between `mlcast` releases. For the full list
of changes, see [CHANGELOG.md](CHANGELOG.md).

## v0.2.0 → Unreleased

- Pass a base config to `mlcast train`: `--config config:convgru_training_experiment` (was the implicit default
  `training_experiment`) or `--config path/to/config.yaml`.
- Replace `SourceDataIndexedDataset` with `SourceDataIndexedSequenceDataset` and set `sequence_steps`
  (`input_steps + forecast_steps`) instead of `input_steps` / `forecast_steps` / `return_mask`; those move to
  `ForecastingDataModule`, which replaces `SourceDataDataModule` and takes `sequence_dataset_factory` instead of
  `dataset_factory`. Config overrides change accordingly (e.g. `data.sequence_dataset_factory.zarr_path`).
- Replace `NowcastLightningModule` with `OutputSpaceForecastingTaskModule`; set `input_steps`,
  `forecast_steps` and `ensemble_size` on the network instead of the module.
- Monitored metric names changed from `val_loss` to `val/loss` (update callbacks and dashboards).

## v0.1.0 → v0.2.0

- Install with a PyTorch extra: `uv sync --extra cpu` (or `gpu-cu126`, `gpu-cu128`, `gpu-cu130`), or
  `pip install "mlcast[cpu]"`. PyTorch is no longer a core dependency, so a plain `uv sync` or
  `pip install mlcast` leaves it out.
- Rename `SourceDataPrecomputedSamplingDataset` to `SourceDataIndexedDataset` and its `csv_path` argument to
  `index_path`. Regenerate sampling indexes as parquet with `mlcast build-sampling-index` (legacy `.csv` indexes
  are still read as-is).
- The default experiment now loads `rainfall_flux` (was `rainfall_rate`) from `./data/sampled_datacubes.parquet`.
- `target_mask` now has shape `(1, C, H, W)`, collapsed over the whole sequence.
