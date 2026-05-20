# LDCast Refactor Plan

0. Config naming and CLI contract
- [x] Rename `training_experiment` to `convgru_training_experiment`.
- [x] Do not keep `training_experiment` as an alias.
- [ ] Reserve `ldcast_training_experiment` as the top-level config name for the new two-stage LDCast workflow.
- [x] Require `mlcast train` users to provide an explicit base config via `--config config:<name>` or `--config /path/to/config.yaml`.
- [x] Update CLI help text to list the included config entry points explicitly.
- [x] Update all docs, examples, tests, and scripts to use `convgru_training_experiment` instead of `training_experiment`.
- [x] Treat `convgru_training_experiment` as the existing ConvGRU forecasting example, not as a special default config.

1. Forecasting and reconstruction data
- [x] Move the existing sampled-sequence source-data logic into `src/mlcast/data/sequence.py`.
- [x] Rename `SourceDataDatasetBase`, `SourceDataPrecomputedSamplingDataset`, and `SourceDataRandomSamplingDataset` to `SourceDataSequenceDatasetBase`, `SourceDataPrecomputedSequenceDataset`, and `SourceDataRandomSequenceDataset` under the sequence data area.
- [x] Remove the old source-data public API rather than keeping compatibility re-exports.
- [x] Keep the existing sampled-sequence implementation as the source-data sequence layer.
- [x] Sequence datasets should own normalization and return normalized tensors of shape `(sequence_steps, channels, height, width)`.
- [x] Replace forecasting-specific sampling parameters in the source-data sequence layer with a single `sequence_steps` parameter.
- [x] Add `src/mlcast/data/forecasting.py`.
- [x] Add a generic `ForecastingDataset` that wraps a base sequence dataset, takes `input_steps` and `forecast_steps`, validates `input_steps + forecast_steps == sequence_steps`, and returns forecasting samples.
- [x] `ForecastingDataset` should derive `target_mask` itself rather than relying on the base sequence dataset to return masks.
- [x] Add `src/mlcast/data/reconstruction.py`.
- [x] Add `ReconstructionDataset`, a generic wrapper around a base sequence dataset that slices each full sequence into all overlapping windows of length `input_steps` and returns only the tensor window.
- [x] Add `src/mlcast/data/datamodules.py`.
- [x] Rename `SourceDataDataModule` to `ForecastingDataModule` in `src/mlcast/data/datamodules.py`.
- [x] `ForecastingDataModule` should remain factory-based and build `ForecastingDataset` instances over the underlying sequence datasets.
- [x] Add `ReconstructionDataModule` to `src/mlcast/data/datamodules.py`; it remains factory-based, builds the underlying sequence datasets, splits them into train/val/test, and wraps each split with `ReconstructionDataset`.
- [x] Keep this generic: no LDCast-specific naming in the module or class names.
- [x] Forecasting should stay one-sequence-to-one-sample.
- [x] Reconstruction should expand each sequence into `sequence_steps - input_steps + 1` overlapping samples.
- [x] Stage 1 should use reconstruction windows of length `input_steps` derived from the full sequence dataset.

2. Autoencoder model architecture
- Autoencoder model split:
  - [x] `src/mlcast/models/autoencoder/encoder.py` for `Encoder` and `EncoderBlock`.
  - [x] `src/mlcast/models/autoencoder/decoder.py` for `Decoder` and `DecoderBlock`.
  - [x] `src/mlcast/models/autoencoder/net.py` for `AutoencoderNet`.
- [x] Use `input_steps` for the stage-1 reconstruction window length; do not introduce names like `autoenc_time_ratio`.
- Autoencoder validation and tests:
  - [x] encoder output shape.
  - [x] decoder output shape.
  - [x] autoencoder reconstruction forward pass.
  - [x] autoencoder improves reconstruction loss on a small generated dataset after a few training steps.

3. Forecasting model contract
- [x] Standardize all forecasting models on init-time `input_steps`, `forecast_steps`, and `ensemble_size`.
- [x] Standardize forecasting model inference on `forward(x)` only; do not pass `forecast_steps` or `ensemble_size` at runtime.
- [x] Refactor the existing ConvGRU path to follow this fixed-shape contract.
- [x] Add config consistency checks that dataset `input_steps` and `forecast_steps` match the configured forecasting model.

4. Diffusion model architecture
- Diffusion model split:
  - [ ] `src/mlcast/models/diffusion/conditioner.py` for latent conditioning blocks and `ConditionerNet`.
  - [ ] `src/mlcast/models/diffusion/denoiser.py` for `DenoiserUNet` and timestep-aware helpers.
  - [ ] `src/mlcast/models/diffusion/net.py` for `LatentDiffusionNet`.
  - [ ] `src/mlcast/models/diffusion/forecasting.py` for `LatentDiffusionForecaster`, the diffusion-specific adapter configured with fixed `input_steps`, `forecast_steps`, and `ensemble_size` and exposing `forward(x)`.
  - [ ] `src/mlcast/models/diffusion/scheduler.py`, `ema.py`, `sampler.py`, `loss.py` for diffusion support code.
- Validation and tests:
  - [ ] diffusion forecasting adapter API.
  - [ ] diffusion model improves loss on a small generated latent dataset after a few training steps.

5. Task wrappers
- [ ] Add `src/mlcast/modules/forecasting.py` and rename `NowcastLightningModule` to `ForecastingModule`.
- [ ] Remove runtime `forecast_steps` and `ensemble_size` arguments from the forecasting Lightning module and its `predict()` API.
- [ ] Add `src/mlcast/modules/reconstruction.py` with a generic `ReconstructionModule` for any reconstruction model.
- [ ] Keep `modules/` for training/task wrappers only; keep `models/` for pure architectures.

6. Training experiment
- [ ] Add a new LDCast-specific training module containing `LDCastTrainingExperiment`.
- [x] Keep `convgru_training_experiment` as the existing ConvGRU forecasting example and one of the explicitly selected included CLI configs.
- [ ] Stage 1 builds the reconstruction dataset, autoencoder model, and reconstruction module, then trains the autoencoder.
- [ ] Stage 2 reuses the same trained in-memory encoder instance, builds the diffusion dataset/model/module, then trains latent diffusion.
- [ ] The shared Fiddle graph should define the encoder once and reference the same object in both stages, but no unresolved Fiddle objects should flow into actual `torch.nn.Module.__init__` calls.
- [ ] The decoder is stage-1 only and is not shared into stage 2.
- [ ] Reuse the same forecasting dataset abstraction in stage 2; do not add a separate latent dataset layer.
- [ ] Add tests for shared object identity and stage sequencing.

7. Audit and migration targets
- [x] Update CLI/help text in `src/mlcast/__main__.py` to require an explicit base config and list the included config entry points.
- [x] Rename `training_experiment` to `convgru_training_experiment` in `src/mlcast/config/base.py` and export it from `src/mlcast/config/__init__.py`.
- [ ] Add the LDCast config entry point to `src/mlcast/config/__init__.py` alongside the existing ConvGRU example config.
- [ ] Keep `src/mlcast/config/orchestrator.py` compatible with both the existing single-stage `Experiment` and the new `LDCastTrainingExperiment` through a common `run()` surface.
- [x] Update docstrings and comments that currently imply `training_experiment` is the only experiment, including `src/mlcast/data/source_data_datamodule.py`, `src/mlcast/config/orchestrator.py`, and related config docs.
- [x] Update docs and scripts that still reference `training_experiment`, including `README.md` and `docs/generate_base_experiment_config_diagram.py`.
- [x] Keep existing ConvGRU CLI/config tests passing while adding separate tests for selecting the LDCast config explicitly.
- [ ] Add real but small-scale end-to-end tests with generated sample data for the autoencoder stage, diffusion stage, and full LDCast stage sequencing.
