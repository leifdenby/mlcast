# LDCast Refactor Plan

0. Config naming and CLI contract
- [ ] Rename `training_experiment` to `convgru_training_experiment`.
- [ ] Do not keep `training_experiment` as an alias.
- [ ] Reserve `ldcast_training_experiment` as the top-level config name for the new two-stage LDCast workflow.
- [ ] Require `mlcast train` users to provide an explicit base config via `--config config:<name>` or `--config /path/to/config.yaml`.
- [ ] Update CLI help text to list the included config entry points explicitly.
- [ ] Update all docs, examples, tests, and scripts to use `convgru_training_experiment` instead of `training_experiment`.
- [ ] Treat `convgru_training_experiment` as the existing ConvGRU forecasting example, not as a special default config.

1. Forecasting and reconstruction data
- [ ] Rename the existing sampled-sequence dataset classes into `src/mlcast/data/forecasting.py`.
- [ ] Rename `SourceDataDatasetBase`, `SourceDataPrecomputedSamplingDataset`, and `SourceDataRandomSamplingDataset` into forecasting-oriented names under the forecasting data area.
- [ ] Remove the old source-data public API rather than keeping compatibility re-exports.
- [ ] Keep the existing sampled-sequence dataset implementation as the forecasting data source.
- [ ] Add `src/mlcast/data/reconstruction.py`.
- [ ] Add `ReconstructionDataset`, a thin wrapper around a `base_forecasting_dataset` that returns only the input tensor `x`.
- [ ] Add `ReconstructionDataModule`, which remains factory-based, builds the underlying forecasting datasets, splits them into train/val/test, and wraps each split with `ReconstructionDataset`.
- [ ] Keep this generic: no LDCast-specific naming in the module or class names.
- [ ] Stage 1 should use only the input window from the forecasting dataset.
- [ ] Allow `forecast_steps == 0` in forecasting datasets, but emit a warning when used.

2. Autoencoder model architecture
- Autoencoder model split:
  - [ ] `src/mlcast/models/autoencoder/encoder.py` for `Encoder` and `EncoderBlock`.
  - [ ] `src/mlcast/models/autoencoder/decoder.py` for `Decoder` and `DecoderBlock`.
  - [ ] `src/mlcast/models/autoencoder/net.py` for `AutoencoderNet`.
- Autoencoder validation and tests:
  - [ ] encoder output shape.
  - [ ] decoder output shape.
  - [ ] autoencoder reconstruction forward pass.
  - [ ] autoencoder improves reconstruction loss on a small generated dataset after a few training steps.

3. Diffusion model architecture
- Diffusion model split:
  - [ ] `src/mlcast/models/diffusion/conditioner.py` for latent conditioning blocks and `ConditionerNet`.
  - [ ] `src/mlcast/models/diffusion/denoiser.py` for `DenoiserUNet` and timestep-aware helpers.
  - [ ] `src/mlcast/models/diffusion/net.py` for `LatentDiffusionNet`.
  - [ ] `src/mlcast/models/diffusion/forecasting.py` for `LatentDiffusionForecaster`, the diffusion-specific adapter that exposes `forward(x, forecast_steps, ensemble_size)`.
  - [ ] `src/mlcast/models/diffusion/scheduler.py`, `ema.py`, `sampler.py`, `loss.py` for diffusion support code.
- Validation and tests:
  - [ ] diffusion forecasting adapter API.
  - [ ] diffusion model improves loss on a small generated latent dataset after a few training steps.

4. Task wrappers
- [ ] Add `src/mlcast/modules/forecasting.py` and rename `NowcastLightningModule` to `ForecastingModule`.
- [ ] Add `src/mlcast/modules/reconstruction.py` with a generic `ReconstructionModule` for any reconstruction model.
- [ ] Keep `modules/` for training/task wrappers only; keep `models/` for pure architectures.

5. Training experiment
- [ ] Add a new LDCast-specific training module containing `LDCastTrainingExperiment`.
- [ ] Keep `convgru_training_experiment` as the existing ConvGRU forecasting example and one of the explicitly selected included CLI configs.
- [ ] Stage 1 builds the reconstruction dataset, autoencoder model, and reconstruction module, then trains the autoencoder.
- [ ] Stage 2 reuses the same trained in-memory encoder instance, builds the diffusion dataset/model/module, then trains latent diffusion.
- [ ] The shared Fiddle graph should define the encoder once and reference the same object in both stages, but no unresolved Fiddle objects should flow into actual `torch.nn.Module.__init__` calls.
- [ ] The decoder is stage-1 only and is not shared into stage 2.
- [ ] Reuse the same forecasting dataset abstraction in stage 2; do not add a separate latent dataset layer.
- [ ] Add tests for shared object identity and stage sequencing.

6. Audit and migration targets
- [ ] Update CLI/help text in `src/mlcast/__main__.py` to require an explicit base config and list the included config entry points.
- [ ] Rename `training_experiment` to `convgru_training_experiment` in `src/mlcast/config/base.py` and export it from `src/mlcast/config/__init__.py`.
- [ ] Add the LDCast config entry point to `src/mlcast/config/__init__.py` alongside the existing ConvGRU example config.
- [ ] Keep `src/mlcast/config/orchestrator.py` compatible with both the existing single-stage `Experiment` and the new `LDCastTrainingExperiment` through a common `run()` surface.
- [ ] Update docstrings and comments that currently imply `training_experiment` is the only experiment, including `src/mlcast/data/source_data_datamodule.py`, `src/mlcast/config/orchestrator.py`, and related config docs.
- [ ] Update docs and scripts that still reference `training_experiment`, including `README.md` and `docs/generate_base_experiment_config_diagram.py`.
- [ ] Keep existing ConvGRU CLI/config tests passing while adding separate tests for selecting the LDCast config explicitly.
- [ ] Add real but small-scale end-to-end tests with generated sample data for the autoencoder stage, diffusion stage, and full LDCast stage sequencing.
