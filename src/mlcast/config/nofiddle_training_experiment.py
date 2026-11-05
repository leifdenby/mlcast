"""Plain-Python (no Fiddle) version of the ConvGRU training experiment.

This builds the same objects as :func:`mlcast.config.base.training_experiment`,
but directly with ordinary constructors and :func:`functools.partial`
instead of a Fiddle config graph. Use it when you want a classic PyTorch
Lightning setup that you edit in code rather than override from the CLI.

Typical usage
-------------
>>> pl_module, datamodule, trainer = build_training_experiment(
...     zarr_path="/path/to/radar.zarr",
...     index_path="/path/to/sampled_datacubes.parquet",
... )
>>> trainer.fit(pl_module, datamodule=datamodule)
>>> trainer.test(pl_module, datamodule=datamodule)
"""

import functools

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.source_data.sampling import ImportanceSelector, UniformSelector
from ..data.source_data_datamodule import SourceDataDataModule
from ..data.source_data_datasets import SourceDataIndexedDataset
from ..models.convgru import ConvGruModel
from ..nowcasting_module import NowcastLightningModule


def build_training_experiment(
    zarr_path: str = "./data/radar.zarr",
    index_path: str = "./data/sampled_datacubes.parquet",
) -> tuple[NowcastLightningModule, SourceDataDataModule, pl.Trainer]:
    """Build the ConvGRU ensemble nowcasting experiment without Fiddle.

    Mirrors :func:`mlcast.config.base.training_experiment` one-to-one; edit
    the values below directly to change the setup.

    Parameters
    ----------
    zarr_path : str, optional
        Path to the source-data zarr store. Default is ``"./data/radar.zarr"``.
    index_path : str, optional
        Path to the sampling-index parquet file. Default is
        ``"./data/sampled_datacubes.parquet"``.

    Returns
    -------
    pl_module : NowcastLightningModule
        Lightning module wrapping the ConvGRU network.
    datamodule : SourceDataDataModule
        Data module providing the train/val/test DataLoaders.
    trainer : pl.Trainer
        Configured Lightning trainer.
    """
    # The data module calls this once per split with subset=, augment= and selector=
    dataset_factory = functools.partial(
        SourceDataIndexedDataset,
        zarr_path=zarr_path,
        index_path=index_path,
        standard_names=["rainfall_flux"],
        input_steps=6,
        forecast_steps=12,
        return_mask=True,
        deterministic=False,
    )

    datamodule = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={"time": {"train": 0.70, "val": 0.15, "test": 0.15}},
        train_selector=ImportanceSelector(),
        eval_selector=UniformSelector(keep_fraction=0.1),
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )

    network = ConvGruModel(
        input_channels=1,
        num_blocks=5,
        noisy_decoder=False,
    )

    pl_module = NowcastLightningModule(
        network=network,
        ensemble_size=2,
        loss_class="crps",
        loss_params={"temporal_lambda": 0.01},
        masked_loss=True,
        optimizer=functools.partial(torch.optim.Adam, lr=1e-4, fused=True),
        lr_scheduler=functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.5, patience=10),
    )

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
            ModelCheckpoint(monitor="train_loss_epoch", save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=100, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=TensorBoardLogger(save_dir="logs", name="mlcast"),
    )

    return pl_module, datamodule, trainer


def main() -> None:
    """Build the experiment, then train and test it."""
    pl_module, datamodule, trainer = build_training_experiment()
    trainer.fit(pl_module, datamodule=datamodule)
    trainer.test(pl_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
