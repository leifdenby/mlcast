import fiddle as fdl
from pytorch_lightning.loggers import MLFlowLogger

from mlcast.config import dmi_500m_10min_convgru_experiment, validate_config


def test_dmi_500m_10min_convgru_experiment_config():
    cfg = dmi_500m_10min_convgru_experiment()  # plain function, already returns fdl.Config

    assert cfg.data.dataset_factory.zarr_path.endswith("DMI_500m_10min_v012.zarr")
    assert cfg.data.dataset_factory.index_path.endswith(".parquet")
    assert cfg.data.dataset_factory.input_steps == 6
    assert cfg.data.dataset_factory.forecast_steps == 18
    assert cfg.data.dataset_factory.standard_names == ["equivalent_reflectivity_factor"]
    assert cfg.pl_module.network.input_channels == 1
    assert cfg.data.batch_size == 32

    assert cfg.data.splits == {
        "time": {
            "train": ("2016-01-01", "2023-12-31"),
            "val": ("2024-01-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-12-31"),
        }
    }

    assert cfg.trainer.logger.__fn_or_cls__ is MLFlowLogger
    assert cfg.trainer.logger.tracking_uri == "https://mlflow.dmidev.org/"
    assert cfg.trainer.logger.experiment_name == "dmi-500m-10min-convgru"

    validate_config(cfg)
    fdl.build(cfg)  # succeeds without touching the filesystem
