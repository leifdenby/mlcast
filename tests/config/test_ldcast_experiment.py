from dataclasses import dataclass

import fiddle as fdl

from mlcast.config import LDCastTrainingExperiment, ldcast_training_experiment, validate_config
from mlcast.config.base import Experiment


@dataclass
class RecordingTrainer:
    """Minimal trainer stub that records fit/test call order."""

    events: list[str]

    def fit(self, pl_module, datamodule=None) -> None:  # type: ignore[no-untyped-def]
        self.events.append(f"fit:{pl_module}:{datamodule}")

    def test(self, pl_module, datamodule=None) -> None:  # type: ignore[no-untyped-def]
        self.events.append(f"test:{pl_module}:{datamodule}")


def test_ldcast_training_experiment_runs_stages_in_order() -> None:
    """LDCastTrainingExperiment should execute stage 1 fully before stage 2."""
    events: list[str] = []
    stage1 = Experiment(pl_module="stage1_module", data="stage1_data", trainer=RecordingTrainer(events=events))
    stage2 = Experiment(pl_module="stage2_module", data="stage2_data", trainer=RecordingTrainer(events=events))
    experiment = LDCastTrainingExperiment(stage1=stage1, stage2=stage2)

    experiment.run()

    assert events == [
        "fit:stage1_module:stage1_data",
        "test:stage1_module:stage1_data",
        "fit:stage2_module:stage2_data",
        "test:stage2_module:stage2_data",
    ]


def test_ldcast_training_experiment_shares_autoencoder_identity() -> None:
    """Stage 1 and stage 2 should reference the same built autoencoder instance."""
    cfg = ldcast_training_experiment.as_buildable()
    validate_config(cfg)

    experiment = fdl.build(cfg)

    assert experiment.stage1.pl_module.network is experiment.stage2.pl_module.autoencoder
