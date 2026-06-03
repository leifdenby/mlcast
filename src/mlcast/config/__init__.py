"""Fiddle-based experiment configuration package.

This package defines the configuration schemas, validation constraints,
and runtime orchestration logic for `mlcast`.
"""

from .archetype.convgru import convgru_training_experiment
from .archetype.ldcast import LDCastTrainingExperiment, ldcast_training_experiment
from .base import Experiment
from .consistency_checks import validate_config
from .fiddlers import (
    set_variables,
    toggle_masking,
    use_anon_s3_dataset,
    use_mlflow_logger,
    use_random_sampler,
    use_ratio_splits,
)
from .loader import load_yaml_config
from .orchestrator import train_from_config

__all__ = [
    "Experiment",
    "LDCastTrainingExperiment",
    "convgru_training_experiment",
    "ldcast_training_experiment",
    "validate_config",
    "train_from_config",
    "load_yaml_config",
    "set_variables",
    "toggle_masking",
    "use_random_sampler",
    "use_ratio_splits",
    "use_mlflow_logger",
    "use_anon_s3_dataset",
]
