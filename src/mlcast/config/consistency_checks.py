"""Cross-parameter validation constraints for Fiddle configurations.

Consistency checks are predicate functions that raise ``ValueError`` when
two or more config parameters that must stay in sync have drifted apart.
Unlike fiddlers (which *mutate* a config to enforce a policy), consistency
checks are *read-only* — they inspect the config and signal problems early,
before ``fdl.build()`` is called and before any heavyweight objects are
instantiated.

Call ``validate_config(cfg)`` explicitly after all fiddlers have been
applied and before handing the config off to the training orchestrator.
"""

import fiddle as fdl
from loguru import logger


def _shared_dataset_factories(dataset_factory: fdl.Buildable | object):
    dataset_factories = getattr(dataset_factory, "dataset_factories", None)
    if dataset_factories is None:
        return None
    return dataset_factories


def _dataset_factory_standard_names(dataset_factory: fdl.Buildable | object) -> list[str]:
    dataset_factories = _shared_dataset_factories(dataset_factory)
    if dataset_factories is None:
        return list(dataset_factory.standard_names)

    standard_names = [list(factory.standard_names) for factory in dataset_factories.values()]
    num_vars_set = {len(names) for names in standard_names}
    if len(num_vars_set) != 1:
        raise ValueError("All dataset factories must have the same number of variables (standard_names).")
    return standard_names[0]


def _dataset_factory_width(dataset_factory: fdl.Buildable | object) -> int:
    dataset_factories = _shared_dataset_factories(dataset_factory)
    if dataset_factories is None:
        return getattr(dataset_factory, "width", 256)

    widths = {getattr(factory, "width", 256) for factory in dataset_factories.values()}
    if len(widths) != 1:
        raise ValueError("All dataset factories must agree on width.")
    return widths.pop()


def _dataset_factory_return_mask(dataset_factory: fdl.Buildable | object) -> bool:
    dataset_factories = _shared_dataset_factories(dataset_factory)
    if dataset_factories is None:
        return bool(dataset_factory.return_mask)

    return_masks = {bool(factory.return_mask) for factory in dataset_factories.values()}
    if len(return_masks) != 1:
        raise ValueError("All dataset factories must agree on return_mask.")
    return return_masks.pop()


def validate_config(cfg: fdl.Config) -> None:
    """Validate cross-system constraints on a Fiddle configuration before training.

    Parameters
    ----------
    cfg : fdl.Config
        Fiddle configuration.

    Raises
    ------
    ValueError
        If any configuration contract is violated.
    """
    dataset_factory = cfg.data.dataset_factory
    network = cfg.pl_module.network
    pl_module = cfg.pl_module

    # Contract 1: Network input_channels == len(dataset_factory.standard_names)
    # If the network does not expose input_channels, emit a warning because
    # this contract cannot be checked.
    num_vars = len(_dataset_factory_standard_names(dataset_factory))
    try:
        net_input_channels = network.input_channels
    except AttributeError:
        logger.warning(
            "Warning: can't ensure network input_channels matches the number of dataset variables, "
            "because network {} doesn't expose 'input_channels'.",
            network.__class__.__name__,
        )
        net_input_channels = None
    if net_input_channels is not None and net_input_channels != num_vars:
        raise ValueError(
            f"Contract 1 violated: network input_channels ({net_input_channels}) "
            f"must equal the number of standard_names ({num_vars})."
        )

    # Contract 2: Dataset width must be divisible by 2 ** network.num_blocks
    # If the network does not expose num_blocks, emit a warning because this
    # contract cannot be checked.
    try:
        num_blocks = network.num_blocks
    except AttributeError:
        logger.warning(
            "Warning: can't ensure dataset width is compatible with the network downsampling factor, "
            "because network {} doesn't expose 'num_blocks'.",
            network.__class__.__name__,
        )
        num_blocks = None
    if num_blocks is not None:
        width = _dataset_factory_width(dataset_factory)
        divisor = 2**num_blocks
        if width % divisor != 0:
            raise ValueError(
                f"Contract 2 violated: Dataset width ({width}) must be divisible by "
                f"2 ** network.num_blocks ({divisor})."
            )

    # Contract 3: Ensemble models require CRPS or AFCRPS
    if pl_module.ensemble_size > 1:
        if str(pl_module.loss_class).lower() not in ["crps", "afcrps"]:
            raise ValueError(
                f"Contract 3 violated: Ensemble models (ensemble_size={pl_module.ensemble_size}) "
                f"require 'crps' or 'afcrps' loss, got '{pl_module.loss_class}'."
            )

    # Contract 4: Dataset return_mask must match model masked_loss
    if _dataset_factory_return_mask(dataset_factory) != bool(pl_module.masked_loss):
        raise ValueError(
            f"Contract 4 violated: dataset_factory.return_mask ({_dataset_factory_return_mask(dataset_factory)}) "
            f"must match pl_module.masked_loss ({pl_module.masked_loss})."
        )
