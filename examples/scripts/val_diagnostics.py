"""Diagnose whether the validation loss can tell checkpoints and runs apart.

The DMI training runs validate on ``limit_val_batches`` batches of a val
loader that is not shuffled and whose candidates are sorted by time, with a
window start offset drawn at random on every pass and fresh ensemble noise.
This script measures what that val subset covers and how noisy its loss is,
and scores checkpoints on a fixed, time-spread val subset instead.

Three checks, selected with ``--check``:

- ``coverage`` (CPU): size of the val set, the dates covered by the first
  ``--head-sizes`` windows (what ``limit_val_batches`` actually scores), and
  the distribution of the index statistics there versus all val windows and
  the importance-selected train windows.
- ``noise`` (GPU): score one checkpoint ``--repeats`` times on the first
  ``--limit-batches`` batches, once as in training (random window offsets and
  ensemble noise) and once with fixed windows and a fixed seed. The spread of
  the first is the noise floor of the logged ``val_loss``.
- ``checkpoints`` (GPU): score each ``--checkpoint`` on the same fixed
  ``--n-windows`` windows strided evenly across the whole val period, overall
  and per quantile bin of the index ``mean`` (dry to wet windows), and on as
  many val windows chosen by the train selector (``trainlike``: the
  rain-weighted mix the model trains on). Persistence and always-dry are
  scored the same way as reference rows (``--no-references`` to skip them).

Run from the repository root on a machine with the DMI data, e.g.::

    uv run python examples/scripts/val_diagnostics.py --check coverage
    uv run python examples/scripts/val_diagnostics.py --check noise \\
        --checkpoint 114/<run_id>/checkpoints/every-3h-epoch=0-step=170000.ckpt
    uv run python examples/scripts/val_diagnostics.py --check checkpoints \\
        --checkpoint a.ckpt --checkpoint b.ckpt

The dataset and model are built from ``dmi_500m_10min_convgru_experiment``
with the overrides given on the command line, so the val selection is the
one training used. Only ``cfg.data`` and ``cfg.pl_module`` are built, so no
MLflow run is started.
"""

import argparse
import copy
import sys

import fiddle as fdl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

STAT_COLUMNS = ("mean", "frac_wet")


def window_start_times(dataset) -> pd.DatetimeIndex:
    """Return the start time of every candidate window in an indexed dataset.

    Parameters
    ----------
    dataset : SourceDataIndexedDataset
        Dataset whose ``candidates["t"]`` index its (time-subset) ``ds``.

    Returns
    -------
    pandas.DatetimeIndex
        One timestamp per candidate row, in candidate order.
    """
    return pd.DatetimeIndex(dataset.ds.indexes[dataset.t_dim][dataset.candidates["t"].to_numpy()])


def coverage_of_head(times: pd.DatetimeIndex, head_sizes: list[int]) -> pd.DataFrame:
    """Summarise the time span covered by the first ``n`` windows, for each ``n``.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Window start times, in the order the val loader yields them.
    head_sizes : list of int
        Numbers of leading windows to summarise (e.g. ``batch_size * limit_val_batches``).
        The full set is always added as the last row.

    Returns
    -------
    pandas.DataFrame
        One row per head size with columns ``n_windows``, ``first``, ``last``,
        ``n_days`` (distinct calendar days) and ``n_months``.
    """
    rows = []
    for n in [*head_sizes, len(times)]:
        head = times[: min(n, len(times))]
        rows.append(
            {
                "n_windows": len(head),
                "first": head.min(),
                "last": head.max(),
                "n_days": head.normalize().nunique(),
                "n_months": head.to_period("M").nunique(),
            }
        )
    return pd.DataFrame(rows)


def describe_stats(groups: dict[str, pd.DataFrame], columns=STAT_COLUMNS) -> pd.DataFrame:
    """Compare the index statistics of several candidate sets.

    Parameters
    ----------
    groups : dict of str to pandas.DataFrame
        Named candidate sets (sampling-index rows).
    columns : sequence of str
        Statistic columns to summarise; those absent from a set are skipped.

    Returns
    -------
    pandas.DataFrame
        Rows ``(group, column)`` with the count, mean, 10/50/90th percentiles
        and the fraction of rows equal to zero.
    """
    rows = []
    for name, df in groups.items():
        for col in columns:
            if col not in df.columns:
                continue
            v = df[col].to_numpy(dtype=float)
            rows.append(
                {
                    "group": name,
                    "column": col,
                    "n": len(v),
                    "mean": np.nanmean(v),
                    "p10": np.nanpercentile(v, 10),
                    "p50": np.nanpercentile(v, 50),
                    "p90": np.nanpercentile(v, 90),
                    "frac_zero": float(np.mean(v == 0)),
                }
            )
    return pd.DataFrame(rows).set_index(["group", "column"])


def strided_positions(n_total: int, n_windows: int) -> np.ndarray:
    """Return ``n_windows`` positions spread evenly over ``range(n_total)``.

    Parameters
    ----------
    n_total : int
        Number of candidates to choose from (sorted by time, so evenly spaced
        positions are spread across the whole period).
    n_windows : int
        Number of positions wanted; capped at ``n_total``.

    Returns
    -------
    numpy.ndarray
        Strictly increasing integer positions.
    """
    n = min(n_windows, n_total)
    return np.unique(np.linspace(0, n_total - 1, n).round().astype(int))


def quantile_bins(values: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Split positions into ``n_bins`` groups by quantile of ``values``.

    Parameters
    ----------
    values : numpy.ndarray
        Per-window statistic (e.g. the index ``mean``).
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    list of numpy.ndarray
        Positions into ``values`` for each bin, lowest values first. Ties are
        broken by position, so every bin has (nearly) the same size.
    """
    order = np.argsort(values, kind="stable")
    return [np.sort(chunk) for chunk in np.array_split(order, n_bins)]


def build_config(args: argparse.Namespace) -> fdl.Config:
    """Build the DMI experiment config with the run's overrides applied."""
    from mlcast.config.dmi import dmi_500m_10min_convgru_experiment

    cfg = dmi_500m_10min_convgru_experiment()
    cfg.data.batch_size = args.batch_size
    cfg.data.dataset_factory.forecast_steps = args.forecast_steps
    cfg.pl_module.ensemble_size = args.ensemble_size
    cfg.pl_module.loss_class = args.loss_class
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    return cfg


# Normalised value of "no rain": the reflectivity normalisation maps everything at
# or below its 0 dBZ floor to `norm_min` (-1), which is also the dataset's NaN fill.
DRY_VALUE = -1.0


class PersistenceNetwork(torch.nn.Module):
    """Reference forecast: the last input frame, repeated for every lead time and member."""

    def forward(self, x: torch.Tensor, steps: int, ensemble_size: int = 1) -> torch.Tensor:
        """Repeat the last input frame.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, time, channels, height, width)``.
        steps : int
            Number of lead times.
        ensemble_size : int
            Number of (identical) members, concatenated on the channel axis like
            ``ConvGruModel``'s.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, steps, ensemble_size * channels, height, width)``.
        """
        return x[:, -1:].repeat(1, steps, ensemble_size, 1, 1)


class AlwaysDryNetwork(torch.nn.Module):
    """Reference forecast: no rain anywhere, at every lead time, for every member."""

    def forward(self, x: torch.Tensor, steps: int, ensemble_size: int = 1) -> torch.Tensor:
        """Return a forecast of all ``DRY_VALUE``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, time, channels, height, width)``.
        steps : int
            Number of lead times.
        ensemble_size : int
            Number of (identical) members.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, steps, ensemble_size * channels, height, width)``.
        """
        b, _, c, h, w = x.shape
        return torch.full((b, steps, ensemble_size * c, h, w), DRY_VALUE, dtype=x.dtype, device=x.device)


REFERENCE_NETWORKS = {"persistence": PersistenceNetwork, "always-dry": AlwaysDryNetwork}


def build_module(cfg: fdl.Config, reference: str | None = None):
    """Build the Lightning module, optionally with a reference network in place of the ConvGRU.

    Parameters
    ----------
    cfg : fdl.Config
        Experiment config; only ``cfg.pl_module`` is built.
    reference : str or None
        A key of ``REFERENCE_NETWORKS``, or ``None`` for the configured model.

    Returns
    -------
    NowcastLightningModule
        Module with the same loss as training.
    """
    module = fdl.build(cfg.pl_module)
    if reference is not None:
        module.network = REFERENCE_NETWORKS[reference]()
    return module


def val_loader(datamodule, dataset) -> DataLoader:
    """Build an unshuffled loader over ``dataset`` with the datamodule's loader settings."""
    return DataLoader(dataset, shuffle=False, **datamodule.dataloader_kwargs)


def score(
    module, loader: DataLoader, ckpt_path: str | None, seed: int | None, limit_batches: int | float = 1.0
) -> float:
    """Return the mean ``val_loss`` of a checkpoint over ``loader``.

    Parameters
    ----------
    module : NowcastLightningModule
        Freshly built module; the checkpoint's weights are loaded into it.
    loader : torch.utils.data.DataLoader
        Validation loader.
    ckpt_path : str or None
        Lightning checkpoint to score, or ``None`` to score ``module`` as it is
        (for the reference forecasts).
    seed : int or None
        If given, ``pl.seed_everything(seed)`` before scoring, which fixes the
        ensemble noise. ``None`` leaves it random, as in training.
    limit_batches : int or float
        Passed to the trainer as ``limit_val_batches``.

    Returns
    -------
    float
        The aggregated ``val_loss``.
    """
    if seed is not None:
        pl.seed_everything(seed, workers=True, verbose=False)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_val_batches=limit_batches,
    )
    (metrics,) = trainer.validate(module, dataloaders=loader, ckpt_path=ckpt_path, verbose=False)
    return float(metrics["val_loss"])


def fix_windows(dataset) -> None:
    """Make every window start at its candidate's ``t`` instead of a random offset.

    ``__getitem__`` only draws a random start when ``steps < dt``, so setting
    ``dt`` to ``steps`` pins ``t_start = t0``.
    """
    dataset.dt = dataset.steps


def check_coverage(args: argparse.Namespace) -> None:
    """Print what the val windows (and the first ``--head-sizes`` of them) cover."""
    cfg = build_config(args)
    dm = fdl.build(cfg.data)
    dm.setup("fit")
    val, train = dm.val_dataset, dm.train_dataset
    times = window_start_times(val)

    logger.info("val: {} windows, train: {} windows", len(val), len(train))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nTime span of the first n val windows (the loader is not shuffled):")
        print(coverage_of_head(times, args.head_sizes).to_string(index=False))

        print("\nVal windows per month (whole val set):")
        print(times.to_period("M").value_counts().sort_index().to_string())

        groups = {f"val head {n}": val.candidates.iloc[:n] for n in args.head_sizes}
        groups["val all"] = val.candidates
        groups["train (importance)"] = train.candidates
        print("\nIndex statistics:")
        print(describe_stats(groups).to_string(float_format=lambda x: f"{x:.4f}"))


def check_noise(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Score one checkpoint repeatedly on the training-time val subset.

    Returns the ``val_loss`` per repeat for the modes ``"as-training"``
    (random window offsets and ensemble noise) and ``"fixed"`` (fixed windows
    and seed, so all repeats should agree).
    """
    if len(args.checkpoint) != 1:
        sys.exit("--check noise takes exactly one --checkpoint")
    ckpt = args.checkpoint[0]
    cfg = build_config(args)
    dm = fdl.build(cfg.data)

    results = {}
    for mode in ("as-training", "fixed"):
        dm.setup("validate")  # fresh dataset each mode, so fix_windows doesn't leak
        if mode == "fixed":
            fix_windows(dm.val_dataset)
        loader = val_loader(dm, dm.val_dataset)
        losses = []
        for i in range(args.repeats):
            module = fdl.build(cfg.pl_module)
            loss = score(
                module, loader, ckpt, seed=args.seed if mode == "fixed" else None, limit_batches=args.limit_batches
            )
            logger.info("{} repeat {}: val_loss={:.5f}", mode, i, loss)
            losses.append(loss)
        results[mode] = np.array(losses)

    print(f"\nval_loss of {ckpt} over {args.repeats} repeats, first {args.limit_batches} batches of {args.batch_size}:")
    for mode, v in results.items():
        print(f"  {mode:12s} mean {v.mean():.5f}  std {v.std(ddof=1):.5f}  min {v.min():.5f}  max {v.max():.5f}")
    return results


def select_trainlike(datamodule, candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return the val candidates the train selector would keep (rain-weighted, like training).

    Parameters
    ----------
    datamodule : SourceDataDataModule
        Supplies ``train_selector``; with none, all candidates are returned.
    candidates : pandas.DataFrame
        Val candidates, sorted by time.
    seed : int
        Seed for the selector's random keep/discard draw.

    Returns
    -------
    pandas.DataFrame
        The kept rows, in time order.
    """
    if datamodule.train_selector is None:
        return candidates
    kept = datamodule.train_selector.select(candidates, np.random.default_rng(seed))
    return candidates.iloc[kept].reset_index(drop=True)


def check_checkpoints(args: argparse.Namespace) -> pd.DataFrame:
    """Score each checkpoint, and the reference forecasts, on fixed, time-spread val windows.

    Two window sets are used, both ``--n-windows`` long and strided across the
    val period: ``uniform`` (the val set as configured, i.e. the real mix of
    dry and wet) and ``trainlike`` (the val windows the train selector would
    keep, i.e. the rain-weighted mix the model trains on).

    Returns one row per checkpoint and per reference forecast (persistence,
    always-dry), with the loss over all uniform windows, per quantile bin of
    their index ``mean`` (dry to wet), and over the trainlike windows.
    """
    cfg = build_config(args)
    dm = fdl.build(cfg.data)
    dm.setup("validate")
    val = dm.val_dataset
    fix_windows(val)
    trainlike = copy.copy(val)
    trainlike.candidates = select_trainlike(dm, val.candidates, seed=args.seed)
    for name, ds in (("uniform", val), ("trainlike", trainlike)):
        pos = strided_positions(len(ds), args.n_windows)
        ds.candidates = ds.candidates.iloc[pos].reset_index(drop=True)
        times = window_start_times(ds)
        logger.info("fixed {} subset: {} windows from {} to {}", name, len(ds), times.min(), times.max())

    # the dataset loads only the index's `mean` statistic (see `_load_sampling_index`)
    bins = quantile_bins(val.candidates["mean"].to_numpy(dtype=float), args.bins)
    subsets = {"all": (val, np.arange(len(val)))}
    for i, b in enumerate(bins):
        m = val.candidates["mean"].iloc[b]
        subsets[f"mean q{i + 1} [{m.min():.3g}, {m.max():.3g}]"] = (val, b)
    m = trainlike.candidates["mean"]
    subsets[f"trainlike [{m.min():.3g}, {m.max():.3g}]"] = (trainlike, np.arange(len(trainlike)))

    entries = [(ckpt, ckpt, None) for ckpt in args.checkpoint]
    if not args.no_references:
        entries += [(ref, None, ref) for ref in REFERENCE_NETWORKS]

    rows = []
    for label, ckpt, reference in entries:
        row = {"forecast": label}
        for name, (ds, idx) in subsets.items():
            loader = val_loader(dm, Subset(ds, idx.tolist()))
            row[name] = score(build_module(cfg, reference), loader, ckpt, seed=args.seed)
            logger.info("{} {}: val_loss={:.5f}", label, name, row[name])
        rows.append(row)

    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 90):
        print(
            f"\nval_loss on fixed windows strided across the val period (seed {args.seed}): "
            f"{len(val)} uniform (all + `mean` bins), {len(trainlike)} trainlike:"
        )
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", choices=["coverage", "noise", "checkpoints"], required=True)
    p.add_argument("--checkpoint", action="append", default=[], help="Lightning checkpoint (repeatable)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--forecast-steps", type=int, default=12)
    p.add_argument("--ensemble-size", type=int, default=10, help="training ensemble size (val always uses 10)")
    p.add_argument("--loss-class", default="afcrps")
    p.add_argument("--num-workers", type=int, default=None, help="default: the config's")
    p.add_argument("--head-sizes", type=int, nargs="+", default=[1600, 2400], help="coverage: leading windows")
    p.add_argument("--limit-batches", type=int, default=100, help="noise: limit_val_batches, as in training")
    p.add_argument("--repeats", type=int, default=5, help="noise: repeats per mode")
    p.add_argument("--n-windows", type=int, default=2000, help="checkpoints: size of the fixed val subset")
    p.add_argument("--bins", type=int, default=3, help="checkpoints: quantile bins of the index `mean`")
    p.add_argument("--seed", type=int, default=0, help="seed for the fixed-noise scores")
    p.add_argument("--no-references", action="store_true", help="checkpoints: skip the persistence and always-dry rows")
    args = p.parse_args(argv)
    if args.check in ("noise", "checkpoints") and not args.checkpoint:
        p.error(f"--check {args.check} needs at least one --checkpoint")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the selected check."""
    args = parse_args(argv)
    {"coverage": check_coverage, "noise": check_noise, "checkpoints": check_checkpoints}[args.check](args)


if __name__ == "__main__":
    main()
