"""Tests for the pure helpers in ``examples/scripts/val_diagnostics.py``."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "examples" / "scripts" / "val_diagnostics.py"
_spec = importlib.util.spec_from_file_location("val_diagnostics", _SCRIPT)
vd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vd)


def test_coverage_of_head_counts_days_and_months():
    # 10-minute windows from 1 Jan 2024: 144 per day
    times = pd.date_range("2024-01-01", periods=144 * 40, freq="10min")
    cov = vd.coverage_of_head(times, [144, 144 * 3])

    assert list(cov["n_windows"]) == [144, 432, 144 * 40]
    assert list(cov["n_days"]) == [1, 3, 40]
    assert list(cov["n_months"]) == [1, 1, 2]
    assert cov["first"].iloc[0] == pd.Timestamp("2024-01-01")
    assert cov["last"].iloc[1] == pd.Timestamp("2024-01-03 23:50")


def test_coverage_of_head_caps_at_full_length():
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    cov = vd.coverage_of_head(times, [100])
    assert list(cov["n_windows"]) == [10, 10]


def test_describe_stats_skips_missing_columns():
    a = pd.DataFrame({"mean": [0.0, 1.0, 2.0, 3.0], "frac_wet": [0.0, 0.0, 0.5, 1.0]})
    b = pd.DataFrame({"mean": [5.0, 5.0]})
    out = vd.describe_stats({"a": a, "b": b})

    assert set(out.index) == {("a", "mean"), ("a", "frac_wet"), ("b", "mean")}
    assert out.loc[("a", "frac_wet"), "frac_zero"] == pytest.approx(0.5)
    assert out.loc[("a", "mean"), "mean"] == pytest.approx(1.5)
    assert out.loc[("b", "mean"), "n"] == 2


@pytest.mark.parametrize("n_total,n_windows", [(10_000, 2000), (2000, 2000), (50, 2000), (7, 3)])
def test_strided_positions_span_whole_range(n_total, n_windows):
    pos = vd.strided_positions(n_total, n_windows)

    assert len(pos) == min(n_total, n_windows)
    assert pos[0] == 0 and pos[-1] == n_total - 1
    assert np.all(np.diff(pos) > 0)


def test_quantile_bins_partition_by_value():
    rng = np.random.default_rng(0)
    values = rng.random(301)
    bins = vd.quantile_bins(values, 3)

    assert sorted(np.concatenate(bins).tolist()) == list(range(301))
    assert [len(b) for b in bins] == [101, 100, 100]
    assert values[bins[0]].max() <= values[bins[1]].min()
    assert values[bins[1]].max() <= values[bins[2]].min()


def test_quantile_bins_with_ties_are_balanced():
    values = np.zeros(10)  # e.g. many completely dry windows
    bins = vd.quantile_bins(values, 2)
    assert [len(b) for b in bins] == [5, 5]


class _FakeDataset:
    def __init__(self, steps, dt):
        self.steps, self.dt = steps, dt


def test_fix_windows_pins_window_start():
    ds = _FakeDataset(steps=18, dt=24)
    vd.fix_windows(ds)
    # `__getitem__` draws a random start only when steps < dt
    assert not ds.steps < ds.dt


def test_persistence_repeats_last_frame_for_every_lead_and_member():
    x = torch.arange(2 * 3 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 3, 1, 4, 4)
    out = vd.PersistenceNetwork()(x, steps=5, ensemble_size=3)

    assert out.shape == (2, 5, 3, 4, 4)
    assert torch.equal(out, x[:, -1:].expand(2, 5, 3, 4, 4))


def test_always_dry_is_dry_everywhere():
    x = torch.rand(2, 3, 1, 4, 4)
    out = vd.AlwaysDryNetwork()(x, steps=5, ensemble_size=3)

    assert out.shape == (2, 5, 3, 4, 4)
    assert torch.all(out == vd.DRY_VALUE)


def test_reference_forecasts_score_zero_on_dry_windows():
    """Persistence and always-dry are exact on a window with no rain at all."""
    x = torch.full((2, 3, 1, 8, 8), vd.DRY_VALUE)
    target = torch.full((2, 4, 1, 8, 8), vd.DRY_VALUE)
    for reference in vd.REFERENCE_NETWORKS:
        module = vd.build_module(_tiny_module_cfg(), reference=reference)
        preds = module(x, forecast_steps=4, ensemble_size=3)
        loss = module.criterion(preds, target, torch.ones(2, 1, 1, 8, 8))
        loss = loss[0] if isinstance(loss, tuple) else loss
        assert float(loss) == pytest.approx(0.0, abs=1e-7), reference


def _tiny_module_cfg():
    """Experiment config whose ``pl_module`` uses the DMI training loss (afCRPS, masked)."""
    from mlcast.config import training_experiment

    cfg = training_experiment.as_buildable()
    cfg.pl_module.network.num_blocks = 1
    cfg.pl_module.loss_class = "afcrps"
    cfg.pl_module.ensemble_size = 3
    return cfg


def test_select_trainlike_uses_train_selector():
    from mlcast.data.source_data.sampling import ImportanceSelector

    candidates = pd.DataFrame({"t": np.arange(6), "mean": [-32.0, -32.0, -10.0, 0.0, 5.0, 20.0]})

    class _DM:
        train_selector = ImportanceSelector()

    kept = vd.select_trainlike(_DM(), candidates, seed=0)
    # dBZ means below 0 get a non-positive weight and are never kept (see STATUS)
    assert (kept["mean"] > -1).all()
    assert 20.0 in kept["mean"].to_numpy()

    _DM.train_selector = None
    assert len(vd.select_trainlike(_DM(), candidates, seed=0)) == len(candidates)


def test_parse_args_requires_checkpoint_for_gpu_checks():
    with pytest.raises(SystemExit):
        vd.parse_args(["--check", "noise"])
    args = vd.parse_args(["--check", "checkpoints", "--checkpoint", "a.ckpt", "--checkpoint", "b.ckpt"])
    assert args.checkpoint == ["a.ckpt", "b.ckpt"]


@pytest.fixture
def tiny_cfg(fp_test_dataset: Path, tmp_path: Path):
    """A small indexed-dataset config on the Italian test zarr, plus a 1-step checkpoint."""
    import fiddle as fdl
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytorch_lightning as pl

    from mlcast.config import training_experiment
    from mlcast.data.source_data.sampling import UniformSelector
    from mlcast.data.source_data.sampling.sampling_index_spec import SamplingParameters, build_schema

    meta = SamplingParameters(
        zarr_path=str(fp_test_dataset),
        data_var="RR",
        time_var="time",
        start_date="2010-01-01",
        end_date="2010-01-02",
        time_step_minutes=15,
        time_depth=4,
        width=32,
        height=32,
        step_t=1,
        step_x=1,
        step_y=1,
        max_nan=0,
        wet_threshold=0.1,
        data_kind="rainrate",
        units="mm/h",
    )
    t = np.arange(0, 96, dtype=np.int32)
    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "t": pa.array(t, pa.int32()),
            "x": pa.array(np.full(len(t), 600), pa.int32()),
            "y": pa.array(np.full(len(t), 600), pa.int32()),
            "nan_count": pa.array(np.zeros(len(t)), pa.int32()),
            "sum": pa.array(rng.random(len(t)), pa.float32()),
            "mean": pa.array(rng.random(len(t)), pa.float32()),
            "frac_wet": pa.array(rng.random(len(t)), pa.float32()),
        },
        schema=build_schema(meta),
    )
    index_path = tmp_path / "index.parquet"
    pq.write_table(table, index_path)

    cfg = training_experiment.as_buildable()
    f = cfg.data.dataset_factory
    f.zarr_path = str(fp_test_dataset.absolute())
    f.index_path = str(index_path)
    f.standard_names = ["rainfall_flux"]
    f.input_steps, f.forecast_steps, f.time_depth = 2, 2, 4
    f.width = f.height = 32
    cfg.data.splits = {
        "time": {
            "train": ("2010-01-01T00:00", "2010-01-01T11:59"),
            "val": ("2010-01-01T12:00", "2010-01-02T00:45"),
            "test": None,
        }
    }
    cfg.data.eval_selector = fdl.Config(UniformSelector)
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.pl_module.network.num_blocks = 2
    cfg.pl_module.ensemble_size = 2

    module = fdl.build(cfg.pl_module)
    trainer = pl.Trainer(
        max_steps=1, limit_val_batches=0, logger=False, enable_checkpointing=False, default_root_dir=tmp_path
    )
    dm = fdl.build(cfg.data)
    trainer.fit(module, datamodule=dm)
    ckpt = tmp_path / "tiny.ckpt"
    trainer.save_checkpoint(ckpt)
    return cfg, str(ckpt)


def test_checks_run_end_to_end(tiny_cfg, monkeypatch, capsys):
    cfg, ckpt = tiny_cfg
    monkeypatch.setattr(vd, "build_config", lambda args: cfg)

    vd.check_coverage(vd.parse_args(["--check", "coverage", "--head-sizes", "4"]))
    out = capsys.readouterr().out
    assert "val head 4" in out and "train (importance)" in out

    noise = vd.check_noise(
        vd.parse_args(["--check", "noise", "--checkpoint", ckpt, "--repeats", "2", "--limit-batches", "2"])
    )
    # fixed windows and seed: repeats agree exactly
    assert noise["fixed"][0] == pytest.approx(noise["fixed"][1], rel=1e-6)
    assert np.all(np.isfinite(noise["as-training"]))

    table = vd.check_checkpoints(
        vd.parse_args(["--check", "checkpoints", "--checkpoint", ckpt, "--n-windows", "6", "--bins", "2"])
    )
    assert list(table["forecast"]) == [ckpt, "persistence", "always-dry"]
    assert table.shape[1] == 1 + 1 + 2 + 1  # forecast, all, two `mean` bins, trainlike
    assert table.columns[-1].startswith("trainlike")
    assert np.all(np.isfinite(table.drop(columns="forecast").to_numpy(dtype=float)))

    no_refs = vd.check_checkpoints(
        vd.parse_args(
            ["--check", "checkpoints", "--checkpoint", ckpt, "--n-windows", "6", "--bins", "2", "--no-references"]
        )
    )
    assert list(no_refs["forecast"]) == [ckpt]


def test_build_config_applies_overrides():
    args = vd.parse_args(["--check", "coverage", "--batch-size", "24", "--num-workers", "2"])
    cfg = vd.build_config(args)

    assert cfg.data.batch_size == 24
    assert cfg.data.num_workers == 2
    assert cfg.data.dataset_factory.forecast_steps == 12
    assert cfg.pl_module.ensemble_size == 10
    assert cfg.pl_module.loss_class == "afcrps"
