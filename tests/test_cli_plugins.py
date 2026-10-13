"""Tests for ``mlcast <command>`` plugin subcommands (the ``mlcast.commands`` entry point group).

Unlike the training CLI tests these call ``cli()`` in-process: the entry point
lookup has to be monkeypatched, and no absl flag state is involved.
"""

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pytest
from loguru import logger

import mlcast.__main__ as mlcast_main


@dataclass
class FakeEntryPoint:
    """Stand-in for ``importlib.metadata.EntryPoint`` whose ``load()`` returns a given object."""

    name: str
    target: Callable[..., object] | None = None
    error: Exception | None = None
    value: str = "fake_plugin:main"
    group: str = mlcast_main.PLUGIN_ENTRY_POINT_GROUP
    loads: list[str] = field(default_factory=list)

    def load(self) -> Callable[..., object]:
        self.loads.append(self.name)
        if self.error is not None:
            raise self.error
        return self.target


def _install_plugins(monkeypatch: pytest.MonkeyPatch, *eps: FakeEntryPoint) -> None:
    """Make ``cli()`` discover exactly ``eps`` as ``mlcast.commands`` entry points."""

    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        return list(eps) if group == mlcast_main.PLUGIN_ENTRY_POINT_GROUP else []

    monkeypatch.setattr(mlcast_main, "entry_points", fake_entry_points)


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    """Run ``mlcast *args`` in-process and return its exit code."""
    monkeypatch.setattr("sys.argv", ["mlcast", *args])
    try:
        mlcast_main.cli()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    """Collect the loguru WARNING messages emitted during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    # The sampling commands call `logger.remove()`, which may already have dropped the sink.
    with contextlib.suppress(ValueError):
        logger.remove(sink_id)


def test_plugin_listed_in_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The plugin appears in ``mlcast --help`` with the first docstring line as help."""

    def hello(argv: list[str]) -> None:
        """Say hello (100% friendly).

        This second line is not part of the one-line help.
        """

    _install_plugins(monkeypatch, FakeEntryPoint("hello", hello))
    assert _run_cli(monkeypatch, "--help") == 0
    out = capsys.readouterr().out
    assert "hello" in out
    assert "Say hello (100% friendly)." in out
    assert "second line" not in out


def test_plugin_receives_remaining_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything after the subcommand name, including ``--help``, goes to the plugin."""
    received: list[list[str]] = []

    def hello(argv: list[str]) -> None:
        """Say hello."""
        received.append(argv)

    _install_plugins(monkeypatch, FakeEntryPoint("hello", hello))
    assert _run_cli(monkeypatch, "hello", "world", "--loud", "--times=3", "--help") == 0
    assert received == [["world", "--loud", "--times=3", "--help"]]


def test_unused_plugins_not_loaded_when_dispatching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running one plugin command does not import the other plugins."""
    other = FakeEntryPoint("other", lambda argv: None)
    _install_plugins(monkeypatch, FakeEntryPoint("hello", lambda argv: None), other)
    assert _run_cli(monkeypatch, "hello") == 0
    assert other.loads == []


@pytest.mark.parametrize(("returned", "expected_code"), [(None, 0), (0, 0), (3, 3)])
def test_plugin_exit_code_propagates(monkeypatch: pytest.MonkeyPatch, returned: int | None, expected_code: int) -> None:
    """A non-zero int returned by the plugin becomes the process exit code."""
    _install_plugins(monkeypatch, FakeEntryPoint("hello", lambda argv: returned))
    assert _run_cli(monkeypatch, "hello") == expected_code


def test_broken_plugin_does_not_break_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A plugin that fails to import is listed with its error; ``--help`` still works."""

    def hello(argv: list[str]) -> None:
        """Say hello."""

    _install_plugins(
        monkeypatch,
        FakeEntryPoint("broken", error=ImportError("No module named 'nope'")),
        FakeEntryPoint("hello", hello),
    )
    assert _run_cli(monkeypatch, "--help") == 0
    out = " ".join(capsys.readouterr().out.split())  # undo argparse line wrapping
    assert "broken (failed to load: No module named 'nope')" in out
    assert "hello Say hello." in out


def test_broken_plugin_dispatch_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running a plugin that fails to import exits with the error message (exit status 1)."""
    _install_plugins(monkeypatch, FakeEntryPoint("broken", error=ImportError("No module named 'nope'")))
    monkeypatch.setattr("sys.argv", ["mlcast", "broken"])
    with pytest.raises(SystemExit) as exc_info:
        mlcast_main.cli()
    assert "No module named 'nope'" in str(exc_info.value.code)


def test_builtin_command_cannot_be_overridden(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], warnings: list[str]
) -> None:
    """A plugin named like a built-in is ignored with a warning; the built-in still runs."""
    hijack = FakeEntryPoint("validate-sampling-index", lambda argv: 42)
    _install_plugins(monkeypatch, hijack)
    # The built-in's own argparse rejects the missing positional argument (exit 2).
    assert _run_cli(monkeypatch, "validate-sampling-index") == 2
    assert hijack.loads == []
    assert "usage: mlcast validate-sampling-index" in capsys.readouterr().err
    assert any("validate-sampling-index" in w and "already taken" in w for w in warnings)
