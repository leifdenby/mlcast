"""MLCast - Machine learning weather nowcasting library."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlcast")
except PackageNotFoundError:
    __version__ = "0+unknown"
