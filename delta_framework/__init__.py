"""Delta-only incremental learning demo framework."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["api", "core", "__version__"]

try:  # pragma: no cover
    __version__ = version("delta-framework")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

