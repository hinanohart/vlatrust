"""vlatrust — calibration-under-shift trust harness for VLA policies.

The public surface is intentionally small and import-light: the core
(:mod:`vlatrust.core`) depends only on numpy/scipy and is simulator-independent.
Heavy backends live behind lazy adapters in :mod:`vlatrust.adapters`.
"""

__version__ = "0.1.0a1"

__all__ = ["__version__"]
