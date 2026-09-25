"""Optional compiled kernels: the ``supermap_kernels`` package (C++, pybind11).

The NumPy/SciPy implementations in this package are the reference and the
fallback. When ``supermap_kernels`` is installed (colcon builds it with the
workspace; offline, ``pip install ./supermap_kernels``), the dense-cloud surface
graph and depth splatting run through it instead. ``kernels`` is None when the
package is missing, was built for another kernel API, or the environment sets
``SUPERMAP_NATIVE_KERNELS=0`` (to compare against, or fall back to, the reference).
"""
from __future__ import annotations

import os

API_VERSION = 1
"""Kernel API this code calls; a build reporting another version is ignored."""

try:
    import supermap_kernels as _module
except ImportError:  # optional: everything falls back to NumPy/SciPy
    _module = None

kernels = None
if getattr(_module, "API_VERSION", None) == API_VERSION and os.environ.get("SUPERMAP_NATIVE_KERNELS", "1") != "0":
    kernels = _module
