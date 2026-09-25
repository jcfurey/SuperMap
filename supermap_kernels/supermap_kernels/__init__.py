"""Compiled kernels for semantic_mapping (C++, pybind11).

Each function reimplements one NumPy/SciPy routine of semantic_mapping, which
stays the reference implementation and the fallback when this package is not
installed: dense-cloud surface normals, edges and components
(``semantic_mapping.dense_cloud``) and depth-buffer splatting
(``semantic_mapping.geometry_utils.splat_depth_buffer``). The kernels release
the GIL, and ``workers`` threads (-1: all cores) never change a result.
"""
from supermap_kernels._native import (  # noqa: F401
    API_VERSION,
    __version__,
    connected_components,
    remap_edges,
    splat_depth_buffer,
    surface_edges,
    surface_normals,
    within,
)
