"""The compiled supermap_kernels agree with the NumPy/SciPy reference they replace.

Skipped when the kernels are not installed (``pip install ./supermap_kernels``
or a colcon build) or disabled with ``SUPERMAP_NATIVE_KERNELS=0``.
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from semantic_mapping import dense_cloud as dc, geometry_utils, native
from semantic_mapping.types import CameraIntrinsics, Detection2D

pytestmark = pytest.mark.skipif(native.kernels is None, reason="supermap_kernels not installed or disabled")
K = native.kernels
PYTHON = dc.DenseCloudConfig(native_kernels=False, max_map_voxels=1_000_000)


def _room(n, seed=0):
    """Noisy walls, floor and a volume of clutter: no two neighbours equidistant."""
    rng = np.random.default_rng(seed)
    k = n // 4
    cloud = np.concatenate([
        np.column_stack([rng.uniform(0, 6, k), rng.uniform(0, 3, k), rng.normal(0, .005, k)]),
        np.column_stack([rng.uniform(0, 6, k), rng.normal(0, .005, k), rng.uniform(0, 2, k)]),
        np.column_stack([rng.uniform(0, 6, k), 3 + rng.normal(0, .005, k), rng.uniform(0, 2, k)]),
        np.column_stack([rng.uniform(0, 6, k), rng.uniform(.5, 2.5, k), rng.uniform(0, .8, k)])])
    cells = np.floor(cloud / PYTHON.voxel_size).astype(np.int64)
    _, first = np.unique(geometry_utils.pack_voxel_keys(cells), return_index=True)
    return cloud[first]


def _surfaces(n, seed):
    """Separate noisy surfaces (floor, table top, wall, ramp): several regions."""
    rng = np.random.default_rng(seed)
    k = n // 4
    noise = lambda: rng.normal(0, .004, k)  # noqa: E731
    x, y = rng.uniform(0, 1, k), rng.uniform(0, 1, k)
    cloud = np.concatenate([
        np.column_stack([6 * x, 3 * y, noise()]),
        np.column_stack([1 + x, 1 + y, .8 + noise()]),
        np.column_stack([6 * x, 3.5 + noise(), .3 + 1.7 * y]),
        np.column_stack([3 + 2 * x, .5 + 2 * y, .3 + .8 * x + noise()])])
    cells = np.floor(cloud / PYTHON.voxel_size).astype(np.int64)
    _, first = np.unique(geometry_utils.pack_voxel_keys(cells), return_index=True)
    return cloud[first]


@pytest.fixture(scope="module")
def room():
    points = _room(60_000)
    normals, reliable = dc._surface_normals(points, cKDTree(points), np.arange(len(points)), PYTHON)
    return points, normals, reliable


def test_normals_match_the_reference(room):
    points, normals, reliable = room
    index = np.arange(len(points))
    native_normals, native_reliable = K.surface_normals(points, index, PYTHON.normal_radius, PYTHON.max_neighbors, -1)
    np.testing.assert_array_equal(native_reliable, reliable)
    alignment = np.abs(np.einsum("ij,ij->i", native_normals, normals))  # eigenvector signs are arbitrary
    assert alignment[reliable].min() > 1 - 1e-12
    subset = np.random.default_rng(1).choice(len(points), 5000, replace=False)
    sub_normals, sub_reliable = K.surface_normals(points, subset, PYTHON.normal_radius, PYTHON.max_neighbors, 1)
    np.testing.assert_array_equal(sub_normals, native_normals[subset])  # thread count and query set never matter
    np.testing.assert_array_equal(sub_reliable, native_reliable[subset])


def test_edges_match_the_reference_in_order(room):
    points, normals, reliable = room
    cosine = np.cos(np.radians(PYTHON.normal_angle_deg))
    for index in (np.arange(len(points)), np.flatnonzero(np.random.default_rng(2).random(len(points)) < .1)):
        expected = dc._surface_edges(points, cKDTree(points), normals, reliable, index, PYTHON)
        for workers in (1, -1):
            actual = K.surface_edges(points, normals, reliable, index, PYTHON.neighbor_radius, PYTHON.max_neighbors,
                                     cosine, PYTHON.plane_tolerance, workers)
            for a, b in zip(actual, expected):
                assert a.dtype == np.int32
                np.testing.assert_array_equal(a, b)


def test_components_are_numbered_like_scipy():
    rng = np.random.default_rng(3)
    for n, e in ((1, 0), (50, 0), (500, 300), (5000, 4000)):
        rows, cols = rng.integers(0, n, e).astype(np.int32), rng.integers(0, n, e).astype(np.int32)
        _, expected = connected_components(coo_matrix((np.ones(e, np.uint8), (rows, cols)), shape=(n, n)), directed=False)
        np.testing.assert_array_equal(K.connected_components(n, rows, cols), expected)
    for n in (0, 5):
        assert K.connected_components(n, np.zeros(0, np.int32), np.zeros(0, np.int32)).tolist() == list(range(n))


def test_within_matches_the_kdtree_and_excludes_the_exact_radius():
    rng = np.random.default_rng(4)
    points, seeds = rng.uniform(0, 5, (20000, 3)), rng.uniform(0, 5, (300, 3))
    for radius in (.05, .2, 1.0):
        np.testing.assert_array_equal(K.within(points, seeds, radius, -1), dc._within(points, seeds, radius))
    exact = np.array([[0., 0, 0], [.25, 0, 0], [np.nextafter(.25, 0), 0, 0]])
    assert K.within(exact, exact[:1], .25, 1).tolist() == dc._within(exact, exact[:1], .25).tolist() == [True, False, True]
    assert not K.within(points, np.zeros((0, 3)), .2, 1).any()


def test_edge_remap_matches_the_numpy_expression():
    rng = np.random.default_rng(5)
    old_n, new_n, e = 1000, 900, 20000
    rows, cols = rng.integers(0, old_n, e).astype(np.int32), rng.integers(0, old_n, e).astype(np.int32)
    old_to_new = np.full(old_n, -1, np.int32)
    survivors = rng.choice(old_n, new_n - 50, replace=False)
    old_to_new[survivors] = rng.permutation(new_n)[:len(survivors)]
    dirty = rng.random(new_n) < .2
    r, c = old_to_new[rows], old_to_new[cols]
    keep = (r >= 0) & (c >= 0)
    keep &= ~dirty[r]
    actual = K.remap_edges(rows, cols, old_to_new, dirty)
    np.testing.assert_array_equal(actual[0], r[keep])
    np.testing.assert_array_equal(actual[1], c[keep])


def test_splat_is_bit_identical(monkeypatch):
    rng = np.random.default_rng(6)
    w, h = 97, 61
    us, vs = rng.integers(0, w, 3000), rng.integers(0, h, 3000)
    z = rng.uniform(.2, 40, 3000)
    z[:6] = [1e-9, 1e300, np.inf, .05, .5, 1.0]  # huge, cap-sized and zero-sized footprints
    us[:4], vs[:4] = [0, w - 1, 0, w - 1], [0, h - 1, h - 1, 0]  # footprints clipped by every border
    cases = [(us, vs, z, 300.0, w, h, .05, 8), (us, vs, z, 300.0, w, h, 0.0, 8), (us, vs, z, 30.0, w, h, .3, 3),
             (us[:0], vs[:0], z[:0], 300.0, w, h, .05, 8)]
    native_results = [geometry_utils.splat_depth_buffer(*case) for case in cases]
    monkeypatch.setattr(native, "kernels", None)
    for case, result in zip(cases, native_results):
        expected = geometry_utils.splat_depth_buffer(*case)
        assert result.dtype == expected.dtype and result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)


def test_kernels_validate_their_inputs():
    points = np.zeros((4, 3))
    with pytest.raises(ValueError):
        K.surface_normals(np.zeros((4, 2)), np.arange(4), .2, 24, 1)
    with pytest.raises(IndexError):
        K.surface_normals(points, np.array([0, 4]), .2, 24, 1)
    with pytest.raises(ValueError):
        K.surface_normals(points, np.arange(4), 0.0, 24, 1)
    with pytest.raises(ValueError):
        K.surface_edges(np.array([[0., 0, 0], [np.nan, 0, 0]]), np.zeros((2, 3)), np.zeros(2, bool), np.arange(2),
                        .2, 24, .9, .04, 1)
    with pytest.raises(ValueError):
        K.within(points, np.array([[1e300, 0, 0]]), .2, 1)
    with pytest.raises(IndexError):
        K.connected_components(3, np.array([0, 3], np.int32), np.array([1, 1], np.int32))
    with pytest.raises(IndexError):
        K.remap_edges(np.array([5], np.int32), np.array([0], np.int32), np.zeros(3, np.int32), np.zeros(3, bool))


def _camera(stamp):
    intr = CameraIntrinsics(300.0, 300.0, 160.0, 120.0, 320, 240)
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    T[:3, 3] = [-2.0, 1.5, 1.0]
    detections = []
    for label, (y0, y1, x0, x1) in (("chair", (100, 200, 50, 150)), ("table", (60, 180, 140, 300))):
        mask = np.zeros((240, 320), bool)
        mask[y0:y1, x0:x1] = True
        detections.append(Detection2D(np.array([x0, y0, x1, y1], float), label, 0.9, mask=mask))
    return dc.CameraLabels(stamp, intr, T, detections)


@pytest.mark.parametrize("mode", ["snapshot", "scan"])
def test_dense_pipeline_gives_the_same_map_with_either_backend(mode):
    clouds = [_surfaces(40_000, seed) for seed in (8, 9)]
    clouds.insert(1, clouds[0] + (clouds[0][:, :1] < .5) * [0, 0, .3])  # a local change at one end of the room
    results = {}
    for backend in (False, True):
        pipeline = dc.DenseCloudPipeline(dc.DenseCloudConfig(input_mode=mode, native_kernels=backend,
                                                             label_propagation_radius=.2))
        runs = []
        for i, cloud in enumerate(clouds):
            pipeline.update(cloud, float(i))
            runs.append((pipeline.annotate(_camera(float(i))), dict(pipeline.last_segmentation)))
        results[backend] = runs
    assert results[True][0][0].stats["segmentation_backend"] == "native"
    assert results[False][0][0].stats["segmentation_backend"] == "python"
    assert results[True][1][1]["incremental"]  # the local change took the incremental path
    assert results[True][0][0].stats["regions"] >= 4 and (results[True][0][0].semantic_ids > 0).sum() > 100
    for (python, python_seg), (compiled, compiled_seg) in zip(results[False], results[True]):
        assert python_seg == compiled_seg
        for field in ("region_ids", "semantic_ids", "confidence", "label_source", "camera_visible", "map_points_world",
                      "map_region_ids", "map_semantic_ids", "map_confidence", "map_label_source"):
            np.testing.assert_array_equal(getattr(compiled, field), getattr(python, field), err_msg=field)
