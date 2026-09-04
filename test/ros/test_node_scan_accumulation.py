"""Sparse LiDAR: accumulating the last N scans through TF densifies the rasterized depth."""
import numpy as np

from test.ros.helpers import feed_frames


def _raster_densities(node, dataset, frames):
    densities = []
    original = node._process_and_publish

    def spy(observation, header):
        densities.append(float((observation.depth > 0).mean()))
        return original(observation, header)

    node._process_and_publish = spy
    feed_frames(node, dataset, frames, point_fraction=0.05)  # 5% of the scan, like a sparse LiDAR
    return densities


def test_accumulated_scans_densify_depth_and_keep_the_map(node_factory, dataset):
    frames = list(dataset)[:6]

    node = node_factory("-p", "pointcloud_accumulate_scans:=1", "-p", "depth_fill_radius_px:=2")
    assert node.pipeline.config.depth_fill_radius_px == 2
    single = _raster_densities(node, dataset, frames)
    assert len(node._scan_history) <= 1
    labels_single = {o.label for o in node.pipeline.object_map.objects.values()}

    node = node_factory("-p", "pointcloud_accumulate_scans:=3", "-p", "depth_fill_radius_px:=2")
    accumulated = _raster_densities(node, dataset, frames)
    assert len(node._scan_history) == 3
    assert np.mean(accumulated[2:]) > 2.0 * np.mean(single)
    assert len(node.pipeline.object_map.objects) >= 5
    assert labels_single <= {o.label for o in node.pipeline.object_map.objects.values()}
