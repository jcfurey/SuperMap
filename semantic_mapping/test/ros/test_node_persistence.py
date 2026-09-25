"""Save and restore through the node's services, then keep mapping."""
import json
from pathlib import Path

import rclpy
from supermap_msgs.srv import LoadMap, SaveMap
from visualization_msgs.msg import Marker

from test.ros.helpers import BackgroundExecutor, feed_frames


def test_save_and_load_services_keep_identities(node_factory, dataset, tmp_path):
    map_dir = tmp_path / "map"
    node = node_factory("-p", f"map_save_path:={map_dir}")
    assert node.save_map_srv is not None and node.load_map_srv is not None
    frames = list(dataset)[:8]

    feed_frames(node, dataset, frames[:5])
    before = {o.instance_id: o.label for o in node.pipeline.object_map.objects.values()}
    assert before

    # Empty path -> map_save_path.
    response = node._on_save_map(SaveMap.Request(), SaveMap.Response())
    assert response.success, response.message
    assert Path(response.path) == map_dir
    header = json.loads(Path(map_dir, "map.json").read_text())
    assert header["num_instances"] == len(before) and header["metadata"]["world_frame"] == "map"

    # Wipe the live map, then restore through the service (empty path, map_load_path empty -> map_save_path).
    markers = []
    node.obj_boxes_pub.publish = markers.append
    node.pipeline.object_map.objects.clear()
    response = node._on_load_map(LoadMap.Request(), LoadMap.Response())
    assert response.success, response.message
    assert response.path == str(map_dir)
    restored = node.pipeline.object_map.objects
    assert set(restored) == set(before)
    assert all(o.status.value in ("occluded", "tentative", "disappeared") for o in restored.values())
    # R17: loading clears whatever RViz still shows from the previous map.
    assert markers and markers[0].markers[0].action == Marker.DELETEALL

    feed_frames(node, dataset, frames[5:])  # re-observation reactivates the same IDs
    after = {o.instance_id: o.status.value for o in node.pipeline.object_map.objects.values()}
    assert set(before) <= set(after)
    assert any(status == "active" for status in after.values())


def test_services_take_explicit_paths_over_ros(node_factory, dataset, tmp_path):
    """Explicit paths win over the configured ones; calls go through the executor like a real client."""
    node = node_factory("-p", f"map_save_path:={tmp_path / 'configured'}")
    feed_frames(node, dataset, list(dataset)[:4])
    count = len(node.pipeline.object_map.objects)
    assert count

    client_node = rclpy.create_node("persistence_test_client")
    try:
        save = client_node.create_client(SaveMap, "/semantic_mapping_node/save_map")
        load = client_node.create_client(LoadMap, "/semantic_mapping_node/load_map")
        with BackgroundExecutor(node, client_node):
            assert save.wait_for_service(timeout_sec=5.0) and load.wait_for_service(timeout_sec=5.0)
            explicit = tmp_path / "explicit"
            response = save.call(SaveMap.Request(path=str(explicit)), timeout_sec=10.0)
            assert response.success and Path(response.path) == explicit
            assert (explicit / "map.json").is_file() and not (tmp_path / "configured").exists()

            node.pipeline.object_map.objects.clear()
            response = load.call(LoadMap.Request(path=str(explicit)), timeout_sec=10.0)
            assert response.success and response.path == str(explicit)
            assert len(node.pipeline.object_map.objects) == count

            response = load.call(LoadMap.Request(path=str(tmp_path / "missing")), timeout_sec=10.0)
            assert not response.success and "load failed" in response.message
    finally:
        client_node.destroy_node()
