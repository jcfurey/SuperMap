"""Save and restore through the node's services, then keep mapping."""
import json
from pathlib import Path

from std_srvs.srv import Trigger

from test.ros.helpers import feed_frames


def test_save_and_load_services_keep_identities(node_factory, dataset, tmp_path):
    map_dir = tmp_path / "map"
    node = node_factory("-p", f"map_save_path:={map_dir}")
    assert node.save_map_srv is not None and node.load_map_srv is not None
    frames = list(dataset)[:8]

    feed_frames(node, dataset, frames[:5])
    before = {o.instance_id: o.label for o in node.pipeline.object_map.objects.values()}
    assert before

    response = node._on_save_map(Trigger.Request(), Trigger.Response())
    assert response.success, response.message
    header = json.loads(Path(map_dir, "map.json").read_text())
    assert header["num_instances"] == len(before) and header["metadata"]["world_frame"] == "map"

    # Wipe the live map, then restore through the service (map_load_path empty -> map_save_path).
    node.pipeline.object_map.objects.clear()
    response = node._on_load_map(Trigger.Request(), Trigger.Response())
    assert response.success, response.message
    restored = node.pipeline.object_map.objects
    assert set(restored) == set(before)
    assert all(o.status.value in ("occluded", "tentative", "disappeared") for o in restored.values())

    feed_frames(node, dataset, frames[5:])  # re-observation reactivates the same IDs
    after = {o.instance_id: o.status.value for o in node.pipeline.object_map.objects.values()}
    assert set(before) <= set(after)
    assert any(status == "active" for status in after.values())
