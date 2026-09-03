import json

import numpy as np
import pytest

from semantic_mapping import persistence
from semantic_mapping.object_map import ObjectMap
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.tracking import current_bbox, current_velocity
from semantic_mapping.types import ObjectStatus
from test.helpers import make_object
from test.test_pipeline import _observation


def _populated_map() -> ObjectMap:
    object_map = ObjectMap()
    chair = make_object(3, "chair", [0, 0, 0, 1, 1, 1])
    chair.label_belief = {"chair": 0.8, "stool": 0.2}
    chair.points_world = np.random.default_rng(0).uniform(size=(20, 3))
    chair.point_log_odds = np.linspace(-2.0, 2.0, 20)
    chair.point_membership = np.linspace(-1.0, 1.0, 20)
    chair.track.state[4:] = [3.0, -1.0]
    chair.hits, chair.frames_since_seen, chair.points_contradicted = 7, 2, 4
    chair.first_seen_stamp, chair.latest_stamp = 1.5, 9.25
    chair.trajectory = [(1.5, np.array([0.5, 0.5, 0.5]), "tentative"), (2.0, np.array([0.6, 0.5, 0.5]), "active")]
    plant = make_object(5, "plant", [2, 2, 0, 3, 3, 1], status=ObjectStatus.DISAPPEARED)  # no points at all
    object_map.objects = {3: chair, 5: plant}
    object_map._next_id = 11
    return object_map


def test_roundtrip_restores_every_field(tmp_path):
    saved = _populated_map()
    persistence.save_map(saved, tmp_path / "map", metadata={"note": "x"})
    restored = ObjectMap()
    header = persistence.load_map(tmp_path / "map", restored, resume=False)

    assert header["num_instances"] == 2 and header["metadata"] == {"note": "x"}
    assert restored._next_id == 11 and set(restored.objects) == {3, 5}
    for instance_id in (3, 5):
        s, r = saved.objects[instance_id], restored.objects[instance_id]
        assert r.label_belief == s.label_belief and r.status == s.status
        np.testing.assert_allclose(r.points_world, s.points_world, atol=1e-6)  # stored as float32
        np.testing.assert_allclose(r.point_log_odds, s.point_log_odds, atol=1e-6)
        np.testing.assert_allclose(r.point_membership, s.point_membership, atol=1e-6)
        np.testing.assert_allclose(r.bbox3d, s.bbox3d)
        np.testing.assert_allclose(r.track.state, s.track.state)
        np.testing.assert_allclose(r.track.covariance, s.track.covariance)
        assert (r.first_seen_stamp, r.latest_stamp) == (s.first_seen_stamp, s.latest_stamp)
        assert (r.frames_since_seen, r.hits, r.points_contradicted) == (s.frames_since_seen, s.hits, s.points_contradicted)
        assert len(r.trajectory) == len(s.trajectory)
        for (rs, rc, rst), (ss, sc, sst) in zip(r.trajectory, s.trajectory):
            assert rs == ss and rst == sst and np.allclose(rc, sc)

    # Saving the restored map reproduces the same records byte-for-byte.
    persistence.save_map(restored, tmp_path / "again")
    first = json.loads((tmp_path / "map" / persistence.MAP_JSON).read_text())["instances"]
    second = json.loads((tmp_path / "again" / persistence.MAP_JSON).read_text())["instances"]
    assert first == second


def test_resume_resets_tracklet_and_demotes_active_to_occluded(tmp_path):
    saved = _populated_map()
    persistence.save_map(saved, tmp_path / "map")
    restored = ObjectMap()
    persistence.load_map(tmp_path / "map", restored)

    chair = restored.objects[3]
    assert chair.status == ObjectStatus.OCCLUDED
    assert np.allclose(current_velocity(chair.track), 0.0)
    np.testing.assert_allclose(current_bbox(chair.track), current_bbox(saved.objects[3].track))
    assert restored.objects[5].status == ObjectStatus.DISAPPEARED  # untouched


def test_next_id_never_reuses_a_saved_id(tmp_path):
    saved = _populated_map()
    saved._next_id = 1  # a counter that lost track
    persistence.save_map(saved, tmp_path / "map")
    restored = ObjectMap()
    persistence.load_map(tmp_path / "map", restored)
    assert restored._next_id == 6


def test_empty_map_roundtrip(tmp_path):
    persistence.save_map(ObjectMap(), tmp_path / "empty")
    restored = ObjectMap()
    restored.objects = {1: make_object(1, "x", [0, 0, 0, 1, 1, 1])}
    persistence.load_map(tmp_path / "empty", restored)
    assert restored.objects == {} and restored._next_id == 1


def test_rejects_unknown_format_version(tmp_path):
    persistence.save_map(_populated_map(), tmp_path / "map")
    header_path = tmp_path / "map" / persistence.MAP_JSON
    header = json.loads(header_path.read_text())
    header["format_version"] = 99
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="format version"):
        persistence.load_map(tmp_path / "map", ObjectMap())


def test_pipeline_continues_from_saved_map_with_the_same_identity(tmp_path):
    first = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2))
    for i in range(5):
        first.process_frame(_observation(stamp=i * 0.1, distance=2.0, with_detection=True))
    (chair_id,) = first.object_map.objects
    hits_before = first.object_map.objects[chair_id].hits
    first.save(tmp_path / "map")

    second = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2))
    header = second.load(tmp_path / "map")
    assert header["metadata"]["frame_index"] == 5
    assert second.object_map.objects[chair_id].status == ObjectStatus.OCCLUDED

    result = None
    for i in range(5, 9):
        result = second.process_frame(_observation(stamp=i * 0.1, distance=2.0, with_detection=True))
    assert [o.instance_id for o in result.objects] == [chair_id]  # re-observed, not re-spawned
    assert result.objects[0].status == ObjectStatus.ACTIVE
    assert result.objects[0].hits > hits_before


def test_embeddings_round_trip(tmp_path):
    saved = _populated_map()
    saved.objects[3].embedding = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    saved.objects[3].embedding_count = 4
    persistence.save_map(saved, tmp_path / "map")
    restored = ObjectMap()
    persistence.load_map(tmp_path / "map", restored, resume=False)
    np.testing.assert_allclose(restored.objects[3].embedding, [0.6, 0.8, 0.0], atol=1e-6)
    assert restored.objects[3].embedding.dtype == np.float32 and restored.objects[3].embedding_count == 4
    assert restored.objects[5].embedding is None and restored.objects[5].embedding_count == 0
