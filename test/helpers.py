"""Shared test fixtures (not a test module itself -- no test_ prefix)."""
import numpy as np

from semantic_mapping.tracking import init_track
from semantic_mapping.types import ObjectInstance, ObjectStatus


def make_object(instance_id: int, label: str, bbox3d, status=ObjectStatus.ACTIVE) -> ObjectInstance:
    bbox3d = np.array(bbox3d, dtype=np.float64)
    return ObjectInstance(
        instance_id=instance_id,
        label_belief={label: 1.0},
        points_world=np.zeros((0, 3)),
        point_log_odds=np.zeros(0),
        bbox3d=bbox3d,
        status=status,
        track=init_track(np.array([0.0, 0.0, 10.0, 10.0])),
        first_seen_stamp=0.0,
        latest_stamp=0.0,
    )
