"""The global instance-level map M_t (Sec. III-IV).

Owns the object registry, point fusion/voxelization, and the lifecycle
transitions (tentative -> active -> occluded -> disappeared) that combine
the geometric-consistency and semantic-fusion evidence into a single
object-level status.
"""
from __future__ import annotations

import numpy as np

from semantic_mapping import geometric_consistency as gc
from semantic_mapping import semantic_fusion as sf
from semantic_mapping.geometry_utils import bbox3d_from_points
from semantic_mapping.tracking import TrackKalmanState, init_track
from semantic_mapping.types import ObjectInstance, ObjectStatus


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deduplicate points onto a voxel grid, keeping the map's point budget bounded."""
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _unique_keys, first_indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first_indices)]


class ObjectMap:
    """Container for the global map M_t and its update rules."""

    def __init__(
        self,
        voxel_size: float = 0.05,
        tau_eps: float = 0.15,
        max_points_per_object: int = 20000,
        prune_log_odds: float = -1.5,
        active_occupied_fraction: float = 0.6,
        disappeared_occupied_fraction: float = 0.2,
        max_occlusion_frames: int = 30,
        min_label_confidence: float = 0.4,
        min_observations_for_confidence_check: int = 5,
    ) -> None:
        self.voxel_size = voxel_size
        self.tau_eps = tau_eps
        self.max_points_per_object = max_points_per_object
        self.prune_log_odds = prune_log_odds
        self.active_occupied_fraction = active_occupied_fraction
        self.disappeared_occupied_fraction = disappeared_occupied_fraction
        self.max_occlusion_frames = max_occlusion_frames
        self.min_label_confidence = min_label_confidence
        self.min_observations_for_confidence_check = min_observations_for_confidence_check

        self.objects: dict[int, ObjectInstance] = {}
        self._next_id = 1

    def spawn(
        self,
        bbox_2d: np.ndarray,
        points_world: np.ndarray,
        label: str,
        score: float,
        stamp: float,
    ) -> ObjectInstance:
        """Create a new tentative instance from an unmatched detection."""
        instance_id = self._next_id
        self._next_id += 1

        fused_points = voxel_downsample(points_world, self.voxel_size)
        log_odds = np.zeros(fused_points.shape[0], dtype=np.float64)
        bbox3d = bbox3d_from_points(fused_points) if fused_points.shape[0] else np.zeros(6)

        instance = ObjectInstance(
            instance_id=instance_id,
            label_belief=sf.new_belief(label, score),
            points_world=fused_points,
            point_log_odds=log_odds,
            bbox3d=bbox3d,
            status=ObjectStatus.TENTATIVE,
            track=init_track(bbox_2d),
            first_seen_stamp=stamp,
            latest_stamp=stamp,
            frames_since_seen=0,
            hits=1,
        )
        self.objects[instance_id] = instance
        return instance

    def update_matched(
        self,
        instance: ObjectInstance,
        track: TrackKalmanState,
        new_points_world: np.ndarray,
        label: str,
        score: float,
        stamp: float,
        K: np.ndarray,
        T_world_from_cam: np.ndarray,
        depth_image: np.ndarray,
    ) -> None:
        """Fuse a newly-associated detection into an existing instance.

        Runs the geometric-consistency update (Eq. 7-9) over the instance's
        existing points, merges in the freshly back-projected points from
        this frame's detection, and fuses the semantic label only for
        points confirmed :attr:`~semantic_mapping.geometric_consistency.GeometricState.OBSERVABLE`.
        """
        instance.track = track
        instance.latest_stamp = stamp
        instance.frames_since_seen = 0
        instance.hits += 1
        instance.status = ObjectStatus.ACTIVE

        if instance.points_world.shape[0] > 0:
            updated_log_odds, states = gc.update_object_points(
                K, T_world_from_cam, depth_image, instance.points_world,
                instance.point_log_odds, self.tau_eps,
            )
            instance.point_log_odds = updated_log_odds
            observable = states == gc.GeometricState.OBSERVABLE
            keep = ~gc.prune_mask(instance.point_log_odds, self.prune_log_odds)
            instance.points_world = instance.points_world[keep]
            instance.point_log_odds = instance.point_log_odds[keep]
            if np.any(observable):
                instance.label_belief = sf.bayesian_label_update(instance.label_belief, label, score)

        merged = np.concatenate([instance.points_world, new_points_world], axis=0) \
            if instance.points_world.shape[0] else new_points_world
        merged = voxel_downsample(merged, self.voxel_size)
        if merged.shape[0] > self.max_points_per_object:
            keep_idx = np.random.default_rng(instance.instance_id).choice(
                merged.shape[0], size=self.max_points_per_object, replace=False,
            )
            merged = merged[keep_idx]

        new_point_count = merged.shape[0] - instance.points_world.shape[0]
        if new_point_count > 0:
            fresh_log_odds = np.zeros(new_point_count, dtype=np.float64)
            instance.point_log_odds = np.concatenate([instance.point_log_odds, fresh_log_odds])
        instance.points_world = merged

        if not instance.label_belief:
            instance.label_belief = sf.new_belief(label, score)
        else:
            instance.label_belief = sf.bayesian_label_update(instance.label_belief, label, score)
        instance.label_belief = sf.prune_low_confidence_labels(instance.label_belief)

        if instance.points_world.shape[0] > 0:
            instance.bbox3d = bbox3d_from_points(instance.points_world)

    def update_unmatched(
        self,
        instance: ObjectInstance,
        K: np.ndarray,
        T_world_from_cam: np.ndarray,
        depth_image: np.ndarray,
    ) -> None:
        """Advance an instance with no detection this frame: re-evaluate geometric
        evidence only (Sec. IV-B.3), which is how disappearances are detected
        even though no 2D detection ever fires "removed".
        """
        instance.frames_since_seen += 1

        if instance.points_world.shape[0] > 0:
            updated_log_odds, _states = gc.update_object_points(
                K, T_world_from_cam, depth_image, instance.points_world,
                instance.point_log_odds, self.tau_eps,
            )
            instance.point_log_odds = updated_log_odds
            keep = ~gc.prune_mask(instance.point_log_odds, self.prune_log_odds)
            instance.points_world = instance.points_world[keep]
            instance.point_log_odds = instance.point_log_odds[keep]
            if instance.points_world.shape[0] > 0:
                instance.bbox3d = bbox3d_from_points(instance.points_world)

        fraction = gc.occupied_fraction(instance.point_log_odds)
        if instance.points_world.shape[0] == 0 or fraction <= self.disappeared_occupied_fraction:
            instance.status = ObjectStatus.DISAPPEARED
        elif fraction < self.active_occupied_fraction or \
                instance.frames_since_seen > self.max_occlusion_frames:
            instance.status = ObjectStatus.OCCLUDED

    def confirm_tentative(self, instance: ObjectInstance, min_hits: int) -> None:
        """Promote a tentative track to active once it has enough corroborating hits."""
        if instance.status != ObjectStatus.TENTATIVE:
            return
        if sf.should_discard_instance(
            instance.label_belief, self.min_label_confidence,
            self.min_observations_for_confidence_check, instance.hits,
        ):
            instance.status = ObjectStatus.DISAPPEARED
            return
        if instance.hits >= min_hits:
            instance.status = ObjectStatus.ACTIVE

    def prune_disappeared(self, grace_period_frames: int = 60) -> list[int]:
        """Drop instances that have been confirmed disappeared for a while, freeing memory.

        Their instance IDs are still retired (never reused), preserving the
        temporal-edge history already recorded in the scene graph.
        """
        to_remove = [
            instance_id for instance_id, instance in self.objects.items()
            if instance.status == ObjectStatus.DISAPPEARED
            and instance.frames_since_seen > grace_period_frames
        ]
        for instance_id in to_remove:
            del self.objects[instance_id]
        return to_remove

    def active_objects(self) -> list[ObjectInstance]:
        return [obj for obj in self.objects.values() if obj.status == ObjectStatus.ACTIVE]
