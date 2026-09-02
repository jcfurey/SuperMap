"""The global instance-level map M_t (Sec. III-IV).

Owns the object registry, point fusion/voxelization, and the lifecycle
transitions (tentative -> active -> occluded -> disappeared) that combine
the geometric-consistency and semantic-fusion evidence into a single
object-level status.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from semantic_mapping import geometric_consistency as gc
from semantic_mapping import semantic_fusion as sf
from semantic_mapping.geometry_utils import bbox3d_from_points, clip_bbox_to_image, invert_se3, iou_3d, project_points
from semantic_mapping.tracking import TrackKalmanState, init_track
from semantic_mapping.types import Detection2D, ObjectInstance, ObjectStatus

_STATUS_RANK = {
    ObjectStatus.DISAPPEARED: 0,
    ObjectStatus.TENTATIVE: 1,
    ObjectStatus.OCCLUDED: 2,
    ObjectStatus.ACTIVE: 3,
}


def voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Indices (ascending) of one representative point per occupied voxel.

    Returning indices rather than points lets callers subset any per-point
    arrays (log-odds, membership) in lockstep, instead of relying on ordering
    assumptions.
    """
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _unique_keys, first_indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(first_indices)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deduplicate points onto a voxel grid, keeping the map's point budget bounded."""
    return points[voxel_downsample_indices(points, voxel_size)]


def _inside_detection(pixels: np.ndarray, detection: Detection2D, margin_px: float) -> np.ndarray:
    """Which projected pixels fall inside the detection's mask (or box, if no mask)."""
    inside = np.zeros(pixels.shape[0], dtype=bool)
    valid = (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0)
    if detection.mask is not None:
        h, w = detection.mask.shape
        ok = valid & (pixels[:, 0] < w) & (pixels[:, 1] < h)
        inside[ok] = detection.mask[pixels[ok, 1], pixels[ok, 0]]
    else:
        x1, y1, x2, y2 = detection.bbox
        inside = valid & (pixels[:, 0] >= x1 - margin_px) & (pixels[:, 0] <= x2 + margin_px) \
            & (pixels[:, 1] >= y1 - margin_px) & (pixels[:, 1] <= y2 + margin_px)
    return inside


class ObjectMap:
    """Container for the global map M_t and its update rules."""

    def __init__(
        self,
        voxel_size: float = 0.05,
        tau_eps: float = 0.15,
        max_points_per_object: int = 20000,
        prune_log_odds: float = -1.5,
        prune_membership: float = -1.5,
        membership_margin_px: float = 2.0,
        active_occupied_fraction: float = 0.6,
        disappeared_occupied_fraction: float = 0.2,
        max_occlusion_frames: int = 30,
        min_label_confidence: float = 0.4,
        min_observations_for_confidence_check: int = 5,
        tentative_max_age: int = 10,
        cull_out_of_view: bool = True,
    ) -> None:
        self.voxel_size = voxel_size
        self.tau_eps = tau_eps
        self.max_points_per_object = max_points_per_object
        self.prune_log_odds = prune_log_odds
        self.prune_membership = prune_membership
        self.membership_margin_px = membership_margin_px
        self.active_occupied_fraction = active_occupied_fraction
        self.disappeared_occupied_fraction = disappeared_occupied_fraction
        self.max_occlusion_frames = max_occlusion_frames
        self.min_label_confidence = min_label_confidence
        self.min_observations_for_confidence_check = min_observations_for_confidence_check
        self.tentative_max_age = tentative_max_age
        self.cull_out_of_view = cull_out_of_view

        self.objects: dict[int, ObjectInstance] = {}
        self._next_id = 1

    # ----------------------------------------------------------------- points
    def _bbox(self, points: np.ndarray) -> np.ndarray:
        """Axis-aligned box of a voxelized point set, padded by half a voxel per
        side: each stored point stands for its whole cell, and a surface seen
        from one side would otherwise be a zero-volume slab."""
        pad = self.voxel_size / 2.0
        return bbox3d_from_points(points) + np.array([-pad, -pad, -pad, pad, pad, pad])

    def _subset_points(self, instance: ObjectInstance, keep: np.ndarray) -> None:
        instance.points_world = instance.points_world[keep]
        instance.point_log_odds = instance.point_log_odds[keep]
        instance.point_membership = instance.point_membership[keep]
        if instance.points_world.shape[0] > 0:
            instance.bbox3d = self._bbox(instance.points_world)

    def _fuse_points(self, instance: ObjectInstance, new_points_world: np.ndarray) -> None:
        """Union new points into the instance, keeping per-point arrays aligned.

        New points enter with neutral (zero) log-odds and membership; voxel
        deduplication prefers the existing point (with its accumulated
        evidence) whenever an old and a new point share a voxel, because the
        existing points come first in the concatenation and np.unique keeps
        the first occurrence.
        """
        if new_points_world.shape[0] == 0:
            return
        n_new = new_points_world.shape[0]
        instance.points_world = np.concatenate([instance.points_world, new_points_world], axis=0)
        instance.point_log_odds = np.concatenate([instance.point_log_odds, np.zeros(n_new)])
        instance.point_membership = np.concatenate([instance.point_membership, np.zeros(n_new)])
        keep = voxel_downsample_indices(instance.points_world, self.voxel_size)
        if keep.size > self.max_points_per_object:
            rng = np.random.default_rng(instance.instance_id)
            keep = np.sort(rng.choice(keep, size=self.max_points_per_object, replace=False))
        self._subset_points(instance, keep)

    @staticmethod
    def boxes_in_view(bboxes: np.ndarray, K: np.ndarray, T_world_from_cam: np.ndarray, image_shape) -> np.ndarray:
        """Batched frustum test: False only for boxes provably outside the image.

        Every point of a box lies inside the convex hull of its corners, so
        with all corners in front of the camera the projected envelope bounds
        every point's projection. A box with corners on both sides of the
        camera plane is kept (its projection is not meaningful; the per-point
        test decides). One vectorized projection covers the whole map, which
        is what keeps the per-frame cost flat when most instances are behind
        the robot.
        """
        boxes = np.asarray(bboxes, dtype=np.float64).reshape(-1, 6)
        n = boxes.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        lo, hi = boxes[:, :3], boxes[:, 3:]
        corner_mask = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
        corners = lo[:, None, :] + (hi - lo)[:, None, :] * corner_mask[None, :, :]  # (n, 8, 3)
        pixels, depths = project_points(K, invert_se3(T_world_from_cam), corners.reshape(-1, 3))
        pixels, depths = pixels.reshape(n, 8, 2), depths.reshape(n, 8)
        all_behind = depths.max(axis=1) <= 0
        straddling = (depths.min(axis=1) <= 0) & ~all_behind
        height, width = image_shape[0], image_shape[1]
        x1, y1 = pixels[:, :, 0].min(axis=1) - 1, pixels[:, :, 1].min(axis=1) - 1
        x2, y2 = pixels[:, :, 0].max(axis=1) + 1, pixels[:, :, 1].max(axis=1) + 1
        overlaps = (x2 > 0) & (y2 > 0) & (x1 < width) & (y1 < height)
        return straddling | (~all_behind & overlaps)

    @classmethod
    def may_be_in_view(cls, bbox3d: np.ndarray, K: np.ndarray, T_world_from_cam: np.ndarray, image_shape) -> bool:
        return bool(cls.boxes_in_view(np.asarray(bbox3d)[None, :], K, T_world_from_cam, image_shape)[0])

    def _apply_evidence(
        self,
        instance: ObjectInstance,
        K: np.ndarray,
        T_world_from_cam: np.ndarray,
        depth_image: np.ndarray,
        detection: Detection2D | None = None,
        in_view: bool | None = None,
    ) -> bool:
        """Run Eq. (7)-(9) over the instance's points and, when a detection is
        associated this frame, the per-point membership update; then prune
        points that either evidence has ruled out. Returns whether any point
        was geometrically confirmed observable.

        Instances whose box is entirely outside the view get no evidence from
        this frame (every point would be OUT_OF_VIEW), so their per-point
        work is skipped outright: in a long deployment most of the map is
        behind the robot at any moment, and this keeps the update cost
        proportional to what the camera sees rather than to the map.
        ``in_view`` passes a visibility verdict computed in batch by the
        caller; ``None`` runs the per-instance test here.
        """
        if instance.points_world.shape[0] == 0 or depth_image is None:
            return False
        if in_view is False:
            return False
        if in_view is None and self.cull_out_of_view \
                and not self.may_be_in_view(instance.bbox3d, K, T_world_from_cam, depth_image.shape):
            return False
        log_odds, states, pixels = gc.update_object_points(
            K, T_world_from_cam, depth_image, instance.points_world, instance.point_log_odds, self.tau_eps,
        )
        observable = states == gc.GeometricState.OBSERVABLE
        instance.point_log_odds = log_odds
        if detection is not None:
            inside = _inside_detection(pixels, detection, self.membership_margin_px)
            instance.point_membership = sf.update_point_membership(instance.point_membership, observable, inside)

        contradicted = gc.prune_mask(instance.point_log_odds, self.prune_log_odds)
        instance.points_contradicted += int(contradicted.sum())
        keep = ~(contradicted | (instance.point_membership < self.prune_membership))
        self._subset_points(instance, keep)
        return bool(np.any(observable))

    # -------------------------------------------------------------- lifecycle
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
        n = fused_points.shape[0]
        instance = ObjectInstance(
            instance_id=instance_id,
            label_belief=sf.new_belief(label, score),
            points_world=fused_points,
            point_log_odds=np.zeros(n, dtype=np.float64),
            point_membership=np.zeros(n, dtype=np.float64),
            bbox3d=self._bbox(fused_points) if n else np.zeros(6),
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
        detection: Detection2D,
        stamp: float,
        K: np.ndarray,
        T_world_from_cam: np.ndarray,
        depth_image: np.ndarray,
    ) -> None:
        """Fuse a newly-associated detection into an existing instance.

        Runs the geometric-consistency update (Eq. 7-9) and the per-point
        membership update over the instance's existing points, merges in the
        freshly back-projected points from this frame's detection, and fuses
        the semantic label (Eq. 10). The label gets one update from the
        observation, plus one extra corroboration when geometry confirmed the
        existing points observable, i.e. the detection landed on the object
        we already had rather than on something new in front of it.
        """
        instance.track = track
        instance.latest_stamp = stamp
        instance.frames_since_seen = 0
        instance.hits += 1
        instance.status = ObjectStatus.ACTIVE
        instance.points_contradicted = 0  # re-detected: contradiction bookkeeping starts over

        corroborated = self._apply_evidence(instance, K, T_world_from_cam, depth_image, detection)
        if corroborated:
            instance.label_belief = sf.bayesian_label_update(instance.label_belief, detection.label, detection.score)

        self._fuse_points(instance, new_points_world)

        instance.label_belief = sf.bayesian_label_update(instance.label_belief, detection.label, detection.score)
        instance.label_belief = sf.prune_low_confidence_labels(instance.label_belief)

    def reactivate(
        self,
        instance: ObjectInstance,
        new_points_world: np.ndarray,
        detection: Detection2D,
        stamp: float,
        K: np.ndarray,
        T_world_from_cam: np.ndarray,
        depth_image: np.ndarray,
    ) -> None:
        """Re-attach a detection that matched in 3D but not in 2D (Sec. IV-B re-activation).

        The 2D tracklet state is re-seeded from the detection -- its prediction
        was what failed -- while the 3D state, label belief, and instance ID
        all carry over untouched.
        """
        self.update_matched(
            instance, init_track(detection.bbox), new_points_world, detection, stamp, K, T_world_from_cam, depth_image,
        )

    def update_unmatched(
        self,
        instance: ObjectInstance,
        K: np.ndarray,
        T_world_from_cam: np.ndarray,
        depth_image: np.ndarray,
        in_view: bool | None = None,
    ) -> None:
        """Advance an instance with no detection this frame: re-evaluate geometric
        evidence only (Sec. IV-B.3), which is how disappearances are detected
        even though no 2D detection ever fires "removed".
        """
        instance.frames_since_seen += 1
        self._apply_evidence(instance, K, T_world_from_cam, depth_image, in_view=in_view)

        if instance.status == ObjectStatus.TENTATIVE and instance.frames_since_seen > self.tentative_max_age:
            # Never corroborated: a one-off false detection, not an object.
            instance.status = ObjectStatus.DISAPPEARED
            return

        # Score survivors against everything that has been ruled out since the
        # last detection, so pruning contradicted points can't launder an
        # object back to "intact" (see ObjectInstance.points_contradicted).
        alive = instance.points_world.shape[0]
        occupied = gc.occupied_fraction(instance.point_log_odds) * alive
        fraction = occupied / (alive + instance.points_contradicted) if alive + instance.points_contradicted else 0.0
        if alive == 0 or fraction <= self.disappeared_occupied_fraction:
            instance.status = ObjectStatus.DISAPPEARED
        elif instance.status != ObjectStatus.TENTATIVE and (
            fraction < self.active_occupied_fraction or instance.frames_since_seen > self.max_occlusion_frames
        ):
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

    # ---------------------------------------------------------------- merging
    def _merge_into(self, keep: ObjectInstance, drop: ObjectInstance) -> None:
        # Concatenate keeping both instances' per-point evidence, then dedupe;
        # `keep` comes first so its points win shared voxels.
        keep.points_world = np.concatenate([keep.points_world, drop.points_world], axis=0)
        keep.point_log_odds = np.concatenate([keep.point_log_odds, drop.point_log_odds])
        keep.point_membership = np.concatenate([keep.point_membership, drop.point_membership])
        idx = voxel_downsample_indices(keep.points_world, self.voxel_size)
        if idx.size > self.max_points_per_object:
            rng = np.random.default_rng(keep.instance_id)
            idx = np.sort(rng.choice(idx, size=self.max_points_per_object, replace=False))
        self._subset_points(keep, idx)

        total_hits = keep.hits + drop.hits
        merged_belief: dict[str, float] = {}
        for belief, weight in ((keep.label_belief, keep.hits), (drop.label_belief, drop.hits)):
            for label, prob in belief.items():
                merged_belief[label] = merged_belief.get(label, 0.0) + weight * prob
        norm = sum(merged_belief.values()) or 1.0
        keep.label_belief = sf.prune_low_confidence_labels({k: v / norm for k, v in merged_belief.items()})

        keep.hits = total_hits
        keep.frames_since_seen = min(keep.frames_since_seen, drop.frames_since_seen)
        keep.first_seen_stamp = min(keep.first_seen_stamp, drop.first_seen_stamp)
        if drop.latest_stamp > keep.latest_stamp:
            keep.latest_stamp = drop.latest_stamp
            keep.track = drop.track
        if _STATUS_RANK[drop.status] > _STATUS_RANK[keep.status]:
            keep.status = drop.status
        del self.objects[drop.instance_id]

    def merge_duplicates(self, iou_threshold: float = 0.3, distance_threshold: float = 0.25) -> list[tuple[int, int]]:
        """Merge label-compatible live instances that occupy the same space.

        Duplicates arise when a detection failed to associate for a frame or
        two and spawned a second instance for the same physical object. The
        older instance ID always survives, preserving the identity that the
        scene graph's temporal edges already reference. Returns (kept, dropped)
        ID pairs.
        """
        merged: list[tuple[int, int]] = []
        live = sorted(
            (o for o in self.objects.values()
             if o.status != ObjectStatus.DISAPPEARED and o.points_world.shape[0] > 0),
            key=lambda o: o.instance_id,
        )
        if len(live) < 2:
            return merged
        # Candidate pairs from a KD-tree on centres: two boxes can only overlap
        # when their centres are closer than the sum of their half-diagonals,
        # so a radius of twice the largest half-diagonal (or the distance
        # threshold, if larger) keeps every pair the exhaustive loop would
        # merge, at O(n log n) instead of O(n^2) pair tests. A `keep` that grows
        # by absorbing a duplicate may reach a third instance beyond the
        # radius; that pair is caught on the next frame.
        centers = np.array([o.center for o in live])
        half_diagonals = np.linalg.norm(np.array([o.bbox3d[3:] - o.bbox3d[:3] for o in live]), axis=1) / 2.0
        radius = max(2.0 * float(half_diagonals.max()), distance_threshold) + 1e-6
        neighbours = cKDTree(centers).query_ball_point(centers, r=radius)
        for i, keep in enumerate(live):
            if keep.instance_id not in self.objects:
                continue
            for j in sorted(neighbours[i]):
                if j <= i:
                    continue
                drop = live[j]
                if drop.instance_id not in self.objects:
                    continue
                if not (set(keep.label_belief) & set(drop.label_belief)):
                    continue
                overlapping = iou_3d(keep.bbox3d, drop.bbox3d) > iou_threshold
                close = float(np.linalg.norm(keep.center - drop.center)) < distance_threshold
                if overlapping or close:
                    self._merge_into(keep, drop)
                    merged.append((keep.instance_id, drop.instance_id))
        return merged

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
