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
from semantic_mapping.appearance import cosine_similarity, update_running_embedding
from semantic_mapping.geometry_utils import bbox3d_from_points, clip_bbox_to_image, invert_se3, iou_3d, project_points
from semantic_mapping.tracking import TrackKalmanState, init_track
from semantic_mapping.types import Detection2D, ObjectInstance, ObjectStatus

_STATUS_RANK = {
    ObjectStatus.DISAPPEARED: 0,
    ObjectStatus.TENTATIVE: 1,
    ObjectStatus.OCCLUDED: 2,
    ObjectStatus.ACTIVE: 3,
}


_VOXEL_KEY_BITS = 21
_VOXEL_KEY_OFFSET = 1 << (_VOXEL_KEY_BITS - 1)
"""Voxel coordinates are packed three-per-int64 with 21 bits each, i.e. any
map within +-2^20 voxels of the origin (+-52 km at 5 cm) takes the fast path."""


def voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Indices (ascending) of one representative point per occupied voxel.

    Returning indices rather than points lets callers subset any per-point
    arrays (log-odds, membership) in lockstep, instead of relying on ordering
    assumptions. The three voxel coordinates are packed into one int64 so the
    de-duplication is a 1-D unique: row-wise ``np.unique(axis=0)`` was the
    single largest cost of the whole pipeline (doc/audit-2026-09.md, P2) and
    the packed key gives the same result about nine times faster. Coordinates
    outside the packable range fall back to the row-wise form.
    """
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    shifted = keys + _VOXEL_KEY_OFFSET
    if np.all((shifted >= 0) & (shifted < (1 << _VOXEL_KEY_BITS))):
        packed = (shifted[:, 0] << (2 * _VOXEL_KEY_BITS)) | (shifted[:, 1] << _VOXEL_KEY_BITS) | shifted[:, 2]
        _unique_keys, first_indices = np.unique(packed, return_index=True)
    else:
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
        embedding: np.ndarray | None = None,
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
        if embedding is not None:
            instance.embedding, instance.embedding_count = update_running_embedding(None, 0, embedding)
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
        # Newly matched tentative tracks still need the configured hit check.
        # Retired tracks also pass through it: a retired identity may have
        # expired before it was ever confirmed. Confirmed ones already have
        # enough hits and are promoted again at the end of this frame.
        instance.status = (ObjectStatus.TENTATIVE
                           if instance.status in (ObjectStatus.TENTATIVE, ObjectStatus.DISAPPEARED)
                           else ObjectStatus.ACTIVE)
        instance.points_contradicted = 0  # re-detected: contradiction bookkeeping starts over

        corroborated = self._apply_evidence(instance, K, T_world_from_cam, depth_image, detection)
        if corroborated:
            instance.label_belief = sf.bayesian_label_update(instance.label_belief, detection.label, detection.score)

        self._fuse_points(instance, new_points_world)

        instance.label_belief = sf.bayesian_label_update(instance.label_belief, detection.label, detection.score)
        instance.label_belief = sf.prune_low_confidence_labels(instance.label_belief)
        if detection.embedding is not None:
            instance.embedding, instance.embedding_count = update_running_embedding(
                instance.embedding, instance.embedding_count, detection.embedding)

    def revive(
        self,
        instance: ObjectInstance,
        new_points_world: np.ndarray,
        detection: Detection2D,
        stamp: float,
        K: np.ndarray,
        T_world_from_cam: np.ndarray,
        depth_image: np.ndarray,
        relocated: bool,
    ) -> None:
        """Bring a retired (disappeared) instance back under its own ID.

        Back in the same place, its old points are kept and re-judged by the
        evidence update like any re-observed object. Relocated, the old
        geometry describes where it *was*, so the point set restarts from
        this detection while the ID, label belief, appearance, and trajectory
        (which now records the move) carry over.
        """
        if relocated or instance.points_world.shape[0] == 0:
            instance.points_world = np.zeros((0, 3), dtype=np.float64)
            instance.point_log_odds = np.zeros(0, dtype=np.float64)
            instance.point_membership = np.zeros(0, dtype=np.float64)
        self.update_matched(
            instance, init_track(detection.bbox), new_points_world, detection, stamp, K, T_world_from_cam, depth_image,
        )

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

        if drop.embedding is not None:
            if keep.embedding is None:
                keep.embedding, keep.embedding_count = drop.embedding, drop.embedding_count
            else:
                total = keep.embedding_count + drop.embedding_count
                mixed = keep.embedding * keep.embedding_count + drop.embedding * drop.embedding_count
                keep.embedding = (mixed / max(float(np.linalg.norm(mixed)), 1e-8)).astype(np.float32)
                keep.embedding_count = total

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
            # Exact per-pair bound: two boxes can only overlap (or be within the
            # distance threshold) when their centres are closer than the sum of
            # their half-diagonals plus the threshold; the global radius above
            # is the loosest such bound and admits far too many pairs at scale.
            candidates = np.array(sorted(j for j in neighbours[i] if j > i), dtype=np.int64)
            if candidates.size == 0:
                continue
            distances = np.linalg.norm(centers[candidates] - centers[i], axis=1)
            candidates = candidates[distances <= half_diagonals[i] + half_diagonals[candidates] + distance_threshold]
            for j in candidates:
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

    def reconcile_retired(
        self, newly_retired: list[ObjectInstance], min_similarity: float,
    ) -> list[tuple[int, int]]:
        """Fold a relocated object's provisional record under its original ID.

        An object moved before its old spot was confirmed empty is, at the
        time it is detected at the new place, still an *occluded* instance,
        so the detection spawns a provisional instance. Once the old spot is
        confirmed empty and the instance retires, this looks for a live
        instance with a compatible label that first appeared after the
        retired one was last seen and matches its appearance, and merges the
        two: the original ID keeps its history and gains the new geometry,
        and the trajectory records the move. Returns (kept, dropped) ID pairs.
        """
        merged: list[tuple[int, int]] = []
        for old in newly_retired:
            if old.embedding is None or old.instance_id not in self.objects:
                continue
            best, best_similarity = None, min_similarity
            for candidate in self.objects.values():
                if candidate is old or candidate.status == ObjectStatus.DISAPPEARED or candidate.embedding is None:
                    continue
                if candidate.first_seen_stamp <= old.latest_stamp or candidate.points_world.shape[0] == 0:
                    continue
                if not (set(candidate.label_belief) & set(old.label_belief)):
                    continue
                similarity = cosine_similarity(candidate.embedding, old.embedding)
                if similarity >= best_similarity:
                    best, best_similarity = candidate, similarity
            if best is not None:
                self._absorb_relocated(old, best)
                merged.append((old.instance_id, best.instance_id))
        return merged

    def _absorb_relocated(self, keep: ObjectInstance, moved: ObjectInstance) -> None:
        """``keep`` (retired) takes over ``moved``'s geometry and tracking state."""
        keep.points_world = moved.points_world
        keep.point_log_odds = moved.point_log_odds
        keep.point_membership = moved.point_membership
        keep.bbox3d = moved.bbox3d
        keep.track = moved.track
        keep.status = moved.status  # the pipeline confirms tentative reconciliations using the combined hits
        keep.latest_stamp = moved.latest_stamp
        keep.frames_since_seen = moved.frames_since_seen
        keep.points_contradicted = moved.points_contradicted
        total_hits = keep.hits + moved.hits
        merged_belief: dict[str, float] = {}
        for belief, weight in ((keep.label_belief, keep.hits), (moved.label_belief, moved.hits)):
            for label, prob in belief.items():
                merged_belief[label] = merged_belief.get(label, 0.0) + weight * prob
        norm = sum(merged_belief.values()) or 1.0
        keep.label_belief = sf.prune_low_confidence_labels({k: v / norm for k, v in merged_belief.items()})
        keep.hits = total_hits
        if moved.embedding is not None and keep.embedding is not None:
            mixed = keep.embedding * keep.embedding_count + moved.embedding * moved.embedding_count
            keep.embedding = (mixed / max(float(np.linalg.norm(mixed)), 1e-8)).astype(np.float32)
            keep.embedding_count += moved.embedding_count
        # History: where it was until it went missing, a gap, then where it turned up.
        left_at = moved.first_seen_stamp
        before = [s for s in keep.trajectory if s[0] < left_at and s[2] != ObjectStatus.DISAPPEARED.value]
        last_center = before[-1][1] if before else keep.trajectory[-1][1] if keep.trajectory else keep.center
        keep.trajectory = before + [(left_at, np.array(last_center, dtype=np.float64), ObjectStatus.DISAPPEARED.value)] \
            + [s for s in moved.trajectory if s[0] >= left_at]
        del self.objects[moved.instance_id]

    def compact_disappeared(self, grace_period_frames: int = 60, max_retired: int = 1000) -> list[int]:
        """Retire disappeared instances: keep their identity, release their points.

        A disappeared instance stays in the map -- its ID, label belief, last
        box, appearance, and trajectory are what re-identification and
        "where was it" queries need -- but after ``grace_period_frames`` its
        per-point arrays are released, since the geometry no longer describes
        anything present. Beyond ``max_retired`` retired instances the least
        recently seen are evicted (their IDs are never reused). Returns the
        evicted IDs. Call once per frame: disappeared instances are not
        visited by the per-frame update, so their age advances here.
        """
        retired = [o for o in self.objects.values() if o.status == ObjectStatus.DISAPPEARED]
        for instance in retired:
            instance.frames_since_seen += 1
            if instance.frames_since_seen > grace_period_frames and instance.points_world.shape[0] > 0:
                instance.points_world = np.zeros((0, 3), dtype=np.float64)
                instance.point_log_odds = np.zeros(0, dtype=np.float64)
                instance.point_membership = np.zeros(0, dtype=np.float64)
        evicted: list[int] = []
        if len(retired) > max_retired:
            for instance in sorted(retired, key=lambda o: o.latest_stamp)[: len(retired) - max_retired]:
                del self.objects[instance.instance_id]
                evicted.append(instance.instance_id)
        return evicted

    def active_objects(self) -> list[ObjectInstance]:
        return [obj for obj in self.objects.values() if obj.status == ObjectStatus.ACTIVE]
