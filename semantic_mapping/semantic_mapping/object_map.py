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
from semantic_mapping.association import DEFAULT_LABEL_MIN_MASS, beliefs_compatible
from semantic_mapping import semantic_fusion as sf
from semantic_mapping.appearance import cosine_similarity, update_running_embedding
from semantic_mapping.geometry_utils import (
    bbox3d_from_points, exceeds_size_limit, invert_se3, iou_3d, pack_voxel_keys, project_points,
)
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
    assumptions. The three voxel coordinates are packed into one int64
    (``pack_voxel_keys``) so the de-duplication is a 1-D unique: row-wise
    ``np.unique(axis=0)`` was the single largest cost of the whole pipeline
    (doc/audit-2026-09.md, P2) and the packed key gives the same result about
    nine times faster. Coordinates outside the packable range fall back to the
    row-wise form.
    """
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    packed = pack_voxel_keys(keys)
    if packed is not None:
        _unique_keys, first_indices = np.unique(packed, return_index=True)
    else:
        _unique_keys, first_indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(first_indices)


_MIN_SUPPORTED_POINTS = 3
"""Fewest supported points a support-based box is computed from; fewer fall back to all points."""


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
        max_points_per_object: int = 5000,
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
        bbox_trim_percentile: float = 0.0,
        dynamic_geometry_labels: tuple[str, ...] | list[str] = (),
        dynamic_geometry_min_extent_fraction: float = 0.0,
        label_min_mass: float = DEFAULT_LABEL_MIN_MASS,
        prune_min_contradictions: int = 2,
        contradiction_window_px: int = 1,
        reconcile_max_distance_m: float = 10.0,
        reconcile_max_gap_sec: float = 120.0,
        bbox_min_support: int = 1,
        bbox_support_radius_m: float = 0.0,
        size_limits: dict[str, tuple[float, ...]] | None = None,
        bbox_support_miss: float = 0.0,
        bbox_support_cull_at: float = 0.0,
        bbox_support_max: float = 0.0,
        existence_hit_gain: float = 0.0,
        existence_miss_penalty: float = 1.0,
        existence_cull_log_odds: float = -3.0,
        existence_max_log_odds: float = 6.0,
        geometric_consistency: bool = True,
        semantic_fusion: bool = True,
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
        self.bbox_trim_percentile = bbox_trim_percentile
        self.dynamic_geometry_labels = frozenset(dynamic_geometry_labels)
        self.dynamic_geometry_min_extent_fraction = dynamic_geometry_min_extent_fraction
        self.label_min_mass = label_min_mass
        """Belief mass two instances must both give a shared label to merge or reconcile."""
        self.prune_min_contradictions = prune_min_contradictions
        self.contradiction_window_px = contradiction_window_px
        self.reconcile_max_distance_m = reconcile_max_distance_m
        self.reconcile_max_gap_sec = reconcile_max_gap_sec
        """Plausibility gate for reconcile_retired (0 disables either bound)."""
        self.bbox_min_support = bbox_min_support
        """Frames a point must be re-observed in to bound a static instance's box; 1 = off."""
        self.bbox_support_radius_m = bbox_support_radius_m or voxel_size * np.sqrt(3.0)
        self.size_limits = dict(size_limits or {})
        """label -> geometry_utils.parse_size_limits limit."""
        self.bbox_support_miss = bbox_support_miss
        """Support a mapped point loses when a matched frame looked at it (in the
        image, not occluded) and its lifted points did not re-hit it; 0 = support only grows."""
        self.bbox_support_cull_at = bbox_support_cull_at
        """With bbox_support_miss, points whose support falls to this value are removed."""
        self.bbox_support_max = bbox_support_max
        """Cap on per-point support so long-supported regions can still decay; 0 = uncapped."""
        self.existence_hit_gain = existence_hit_gain
        """Existence log-odds gained per matched detection, times its score; 0 = off."""
        self.existence_miss_penalty = existence_miss_penalty
        self.existence_cull_log_odds = existence_cull_log_odds
        self.existence_max_log_odds = existence_max_log_odds
        self.geometric_consistency = geometric_consistency
        """Table V ablation switch: False skips Eq. (7)-(9), so map points are never judged or pruned and
        no object is retired by geometry."""
        self.semantic_fusion = semantic_fusion
        """Table V ablation switch: False replaces Eq. (10) with the latest detection's label and skips the
        per-point mask-membership pruning."""
        self.stats = {"size_rejected_observations": 0, "size_refused_associations": 0, "size_refused_merges": 0,
                      "mask_completions": 0, "ground_contact_completions": 0,
                      "support_culled_points": 0, "existence_culled": 0,
                      "depth_split_matches": 0, "relabel_matches": 0}
        """Cumulative counts of size-limit, mask-completion and association decisions (the pipeline counts all
        but merges)."""

        self.objects: dict[int, ObjectInstance] = {}
        self._next_id = 1
        self._prepared_evidence: tuple | None = None

    @property
    def effective_prune_log_odds(self) -> float:
        """``prune_log_odds``, lowered so one contradicting frame cannot prune a
        fresh point (gc.min_contradiction_prune_threshold)."""
        return gc.min_contradiction_prune_threshold(self.prune_log_odds, self.prune_min_contradictions)

    # ----------------------------------------------------------------- points
    def _bbox(self, points: np.ndarray) -> np.ndarray:
        """Axis-aligned box of a voxelized point set, padded by half a voxel per
        side: each stored point stands for its whole cell, and a surface seen
        from one side would otherwise be a zero-volume slab."""
        pad = self.voxel_size / 2.0
        return bbox3d_from_points(points, self.bbox_trim_percentile) + np.array([-pad, -pad, -pad, pad, pad, pad])

    @property
    def tracks_support(self) -> bool:
        return self.bbox_min_support > 1

    @staticmethod
    def _support(instance: ObjectInstance) -> np.ndarray:
        """Per-point support aligned with the points; points without a record count once."""
        if instance.point_support.shape[0] != instance.points_world.shape[0]:
            instance.point_support = np.ones(instance.points_world.shape[0])
        return instance.point_support

    def _instance_bbox(self, instance: ObjectInstance) -> np.ndarray:
        """Reported box: with support tracking, a static instance that has been
        detected ``bbox_min_support`` times is bounded by the points re-observed
        in at least that many frames, so unconfirmed outliers (ground, background
        caught in one mask) stop inflating it and the box shrinks once they are
        not re-supported. Young instances, moving classes, and instances with
        too few supported points use all points."""
        if (self.tracks_support and instance.hits >= self.bbox_min_support
                and instance.label not in self.dynamic_geometry_labels):
            supported = self._support(instance) >= self.bbox_min_support
            if np.count_nonzero(supported) >= _MIN_SUPPORTED_POINTS:
                return self._bbox(instance.points_world[supported])
        return self._bbox(instance.points_world)

    def _reinforce_support(
        self,
        instance: ObjectInstance,
        new_points_world: np.ndarray,
        K: np.ndarray | None = None,
        T_world_from_cam: np.ndarray | None = None,
        depth_image: np.ndarray | None = None,
    ) -> None:
        """Count this frame for every mapped point that one of its lifted points lies near.

        With ``bbox_support_miss``, a point this frame looked at but did not
        re-hit loses support, so the box trims regions later observations no
        longer confirm; support stays two-sided evidence, not a hit counter.
        "Looked at" means it projects inside the image in front of the camera
        and the depth does not show a nearer surface on its pixel (occluded
        points and points outside a truncated view keep their support).
        """
        if not self.tracks_support or not len(new_points_world) or not len(instance.points_world):
            return
        distances, _ = cKDTree(new_points_world).query(
            instance.points_world, distance_upper_bound=self.bbox_support_radius_m)
        hit = distances <= self.bbox_support_radius_m
        support = self._support(instance) + hit
        if self.bbox_support_miss > 0 and K is not None and T_world_from_cam is not None and depth_image is not None:
            pixels, depths = project_points(K, invert_se3(T_world_from_cam), instance.points_world)
            h, w = depth_image.shape[:2]
            seen = (depths > 0) & (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] < h)
            states, _, _ = gc.project_and_classify(K, T_world_from_cam, depth_image, instance.points_world,
                                                   self.tau_eps, self.contradiction_window_px)
            seen &= states != gc.GeometricState.UNOBSERVABLE
            support = support - self.bbox_support_miss * (seen & ~hit)
        if self.bbox_support_max > 0:
            support = np.minimum(support, self.bbox_support_max)
        instance.point_support = support
        if self.bbox_support_miss > 0:
            keep = support > self.bbox_support_cull_at
            if not keep.all():
                self.stats["support_culled_points"] += int(np.count_nonzero(~keep))
                self._subset_points(instance, keep)

    def growth_exceeds_limit(self, instance: ObjectInstance, bbox3d: np.ndarray, labels=()) -> bool:
        """Whether adding ``bbox3d`` (unpadded) would grow ``instance`` past a size
        limit of its label or of ``labels``. An instance with no geometry, or one
        the addition does not grow, is never refused."""
        limits = [self.size_limits[label] for label in {instance.label, *labels} if label in self.size_limits]
        if not limits or not len(instance.points_world):
            return False
        pad = self.voxel_size / 2.0
        box = np.asarray(bbox3d, dtype=np.float64) + np.array([-pad, -pad, -pad, pad, pad, pad])
        union = np.concatenate([np.minimum(instance.bbox3d[:3], box[:3]), np.maximum(instance.bbox3d[3:], box[3:])])
        if np.all(union == instance.bbox3d):
            return False
        return any(exceeds_size_limit(union, limit) for limit in limits)

    @property
    def tracks_existence(self) -> bool:
        return self.existence_hit_gain > 0

    def _raise_existence(self, instance: ObjectInstance, score: float) -> None:
        if self.tracks_existence:
            instance.existence_log_odds = min(
                instance.existence_log_odds + self.existence_hit_gain * float(score), self.existence_max_log_odds)

    def _subset_points(self, instance: ObjectInstance, keep: np.ndarray) -> None:
        if self.tracks_support:
            instance.point_support = self._support(instance)[keep]
        instance.points_world = instance.points_world[keep]
        instance.point_log_odds = instance.point_log_odds[keep]
        instance.point_membership = instance.point_membership[keep]
        if instance.points_world.shape[0] > 0:
            instance.bbox3d = self._instance_bbox(instance)

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
        if self.tracks_support:
            instance.point_support = np.concatenate([self._support(instance), np.ones(n_new)])
        instance.points_world = np.concatenate([instance.points_world, new_points_world], axis=0)
        instance.point_log_odds = np.concatenate([instance.point_log_odds, np.zeros(n_new)])
        instance.point_membership = np.concatenate([instance.point_membership, np.zeros(n_new)])
        keep = self._cap_indices(instance, voxel_downsample_indices(instance.points_world, self.voxel_size),
                                 n_existing=instance.points_world.shape[0] - n_new)
        self._subset_points(instance, keep)

    def _cap_indices(self, instance: ObjectInstance, keep: np.ndarray, n_existing: int | None = None) -> np.ndarray:
        """Bound ``keep`` (indices into the instance's per-point arrays) to the point budget.

        A uniform resample threw away points that carry accumulated evidence
        and replaced them with neutral new ones, so large objects flickered
        and re-learned their occupancy every frame. Priority instead goes to
        points with evidence (non-zero log-odds or membership), then to the
        rest of the existing points (indices below ``n_existing``; all when
        None), and only the remaining budget is filled from new points.
        Within the tier that overflows, a seeded uniform sample keeps the
        spatial coverage even.
        """
        cap = self.max_points_per_object
        if keep.size <= cap:
            return keep
        n_existing = instance.points_world.shape[0] if n_existing is None else n_existing
        evidence = (instance.point_log_odds[keep] != 0) | (instance.point_membership[keep] != 0)
        if self.tracks_support:
            evidence |= self._support(instance)[keep] > 1
        existing = keep < n_existing
        rng = np.random.default_rng(instance.instance_id)
        chosen: list[np.ndarray] = []
        budget = cap
        for tier in (keep[existing & evidence], keep[existing & ~evidence], keep[~existing]):
            if budget <= 0:
                break
            if tier.size > budget:
                tier = rng.choice(tier, size=budget, replace=False)
            chosen.append(tier)
            budget -= tier.size
        return np.sort(np.concatenate(chosen))

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
        depth_image: np.ndarray | None,
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
        if instance.points_world.shape[0] == 0 or depth_image is None or not self.geometric_consistency:
            return False
        if in_view is False:
            return False
        if in_view is None and self.cull_out_of_view \
                and not self.may_be_in_view(instance.bbox3d, K, T_world_from_cam, depth_image.shape):
            return False
        log_odds, states, pixels = gc.update_object_points(
            K, T_world_from_cam, depth_image, instance.points_world, instance.point_log_odds, self.tau_eps,
            contradiction_window_px=self.contradiction_window_px,
            classification=self._prepared_classification(instance, K, T_world_from_cam, depth_image),
        )
        observable = states == gc.GeometricState.OBSERVABLE
        instance.point_log_odds = log_odds
        if detection is not None and self.semantic_fusion:
            inside = _inside_detection(pixels, detection, self.membership_margin_px)
            instance.point_membership = sf.update_point_membership(instance.point_membership, observable, inside)

        contradicted = gc.prune_mask(instance.point_log_odds, self.effective_prune_log_odds)
        if self._preserve_compact_body(instance):
            # Keep the last supported body intact while accumulating evidence.
            # Unknown/occluded points retain their prior; contradicted points
            # can retire the whole body instead of leaving a floor fragment.
            return bool(np.any(observable))
        instance.points_contradicted += int(contradicted.sum())
        keep = ~(contradicted | (instance.point_membership < self.prune_membership))
        if not np.all(keep):
            self._subset_points(instance, keep)
        return bool(np.any(observable))

    def prepare_evidence(self, instances, K: np.ndarray, T_world_from_cam: np.ndarray,
                         depth_image: np.ndarray | None) -> None:
        """Project and classify the points of every instance about to receive evidence, in one pass.

        :meth:`_apply_evidence` then uses an instance's prepared
        classification instead of projecting its points itself, but only
        while the instance still holds the very point array that was
        projected and is updated with the same ``K``, pose and depth image
        objects; anything else is classified on the spot. Clear with
        :meth:`discard_prepared_evidence`.
        """
        self._prepared_evidence = None
        if depth_image is None or not self.geometric_consistency:
            return
        chosen = [instance for instance in instances if instance.points_world.shape[0] > 0]
        if not chosen:
            return
        results = gc.project_and_classify_many(
            K, T_world_from_cam, depth_image, [instance.points_world for instance in chosen], self.tau_eps,
            self.contradiction_window_px)
        self._prepared_evidence = (K, T_world_from_cam, depth_image, {
            instance.instance_id: (instance.points_world, states, pixels)
            for instance, (states, _delta_d, pixels) in zip(chosen, results)
        })

    def discard_prepared_evidence(self) -> None:
        self._prepared_evidence = None

    def _prepared_classification(self, instance: ObjectInstance, K: np.ndarray, T_world_from_cam: np.ndarray,
                                 depth_image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        prepared = self._prepared_evidence
        if prepared is None or prepared[0] is not K or prepared[1] is not T_world_from_cam \
                or prepared[2] is not depth_image:
            return None
        entry = prepared[3].get(instance.instance_id)
        if entry is None or entry[0] is not instance.points_world:
            return None
        return entry[1], entry[2]

    def _preserve_compact_body(self, instance: ObjectInstance) -> bool:
        return instance.label in self.dynamic_geometry_labels and self.dynamic_geometry_min_extent_fraction > 0

    def _body_contradicted(self, instance: ObjectInstance) -> bool:
        if not self._preserve_compact_body(instance) or not len(instance.points_world):
            return False
        supported = ((instance.point_log_odds >= self.effective_prune_log_odds)
                     & (instance.point_membership >= self.prune_membership))
        if supported.all():
            return False
        if not supported.any():
            return True
        reference = np.ptp(instance.points_world, axis=0)
        remaining = np.ptp(instance.points_world[supported], axis=0)
        # Ignore thin/unobserved axes: a single viewed surface is not a full
        # volume. Compare only axes spanning at least four voxel cells.
        measured = reference >= 4 * self.voxel_size
        return bool(np.any(remaining[measured] < reference[measured] * self.dynamic_geometry_min_extent_fraction))

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
            point_support=np.ones(n) if self.tracks_support else np.zeros(0),
            bbox3d=self._bbox(fused_points) if n else np.zeros(6),
            status=ObjectStatus.TENTATIVE,
            track=init_track(bbox_2d),
            first_seen_stamp=stamp,
            latest_stamp=stamp,
            frames_since_seen=0,
            hits=1,
            geometry_stamp=stamp if n else None,
        )
        if embedding is not None:
            instance.embedding, instance.embedding_count = update_running_embedding(None, 0, embedding)
        self._raise_existence(instance, score)
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
        in_view: bool | None = None,
    ) -> None:
        """Fuse a newly-associated detection into an existing instance.

        Runs the geometric-consistency update (Eq. 7-9) and the per-point
        membership update over the instance's existing points, merges in the
        freshly back-projected points from this frame's detection, and fuses
        the semantic label (Eq. 10) exactly once. When geometry confirmed the
        existing points observable -- the detection landed on the object we
        already had rather than on something new in front of it -- that
        update uses the sharper corroborated likelihood
        (``sf.P_SELF_CORROBORATED``). ``in_view`` passes the caller's batched
        frustum verdict through to the evidence update.
        """
        instance.track = track
        instance.latest_stamp = stamp
        instance.frames_since_seen = 0
        instance.missed_detection_frames = 0
        instance.hits += 1
        self._raise_existence(instance, detection.score)
        # Newly matched tentative tracks still need the configured hit check.
        # Retired tracks also pass through it: a retired identity may have
        # expired before it was ever confirmed. Confirmed ones already have
        # enough hits and are promoted again at the end of this frame.
        instance.status = (ObjectStatus.TENTATIVE
                           if instance.status in (ObjectStatus.TENTATIVE, ObjectStatus.DISAPPEARED)
                           else ObjectStatus.ACTIVE)
        has_geometry = len(new_points_world) > 0
        dynamic = detection.label in self.dynamic_geometry_labels
        if has_geometry or not dynamic:
            instance.points_contradicted = 0
        # A visible 2D silhouette with inadequate depth is not a measurement
        # of its body or evidence that the last body has become empty space.
        corroborated = (self._apply_evidence(instance, K, T_world_from_cam, depth_image, detection, in_view=in_view)
                        if has_geometry or not dynamic else False)

        if detection.label in self.dynamic_geometry_labels and len(new_points_world):
            # A moving object's past locations belong in its trajectory, not
            # in its current volume. Missing depth must not erase the last
            # supported geometry, and identity/semantic evidence stay intact.
            instance.points_world = np.zeros((0, 3), dtype=np.float64)
            instance.point_log_odds = np.zeros(0)
            instance.point_membership = np.zeros(0)
            instance.point_support = np.zeros(0)
        else:
            self._reinforce_support(instance, new_points_world, K, T_world_from_cam, depth_image)
        self._fuse_points(instance, new_points_world)
        if has_geometry:
            instance.geometry_stamp = stamp
        elif dynamic and instance.status != ObjectStatus.TENTATIVE:
            instance.status = ObjectStatus.OCCLUDED

        if self.semantic_fusion:
            instance.label_belief = sf.bayesian_label_update(
                instance.label_belief, detection.label, detection.score,
                p_self=sf.P_SELF_CORROBORATED if corroborated else sf.P_SELF)
            instance.label_belief = sf.prune_low_confidence_labels(instance.label_belief)
        else:
            instance.label_belief = sf.new_belief(detection.label, detection.score)  # the latest detection decides
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
            instance.point_support = np.zeros(0)
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
        in_view: bool | None = None,
    ) -> None:
        """Re-attach a detection that matched in 3D but not in 2D (Sec. IV-B re-activation).

        The 2D tracklet state is re-seeded from the detection -- its prediction
        was what failed -- while the 3D state, label belief, and instance ID
        all carry over untouched.
        """
        self.update_matched(
            instance, init_track(detection.bbox), new_points_world, detection, stamp, K, T_world_from_cam, depth_image,
            in_view=in_view,
        )

    def update_unmatched(
        self,
        instance: ObjectInstance,
        K: np.ndarray,
        T_world_from_cam: np.ndarray,
        depth_image: np.ndarray | None,
        in_view: bool | None = None,
        detections_evaluated: bool = True,
    ) -> None:
        """Advance an instance with no detection this frame: re-evaluate geometric
        evidence only (Sec. IV-B.3), which is how disappearances are detected
        even though no 2D detection ever fires "removed".

        A detector miss is only charged when the detector could have seen the
        instance: the frame's detections were evaluated, the instance is in
        view, and -- for an instance with mapped points and a depth image --
        at least one point was confirmed observable. An object that was seen
        once and then left the view (or hid behind something) is not a
        false positive. Without depth, visibility falls back to ``in_view``
        (the frustum test; 2D-only tracks count as visible) so tentative
        tracks still expire on detector misses alone. ``depth_image=None``
        applies no geometric evidence and leaves the status untouched apart
        from that expiry.
        """
        instance.frames_since_seen += 1
        observed = self._apply_evidence(instance, K, T_world_from_cam, depth_image, in_view=in_view)
        if depth_image is None or instance.points_world.shape[0] == 0 or not self.geometric_consistency:
            visible = in_view is not False
        else:
            visible = observed
        if detections_evaluated and visible:
            instance.missed_detection_frames += 1
            if self.tracks_existence:
                instance.existence_log_odds -= self.existence_miss_penalty
                if instance.existence_log_odds < self.existence_cull_log_odds:
                    # Repeatedly visible to the detector and not re-detected:
                    # culled even when confirmed, whatever its geometry says.
                    self.stats["existence_culled"] += 1
                    instance.status = ObjectStatus.DISAPPEARED
                    return

        if instance.status == ObjectStatus.TENTATIVE and instance.missed_detection_frames > self.tentative_max_age:
            # Never corroborated: a one-off false detection, not an object.
            instance.status = ObjectStatus.DISAPPEARED
            return
        if depth_image is None:
            return

        # Score survivors against everything that has been ruled out since the
        # last detection, so pruning contradicted points can't launder an
        # object back to "intact" (see ObjectInstance.points_contradicted).
        alive = instance.points_world.shape[0]
        if alive == 0 and instance.status == ObjectStatus.TENTATIVE and instance.geometry_stamp is None:
            return  # retain the 2D-only track until its detector-miss budget expires
        occupied = gc.occupied_fraction(instance.point_log_odds) * alive
        denominator = alive if self._preserve_compact_body(instance) else alive + instance.points_contradicted
        fraction = occupied / denominator if denominator else 0.0
        if alive == 0 or fraction <= self.disappeared_occupied_fraction or self._body_contradicted(instance):
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
            if instance.label in self.dynamic_geometry_labels:
                if not len(instance.points_world):
                    return  # a 2D track has no mapped location yet
                instance.status = (ObjectStatus.ACTIVE if instance.geometry_stamp == instance.latest_stamp
                                   else ObjectStatus.OCCLUDED)
            else:
                instance.status = ObjectStatus.ACTIVE

    # ---------------------------------------------------------------- merging
    def _merge_into(self, keep: ObjectInstance, drop: ObjectInstance) -> None:
        if keep.label in self.dynamic_geometry_labels and drop.label in self.dynamic_geometry_labels:
            # Preserve the oldest ID without reviving an older body position.
            newest = max((keep, drop), key=lambda o: (o.geometry_stamp if o.geometry_stamp is not None else -np.inf,
                                                     len(o.points_world)))
            for name in ('points_world', 'point_log_odds', 'point_membership', 'point_support', 'bbox3d'):
                setattr(keep, name, getattr(newest, name).copy())
            keep.geometry_stamp = newest.geometry_stamp
        else:
            # Keep both instances' per-point evidence, then dedupe; `keep`
            # comes first so its points win shared voxels.
            support = None
            if self.tracks_support:
                support = np.concatenate([self._support(keep), self._support(drop)])
            keep.points_world = np.concatenate([keep.points_world, drop.points_world], axis=0)
            keep.point_log_odds = np.concatenate([keep.point_log_odds, drop.point_log_odds])
            keep.point_membership = np.concatenate([keep.point_membership, drop.point_membership])
            if support is not None and len(support):
                # A shared voxel keeps the better-supported record of the two.
                _, cells = np.unique(np.floor(keep.points_world / self.voxel_size).astype(np.int64), axis=0,
                                     return_inverse=True)
                cells = cells.ravel()
                best = np.zeros(cells.max() + 1)
                np.maximum.at(best, cells, support)
                keep.point_support = best[cells]
            idx = self._cap_indices(keep, voxel_downsample_indices(keep.points_world, self.voxel_size))
            keep.hits += drop.hits  # the merged box is judged by the merged hit count
            self._subset_points(keep, idx)
            keep.hits -= drop.hits
            stamps = [o.geometry_stamp for o in (keep, drop) if o.geometry_stamp is not None]
            keep.geometry_stamp = max(stamps, default=None)

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
        keep.existence_log_odds = max(keep.existence_log_odds, drop.existence_log_odds)
        keep.missed_detection_frames = min(keep.missed_detection_frames, drop.missed_detection_frames)
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
                if not beliefs_compatible(keep.label_belief, drop.label_belief, self.label_min_mass):
                    continue
                overlapping = iou_3d(keep.bbox3d, drop.bbox3d) > iou_threshold
                close = float(np.linalg.norm(keep.center - drop.center)) < distance_threshold
                if (overlapping or close) and self.size_limits:
                    pad = self.voxel_size / 2.0
                    unpadded = drop.bbox3d + np.array([pad, pad, pad, -pad, -pad, -pad])
                    if self.growth_exceeds_limit(keep, unpadded, (drop.label,)):
                        self.stats["size_refused_merges"] += 1
                        continue
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

        Appearance alone is weak evidence for "the same object" (two chairs
        of one model look identical), so the candidate must also be
        physically plausible: within ``reconcile_max_distance_m`` of where
        the retired instance was, and first seen within
        ``reconcile_max_gap_sec`` of when it was last seen (0 disables
        either bound).
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
                if not beliefs_compatible(candidate.label_belief, old.label_belief, self.label_min_mass):
                    continue
                if (self.reconcile_max_gap_sec > 0
                        and candidate.first_seen_stamp - old.latest_stamp > self.reconcile_max_gap_sec):
                    continue
                if (self.reconcile_max_distance_m > 0
                        and float(np.linalg.norm(candidate.center - old.center)) > self.reconcile_max_distance_m):
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
        keep.point_support = moved.point_support
        keep.bbox3d = moved.bbox3d
        keep.track = moved.track
        keep.status = moved.status  # the pipeline confirms tentative reconciliations using the combined hits
        keep.latest_stamp = moved.latest_stamp
        keep.frames_since_seen = moved.frames_since_seen
        keep.missed_detection_frames = moved.missed_detection_frames
        keep.points_contradicted = moved.points_contradicted
        keep.geometry_stamp = moved.geometry_stamp
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
                instance.point_support = np.zeros(0)
        evicted: list[int] = []
        if len(retired) > max_retired:
            for instance in sorted(retired, key=lambda o: o.latest_stamp)[: len(retired) - max_retired]:
                del self.objects[instance.instance_id]
                evicted.append(instance.instance_id)
        return evicted

    def shift_stamps(self, offset: float) -> None:
        """Add ``offset`` seconds to every stored timestamp (clock-epoch rebase, see
        SemanticMappingPipeline.load): latest/first-seen/geometry stamps and the
        trajectory, so ages and orderings between instances are preserved."""
        if offset == 0:
            return
        for obj in self.objects.values():
            obj.first_seen_stamp += offset
            obj.latest_stamp += offset
            if obj.geometry_stamp is not None:
                obj.geometry_stamp += offset
            obj.trajectory = [(stamp + offset, center, status) for stamp, center, status in obj.trajectory]

    def newest_stamp(self) -> float | None:
        """Newest timestamp stored anywhere in the map, or None for an empty map."""
        stamps = [s for obj in self.objects.values()
                  for s in (obj.latest_stamp, obj.first_seen_stamp, obj.geometry_stamp,
                            *(t[0] for t in obj.trajectory[-1:])) if s is not None]
        return max(stamps, default=None)

    def active_objects(self) -> list[ObjectInstance]:
        return [obj for obj in self.objects.values() if obj.status == ObjectStatus.ACTIVE]
