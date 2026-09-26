"""Online pipeline orchestration: wires detection, association, tracking,
geometric consistency, semantic fusion, the object map, and the scene graph
into the single per-frame update described by Sec. III, Eq. (1)-(2).

This module is imported by both the ROS2 node (live mode) and the offline
example runner, so all sensor I/O concerns live outside it: callers hand in
an :class:`~semantic_mapping.types.Observation` already carrying pose,
intrinsics, RGB/depth, and (asynchronously arriving) detections, and get
back the updated map and scene graph.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from semantic_mapping import association, persistence, scene_graph as sg, tracking
from semantic_mapping.appearance import build_embedder
from semantic_mapping.geometry_utils import (
    GROUND_SURFACE_LABELS,
    back_project_depth,
    bbox3d_from_points,
    clip_bbox_to_image,
    exceeds_size_limit,
    fill_sparse_depth,
    fit_ground_plane,
    foreground_depth_mask,
    parse_size_limits,
    transform_points,
)
from semantic_mapping.object_map import ObjectMap
from semantic_mapping.types import Detection2D, ObjectInstance, ObjectStatus, Observation


GROUND_FIT_MAX_SAMPLES = 2048
"""Readings a local ground fit uses at most (evenly strided), so dense depth
images cost no more per detection than a sparse LiDAR raster."""

LOADED_MAP_EPOCH_GAP_SEC = 1e-3
"""After a clock-epoch rebase, how long before the first new observation the
loaded map's newest stamp is placed (SemanticMappingPipeline.load)."""


def _box_pixels(bbox: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Integer pixel bounds ``(x1, y1, x2, y2)`` of a box, clipped to the image."""
    h, w = image_shape
    x1, y1, x2, y2 = (int(np.clip(v, 0, limit)) for v, limit in zip(np.asarray(bbox).astype(int), (w, h, w, h)))
    return x1, y1, x2, y2


@dataclass
class PipelineConfig:
    voxel_size: float = 0.05
    min_depth_m: float = 0.0
    max_depth_m: float = 0.0
    """Usable optical depth interval for geometry and evidence; max 0 disables the upper bound."""
    bbox_trim_percentile: float = 0.0
    """Optional per-axis percentile trimming for object bounds; 0 retains full extents."""
    mask_depth_mad_factor: float = 0.0
    """Optional robust depth gate inside instance masks; 0 disables it for full-depth objects."""
    mask_depth_min_tolerance_m: float = 0.15
    foreground_depth_gap_m: float = 0.0
    """Separate masked sensor depths into layers; 0 disables foreground selection."""
    foreground_depth_min_points: int = 5
    foreground_depth_min_fraction: float = 0.1
    foreground_depth_min_image_span: float = 0.0
    """Minimum fraction of the visible mask span covered by real returns in each image axis."""
    foreground_depth_max_extent_m: float = 0.0
    """Optional compact-object bound per camera axis; reject implausible layers, never clip them."""
    foreground_depth_labels: list[str] = field(default_factory=lambda: ["person"])
    foreground_depth_largest_labels: list[str] = field(default_factory=list)
    """Of foreground_depth_labels, those that keep the supported layer with the
    most real returns instead of the nearest (which is often the ground strip
    in front of the object). Empty = nearest for all."""
    ground_exclusion: bool = True
    """Drop an instance mask's readings on the local ground before layer
    selection and back-projection when at least ``ground_exclusion_min_returns``
    of them stand above it, for every class but ``ground_surface_labels``.
    A mask bleeds onto the ground at an object's base, and with a LiDAR the
    ground rings there are nearer than the object: they were taken as its
    depth, so boxes sat on the ring in front of the object and slid along
    with the sensor. Masks with no ground fitted below the camera (e.g. a
    world frame that is not z up) or lying flat on it keep every reading."""
    ground_exclusion_min_returns: int = 3
    ground_surface_labels: list[str] = field(default_factory=lambda: list(GROUND_SURFACE_LABELS))
    ground_removal_labels: list[str] = field(default_factory=list)
    """Drop masked depth samples on the local ground surface before layer
    selection and back-projection (world z up; geometry_utils.fit_ground_plane
    over the real returns in and around the mask), unconditionally (unlike
    ``ground_exclusion``: even when nothing is left). Empty disables it."""
    ground_clearance_m: float = 0.15
    """Samples less than this above the fitted ground are dropped; the rest of an object on the ground is kept."""
    ground_context_px: int = 20
    """Margin around the mask's box whose returns also inform the ground fit (the ground in front of and beside it)."""
    ground_max_slope: float = 0.25
    """Steepest local ground accepted (rise over run); steeper fits fall back to a level ground."""
    mask_completion_labels: list[str] = field(default_factory=list)
    """Complete these classes' instance masks: when the kept layer (after
    ground removal and layer selection) has at least
    ``mask_completion_min_returns`` real returns, every mask pixel without a
    reading (on a ``mask_completion_stride_px`` grid, plus the silhouette's
    extreme pixels) is back-projected at their median range, so the box's
    width/height follow the mask and its depth extent the real returns.
    Completed points are geometry only: they never enter layer selection or
    the evidence depth. Empty disables it."""
    mask_completion_min_returns: int = 3
    mask_completion_stride_px: int = 4
    ground_contact_depth_labels: list[str] = field(default_factory=list)
    """Masks of these classes with fewer than ``mask_completion_min_returns``
    kept returns are completed at the range where the ray through the mask's
    bottom-centre pixel meets the local ground (fitted as for ground removal;
    a level fallback or a hit behind the camera or beyond max_depth_m gives
    no geometry). Empty disables it."""
    bbox_min_support: int = 1
    """Report a static instance's box from points re-observed in at least this
    many frames (ObjectMap.point_support); young instances use all points. 1
    disables support tracking."""
    bbox_support_radius_m: float = 0.0
    """A frame supports a mapped point when one of its lifted points lies this
    close; 0 = one voxel diagonal. Sparse LiDAR rings move between frames, so
    outdoors use a few ring spacings (e.g. 0.3)."""
    bbox_support_miss: float = 0.0
    """With support tracking, a mapped point that a matched frame looked at (in
    the image, in front, not occluded by nearer depth) but did not re-hit loses
    this much support, so boxes trim regions later views no longer confirm and
    expand once new regions reach bbox_min_support. 0 = support only grows."""
    bbox_support_cull_at: float = 0.0
    """With bbox_support_miss, points whose support falls to this value are removed."""
    bbox_support_max: float = 0.0
    """Cap on per-point support so a long-supported wrong region can still decay; 0 = uncapped."""
    existence_hit_gain: float = 0.0
    """Per-instance existence log-odds gained per matched detection, times its
    score. Frames in which the detector could have seen the instance and did
    not subtract existence_miss_penalty; below existence_cull_log_odds the
    instance is culled (disappeared), confirmed or not. 0 disables."""
    existence_miss_penalty: float = 1.0
    existence_cull_log_odds: float = -3.0
    existence_max_log_odds: float = 6.0
    """Cap so a long-lived instance can still be culled after sustained misses."""
    class_size_limits: list[str] = field(default_factory=list)
    """Per-class size priors as 'label:D' (max 3D box diagonal) or 'label:H,V'
    (max footprint diagonal, max height), metres. Oversized observations lose
    their geometry (the 2D detection is kept), and associations or merges that
    would grow an instance past its limit are refused. Empty disables it."""
    dynamic_geometry_enabled: bool = False
    dynamic_geometry_labels: list[str] = field(default_factory=lambda: ["person"])
    """Use the latest supported geometry for these moving classes; keep identity/history."""
    dynamic_geometry_min_extent_fraction: float = 0.0
    """Retire a contradicted compact body as a whole when a supported axis collapses; 0 disables."""
    tau_eps: float = 0.15
    max_points_per_object: int = 5000
    prune_log_odds: float = -1.5
    prune_min_contradictions: int = 2
    """Contradicting frames a fresh point needs before it is pruned; lowers
    ``prune_log_odds`` where needed (gc.min_contradiction_prune_threshold). 1
    restores single-frame pruning."""
    contradiction_window_px: int = 1
    """Radius of the window whose nearest depth reading must also lie behind a
    point to contradict it (gc.project_and_classify); protects silhouettes. 0 = off."""
    prune_membership: float = -1.5
    membership_margin_px: float = 2.0
    active_occupied_fraction: float = 0.6
    disappeared_occupied_fraction: float = 0.2
    max_occlusion_frames: int = 30
    min_label_confidence: float = 0.4
    min_observations_for_confidence_check: int = 5
    min_hits_to_confirm: int = 2
    tentative_max_age: int = 10
    association_iou_threshold: float = 0.3
    high_score_threshold: float = 0.5
    low_score_iou_threshold: float = 0.2
    reactivation_iou_threshold: float = 0.05
    reactivation_margin_m: float = 0.25
    label_compatibility_min_mass: float = 0.1
    """Belief mass the detection's label must hold in an instance for the
    label-gated association stages to fuse it there, and that a shared label
    must hold in both instances for them to merge or reconcile
    (association.labels_compatible). The relabelling stage is not label-gated."""
    use_2d_tracker: bool = True
    """Table V ablation switch ("W/o 2D Tracker"): False associates in 3D only
    (re-activation and re-identification); the 2D stages and relabelling are skipped."""
    use_semantic_fusion: bool = True
    """Table V ablation switch ("W/o Semantic Fusion"): False keeps the latest
    detection's label instead of Eq. (10) and skips per-point membership pruning."""
    use_geometric_consistency: bool = True
    """Table V ablation switch ("W/o Geometric Consistency Update"): False skips
    Eq. (7)-(9); map points are never judged or pruned and objects never retire by geometry."""
    match_max_gap_m: float = 1.0
    """2D→3D validation (Fig. 2): a 2D match whose detection lifts farther than
    this from the instance's box is split (association.split_depth_inconsistent).
    0 disables."""
    relabel_min_overlap: float = 0.5
    """A high-confidence detection with a label the instance has not taken joins
    a visible, unmatched track when this fraction of the smaller 3D box lies
    inside the other and its 2D box overlaps the track's prediction or the part
    of the instance the depth shows this frame; Eq. (10) then weighs the new
    label (association.associate_relabel). 0 disables."""
    min_points_for_3d_association: int = 5
    merge_iou_threshold: float = 0.3
    merge_distance_m: float = 0.25
    disappeared_prune_grace_frames: int = 60
    """Frames after which a disappeared instance's points are released; its
    identity stays in the map for re-identification (ObjectMap.compact_disappeared)."""

    max_retired_instances: int = 1000
    appearance_embedder: str = "color_histogram"
    """none | color_histogram | clip: descriptor attached to detections and kept
    per instance for re-identification (semantic_mapping.appearance)."""

    embedder_device: str = "cuda"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    reid_enabled: bool = True
    reid_min_similarity: float = 0.85
    """Appearance similarity below which a retired instance is not the same object."""

    reid_max_age_sec: float = 0.0
    """How long after it was last seen a retired instance may be re-identified (0 = unlimited)."""
    reconcile_max_distance_m: float = 10.0
    reconcile_max_gap_sec: float = 120.0
    """Plausibility gate for a relocation, both when folding a provisional
    instance into a just-retired one (ObjectMap.reconcile_retired) and when
    re-identifying a retired instance elsewhere by appearance (association.
    reidentify): how far it may have moved and how long after it was last seen
    it may have turned up. 0 disables either bound."""
    scene_graph_cluster_radius: float = 2.0
    scene_graph_z_tolerance: float = 0.1
    scene_graph_xy_iou_threshold: float = 0.05
    scene_graph_beside_max_distance: float = 1.0
    scene_graph_on_min_footprint_fraction: float = 0.5
    """``on`` also holds when this fraction of the upper object's footprint lies
    over the support, so a small object on a large table qualifies despite a tiny
    IoU_xy (scene_graph._pair_relations). 0 keeps the paper's IoU test alone."""
    scene_graph_support_classes: list[str] = field(default_factory=lambda: list(sg.DEFAULT_SUPPORT_CLASSES))
    max_points_per_detection: int = 4000
    size_prior_weight: float = 0.0
    """How far the predicted 2D box size follows the projected, image-clipped
    3D box (0 = Kalman size only, 1 = projection only); see tracking.predict."""
    cull_out_of_view: bool = True
    """Skip the per-point geometric update for instances whose 3D box lies
    entirely outside the current view (ObjectMap.may_be_in_view)."""

    depth_fill_radius_px: int = 0
    """Fill pixels without a depth reading from valid neighbours within this
    radius (geometry_utils.fill_sparse_depth). 0 for dense depth sources; 2-3
    for a LiDAR scan rasterized into the camera. The geometric-consistency
    evidence uses the image filled from all readings; a detection is
    back-projected through depth filled only from readings inside its own
    mask (or box). Background returns already inside the silhouette require
    separate foreground selection."""

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_depth_m) or self.min_depth_m < 0:
            raise ValueError("min_depth_m must be finite and nonnegative")
        if not np.isfinite(self.max_depth_m) or self.max_depth_m < 0:
            raise ValueError("max_depth_m must be finite and nonnegative")
        if self.max_depth_m and self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m (or be zero to disable)")
        if not 0 <= self.bbox_trim_percentile < 50:
            raise ValueError("bbox_trim_percentile must be in [0, 50)")
        if not np.isfinite(self.mask_depth_mad_factor) or self.mask_depth_mad_factor < 0:
            raise ValueError("mask_depth_mad_factor must be finite and nonnegative")
        if not np.isfinite(self.mask_depth_min_tolerance_m) or self.mask_depth_min_tolerance_m <= 0:
            raise ValueError("mask_depth_min_tolerance_m must be finite and positive")
        if not np.isfinite(self.foreground_depth_gap_m) or self.foreground_depth_gap_m < 0:
            raise ValueError("foreground_depth_gap_m must be finite and nonnegative")
        if self.foreground_depth_min_points < 1 or int(self.foreground_depth_min_points) != self.foreground_depth_min_points:
            raise ValueError("foreground_depth_min_points must be a positive integer")
        if not 0 <= self.foreground_depth_min_fraction <= 1:
            raise ValueError("foreground_depth_min_fraction must be in [0, 1]")
        for name in ('foreground_depth_min_image_span', 'dynamic_geometry_min_extent_fraction'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not np.isfinite(self.foreground_depth_max_extent_m) or self.foreground_depth_max_extent_m < 0:
            raise ValueError("foreground_depth_max_extent_m must be finite and nonnegative")
        for name in ('foreground_depth_largest_labels', 'ground_removal_labels', 'class_size_limits',
                     'mask_completion_labels', 'ground_contact_depth_labels', 'ground_surface_labels'):
            if isinstance(getattr(self, name), str) or not all(isinstance(v, str) for v in getattr(self, name)):
                raise ValueError(f"{name} must be a list of strings")
        for name in ('ground_clearance_m', 'ground_max_slope'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.bbox_support_radius_m) or self.bbox_support_radius_m < 0:
            raise ValueError("bbox_support_radius_m must be finite and nonnegative")
        parse_size_limits(self.class_size_limits)
        for name in ('bbox_support_miss', 'bbox_support_max', 'existence_hit_gain', 'existence_miss_penalty'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ('bbox_support_cull_at', 'existence_cull_log_odds', 'existence_max_log_odds'):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.existence_cull_log_odds >= self.existence_max_log_odds:
            raise ValueError("existence_cull_log_odds must be below existence_max_log_odds")
        if self.bbox_support_max and self.bbox_support_max < self.bbox_min_support:
            raise ValueError("bbox_support_max must be 0 or at least bbox_min_support")
        # Counts and budgets that the per-frame update divides by, indexes
        # with, or feeds to bbox3d_from_points: a zero here used to be
        # accepted and crash mid-frame after the map had already changed.
        for name in ('min_points_for_3d_association', 'max_points_per_object', 'max_points_per_detection',
                     'min_hits_to_confirm', 'prune_min_contradictions', 'max_retired_instances', 'bbox_min_support',
                     'mask_completion_min_returns', 'mask_completion_stride_px', 'ground_exclusion_min_returns'):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ('tentative_max_age', 'max_occlusion_frames', 'disappeared_prune_grace_frames',
                     'contradiction_window_px', 'depth_fill_radius_px', 'ground_context_px'):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ('voxel_size', 'tau_eps'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ('label_compatibility_min_mass', 'relabel_min_overlap'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not np.isfinite(self.match_max_gap_m) or self.match_max_gap_m < 0:
            raise ValueError("match_max_gap_m must be finite and nonnegative")
        for name in ('reconcile_max_distance_m', 'reconcile_max_gap_sec', 'reid_max_age_sec'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @classmethod
    def from_dict(cls, params: dict) -> "PipelineConfig":
        """Build a config from a flat dict (e.g. a loaded YAML ``ros__parameters``
        block), silently ignoring keys that aren't pipeline fields (topics,
        detector settings, etc. live alongside these in the same file).
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in known})


@dataclass
class FrameResult:
    objects: list[ObjectInstance] = field(default_factory=list)
    stamp: float = 0.0
    """Timestamp of the observation this result reflects; the reference for
    "not seen for N s" ages of occluded instances."""
    scene_graph: sg.SceneGraph | None = None
    detection_instance_ids: list[int] = field(default_factory=list)
    """For each detection in the processed observation, the instance ID it was
    fused into (matched, re-activated, or newly spawned), or -1 if it was
    discarded (e.g. a low-confidence detection with no existing track)."""

    timings: dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds spent in each stage of this update (``depth``: range
    gating and sparse filling of the frame's depth, ``embed``, ``predict``,
    ``backproject``, ``associate``, ``map_update``, ``scene_graph``, ``total``),
    the raw material for the Sec. V-H runtime accounting."""


class SemanticMappingPipeline:
    """Maintains M_t and G across frames; see Sec. III for the underlying model."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.object_map = ObjectMap(
            voxel_size=self.config.voxel_size,
            tau_eps=self.config.tau_eps,
            max_points_per_object=self.config.max_points_per_object,
            prune_log_odds=self.config.prune_log_odds,
            prune_membership=self.config.prune_membership,
            membership_margin_px=self.config.membership_margin_px,
            active_occupied_fraction=self.config.active_occupied_fraction,
            disappeared_occupied_fraction=self.config.disappeared_occupied_fraction,
            max_occlusion_frames=self.config.max_occlusion_frames,
            min_label_confidence=self.config.min_label_confidence,
            min_observations_for_confidence_check=self.config.min_observations_for_confidence_check,
            tentative_max_age=self.config.tentative_max_age,
            cull_out_of_view=self.config.cull_out_of_view,
            bbox_trim_percentile=self.config.bbox_trim_percentile,
            dynamic_geometry_labels=self.config.dynamic_geometry_labels if self.config.dynamic_geometry_enabled else (),
            dynamic_geometry_min_extent_fraction=self.config.dynamic_geometry_min_extent_fraction,
            label_min_mass=self.config.label_compatibility_min_mass,
            prune_min_contradictions=self.config.prune_min_contradictions,
            contradiction_window_px=self.config.contradiction_window_px,
            reconcile_max_distance_m=self.config.reconcile_max_distance_m,
            reconcile_max_gap_sec=self.config.reconcile_max_gap_sec,
            bbox_min_support=self.config.bbox_min_support,
            bbox_support_radius_m=self.config.bbox_support_radius_m,
            size_limits=parse_size_limits(self.config.class_size_limits),
            bbox_support_miss=self.config.bbox_support_miss,
            bbox_support_cull_at=self.config.bbox_support_cull_at,
            bbox_support_max=self.config.bbox_support_max,
            existence_hit_gain=self.config.existence_hit_gain,
            existence_miss_penalty=self.config.existence_miss_penalty,
            existence_cull_log_odds=self.config.existence_cull_log_odds,
            existence_max_log_odds=self.config.existence_max_log_odds,
            geometric_consistency=self.config.use_geometric_consistency,
            semantic_fusion=self.config.use_semantic_fusion,
        )
        self._frame_index = 0
        self._last_stamp: float | None = None
        self._rebase_loaded_stamps = False
        self.embedder = build_embedder(
            self.config.appearance_embedder, device=self.config.embedder_device,
            model_name=self.config.clip_model, pretrained=self.config.clip_pretrained,
        )

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        """Write the current map M_t to a directory (see :mod:`semantic_mapping.persistence`)."""
        return persistence.save_map(
            self.object_map, path, metadata={"frame_index": self._frame_index, **(metadata or {})},
        )

    def load(self, path: str | Path, resume: bool = True) -> dict:
        """Replace the current map with a saved one and continue from it.

        Instance IDs keep counting from where the saved session stopped, so
        histories recorded by downstream consumers stay valid. Returns the
        saved header.

        The new session may run on a different clock epoch (sim time from a
        bag, a rebooted machine). Loaded stamps are therefore checked against
        the first observation fused after the load: if the saved map's newest
        stamp lies at or after it, every stored stamp is shifted so that the
        saved session ends just before the new one begins (the downtime
        between the two is unknown and taken as zero). Stamps already in the
        past of the new clock are kept, so they simply read as old. Either
        way ages, Kalman time steps, re-identification age limits, and
        reconciliation gaps never see negative or mixed-epoch intervals.
        """
        header = persistence.load_map(path, self.object_map, resume=resume)
        self._frame_index = int(header.get("metadata", {}).get("frame_index", 0))
        self.begin_new_epoch()
        return header

    def begin_new_epoch(self) -> None:
        """Accept observations from a new clock epoch (map load, time jump backwards).

        Stored stamps are rebased against the next fused observation, as
        described in :meth:`load`.
        """
        self._last_stamp = None
        self._rebase_loaded_stamps = True

    def _rebase_stamps_for(self, stamp: float) -> None:
        if not self._rebase_loaded_stamps:
            return
        self._rebase_loaded_stamps = False
        newest = self.object_map.newest_stamp()
        if newest is not None and newest >= stamp:
            self.object_map.shift_stamps(stamp - newest - LOADED_MAP_EPOCH_GAP_SEC)

    @staticmethod
    def _fill_within(depth: np.ndarray, region: np.ndarray, radius_px: int) -> np.ndarray:
        """Depth filled only from readings inside ``region`` (elsewhere invalid),
        computed on the region's bounding crop to keep the per-detection cost small."""
        ys, xs = np.nonzero(region)
        if xs.size == 0:
            return depth
        h, w = depth.shape
        y1, y2 = max(int(ys.min()) - radius_px, 0), min(int(ys.max()) + radius_px + 1, h)
        x1, x2 = max(int(xs.min()) - radius_px, 0), min(int(xs.max()) + radius_px + 1, w)
        crop = np.where(region[y1:y2, x1:x2], depth[y1:y2, x1:x2], 0.0)
        filled = np.zeros_like(depth, dtype=np.float64)
        filled[y1:y2, x1:x2] = fill_sparse_depth(crop, radius_px)
        return filled

    def _remove_ground(
        self, depth: np.ndarray, mask: np.ndarray, K: np.ndarray, T_world_from_cam: np.ndarray,
        remove: bool = True, min_off_ground: int = 0, origin: tuple[int, int] = (0, 0),
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``depth`` without the masked readings that lie on the local ground,
        and the ground plane if it was fitted (None for the level fallback).

        The ground is fitted (fit_ground_plane) to the real returns in the
        mask's box grown by ``ground_context_px``, i.e. including the ground
        in front of and beside the object; readings inside the mask less than
        ``ground_clearance_m`` above it (or below it) become invalid, unless
        ``remove`` is False (plane only). ``min_off_ground > 0`` removes them
        only from a fitted ground below the camera, and only when at least
        that many masked readings stand above it (ground_exclusion).
        ``origin`` is the (column, row) of ``depth[0, 0]`` in the image ``K``
        describes; a crop must include the ``ground_context_px`` margin.
        """
        cfg = self.config
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return depth, None
        h, w = depth.shape
        m = cfg.ground_context_px
        y1, y2 = max(int(ys.min()) - m, 0), min(int(ys.max()) + m + 1, h)
        x1, x2 = max(int(xs.min()) - m, 0), min(int(xs.max()) + m + 1, w)
        crop = depth[y1:y2, x1:x2]
        valid = np.isfinite(crop) & (crop > 0)
        vs, us = np.nonzero(valid)
        if vs.size == 0:
            return depth, None
        u0, v0 = x1 + origin[0], y1 + origin[1]

        def world_of(rows, cols):
            z = crop[rows, cols].astype(np.float64)
            cam = np.stack(((cols + u0 - K[0, 2]) * z / K[0, 0], (rows + v0 - K[1, 2]) * z / K[1, 1], z), axis=1)
            return transform_points(T_world_from_cam, cam)

        sample = slice(None)
        if vs.size > GROUND_FIT_MAX_SAMPLES:
            sample = np.linspace(0, vs.size - 1, GROUND_FIT_MAX_SAMPLES).astype(np.int64)
        plane, fitted = fit_ground_plane(world_of(vs[sample], us[sample]), cfg.ground_clearance_m,
                                         cfg.ground_max_slope, return_fitted=True)
        masked = mask[vs + y1, us + x1]
        vs, us = vs[masked], us[masked]
        world = world_of(vs, us)
        ground = world[:, 2] - world[:, :2] @ plane[:2] - plane[2] < cfg.ground_clearance_m
        if min_off_ground > 0:
            center = T_world_from_cam[:3, 3]
            below_camera = fitted and center[2] > center[:2] @ plane[:2] + plane[2] + cfg.ground_clearance_m
            remove = remove and below_camera and np.count_nonzero(~ground) >= min_off_ground
        if not remove or not ground.any():
            return depth, plane if fitted else None
        filtered = np.array(depth, dtype=np.float64, copy=True)
        filtered[vs[ground] + y1, us[ground] + x1] = 0.0
        return filtered, plane if fitted else None

    def _ground_contact_range(self, mask: np.ndarray, K: np.ndarray, T_world_from_cam: np.ndarray,
                              plane: np.ndarray, origin: tuple[int, int] = (0, 0),
                              image_height: int | None = None) -> float | None:
        """Camera depth at which the ray through the mask's bottom-centre pixel meets the ground plane.
        A cropped ``mask`` gives its ``origin`` (column, row) and the full ``image_height``."""
        ys, xs = np.nonzero(mask)
        bottom = int(ys.max())
        v = bottom + origin[1]
        if v >= (mask.shape[0] if image_height is None else image_height) - 1:
            return None  # cut off by the image: the bottom is not the ground contact
        u = float(np.mean(xs[ys == bottom] + origin[0]))
        direction = T_world_from_cam[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        origin = T_world_from_cam[:3, 3]
        denominator = direction[2] - plane[0] * direction[0] - plane[1] * direction[1]
        if abs(denominator) < 1e-9:
            return None
        t = (plane[0] * origin[0] + plane[1] * origin[1] + plane[2] - origin[2]) / denominator
        cfg = self.config
        if not np.isfinite(t) or t <= max(cfg.min_depth_m, 0.0) or (cfg.max_depth_m > 0 and t > cfg.max_depth_m):
            return None
        return float(t)

    def _complete_mask(self, mask: np.ndarray, depth: np.ndarray, K: np.ndarray, z: float, budget: int,
                       T_world_from_cam: np.ndarray, plane: np.ndarray | None,
                       origin: tuple[int, int] = (0, 0)) -> np.ndarray:
        """Camera-frame points at depth ``z`` for mask pixels without a reading in
        ``depth``: a ``mask_completion_stride_px`` grid plus the silhouette's
        extreme pixels, at most ``budget`` (extremes first). With a ground
        ``plane``, pixels that would complete below the ground (a mask bled
        onto the ground in front) are left out first. ``origin`` is the
        (column, row) of ``mask[0, 0]`` in the image ``K`` describes."""
        if budget <= 0:
            return np.zeros((0, 3))
        empty = mask & ~(np.isfinite(depth) & (depth > 0))
        vs, us = np.nonzero(empty)
        us, vs = us + origin[0], vs + origin[1]
        if plane is not None and vs.size:
            cam = np.stack(((us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], np.full(us.size, z)), axis=1)
            world = transform_points(T_world_from_cam, cam)
            above = world[:, 2] >= world[:, :2] @ plane[:2] + plane[2]
            vs, us = vs[above], us[above]
        if vs.size == 0:
            return np.zeros((0, 3))
        s = self.config.mask_completion_stride_px
        extremes = np.unique([np.argmin(us), np.argmax(us), np.argmin(vs), np.argmax(vs)])
        grid = np.flatnonzero((vs % s == 0) & (us % s == 0))
        grid = np.setdiff1d(grid, extremes)
        room = budget - extremes.size
        if room < grid.size:
            grid = np.random.default_rng(0).choice(grid, size=max(room, 0), replace=False)
        chosen = np.concatenate((extremes, grid))[:budget]
        us, vs = us[chosen].astype(np.float64), vs[chosen].astype(np.float64)
        return np.stack(((us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], np.full(us.size, z)), axis=1)

    def _detection_region(self, detection: Detection2D, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Rows ``[y1, y2)`` and columns ``[x1, x2)`` of the detection's pixels (its
        mask, or its box without one) grown by the margin the ground fit and
        per-mask depth filling read around them, clipped to the image. Empty
        (``y1 == y2``) when the detection covers no pixel."""
        h, w = image_shape
        if detection.mask is not None:
            bounds = detection.mask_bounds()
            if bounds is None:
                return 0, 0, 0, 0
            y1, y2, x1, x2 = bounds
        else:
            x1, y1, x2, y2 = _box_pixels(detection.bbox, image_shape)
            if x2 <= x1 or y2 <= y1:
                return 0, 0, 0, 0
        m = max(self.config.ground_context_px, self.config.depth_fill_radius_px)
        return max(y1 - m, 0), min(y2 + m, h), max(x1 - m, 0), min(x2 + m, w)

    def _detection_points_world(
        self, detection: Detection2D, depth: np.ndarray, K: np.ndarray, T_world_from_cam: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """World points of one detection, lifted through ``depth``.

        Every step reads only the detection's pixels and a margin around them,
        so it runs on that crop (``region``, default :meth:`_detection_region`)
        instead of the whole image: per-detection cost follows the object, not
        the camera resolution. Pixel coordinates are offset back to the full
        image, so any crop that contains the region gives the same points.
        """
        y1, y2, x1, x2 = self._detection_region(detection, depth.shape) if region is None else region
        if y2 <= y1 or x2 <= x1:
            return np.zeros((0, 3))
        has_instance_mask = detection.mask is not None
        if has_instance_mask:
            mask = detection.mask[y1:y2, x1:x2]
        else:
            bx1, by1, bx2, by2 = _box_pixels(detection.bbox, depth.shape)
            mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
            mask[max(by1 - y1, 0):max(by2 - y1, 0), max(bx1 - x1, 0):max(bx2 - x1, 0)] = True
        image_height = depth.shape[0]
        depth = depth[y1:y2, x1:x2]
        origin = (x1, y1)

        cfg = self.config
        plane = None
        remove = detection.label in cfg.ground_removal_labels
        contact = has_instance_mask and detection.label in cfg.ground_contact_depth_labels
        exclude = (cfg.ground_exclusion and has_instance_mask and not remove
                   and detection.label not in cfg.ground_surface_labels)
        if remove or contact or exclude:
            depth, plane = self._remove_ground(
                depth, mask, K, T_world_from_cam, remove=remove or exclude,
                min_off_ground=cfg.ground_exclusion_min_returns if exclude else 0, origin=origin)
        if cfg.foreground_depth_gap_m > 0 and detection.label in cfg.foreground_depth_labels:
            # Count real returns, not pixels synthesized by sparse filling.
            pixels = min_span = None
            if cfg.foreground_depth_min_image_span > 0:
                ys, xs = np.nonzero(mask)
                pixels = np.column_stack((xs + origin[0], ys + origin[1]))
                if len(pixels):
                    min_span = (np.ptp(pixels, axis=0) + 1) * cfg.foreground_depth_min_image_span
            selected = foreground_depth_mask(
                depth[mask], cfg.foreground_depth_gap_m,
                cfg.foreground_depth_min_points, cfg.foreground_depth_min_fraction,
                pixels=pixels, min_span=min_span,
                select="largest" if detection.label in cfg.foreground_depth_largest_labels else "nearest",
            )
            filtered = np.zeros_like(depth)
            filtered[mask] = np.where(selected, depth[mask], 0.0)
            depth = filtered
        completion_range = None
        if has_instance_mask and (contact or detection.label in cfg.mask_completion_labels):
            # Range for the silhouette from kept real returns only (before filling).
            kept = depth[mask]
            kept = kept[np.isfinite(kept) & (kept > 0)]
            if kept.size >= cfg.mask_completion_min_returns:
                if detection.label in cfg.mask_completion_labels:
                    completion_range = float(np.median(kept))
                    self.object_map.stats["mask_completions"] += 1
            elif contact and plane is not None:
                completion_range = self._ground_contact_range(mask, K, T_world_from_cam, plane, origin, image_height)
                if completion_range is not None:
                    self.object_map.stats["ground_contact_completions"] += 1
        if self.config.depth_fill_radius_px > 0:
            depth = self._fill_within(depth, mask, self.config.depth_fill_radius_px)
        points_cam = back_project_depth(
            K, depth, mask=mask, max_points=self.config.max_points_per_detection,
            depth_mad_factor=self.config.mask_depth_mad_factor if has_instance_mask else 3.0,
            depth_min_tolerance=self.config.mask_depth_min_tolerance_m if has_instance_mask else 0.05,
            pixel_origin=origin,
        )
        if completion_range is not None:
            points_cam = np.concatenate((points_cam, self._complete_mask(
                mask, depth, K, completion_range, cfg.max_points_per_detection - len(points_cam),
                T_world_from_cam, plane, origin)))
        if (cfg.foreground_depth_max_extent_m > 0 and detection.label in cfg.foreground_depth_labels
                and len(points_cam)):
            bounds = bbox3d_from_points(points_cam, cfg.bbox_trim_percentile)
            if np.any(bounds[3:] - bounds[:3] > cfg.foreground_depth_max_extent_m):
                return np.zeros((0, 3))
        points_world = transform_points(T_world_from_cam, points_cam)
        limit = self.object_map.size_limits.get(detection.label)
        if limit is not None and len(points_world) \
                and exceeds_size_limit(bbox3d_from_points(points_world, cfg.bbox_trim_percentile), limit):
            # Too big for its class: background or ground, not the object. The
            # 2D detection still associates; it just carries no geometry.
            self.object_map.stats["size_rejected_observations"] += 1
            return np.zeros((0, 3))
        return points_world

    def _refuse_oversized(self, result: association.AssociationResult, objects: list[ObjectInstance],
                          detections: list[Detection2D], det_points: list[np.ndarray]) -> None:
        """Unmatch pairs whose fusion would grow the instance past its class size limit."""
        if not self.object_map.size_limits:
            return
        kept = []
        for track_idx, det_idx in result.matches:
            points = det_points[det_idx]
            box = bbox3d_from_points(points, self.config.bbox_trim_percentile) if len(points) else None
            if box is not None and self.object_map.growth_exceeds_limit(
                    objects[track_idx], box, (detections[det_idx].label,)):
                self.object_map.stats["size_refused_associations"] += 1
                result.unmatched_tracks.append(track_idx)
                result.unmatched_detections.append(det_idx)
            else:
                kept.append((track_idx, det_idx))
        result.matches = kept

    def _split_depth_inconsistent(self, result: association.AssociationResult, objects: list[ObjectInstance],
                                  det_boxes3d: list[np.ndarray | None]) -> None:
        """2D→3D validation of a 2D association stage (association.split_depth_inconsistent)."""
        self.object_map.stats["depth_split_matches"] += association.split_depth_inconsistent(
            result, det_boxes3d, objects, self.config.match_max_gap_m)

    def process_frame(self, observation: Observation) -> FrameResult:
        """Run one full P(I_t, M_t, P_t | M_t-1, Q_t) update step (Eq. 2), given
        that pose P_t was already estimated upstream (Sec. IV-A) and is
        carried on ``observation.pose``.
        """
        if not np.isfinite(observation.stamp):
            raise ValueError("observation timestamp must be finite")
        if self._last_stamp is not None and observation.stamp <= self._last_stamp:
            raise ValueError("observations must be fused in strictly increasing timestamp order")
        image_shape = (observation.intrinsics.height, observation.intrinsics.width)
        for detection in observation.detections:
            detection.validate_mask(image_shape)
        # Everything above validates; state changes start here.
        self._rebase_stamps_for(observation.stamp)
        self._last_stamp = observation.stamp
        self._frame_index += 1
        t_start = time.perf_counter()
        K = observation.intrinsics.K
        T_world_from_cam = observation.pose.T_world_from_frame
        # Shallow copies: embeddings computed here must not be written back
        # into the caller's detections, where a later run with another
        # embedder would silently reuse them. Masks are shared, read-only.
        detections = [dataclasses.replace(d) for d in observation.detections]
        cfg = self.config
        depth = observation.depth
        if depth is not None and (cfg.min_depth_m > 0 or cfg.max_depth_m > 0):
            valid = np.isfinite(depth) & (depth > 0) & (depth >= cfg.min_depth_m)
            if cfg.max_depth_m > 0:
                valid &= depth <= cfg.max_depth_m
            # Out-of-range readings mean unknown, not free space. The same
            # filtered depth must drive both back-projection and contradiction
            # evidence, and the caller's observation must remain unmodified.
            depth = np.where(valid, depth, 0.0)
        # Evidence (Eq. 7-9) runs on depth filled from every reading; detections
        # are back-projected through the raw depth, filled per detection from
        # readings inside their own mask (see _detection_points_world).
        evidence_depth = depth
        if depth is not None and cfg.depth_fill_radius_px > 0:
            evidence_depth = fill_sparse_depth(depth, cfg.depth_fill_radius_px)
        t_depth = time.perf_counter()

        live_objects = [
            obj for obj in self.object_map.objects.values() if obj.status != ObjectStatus.DISAPPEARED
        ]
        retired_before = {o.instance_id for o in self.object_map.objects.values() if o.status == ObjectStatus.DISAPPEARED}

        if self.embedder is not None and observation.rgb is not None:
            missing = [d for d in detections if d.embedding is None]
            if missing:
                for detection, embedding in zip(missing, self.embedder.embed(observation.rgb, missing)):
                    detection.embedding = embedding
        t_embed = time.perf_counter()

        image_size = (observation.intrinsics.width, observation.intrinsics.height)
        # One batched frustum test decides which instances can be seen at all;
        # the rest skip prediction, association, and per-point evidence this
        # frame, so the update cost follows the view, not the map size.
        in_view = np.ones(len(live_objects), dtype=bool)
        if cfg.cull_out_of_view and live_objects:
            has_points = np.array([o.points_world.shape[0] > 0 for o in live_objects])
            boxes = np.array([o.bbox3d for o in live_objects], dtype=np.float64).reshape(-1, 6)
            in_view = ~has_points | self.object_map.boxes_in_view(
                boxes, K, T_world_from_cam, (image_size[1], image_size[0]))
        visible_indices = [i for i, flag in enumerate(in_view) if flag]

        predicted_tracks: list[tracking.TrackKalmanState] = []
        predicted_bboxes: list[np.ndarray] = []
        for obj, flag in zip(live_objects, in_view):
            if not flag:
                predicted_tracks.append(obj.track)
                predicted_bboxes.append(tracking.current_bbox(obj.track))
                continue
            # obj.track is the state at the last match (latest_stamp); a
            # prediction is never stored, so each frame predicts once over
            # the whole gap instead of compounding one prediction per frame.
            dt = max(observation.stamp - obj.latest_stamp, 1e-3)
            predicted = tracking.predict(
                obj.track, dt, K=K, T_world_from_cam=T_world_from_cam,
                object_centroid_world=obj.center if len(obj.points_world) else None,
                object_bbox3d_world=obj.bbox3d if obj.points_world.shape[0] > 0 else None,
                image_size=image_size, size_prior_weight=cfg.size_prior_weight,
            )
            predicted_tracks.append(predicted)
            # Detections never extend past the frame, so score the prediction's visible part.
            bbox = tracking.current_bbox(predicted)
            visible = clip_bbox_to_image(bbox, *image_size)
            predicted_bboxes.append(bbox if visible is None else visible)
        t_predict = time.perf_counter()

        # Back-project every detection once; the 3D boxes feed the re-activation
        # stage, the points feed whichever instance the detection ends up in.
        det_points: list[np.ndarray] = []
        det_boxes3d: list[np.ndarray | None] = []
        for detection in detections:
            points = (self._detection_points_world(detection, depth, K, T_world_from_cam)
                      if depth is not None else np.zeros((0, 3)))
            det_points.append(points)
            det_boxes3d.append(
                bbox3d_from_points(points, cfg.bbox_trim_percentile)
                if points.shape[0] >= max(cfg.min_points_for_3d_association, 1) else None
            )
        t_backproject = time.perf_counter()
        detection_bboxes = [d.bbox for d in detections]
        detection_labels = [d.label for d in detections]
        track_beliefs = [o.label_belief for o in live_objects]
        high = [i for i, d in enumerate(detections) if d.score >= cfg.high_score_threshold]
        low = [i for i, d in enumerate(detections) if d.score < cfg.high_score_threshold]

        if cfg.use_2d_tracker:
            # Stage 1: 2D, high-confidence detections, gated by motion and label.
            stage1 = association.associate(
                predicted_tracks, predicted_bboxes, detection_bboxes,
                iou_threshold=cfg.association_iou_threshold, candidate_tracks=visible_indices,
                candidate_detections=high, track_label_beliefs=track_beliefs, detection_labels=detection_labels,
                label_min_mass=cfg.label_compatibility_min_mass,
            )
            self._refuse_oversized(stage1, live_objects, detections, det_points)
            self._split_depth_inconsistent(stage1, live_objects, det_boxes3d)
            # Stage 2 (ByteTrack): leftover tracks vs. low-confidence detections, looser IoU, no motion gate.
            stage2 = association.associate(
                predicted_tracks, predicted_bboxes, detection_bboxes,
                iou_threshold=cfg.low_score_iou_threshold, use_mahalanobis_gate=False,
                candidate_tracks=stage1.unmatched_tracks, candidate_detections=low,
                track_label_beliefs=track_beliefs, detection_labels=detection_labels,
                label_min_mass=cfg.label_compatibility_min_mass,
            )
            self._refuse_oversized(stage2, live_objects, detections, det_points)
            self._split_depth_inconsistent(stage2, live_objects, det_boxes3d)
        else:
            # Table V "W/o 2D Tracker": only the 3D stages associate.
            stage1 = association.AssociationResult(unmatched_tracks=list(visible_indices), unmatched_detections=high)
            stage2 = association.AssociationResult(unmatched_tracks=list(visible_indices))
        # Stage 3: 3D-aware re-activation for high-confidence detections still unmatched.
        stage3 = association.associate_3d(
            det_boxes3d, detection_labels, live_objects,
            iou_threshold=cfg.reactivation_iou_threshold, containment_margin=cfg.reactivation_margin_m,
            candidate_objects=stage2.unmatched_tracks, candidate_detections=stage1.unmatched_detections,
            label_min_mass=cfg.label_compatibility_min_mass,
        )
        self._refuse_oversized(stage3, live_objects, detections, det_points)
        # Every in-view live instance receives this frame's evidence below
        # (matched, re-activated or unmatched); classify all their points at
        # once. Association does not change them, and relabelling reads the
        # classification to see where each candidate is visible.
        self.object_map.prepare_evidence(
            [obj for obj, flag in zip(live_objects, in_view) if flag], K, T_world_from_cam, evidence_depth)
        # Stage 4, relabelling: the same object under a label it has not taken yet (Eq. 10 decides).
        relabel_overlap = cfg.relabel_min_overlap if cfg.use_2d_tracker else 0.0
        visible_bboxes = None
        if relabel_overlap > 0 and evidence_depth is not None and stage3.unmatched_detections:
            visible_bboxes = [None] * len(live_objects)
            for track_idx in stage3.unmatched_tracks:
                visible_bboxes[track_idx] = self.object_map.visible_bbox(
                    live_objects[track_idx], K, T_world_from_cam, evidence_depth)
        relabel = association.associate_relabel(
            predicted_tracks, predicted_bboxes, detection_bboxes, det_boxes3d, live_objects,
            iou_threshold=cfg.association_iou_threshold, min_overlap=relabel_overlap,
            pad=cfg.voxel_size / 2, candidate_tracks=stage3.unmatched_tracks,
            candidate_detections=stage3.unmatched_detections, visible_bboxes=visible_bboxes,
        )
        self._refuse_oversized(relabel, live_objects, detections, det_points)
        self.object_map.stats["relabel_matches"] += len(relabel.matches)

        t_associate = time.perf_counter()

        detection_instance_ids = [-1] * len(detections)
        for track_idx, det_idx in stage1.matches + stage2.matches + relabel.matches:
            detection = detections[det_idx]
            updated_track = tracking.update(predicted_tracks[track_idx], detection.bbox)
            # Matched instances are all in view: pass the batched verdict on
            # instead of repeating the frustum test per instance.
            self.object_map.update_matched(
                live_objects[track_idx], updated_track, det_points[det_idx], detection,
                observation.stamp, K, T_world_from_cam, evidence_depth, in_view=bool(in_view[track_idx]),
            )
            detection_instance_ids[det_idx] = live_objects[track_idx].instance_id
        for track_idx, det_idx in stage3.matches:
            self.object_map.reactivate(
                live_objects[track_idx], det_points[det_idx], detections[det_idx],
                observation.stamp, K, T_world_from_cam, evidence_depth, in_view=bool(in_view[track_idx]),
            )
            detection_instance_ids[det_idx] = live_objects[track_idx].instance_id

        # Unmatched instances keep the track state of their last match (see
        # the prediction loop above). Without depth, update_unmatched applies
        # no evidence but still counts detector misses, so tentative tracks
        # expire; out-of-view instances are never charged a miss.
        for track_idx in relabel.unmatched_tracks:
            self.object_map.update_unmatched(
                live_objects[track_idx], K, T_world_from_cam, evidence_depth, in_view=bool(in_view[track_idx]),
                detections_evaluated=observation.detections_evaluated)
        for track_idx in np.nonzero(~in_view)[0]:
            self.object_map.update_unmatched(
                live_objects[track_idx], K, T_world_from_cam, evidence_depth, in_view=False,
                detections_evaluated=observation.detections_evaluated)
        self.object_map.discard_prepared_evidence()

        # Stage 5: re-identification against retired instances, so an object
        # that was removed and comes back -- in place or elsewhere -- keeps its ID.
        unmatched_detections = relabel.unmatched_detections
        if cfg.reid_enabled and unmatched_detections:
            retired = [o for o in self.object_map.objects.values() if o.status == ObjectStatus.DISAPPEARED]
            if retired:
                stage4, by_place = association.reidentify(
                    det_boxes3d, detection_labels, [d.embedding for d in detections], retired,
                    min_similarity=cfg.reid_min_similarity, iou_threshold=cfg.reactivation_iou_threshold,
                    containment_margin=cfg.reactivation_margin_m, max_age_sec=cfg.reid_max_age_sec,
                    now=observation.stamp, candidate_detections=unmatched_detections,
                    label_min_mass=cfg.label_compatibility_min_mass,
                    relocation_max_distance=cfg.reconcile_max_distance_m,
                    relocation_max_gap_sec=cfg.reconcile_max_gap_sec,
                )
                for obj_idx, det_idx in stage4.matches:
                    self.object_map.revive(
                        retired[obj_idx], det_points[det_idx], detections[det_idx], observation.stamp,
                        K, T_world_from_cam, evidence_depth, relocated=(obj_idx, det_idx) not in by_place,
                    )
                    detection_instance_ids[det_idx] = retired[obj_idx].instance_id
                unmatched_detections = stage4.unmatched_detections

        # Only high-confidence detections may start a new object (ByteTrack).
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            spawned = self.object_map.spawn(
                detection.bbox, det_points[det_idx], detection.label, detection.score, observation.stamp,
                embedding=detection.embedding,
            )
            detection_instance_ids[det_idx] = spawned.instance_id

        for obj in self.object_map.objects.values():
            self.object_map.confirm_tentative(obj, cfg.min_hits_to_confirm)
        merged = dict(
            (dropped, kept)
            for kept, dropped in self.object_map.merge_duplicates(cfg.merge_iou_threshold, cfg.merge_distance_m)
        )
        detection_instance_ids = [merged.get(i, i) for i in detection_instance_ids]
        for obj in self.object_map.objects.values():
            sg.record_trajectory_sample(obj, observation.stamp)

        # An object that moved before its old spot was confirmed empty got a
        # provisional ID meanwhile; now that the old instance has retired,
        # reconcile the two under the original ID (ObjectMap.reconcile_retired).
        if cfg.reid_enabled:
            newly_retired = [
                o for o in self.object_map.objects.values()
                if o.status == ObjectStatus.DISAPPEARED and o.instance_id not in retired_before
            ]
            if newly_retired:
                reconciled = dict(
                    (dropped, kept)
                    for kept, dropped in self.object_map.reconcile_retired(newly_retired, cfg.reid_min_similarity)
                )
                detection_instance_ids = [reconciled.get(i, i) for i in detection_instance_ids]
                for kept in set(reconciled.values()):
                    obj = self.object_map.objects[kept]
                    self.object_map.confirm_tentative(obj, cfg.min_hits_to_confirm)
                    sg.record_trajectory_sample(obj, observation.stamp)

        self.object_map.compact_disappeared(cfg.disappeared_prune_grace_frames, cfg.max_retired_instances)
        t_update = time.perf_counter()

        graph = sg.build_scene_graph(
            list(self.object_map.objects.values()),
            cluster_radius=self.config.scene_graph_cluster_radius,
            z_tolerance=self.config.scene_graph_z_tolerance,
            xy_iou_threshold=self.config.scene_graph_xy_iou_threshold,
            beside_max_distance=self.config.scene_graph_beside_max_distance,
            support_classes=self.config.scene_graph_support_classes,
            on_min_footprint_fraction=self.config.scene_graph_on_min_footprint_fraction,
        )

        t_graph = time.perf_counter()

        return FrameResult(
            objects=list(self.object_map.objects.values()),
            stamp=float(observation.stamp),
            scene_graph=graph,
            detection_instance_ids=detection_instance_ids,
            timings={
                "depth": t_depth - t_start,
                "embed": t_embed - t_depth,
                "predict": t_predict - t_embed,
                "backproject": t_backproject - t_predict,
                "associate": t_associate - t_backproject,
                "map_update": t_update - t_associate,
                "scene_graph": t_graph - t_update,
                "total": t_graph - t_start,
            },
        )
