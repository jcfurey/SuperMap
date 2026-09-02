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

import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from semantic_mapping import association, persistence, scene_graph as sg, tracking
from semantic_mapping.geometry_utils import (
    back_project_depth,
    bbox3d_from_points,
    clip_bbox_to_image,
    depth_consistency_mask,
    fill_sparse_depth,
    transform_points,
)
from semantic_mapping.object_map import ObjectMap
from semantic_mapping.types import Detection2D, ObjectInstance, ObjectStatus, Observation


@dataclass
class PipelineConfig:
    voxel_size: float = 0.05
    tau_eps: float = 0.15
    max_points_per_object: int = 20000
    prune_log_odds: float = -1.5
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
    min_points_for_3d_association: int = 5
    merge_iou_threshold: float = 0.3
    merge_distance_m: float = 0.25
    disappeared_prune_grace_frames: int = 60
    scene_graph_cluster_radius: float = 2.0
    scene_graph_z_tolerance: float = 0.1
    scene_graph_xy_iou_threshold: float = 0.05
    scene_graph_beside_max_distance: float = 1.0
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
    mask (or box), so background never leaks into an object's point set."""

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
    scene_graph: sg.SceneGraph | None = None
    detection_instance_ids: list[int] = field(default_factory=list)
    """For each detection in the processed observation, the instance ID it was
    fused into (matched, re-activated, or newly spawned), or -1 if it was
    discarded (e.g. a low-confidence detection with no existing track)."""

    timings: dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds spent in each stage of this update (``predict``,
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
        )
        self._frame_index = 0
        self._edge_cache = sg.SpatialEdgeCache()

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
        """
        header = persistence.load_map(path, self.object_map, resume=resume)
        self._frame_index = int(header.get("metadata", {}).get("frame_index", 0))
        return header

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

    def _detection_points_world(
        self, detection: Detection2D, depth: np.ndarray, K: np.ndarray, T_world_from_cam: np.ndarray,
    ) -> np.ndarray:
        has_instance_mask = detection.mask is not None
        if has_instance_mask:
            mask = detection.mask
        else:
            mask = np.zeros(depth.shape, dtype=bool)
            x1, y1, x2, y2 = detection.bbox.astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, depth.shape[1]), min(y2, depth.shape[0])
            mask[y1:y2, x1:x2] = True

        if self.config.depth_fill_radius_px > 0:
            depth = self._fill_within(depth, mask, self.config.depth_fill_radius_px)
        points_cam = back_project_depth(K, depth, mask=mask)
        if not has_instance_mask and points_cam.shape[0] > 0:
            # An axis-aligned box (unlike a segmentation mask) commonly includes
            # background around the object's true silhouette; reject it so it
            # doesn't drag the fused 3D point set toward whatever is behind the
            # object (Sec. IV-B.3 depends on a clean per-instance point set).
            points_cam = points_cam[depth_consistency_mask(points_cam[:, 2])]
        if points_cam.shape[0] > self.config.max_points_per_detection:
            rng = np.random.default_rng(0)
            idx = rng.choice(points_cam.shape[0], size=self.config.max_points_per_detection, replace=False)
            points_cam = points_cam[idx]
        return transform_points(T_world_from_cam, points_cam)

    def process_frame(self, observation: Observation) -> FrameResult:
        """Run one full P(I_t, M_t, P_t | M_t-1, Q_t) update step (Eq. 2), given
        that pose P_t was already estimated upstream (Sec. IV-A) and is
        carried on ``observation.pose``.
        """
        self._frame_index += 1
        t_start = time.perf_counter()
        K = observation.intrinsics.K
        T_world_from_cam = observation.pose.T_world_from_frame
        detections = observation.detections
        cfg = self.config
        depth = observation.depth
        # Evidence (Eq. 7-9) runs on depth filled from every reading; detections
        # are back-projected through the raw depth, filled per detection from
        # readings inside their own mask (see _detection_points_world).
        evidence_depth = depth
        if depth is not None and cfg.depth_fill_radius_px > 0:
            evidence_depth = fill_sparse_depth(depth, cfg.depth_fill_radius_px)

        live_objects = [
            obj for obj in self.object_map.objects.values() if obj.status != ObjectStatus.DISAPPEARED
        ]

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
            dt = max(observation.stamp - obj.latest_stamp, 1e-3)
            predicted = tracking.predict(
                obj.track, dt, K=K, T_world_from_cam=T_world_from_cam, object_centroid_world=obj.center,
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
                bbox3d_from_points(points) if points.shape[0] >= cfg.min_points_for_3d_association else None
            )
        t_backproject = time.perf_counter()
        detection_bboxes = [d.bbox for d in detections]
        high = [i for i, d in enumerate(detections) if d.score >= cfg.high_score_threshold]
        low = [i for i, d in enumerate(detections) if d.score < cfg.high_score_threshold]

        # Stage 1: 2D, high-confidence detections, gated.
        stage1 = association.associate(
            predicted_tracks, predicted_bboxes, detection_bboxes,
            iou_threshold=cfg.association_iou_threshold, candidate_tracks=visible_indices,
            candidate_detections=high,
        )
        # Stage 2 (ByteTrack): leftover tracks vs. low-confidence detections, looser IoU, no gate.
        stage2 = association.associate(
            predicted_tracks, predicted_bboxes, detection_bboxes,
            iou_threshold=cfg.low_score_iou_threshold, use_mahalanobis_gate=False,
            candidate_tracks=stage1.unmatched_tracks, candidate_detections=low,
        )
        # Stage 3: 3D-aware re-activation for high-confidence detections still unmatched.
        stage3 = association.associate_3d(
            det_boxes3d, [d.label for d in detections], live_objects,
            iou_threshold=cfg.reactivation_iou_threshold, containment_margin=cfg.reactivation_margin_m,
            candidate_objects=stage2.unmatched_tracks, candidate_detections=stage1.unmatched_detections,
        )

        t_associate = time.perf_counter()

        detection_instance_ids = [-1] * len(detections)
        for track_idx, det_idx in stage1.matches + stage2.matches:
            detection = detections[det_idx]
            updated_track = tracking.update(predicted_tracks[track_idx], detection.bbox)
            self.object_map.update_matched(
                live_objects[track_idx], updated_track, det_points[det_idx], detection,
                observation.stamp, K, T_world_from_cam, evidence_depth,
            )
            detection_instance_ids[det_idx] = live_objects[track_idx].instance_id
        for track_idx, det_idx in stage3.matches:
            self.object_map.reactivate(
                live_objects[track_idx], det_points[det_idx], detections[det_idx],
                observation.stamp, K, T_world_from_cam, evidence_depth,
            )
            detection_instance_ids[det_idx] = live_objects[track_idx].instance_id

        for track_idx in stage3.unmatched_tracks:
            obj = live_objects[track_idx]
            obj.track = predicted_tracks[track_idx]
            if depth is not None:
                self.object_map.update_unmatched(obj, K, T_world_from_cam, evidence_depth)
            else:
                obj.frames_since_seen += 1
        for track_idx in np.nonzero(~in_view)[0]:
            obj = live_objects[track_idx]
            if depth is not None:
                self.object_map.update_unmatched(obj, K, T_world_from_cam, evidence_depth, in_view=False)
            else:
                obj.frames_since_seen += 1

        # Only high-confidence detections may start a new object (ByteTrack).
        for det_idx in stage3.unmatched_detections:
            detection = detections[det_idx]
            spawned = self.object_map.spawn(
                detection.bbox, det_points[det_idx], detection.label, detection.score, observation.stamp,
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

        self.object_map.prune_disappeared(cfg.disappeared_prune_grace_frames)
        t_update = time.perf_counter()

        graph = sg.build_scene_graph(
            list(self.object_map.objects.values()),
            cluster_radius=self.config.scene_graph_cluster_radius,
            z_tolerance=self.config.scene_graph_z_tolerance,
            xy_iou_threshold=self.config.scene_graph_xy_iou_threshold,
            beside_max_distance=self.config.scene_graph_beside_max_distance,
            support_classes=self.config.scene_graph_support_classes,
            cache=self._edge_cache,
        )

        t_graph = time.perf_counter()

        return FrameResult(
            objects=list(self.object_map.objects.values()),
            scene_graph=graph,
            detection_instance_ids=detection_instance_ids,
            timings={
                "predict": t_predict - t_start,
                "backproject": t_backproject - t_predict,
                "associate": t_associate - t_backproject,
                "map_update": t_update - t_associate,
                "scene_graph": t_graph - t_update,
                "total": t_graph - t_start,
            },
        )
