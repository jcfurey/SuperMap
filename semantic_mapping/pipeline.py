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

from dataclasses import dataclass, field, fields

import numpy as np

from semantic_mapping import association, scene_graph as sg, tracking
from semantic_mapping.geometry_utils import back_project_depth, depth_consistency_mask, invert_se3, transform_points
from semantic_mapping.object_map import ObjectMap
from semantic_mapping.types import Detection2D, ObjectInstance, ObjectStatus, Observation


@dataclass
class PipelineConfig:
    voxel_size: float = 0.05
    tau_eps: float = 0.15
    max_points_per_object: int = 20000
    prune_log_odds: float = -1.5
    active_occupied_fraction: float = 0.6
    disappeared_occupied_fraction: float = 0.2
    max_occlusion_frames: int = 30
    min_label_confidence: float = 0.4
    min_observations_for_confidence_check: int = 5
    min_hits_to_confirm: int = 2
    association_iou_threshold: float = 0.3
    disappeared_prune_grace_frames: int = 60
    scene_graph_cluster_radius: float = 2.0
    scene_graph_z_tolerance: float = 0.1
    scene_graph_xy_iou_threshold: float = 0.05
    scene_graph_beside_max_distance: float = 1.0
    max_points_per_detection: int = 4000

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


class SemanticMappingPipeline:
    """Maintains M_t and G across frames; see Sec. III for the underlying model."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.object_map = ObjectMap(
            voxel_size=self.config.voxel_size,
            tau_eps=self.config.tau_eps,
            max_points_per_object=self.config.max_points_per_object,
            prune_log_odds=self.config.prune_log_odds,
            active_occupied_fraction=self.config.active_occupied_fraction,
            disappeared_occupied_fraction=self.config.disappeared_occupied_fraction,
            max_occlusion_frames=self.config.max_occlusion_frames,
            min_label_confidence=self.config.min_label_confidence,
            min_observations_for_confidence_check=self.config.min_observations_for_confidence_check,
        )
        self._frame_index = 0

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
        K = observation.intrinsics.K
        T_world_from_cam = observation.pose.T_world_from_frame
        T_cam_from_world = invert_se3(T_world_from_cam)

        live_ids = [
            instance_id for instance_id, obj in self.object_map.objects.items()
            if obj.status != ObjectStatus.DISAPPEARED
        ]
        live_objects = [self.object_map.objects[i] for i in live_ids]

        predicted_tracks: list[tracking.TrackKalmanState] = []
        predicted_bboxes: list[np.ndarray] = []
        for obj in live_objects:
            dt = max(observation.stamp - obj.latest_stamp, 1e-3)
            predicted = tracking.predict(
                obj.track, dt, K=K, T_world_from_cam=T_world_from_cam, object_centroid_world=obj.center,
            )
            predicted_tracks.append(predicted)
            predicted_bboxes.append(tracking.current_bbox(predicted))

        detection_bboxes = [d.bbox for d in observation.detections]
        result = association.associate(
            predicted_tracks, predicted_bboxes, detection_bboxes,
            iou_threshold=self.config.association_iou_threshold,
        )

        matched_track_indices = set()
        for track_idx, det_idx in result.matches:
            obj = live_objects[track_idx]
            detection = observation.detections[det_idx]
            updated_track = tracking.update(predicted_tracks[track_idx], detection.bbox)
            points_world = np.zeros((0, 3))
            if observation.depth is not None:
                points_world = self._detection_points_world(detection, observation.depth, K, T_world_from_cam)
            self.object_map.update_matched(
                obj, updated_track, points_world, detection.label, detection.score,
                observation.stamp, K, T_world_from_cam, observation.depth,
            )
            matched_track_indices.add(track_idx)

        if observation.depth is not None:
            for track_idx in result.unmatched_tracks:
                obj = live_objects[track_idx]
                obj.track = predicted_tracks[track_idx]
                self.object_map.update_unmatched(obj, K, T_world_from_cam, observation.depth)
        else:
            for track_idx in result.unmatched_tracks:
                obj = live_objects[track_idx]
                obj.track = predicted_tracks[track_idx]
                obj.frames_since_seen += 1

        for det_idx in result.unmatched_detections:
            detection = observation.detections[det_idx]
            points_world = np.zeros((0, 3))
            if observation.depth is not None:
                points_world = self._detection_points_world(detection, observation.depth, K, T_world_from_cam)
            self.object_map.spawn(detection.bbox, points_world, detection.label, detection.score, observation.stamp)

        for obj in self.object_map.objects.values():
            self.object_map.confirm_tentative(obj, self.config.min_hits_to_confirm)
            sg.record_trajectory_sample(obj, observation.stamp)

        self.object_map.prune_disappeared(self.config.disappeared_prune_grace_frames)

        graph = sg.build_scene_graph(
            list(self.object_map.objects.values()),
            cluster_radius=self.config.scene_graph_cluster_radius,
            z_tolerance=self.config.scene_graph_z_tolerance,
            xy_iou_threshold=self.config.scene_graph_xy_iou_threshold,
            beside_max_distance=self.config.scene_graph_beside_max_distance,
        )

        return FrameResult(objects=list(self.object_map.objects.values()), scene_graph=graph)
