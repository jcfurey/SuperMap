"""Spatio-temporal consistency metrics (Sec. V-D) and map precision/recall (Sec. V-E).

Ground truth format (``scene_ground_truth.json``)::

    {
      "num_frames": 60,
      "objects": [
        {
          "label": "chair",
          "identity": "chair#5",    # optional: entries of one physical object share it
          "bbox3d": [xmin, ymin, zmin, xmax, ymax, zmax],   # world frame
          "appear_frame": 0,        # first frame the object is physically present here
          "disappear_frame": 33,    # first frame it is gone again (num_frames if never removed)
          "visible_frames": [0, 1, 2, ...]   # frames where it is in the camera's view
        }, ...
      ]
    }

An object that is moved, or taken away and brought back, has one entry per
presence phase with the same ``identity``; the identity-consistency metric
below checks that a single instance ID served every phase (Sec. IV-B,
identities stable across relocation).

True-positive criterion (paper, Sec. V-D): 3D IoU > 0.1, centroid distance
< 0.3 m, and the correct semantic label. Metrics per ground-truth object:

* ``detection_recall`` -- fraction of frames in the *appearance interval*
  (present and visible) in which the map contains a matching, not-disappeared
  instance.
* ``change_recall`` -- over the appearance interval plus the *disappearance
  interval* (frames after removal in which the object's former location is
  in view), the fraction of frames that are correct: detected while present,
  and no instance still asserting presence (active/tentative) once removed.
  Absence intervals are only scored for objects that were present first, so
  an object the system simply never mapped doesn't get "credit" for its
  absence (the artifact the paper points out in DualMap's numbers).
* ``fragments`` -- number of distinct instance IDs that ever matched the
  object; 1 means a single stable identity for the whole sequence.

Plus a final-map precision / recall / F1 in the spirit of the Sec. V-E
ablation: each not-disappeared instance in the final map is a TP if it is
the best match for some ground-truth object present at the end.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from semantic_mapping.geometry_utils import centroid, invert_se3, iou_3d, project_point
from semantic_mapping.types import CameraIntrinsics, ObjectInstance, ObjectStatus

DEFAULT_IOU_THRESHOLD = 0.1
DEFAULT_CENTROID_THRESHOLD_M = 0.3

PRESENT_STATUSES = (ObjectStatus.ACTIVE, ObjectStatus.OCCLUDED, ObjectStatus.TENTATIVE)
ASSERTING_STATUSES = (ObjectStatus.ACTIVE, ObjectStatus.TENTATIVE)


@dataclass
class GroundTruthObject:
    label: str
    bbox3d: np.ndarray
    appear_frame: int
    disappear_frame: int
    visible_frames: set[int]
    identity: str = ""

    @property
    def center(self) -> np.ndarray:
        return centroid(self.bbox3d)

    def present_at(self, frame_id: int) -> bool:
        return self.appear_frame <= frame_id < self.disappear_frame


def load_ground_truth(path: str | Path) -> tuple[int, list[GroundTruthObject]]:
    data = json.loads(Path(path).read_text())
    objects = [
        GroundTruthObject(
            label=o["label"],
            bbox3d=np.array(o["bbox3d"], dtype=np.float64),
            appear_frame=int(o["appear_frame"]),
            disappear_frame=int(o["disappear_frame"]),
            visible_frames=set(int(f) for f in o.get("visible_frames", [])),
            identity=str(o.get("identity") or f"{o['label']}#{i}"),
        )
        for i, o in enumerate(data["objects"])
    ]
    return int(data["num_frames"]), objects


def is_match(
    instance: ObjectInstance,
    gt: GroundTruthObject,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    centroid_threshold: float = DEFAULT_CENTROID_THRESHOLD_M,
) -> bool:
    if instance.label != gt.label or instance.points_world.shape[0] == 0:
        return False
    if iou_3d(instance.bbox3d, gt.bbox3d) <= iou_threshold:
        return False
    return float(np.linalg.norm(instance.center - gt.center)) < centroid_threshold


def location_in_view(
    gt: GroundTruthObject,
    intrinsics: CameraIntrinsics,
    T_world_from_cam: np.ndarray,
    depth_image: np.ndarray | None = None,
    occlusion_tolerance: float = 0.15,
) -> bool:
    """Whether the object's former location can currently be observed.

    Projects the ground-truth centroid into the camera and, when a depth image
    is given, also requires that nothing else stands in front of it: a removed
    object's spot that is hidden behind another object cannot be confirmed
    empty, so such frames don't count against the system in ``change_recall``.
    """
    pixel, z = project_point(intrinsics.K, invert_se3(T_world_from_cam), gt.center)
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    if not (z > 0 and 0 <= u < intrinsics.width and 0 <= v < intrinsics.height):
        return False
    if depth_image is None:
        return True
    sensor_depth = depth_image[v, u]
    if not np.isfinite(sensor_depth) or sensor_depth <= 0:
        return False
    return sensor_depth >= z - occlusion_tolerance


@dataclass
class ObjectStats:
    label: str
    appearance_frames: int = 0
    appearance_hits: int = 0
    absence_frames: int = 0
    absence_hits: int = 0
    matched_ids: set[int] = field(default_factory=set)

    @property
    def detection_recall(self) -> float:
        return self.appearance_hits / self.appearance_frames if self.appearance_frames else float("nan")

    @property
    def change_recall(self) -> float:
        total = self.appearance_frames + self.absence_frames
        return (self.appearance_hits + self.absence_hits) / total if total else float("nan")

    @property
    def fragments(self) -> int:
        return len(self.matched_ids)


class SequenceEvaluator:
    def __init__(
        self,
        ground_truth: list[GroundTruthObject],
        intrinsics: CameraIntrinsics,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        centroid_threshold: float = DEFAULT_CENTROID_THRESHOLD_M,
    ) -> None:
        self.ground_truth = ground_truth
        self.intrinsics = intrinsics
        self.iou_threshold = iou_threshold
        self.centroid_threshold = centroid_threshold
        self.stats = [ObjectStats(label=gt.label) for gt in ground_truth]
        self.total_ids_created = 0
        self._last_objects: list[ObjectInstance] = []
        self._last_frame_id = -1

    def _matches(self, objects: list[ObjectInstance], gt: GroundTruthObject, statuses) -> list[ObjectInstance]:
        return [
            o for o in objects
            if o.status in statuses and is_match(o, gt, self.iou_threshold, self.centroid_threshold)
        ]

    def observe(
        self,
        frame_id: int,
        T_world_from_cam: np.ndarray,
        objects: list[ObjectInstance],
        depth_image: np.ndarray | None = None,
    ) -> None:
        """Score one frame's map state against ground truth.

        ``depth_image`` (optional) lets the disappearance interval exclude
        frames in which the object's former location is occluded.
        """
        self._last_objects = objects
        self._last_frame_id = frame_id
        self.total_ids_created = max(self.total_ids_created, *(o.instance_id for o in objects), 0)

        for gt, stats in zip(self.ground_truth, self.stats):
            if gt.present_at(frame_id):
                if frame_id not in gt.visible_frames:
                    continue
                stats.appearance_frames += 1
                matched = self._matches(objects, gt, PRESENT_STATUSES)
                if matched:
                    stats.appearance_hits += 1
                    stats.matched_ids.update(o.instance_id for o in matched)
            elif frame_id >= gt.disappear_frame and not self._back_in_place(gt, frame_id) and location_in_view(
                gt, self.intrinsics, T_world_from_cam, depth_image,
            ):
                stats.absence_frames += 1
                if not self._matches(objects, gt, ASSERTING_STATUSES):
                    stats.absence_hits += 1

    def _back_in_place(self, gt: GroundTruthObject, frame_id: int) -> bool:
        """Whether the same physical object is present again at this entry's
        place (a later phase of the same identity), in which case asserting
        presence there is correct, not a missed disappearance."""
        return any(
            other is not gt and other.identity == gt.identity and other.present_at(frame_id)
            and float(np.linalg.norm(other.center - gt.center)) < self.centroid_threshold
            for other in self.ground_truth
        )

    def identity_consistency(self) -> dict:
        """Among identities with several phases that were each matched at least
        once, the fraction served by one and the same instance ID throughout."""
        phases: dict[str, list[set[int]]] = {}
        for gt, stats in zip(self.ground_truth, self.stats):
            phases.setdefault(gt.identity, []).append(stats.matched_ids)
        multi = {k: v for k, v in phases.items() if len(v) > 1}
        evaluated = {k: v for k, v in multi.items() if all(v)}
        consistent = [k for k, v in evaluated.items() if set.intersection(*v)]
        return {
            "identities_with_phases": len(multi),
            "evaluated": len(evaluated),
            "consistent": len(consistent),
            "rate": len(consistent) / len(evaluated) if evaluated else float("nan"),
            "inconsistent": sorted(k for k in evaluated if k not in consistent),
        }

    def final_map_prf(self) -> tuple[float, float, float, int, int, int]:
        """Precision / recall / F1 of the final map plus (tp, n_predictions, n_gt)."""
        present_gt = [gt for gt in self.ground_truth if gt.present_at(self._last_frame_id)]
        predictions = [o for o in self._last_objects if o.status in PRESENT_STATUSES]

        # Greedy one-to-one assignment by 3D IoU so duplicates count as false positives.
        candidates = []
        for pi, pred in enumerate(predictions):
            for gi, gt in enumerate(present_gt):
                if is_match(pred, gt, self.iou_threshold, self.centroid_threshold):
                    candidates.append((iou_3d(pred.bbox3d, gt.bbox3d), pi, gi))
        candidates.sort(reverse=True)
        used_pred, used_gt = set(), set()
        for _, pi, gi in candidates:
            if pi in used_pred or gi in used_gt:
                continue
            used_pred.add(pi)
            used_gt.add(gi)

        tp = len(used_pred)
        precision = tp / len(predictions) if predictions else float("nan")
        recall = tp / len(present_gt) if present_gt else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if predictions and present_gt and (precision + recall) > 0 else 0.0)
        return precision, recall, f1, tp, len(predictions), len(present_gt)

    def summary(self) -> dict:
        precision, recall, f1, tp, n_pred, n_gt = self.final_map_prf()
        per_object = [
            {
                "label": s.label,
                "identity": gt.identity,
                "detection_recall": s.detection_recall,
                "change_recall": s.change_recall,
                "fragments": s.fragments,
                "appearance_frames": s.appearance_frames,
                "absence_frames": s.absence_frames,
            }
            for gt, s in zip(self.ground_truth, self.stats)
        ]
        det = [s.detection_recall for s in self.stats if s.appearance_frames]
        chg = [s.change_recall for s in self.stats if (s.appearance_frames + s.absence_frames)]
        return {
            "per_object": per_object,
            "mean_detection_recall": float(np.mean(det)) if det else float("nan"),
            "mean_change_recall": float(np.mean(chg)) if chg else float("nan"),
            "mean_fragments": float(np.mean([s.fragments for s in self.stats])),
            "total_instance_ids_created": self.total_ids_created,
            "identity_consistency": self.identity_consistency(),
            "final_map": {
                "precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "predictions": n_pred, "ground_truth": n_gt,
            },
        }


def format_summary(summary: dict) -> str:
    lines = [f"{'object':<16} {'det. recall':>11} {'change recall':>13} {'fragments':>9}"]
    for row in summary["per_object"]:
        lines.append(
            f"{row.get('identity', row['label']):<16} {row['detection_recall']:>11.3f} "
            f"{row['change_recall']:>13.3f} {row['fragments']:>9d}"
        )
    fm = summary["final_map"]
    lines += [
        "",
        f"mean detection recall : {summary['mean_detection_recall']:.3f}",
        f"mean change recall    : {summary['mean_change_recall']:.3f}",
        f"mean fragments / obj  : {summary['mean_fragments']:.2f}  "
        f"(total instance IDs created: {summary['total_instance_ids_created']})",
        (f"identity kept across relocation / return: {summary['identity_consistency']['consistent']} / "
         f"{summary['identity_consistency']['evaluated']}"
         + (f"  (lost: {', '.join(summary['identity_consistency']['inconsistent'])})"
            if summary['identity_consistency']['inconsistent'] else "")),
        f"final map             : precision {fm['precision']:.3f}  recall {fm['recall']:.3f}  F1 {fm['f1']:.3f}  "
        f"(tp {fm['tp']} / {fm['predictions']} predictions / {fm['ground_truth']} ground truth)",
    ]
    return "\n".join(lines)
