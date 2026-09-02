"""Class- and instance-level segmentation metrics (Sec. V-B, Tables II-III).

The paper benchmarks the object map on ScanNet with class-level mIoU,
frequency-weighted mIoU, and accuracy (with and without background classes),
and instance-level mAP at 3D IoU 0.25 / 0.5 per class. Ground truth is a
labeled point set (ScanNet's annotated mesh vertices, or the synthetic
scene's rendered surfaces); predictions are the map's not-disappeared
instances.

Map instances are transferred onto the ground-truth points by nearest
neighbour: every ground-truth point takes the label and instance of the
closest map point within ``max_distance``, or stays unlabeled. The metrics
are then computed on that single common support, independent of the map's
own voxel density, which is the protocol object-centric baselines use on
ScanNet (predictions are scored as masks over the annotated vertices).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from semantic_mapping.evaluation import PRESENT_STATUSES
from semantic_mapping.types import ObjectInstance

UNLABELED = ""
"""Label of ground-truth points that carry no annotation (never evaluated)."""

DEFAULT_MAX_DISTANCE_M = 0.1
DEFAULT_BACKGROUND_CLASSES = ("wall", "floor", "ceiling")
DEFAULT_AP_THRESHOLDS = (0.25, 0.5)


@dataclass
class GroundTruthPoints:
    """Annotated points: a semantic label and an instance id per point."""

    points: np.ndarray
    """(N, 3) world-frame coordinates."""

    labels: np.ndarray
    """(N,) class name per point (``UNLABELED`` for unannotated points)."""

    instance_ids: np.ndarray
    """(N,) instance id per point, -1 where the point belongs to no annotated instance."""

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        self.labels = np.asarray(self.labels, dtype=str).reshape(-1)
        self.instance_ids = np.asarray(self.instance_ids, dtype=np.int64).reshape(-1)
        n = self.points.shape[0]
        if self.labels.shape[0] != n or self.instance_ids.shape[0] != n:
            raise ValueError("points, labels, and instance_ids must have the same length")

    @classmethod
    def load(cls, path: str | Path) -> "GroundTruthPoints":
        data = np.load(path, allow_pickle=False)
        return cls(points=data["points"], labels=data["labels"], instance_ids=data["instance_ids"])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, points=self.points.astype(np.float32), labels=self.labels, instance_ids=self.instance_ids.astype(np.int32),
        )


def normalize_label(label: str, aliases: dict[str, str] | None = None) -> str:
    """Canonical form used to compare detector vocabulary with ground-truth classes."""
    key = " ".join(str(label).lower().strip().split())
    if aliases:
        key = aliases.get(key, key)
    return key


def normalize_labels(labels: np.ndarray, aliases: dict[str, str] | None = None) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    mapped = np.array([normalize_label(u, aliases) if u != UNLABELED else UNLABELED for u in unique], dtype=str)
    return mapped[inverse]


@dataclass
class LabelTransfer:
    """Map instances projected onto the ground-truth support."""

    pred_labels: np.ndarray
    """(N,) predicted class per ground-truth point (``UNLABELED`` when no map point is near)."""

    pred_instance_ids: np.ndarray
    """(N,) map instance id per ground-truth point, -1 when unassigned."""

    instances: dict[int, ObjectInstance] = field(default_factory=dict)
    """The evaluated map instances, by id (including ones that captured no ground-truth point)."""

    unsupported_points: dict[int, int] = field(default_factory=dict)
    """Per instance: how many of its own map points have no ground-truth point
    within ``max_distance``. Volume the map asserts where the annotation has
    no surface; large values flag inflated or hallucinated instances."""


def transfer_labels(
    objects: list[ObjectInstance],
    gt: GroundTruthPoints,
    max_distance: float = DEFAULT_MAX_DISTANCE_M,
    aliases: dict[str, str] | None = None,
    statuses=PRESENT_STATUSES,
) -> LabelTransfer:
    """Assign each ground-truth point to the nearest map point's instance."""
    n_gt = gt.points.shape[0]
    pred_labels = np.full(n_gt, UNLABELED, dtype=object)
    pred_instance_ids = np.full(n_gt, -1, dtype=np.int64)
    evaluated = {o.instance_id: o for o in objects if o.status in statuses}
    with_points = [o for o in evaluated.values() if o.points_world.shape[0] > 0]
    unsupported: dict[int, int] = {o.instance_id: 0 for o in evaluated.values()}
    if not with_points or n_gt == 0:
        return LabelTransfer(pred_labels.astype(str), pred_instance_ids, evaluated, unsupported)

    map_points = np.concatenate([o.points_world for o in with_points], axis=0)
    owners = np.concatenate([np.full(o.points_world.shape[0], o.instance_id) for o in with_points])
    distances, nearest = cKDTree(map_points).query(gt.points, distance_upper_bound=max_distance)
    found = np.isfinite(distances)
    pred_instance_ids[found] = owners[nearest[found]]
    label_of = {o.instance_id: normalize_label(o.label, aliases) for o in with_points}
    for instance_id in np.unique(pred_instance_ids[found]):
        pred_labels[pred_instance_ids == instance_id] = label_of[int(instance_id)]

    back, _ = cKDTree(gt.points).query(map_points, distance_upper_bound=max_distance)
    for instance_id, count in zip(*np.unique(owners[~np.isfinite(back)], return_counts=True)):
        unsupported[int(instance_id)] = int(count)
    return LabelTransfer(pred_labels.astype(str), pred_instance_ids, evaluated, unsupported)


def class_level_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    classes: list[str] | None = None,
    exclude_classes=(),
) -> dict:
    """mIoU, frequency-weighted mIoU, overall and mean class accuracy (Table II).

    Ground-truth points that are unlabeled or belong to ``exclude_classes``
    are left out of the evaluation entirely (the "without background"
    setting); everything else is scored against ``classes`` (default: every
    class that has ground truth). A prediction of a class outside that set
    counts as a miss for the point's true class, not as a false positive
    anywhere.
    """
    gt_labels = np.asarray(gt_labels, dtype=str)
    pred_labels = np.asarray(pred_labels, dtype=str)
    in_scope = (gt_labels != UNLABELED) & ~np.isin(gt_labels, list(exclude_classes))
    if classes is None:
        classes = sorted(np.unique(gt_labels[in_scope]).tolist())

    per_class: dict[str, dict] = {}
    for name in classes:
        is_gt = in_scope & (gt_labels == name)
        is_pred = in_scope & (pred_labels == name)
        tp = int(np.sum(is_gt & is_pred))
        fp = int(np.sum(is_pred & ~is_gt))
        fn = int(np.sum(is_gt & ~is_pred))
        support = int(np.sum(is_gt))
        per_class[name] = {
            "iou": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
            "acc": tp / support if support else float("nan"),
            "support": support, "tp": tp, "fp": fp, "fn": fn,
        }

    scored = {c: m for c, m in per_class.items() if m["support"] > 0}
    total_support = sum(m["support"] for m in scored.values())
    n_points = int(np.sum(in_scope))
    return {
        "classes": per_class,
        "miou": float(np.mean([m["iou"] for m in scored.values()])) if scored else float("nan"),
        "fmiou": (sum(m["support"] * m["iou"] for m in scored.values()) / total_support
                  if total_support else float("nan")),
        "acc": float(np.sum(in_scope & (gt_labels == pred_labels)) / n_points) if n_points else float("nan"),
        "mean_class_acc": float(np.mean([m["acc"] for m in scored.values()])) if scored else float("nan"),
        "num_points": n_points,
        "num_classes": len(scored),
    }


def _average_precision(is_tp: np.ndarray, n_gt: int) -> float:
    """Area under the precision-recall curve (all-point interpolation), for
    predictions already sorted by descending confidence."""
    if n_gt == 0:
        return float("nan")
    if is_tp.size == 0:
        return 0.0
    tp = np.cumsum(is_tp)
    fp = np.cumsum(~is_tp)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    # Precision envelope: at each recall level, the best precision achievable at that recall or higher.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * envelope))


def instance_level_ap(
    gt: GroundTruthPoints,
    transfer: LabelTransfer,
    iou_thresholds=DEFAULT_AP_THRESHOLDS,
    classes: list[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> dict:
    """Per-class average precision of map instances at point-set IoU thresholds (Table III).

    A map instance is the set of ground-truth points transferred to it; its
    IoU with a ground-truth instance is computed over those point sets.
    Predictions are ranked by label confidence and matched greedily to the
    unmatched ground-truth instance of the same class with the highest IoU.
    """
    gt_labels = normalize_labels(gt.labels, aliases)
    gt_ids = gt.instance_ids
    annotated = gt_ids >= 0

    gt_instances: dict[int, tuple[str, np.ndarray]] = {}
    for gid in np.unique(gt_ids[annotated]):
        member = gt_ids == gid
        labels, counts = np.unique(gt_labels[member], return_counts=True)
        gt_instances[int(gid)] = (str(labels[np.argmax(counts)]), member)

    pred_instances: dict[int, tuple[str, float, np.ndarray]] = {}
    for pid, obj in transfer.instances.items():
        member = transfer.pred_instance_ids == pid
        pred_instances[pid] = (normalize_label(obj.label, aliases), float(obj.label_confidence), member)

    if classes is None:
        classes = sorted({name for name, _ in gt_instances.values() if name != UNLABELED})

    results: dict[str, dict] = {}
    for threshold in iou_thresholds:
        per_class: dict[str, float] = {}
        for name in classes:
            gts = [gid for gid, (label, _) in gt_instances.items() if label == name]
            preds = sorted(
                (pid for pid, (label, _, _) in pred_instances.items() if label == name),
                key=lambda pid: (-pred_instances[pid][1], pid),
            )
            matched: set[int] = set()
            is_tp = np.zeros(len(preds), dtype=bool)
            for rank, pid in enumerate(preds):
                pred_member = pred_instances[pid][2]
                best_iou, best_gid = 0.0, None
                for gid in gts:
                    if gid in matched:
                        continue
                    gt_member = gt_instances[gid][1]
                    inter = int(np.sum(pred_member & gt_member))
                    union = int(np.sum(pred_member | gt_member))
                    iou = inter / union if union else 0.0
                    if iou > best_iou:
                        best_iou, best_gid = iou, gid
                if best_gid is not None and best_iou >= threshold:
                    matched.add(best_gid)
                    is_tp[rank] = True
            per_class[name] = _average_precision(is_tp, len(gts))
        scored = [ap for ap in per_class.values() if not np.isnan(ap)]
        results[f"ap{int(round(threshold * 100))}"] = {
            "threshold": threshold,
            "per_class": per_class,
            "map": float(np.mean(scored)) if scored else float("nan"),
        }
    return {"classes": classes, "num_gt_instances": len(gt_instances), "num_predictions": len(pred_instances), **results}


def segmentation_report(
    objects: list[ObjectInstance],
    gt: GroundTruthPoints,
    background_classes=DEFAULT_BACKGROUND_CLASSES,
    aliases: dict[str, str] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE_M,
    ap_thresholds=DEFAULT_AP_THRESHOLDS,
    instance_classes: list[str] | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Everything Tables II and III report, for one final map against one annotated scene."""
    transfer = transfer_labels(objects, gt, max_distance=max_distance, aliases=aliases)
    gt_labels = normalize_labels(gt.labels, aliases)
    background = [normalize_label(c, aliases) for c in background_classes]
    return {
        "max_distance": max_distance,
        "num_gt_points": int(gt.points.shape[0]),
        "num_transferred_points": int(np.sum(transfer.pred_instance_ids >= 0)),
        "class_level": {
            "without_background": class_level_metrics(gt_labels, transfer.pred_labels, class_names, background),
            "with_background": class_level_metrics(gt_labels, transfer.pred_labels, class_names),
        },
        "instance_level": instance_level_ap(gt, transfer, ap_thresholds, instance_classes, aliases),
        "unsupported_points": {str(k): v for k, v in transfer.unsupported_points.items()},
    }


def format_segmentation_report(report: dict) -> str:
    cl = report["class_level"]
    lines = [
        f"{'class-level':<22} {'mIoU':>7} {'f-mIoU':>7} {'Acc':>7} {'mAcc':>7} {'classes':>8} {'points':>8}",
    ]
    for name, key in (("without background", "without_background"), ("with background", "with_background")):
        m = cl[key]
        lines.append(f"{name:<22} {m['miou']:>7.3f} {m['fmiou']:>7.3f} {m['acc']:>7.3f} {m['mean_class_acc']:>7.3f} "
                     f"{m['num_classes']:>8d} {m['num_points']:>8d}")
    lines.append("")
    lines.append(f"{'per class (w/o bg)':<22} {'IoU':>7} {'Acc':>7} {'support':>8}")
    for name, m in sorted(cl["without_background"]["classes"].items()):
        if m["support"]:
            lines.append(f"{name:<22} {m['iou']:>7.3f} {m['acc']:>7.3f} {m['support']:>8d}")

    inst = report["instance_level"]
    ap_keys = [k for k in inst if k.startswith("ap")]
    lines += ["", f"{'instance-level':<22} " + " ".join(f"{k.upper():>7}" for k in ap_keys)]
    for name in inst["classes"]:
        lines.append(f"{name:<22} " + " ".join(f"{inst[k]['per_class'].get(name, float('nan')):>7.3f}" for k in ap_keys))
    lines.append(f"{'mAP':<22} " + " ".join(f"{inst[k]['map']:>7.3f}" for k in ap_keys)
                 + f"   ({inst['num_predictions']} predictions / {inst['num_gt_instances']} ground-truth instances)")
    lines.append(f"\nlabel transfer: {report['num_transferred_points']} / {report['num_gt_points']} ground-truth points "
                 f"within {report['max_distance']:.2f} m of a map point")
    return "\n".join(lines)
