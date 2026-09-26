"""Checks of this package against the results its paper reports (doc/paper.pdf, Sec. V).

The paper's numbers (``config/paper_results.yaml``) come from data that is
only partly public: ScanNet for Tables II-III, and the authors' own robot
captures for Tables IV-V. Each check therefore says which of four kinds it is:

* **reproduction**: measured on the paper's own data, so the value itself is
  compared, within a tolerance ("reproduced", "above paper", "below paper").
* **floor**: the paper's value was measured on harder data than the run used
  (real sensors against a synthetic scene, a GPU laptop against a CPU), so the
  measurement should not fall below it ("at or above paper", "below paper").
* **claim**: a statement the paper draws from its results, such as "the full
  system outperforms every ablation", checked on the data at hand ("holds",
  "fails").
* **protocol**: the evaluation itself matches the paper's definitions.

A check the available data cannot support is listed as "not run", with the
reason, so the report always accounts for every result in the paper.
``examples/paper_harness.py`` runs the checks and prints the report.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from semantic_mapping import evaluation
from semantic_mapping.segmentation_metrics import normalize_label

REPRODUCED = "reproduced"
ABOVE = "above paper"
BELOW = "below paper"
AT_OR_ABOVE = "at or above paper"
HOLDS = "holds"
FAILS = "fails"
NOT_RUN = "not run"

TABLE5_FULL = "All (proposed)"
TABLE5_ABLATIONS = ("W/o 2D Tracker", "W/o Semantic Fusion", "W/o Geometric Consistency Update")
TABLE5_OVERRIDES = {
    "W/o 2D Tracker": {"use_2d_tracker": False},
    "W/o Semantic Fusion": {"use_semantic_fusion": False},
    "W/o Geometric Consistency Update": {"use_geometric_consistency": False},
    TABLE5_FULL: {},
}
"""PipelineConfig overrides of each Table V configuration."""


@dataclass
class Check:
    """One comparison with the paper."""

    id: str
    source: str
    """Where the paper reports it, e.g. "Table IV" or "Sec. V-C"."""
    kind: str
    """"reproduction", "floor", "claim" or "protocol"."""
    description: str
    paper: float | str | None
    measured: float | str | None
    verdict: str
    data: str
    """What the measurement ran on."""
    detail: str = ""
    known_deviation: str = ""
    """Why this check is expected to differ from the paper (``config/paper_deviations.yaml``)."""

    def missed(self, strict: bool = False) -> bool:
        """A floor or claim that does not hold; with ``strict``, also a value that was not reproduced."""
        if self.verdict == FAILS or (self.verdict == BELOW and self.kind == "floor"):
            return True
        return strict and self.kind == "reproduction" and self.verdict in (ABOVE, BELOW)

    def failed(self, strict: bool = False) -> bool:
        """A miss that is not a documented known deviation."""
        return self.missed(strict) and not self.known_deviation


def default_results_path() -> Path:
    """``config/paper_results.yaml`` of the source tree, or of the installed package."""
    source = Path(__file__).resolve().parents[1] / "config" / "paper_results.yaml"
    if source.exists():
        return source
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("semantic_mapping")) / "config" / "paper_results.yaml"


def load_paper_results(path: str | Path | None = None) -> dict:
    with open(path or default_results_path()) as f:
        return yaml.safe_load(f)


def load_known_deviations(path: str | Path | None = None) -> dict[str, str]:
    """Check id -> why it differs from the paper (default: ``config/paper_deviations.yaml``)."""
    path = Path(path) if path is not None else default_results_path().with_name("paper_deviations.yaml")
    if not path.exists():
        return {}
    with open(path) as f:
        return {str(k): " ".join(str(v).split()) for k, v in (yaml.safe_load(f) or {}).items()}


def apply_known_deviations(checks: list[Check], deviations: dict[str, str], strict: bool = False) -> list[str]:
    """Mark checks that miss the paper for a documented reason; returns the listed
    ids that no longer miss (or were not run), whose entries can be retired."""
    by_id = {check.id: check for check in checks}
    stale = []
    for check_id, reason in deviations.items():
        check = by_id.get(check_id)
        if check is None or check.verdict == NOT_RUN:
            continue
        if check.missed(strict):
            check.known_deviation = reason
        else:
            stale.append(check_id)
    return stale


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def compare(paper: float, measured: float, kind: str, tolerance: float = 0.0) -> str:
    """Verdict of one value: a reproduction within ``tolerance``, or a floor."""
    if not _finite(measured):
        return NOT_RUN
    if kind == "reproduction":
        if abs(measured - paper) <= tolerance:
            return REPRODUCED
        return ABOVE if measured > paper else BELOW
    return AT_OR_ABOVE if measured >= paper else BELOW


def not_run(id: str, source: str, description: str, paper, reason: str, kind: str = "reproduction") -> Check:
    return Check(id, source, kind, description, paper, None, NOT_RUN, "", reason)


# ---------------------------------------------------------------- protocol
def protocol_checks(reference: dict) -> list[Check]:
    """The Sec. V-D true-positive criterion the evaluator applies is the paper's."""
    protocol = reference["protocol"]
    checks = []
    for key, value, name in (
        ("tp_iou_3d", evaluation.DEFAULT_IOU_THRESHOLD, "3D IoU threshold of a true positive"),
        ("tp_centroid_m", evaluation.DEFAULT_CENTROID_THRESHOLD_M, "centroid distance threshold (m)"),
    ):
        verdict = HOLDS if math.isclose(value, protocol[key]) else FAILS
        checks.append(Check(f"protocol.{key}", "Sec. V-D", "protocol", name, protocol[key], value, verdict,
                            "semantic_mapping/evaluation.py"))
    return checks


# ------------------------------------------------------ Table IV / Sec. V-C
def _gt_index_by_label(evaluator: evaluation.SequenceEvaluator, label: str) -> tuple[int | None, str]:
    matches = [i for i, gt in enumerate(evaluator.ground_truth) if gt.label == label]
    if len(matches) != 1:
        return None, f"{len(matches)} ground-truth objects are labelled {label!r}; Table IV needs exactly one"
    return matches[0], ""


def table4_checks(evaluator: evaluation.SequenceEvaluator, reference: dict, data: str,
                  reproduction: bool = False, tolerance: float = 0.05) -> list[Check]:
    """Table IV: per-object detection and change-detection recall of the six changed objects.

    On the paper's own capture the values are compared (``reproduction``);
    anywhere else the paper's values are floors, since its real-world run was
    the harder setting. Against DualMap's row the paper claims higher recall
    for every object and metric, which only its own data can test.
    """
    table = reference["table4"]
    ours = table["methods"]["SuperMap"]
    kind = "reproduction" if reproduction else "floor"
    per_object = {row["identity"]: row for row in evaluator.summary()["per_object"]}
    checks, measured_all = [], {}
    for label in table["appeared"] + table["disappeared"]:
        index, reason = _gt_index_by_label(evaluator, label)
        row = per_object[evaluator.ground_truth[index].identity] if index is not None else None
        for metric, key in (("detection", "detection_recall"), ("change", "change_recall")):
            description = f"{label}: {metric} recall"
            check_id = f"table4.{label}.{metric}"
            if row is None:
                checks.append(not_run(check_id, "Table IV", description, ours[label][metric], reason, kind))
                continue
            value = float(row[key])
            measured_all[(label, metric)] = value
            checks.append(Check(check_id, "Table IV", kind, description, ours[label][metric], value,
                                compare(ours[label][metric], value, kind, tolerance), data))
    dualmap = table["methods"]["DualMap"]
    claim = "SuperMap's recall is at least DualMap's for every changed object, both metrics"
    if reproduction and measured_all:
        worse = [f"{label} {metric}" for (label, metric), value in measured_all.items()
                 if value < dualmap[label][metric]]
        checks.append(Check("table4.beats_dualmap", "Table IV", "claim", claim, "holds", "fails" if worse else "holds",
                            FAILS if worse else HOLDS, data, "below DualMap: " + ", ".join(worse) if worse else ""))
    else:
        checks.append(not_run("table4.beats_dualmap", "Table IV", claim, "holds",
                              "DualMap's numbers were measured on the paper's capture only", "claim"))
    return checks


def _majority_id(stats: evaluation.ObjectStats) -> int | None:
    if not stats.matched_id_frames:
        return None
    return max(stats.matched_id_frames.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def identity_checks(evaluator: evaluation.SequenceEvaluator, reference: dict, data: str) -> list[Check]:
    """Sec. V-C / Fig. 3 and Sec. IV-B: what the paper says happens to identities across scene changes.

    * Objects that disappear keep one instance ID while present, and the
      instance is still in the map under that ID afterwards (whether the
      removal was confirmed is change-detection recall, Table IV).
    * Newly introduced objects receive new IDs.
    * Objects that stay put keep their original IDs through the changes.
    * An object moved elsewhere, or taken away and brought back, keeps its ID
      (Sec. IV-B; scored by the evaluator's identity consistency).
    """
    table = reference["table4"]
    final = {o.instance_id: o for o in evaluator.final_objects}
    checks = []

    def add(check_id, source, description, ok, detail=""):
        checks.append(Check(check_id, source, "claim", description, "holds", "holds" if ok else "fails",
                            HOLDS if ok else FAILS, data, detail))

    for label in table["disappeared"]:
        index, reason = _gt_index_by_label(evaluator, label)
        description = f"{label} (removed): one ID while present, kept after the removal"
        if index is None:
            checks.append(not_run(f"identity.{label}.kept", "Sec. V-C", description, "holds", reason, "claim"))
            continue
        stats = evaluator.stats[index]
        instance = final.get(_majority_id(stats))
        add(f"identity.{label}.kept", "Sec. V-C", description, stats.fragments == 1 and instance is not None,
            f"IDs {sorted(stats.matched_ids)}; final status "
            f"{instance.status.name if instance is not None else 'gone from the map'}")

    for label in table["appeared"]:
        index, reason = _gt_index_by_label(evaluator, label)
        description = f"{label} (introduced): gets a new ID"
        new = evaluator.got_new_identity(index) if index is not None else None
        if new is None:
            checks.append(not_run(f"identity.{label}.new", "Sec. V-C", description, "holds",
                                  reason or "never matched, or present from the start", "claim"))
            continue
        add(f"identity.{label}.new", "Sec. V-C", description, new,
            f"IDs {sorted(evaluator.stats[index].matched_ids)}")

    static = [i for i, gt in enumerate(evaluator.ground_truth)
              if gt.appear_frame == 0 and gt.disappear_frame >= max(evaluator.max_id_by_frame, default=0) + 1
              and sum(1 for other in evaluator.ground_truth if other.identity == gt.identity) == 1]
    fragmented = [evaluator.ground_truth[i].identity for i in static if evaluator.stats[i].fragments > 1]
    unseen = [evaluator.ground_truth[i].identity for i in static if evaluator.stats[i].fragments == 0]
    add("identity.static.kept", "Sec. V-C", "objects that stay put keep one ID through the changes",
        bool(static) and not fragmented,
        f"{len(static)} static objects" + (f"; fragmented: {', '.join(fragmented)}" if fragmented else "")
        + (f"; never matched: {', '.join(unseen)}" if unseen else ""))

    consistency = evaluator.identity_consistency()
    if consistency["evaluated"]:
        add("identity.relocation", "Sec. IV-B", "moved or returned objects keep their ID",
            consistency["consistent"] == consistency["evaluated"],
            f"{consistency['consistent']} of {consistency['evaluated']} identities kept")
    else:
        checks.append(not_run("identity.relocation", "Sec. IV-B", "moved or returned objects keep their ID", "holds",
                              "no object with several presence phases", "claim"))
    return checks


# ---------------------------------------------------------------- Table V
def table5_checks(runs: list[dict[str, dict]], reference: dict, data: str,
                  reproduction: bool = False, tolerance: float = 0.05) -> list[Check]:
    """Table V: final-map precision / recall / F1 of the full system and each ablation.

    ``runs`` holds one {configuration: {precision, recall, f1}} per scene or
    seed; means are compared. The paper's claim is that the full system
    "outperforms all baseline configurations across every metric": its F1
    must be higher than each ablation's, and its precision and recall no
    lower. Values are compared only on the paper's own capture.
    """
    configs = reference["table5"]["configurations"]
    mean = {name: {m: float(np.nanmean([run[name][m] for run in runs])) for m in ("precision", "recall", "f1")}
            for name in configs if all(name in run for run in runs)}
    checks = []
    for name, paper in configs.items():
        for metric in ("precision", "recall", "f1"):
            check_id = f"table5.{name}.{metric}"
            description = f"{name}: {metric}"
            if name not in mean:
                checks.append(not_run(check_id, "Table V", description, paper[metric], "configuration not run"))
            elif reproduction:
                value = mean[name][metric]
                checks.append(Check(check_id, "Table V", "reproduction", description, paper[metric], value,
                                    compare(paper[metric], value, "reproduction", tolerance), data))
            else:
                checks.append(Check(check_id, "Table V", "reproduction", description, paper[metric],
                                    mean[name][metric], NOT_RUN, data,
                                    "not compared: the paper's 41-object capture is not public"))
    if TABLE5_FULL not in mean:
        return checks
    full = mean[TABLE5_FULL]
    for name in TABLE5_ABLATIONS:
        if name not in mean:
            continue
        ablated = mean[name]
        f1_wins = sum(run[TABLE5_FULL]["f1"] > run[name]["f1"] for run in runs)
        ok = full["f1"] > ablated["f1"] and full["precision"] >= ablated["precision"] \
            and full["recall"] >= ablated["recall"]
        lower = [m for m in ("precision", "recall") if full[m] < ablated[m]]
        checks.append(Check(
            f"table5.outperforms.{name}", "Table V", "claim",
            f"All beats {name}: higher F1, precision and recall no lower",
            "holds", "holds" if ok else "fails", HOLDS if ok else FAILS, data,
            f"F1 {full['f1']:.3f} vs {ablated['f1']:.3f} (higher in {f1_wins}/{len(runs)} runs)"
            + (f"; lower {', '.join(lower)}" if lower else "")))
    return checks


# ----------------------------------------------------------- Tables II, III
def table2_checks(report: dict | None, reference: dict, data: str, tolerance_pp: float = 2.0) -> list[Check]:
    """Table II: class-level mIoU / f-mIoU / accuracy on ScanNet, without and with background.

    Values are compared in percentage points. The text's claim: SuperMap's
    accuracy outperforms the object-centric baselines ConceptGraphs and
    ConceptFusion.
    """
    methods = reference["table2"]["methods"]
    ours = methods["SuperMap"]
    checks = []
    for setting in ("without_background", "with_background"):
        for metric in ("miou", "fmiou", "acc"):
            check_id = f"table2.{setting}.{metric}"
            description = f"{setting.replace('_', ' ')}: {metric}"
            paper = ours[setting][metric]
            if report is None:
                checks.append(not_run(check_id, "Table II", description, paper, "needs annotated ScanNet scenes"))
                continue
            value = 100.0 * report["class_level"][setting][metric]
            checks.append(Check(check_id, "Table II", "reproduction", description + " (%)", paper, value,
                                compare(paper, value, "reproduction", tolerance_pp), data))
    claim = "accuracy (without background) above ConceptGraphs and ConceptFusion"
    if report is None:
        checks.append(not_run("table2.acc_beats_object_baselines", "Table II", claim, "holds",
                              "needs annotated ScanNet scenes", "claim"))
    else:
        value = 100.0 * report["class_level"]["without_background"]["acc"]
        baseline = max(methods[m]["without_background"]["acc"] for m in ("ConceptGraphs", "ConceptFusion"))
        ok = _finite(value) and value > baseline
        checks.append(Check("table2.acc_beats_object_baselines", "Table II", "claim", claim, "holds",
                            "holds" if ok else "fails", HOLDS if ok else FAILS, data,
                            f"{value:.2f} % vs {baseline:.2f} %"))
    return checks


def table3_checks(report: dict | None, reference: dict, data: str, aliases: dict | None = None,
                  tolerance_pp: float = 5.0) -> list[Check]:
    """Table III: instance-level AP50 / AP25 per class on ScanNet, and the claim
    that SuperMap outperforms HOV-SG and ConceptGraphs on every class."""
    table = reference["table3"]
    ours = table["methods"]["SuperMap"]
    checks, worse, compared = [], [], 0
    per_class = report["instance_level"] if report is not None else None
    for name in table["classes"]:
        key = normalize_label(name, aliases)
        for metric in ("ap50", "ap25"):
            check_id = f"table3.{name}.{metric}"
            description = f"{name}: {metric.upper()}"
            paper = ours[name][metric]
            value = per_class[metric]["per_class"].get(key, float("nan")) if per_class is not None else float("nan")
            if not _finite(value):
                reason = "needs annotated ScanNet scenes" if report is None else f"no ground-truth {key!r} instance"
                checks.append(not_run(check_id, "Table III", description, paper, reason))
                continue
            value *= 100.0
            compared += 1
            checks.append(Check(check_id, "Table III", "reproduction", description + " (%)", paper, value,
                                compare(paper, value, "reproduction", tolerance_pp), data))
            best = max(table["methods"][m][name][metric] for m in ("HOV-SG", "ConceptGraphs"))
            if value < best:
                worse.append(f"{name} {metric.upper()} {value:.1f} < {best:.1f}")
    claim = "AP at least HOV-SG's and ConceptGraphs' for every class"
    if not compared:
        checks.append(not_run("table3.beats_baselines", "Table III", claim, "holds",
                              "needs annotated ScanNet scenes", "claim"))
    else:
        checks.append(Check("table3.beats_baselines", "Table III", "claim", claim, "holds",
                            "fails" if worse else "holds", FAILS if worse else HOLDS, data, "; ".join(worse)))
    return checks


# ------------------------------------------------------------- Sec. V-H
def runtime_checks(timings_ms: dict[str, float] | None, reference: dict, data: str) -> list[Check]:
    """Sec. V-H: 3D mapping at 3 Hz and scene-graph updates at 5 Hz, as floors
    (the paper's GPU laptop against whatever this runs on)."""
    rates = reference["runtime"]["rates_hz"]
    checks = []
    if not timings_ms:
        for key in ("mapping_3d", "scene_graph_4d"):
            checks.append(not_run(f"runtime.{key}", "Sec. V-H", key.replace("_", " ") + " rate (Hz)",
                                  rates[key], "no timings", "floor"))
        return checks
    mapping_ms = timings_ms["total"] - timings_ms.get("scene_graph", 0.0)
    for key, latency in (("mapping_3d", mapping_ms), ("scene_graph_4d", timings_ms.get("scene_graph", 0.0))):
        rate = 1000.0 / latency if latency > 0 else float("inf")
        checks.append(Check(f"runtime.{key}", "Sec. V-H", "floor", key.replace("_", " ") + " rate (Hz)",
                            rates[key], rate, AT_OR_ABOVE if rate >= rates[key] else BELOW, data,
                            f"mean {latency:.1f} ms per frame"))
    return checks


# ----------------------------------------------------------------- running
@dataclass
class SequenceRun:
    """One pass of the pipeline over a sequence."""

    evaluator: evaluation.SequenceEvaluator | None
    final_objects: list
    timings_ms: dict[str, float]
    """Mean per-stage latency (FrameResult.timings) in milliseconds."""


def run_sequence(dataset, detector, prompts, params: dict, ground_truth=None, detector_rate_hz: float = 0.0,
                 on_frame=None) -> SequenceRun:
    """Map ``dataset`` with ``params`` and, given Sec. V-D ground truth, score every frame."""
    from semantic_mapping.datasets import run_sequence as drive
    from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline

    pipeline = SemanticMappingPipeline(PipelineConfig.from_dict(params))
    evaluator = (evaluation.SequenceEvaluator(ground_truth, dataset.intrinsics)
                 if ground_truth is not None else None)
    timings: dict[str, list[float]] = {}
    result = None
    for frame, detections, result in drive(dataset, pipeline, detector, prompts, detector_rate_hz=detector_rate_hz):
        if on_frame is not None:
            on_frame(frame, detections)
        if evaluator is not None:
            evaluator.observe(frame.frame_id, frame.T_world_from_cam, result.objects, depth_image=frame.depth,
                              stamp=frame.stamp)
        for key, value in result.timings.items():
            timings.setdefault(key, []).append(value)
    return SequenceRun(evaluator, result.objects if result is not None else [],
                       {key: 1e3 * float(np.mean(values)) for key, values in timings.items()})


def ablation_row(run: SequenceRun) -> dict:
    final = run.evaluator.summary()["final_map"]
    return {"precision": final["precision"], "recall": final["recall"], "f1": final["f1"]}


# ------------------------------------------------------------------ report
def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value == float("inf") else "n/a"
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"
    return str(value)


def format_report(checks: list[Check], strict: bool = False) -> str:
    """The checks as a Markdown table, grouped by where the paper reports them, with a verdict summary."""
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.verdict] = counts.get(check.verdict, 0) + 1
    failed = [c for c in checks if c.failed(strict)]
    known = [c for c in checks if c.missed(strict) and c.known_deviation]
    lines = [
        "| source | check | kind | paper | measured | verdict | data / detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for check in checks:
        verdict = check.verdict
        if check.failed(strict):
            verdict = f"**{verdict}**"
        elif check.known_deviation and check.missed(strict):
            verdict += " (known)"
        detail = "; ".join(part for part in (check.data, check.detail) if part)
        if check.known_deviation and check.missed(strict):
            detail += f"; known deviation: {check.known_deviation}"
        lines.append(f"| {check.source} | {check.description} | {check.kind} | {_fmt(check.paper)} | "
                     f"{_fmt(check.measured)} | {verdict} | {detail} |")
    summary = ", ".join(f"{n} {verdict}" for verdict, n in sorted(counts.items()))
    lines += ["", f"{len(checks)} checks: {summary}. "
              + (f"{len(known)} known deviation{'s' if len(known) != 1 else ''}. " if known else "")
              + (f"{len(failed)} failed: " + ", ".join(c.id for c in failed) if failed else "None failed.")]
    return "\n".join(lines)


def to_json(checks: list[Check]) -> list[dict]:
    rows = []
    for check in checks:
        row = asdict(check)
        for key in ("paper", "measured"):
            if isinstance(row[key], float) and not math.isfinite(row[key]):
                row[key] = None
        rows.append(row)
    return rows
