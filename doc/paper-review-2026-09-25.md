# SuperMap paper review — 2026-09-25

This is a read-only comparison of `jcfurey/SuperMap` on branch `dev` at
`ac159e9` against the paper (`doc/paper.pdf`, RSS 2026). It covers the method
(Sec. IV, Eq. 3–10) and the evaluation protocol (Sec. V).

Upstream released no code, so this package is the fork's own reconstruction
from the paper. Every deviation below was therefore introduced here. The
earlier [review](review-2026-09-24.md) was done at `63d9b6e`. Items that came
in after it are marked with the commit that introduced them.

**Kinds of item:**

- **Contradiction**: the code does something other than what the paper states.
- **Reinterpretation**: the paper is ambiguous or silent, and the code picks one
  reading.
- **Extension**: the code adds behaviour the paper does not describe.

**Verification:** **Checked** means the code was read and the behaviour
confirmed; **Reproduced** means a script showed it end to end. Line numbers
refer to `ac159e9`. Fixes are recorded under [Resolution](#resolution).

## 1. Contradictions, on by default

| ID | Location | Paper | Code | Status |
|---|---|---|---|---|
| D1 | `association.py:127-140`, `semantic_fusion.py:28-38` (7ce1887) | Eq. 10 fuses detector labels so that "transient misclassifications" are suppressed. | A new instance starts with the belief `{label: 1.0}`. Every association stage admits a detection only when its label already holds `label_compatibility_min_mass` (0.1) of the belief. A label the instance has never seen therefore never reaches it, even with the mass set to 0. The new-label branch of Eq. 10 never runs, and every belief stays at one label with probability 1.0. Three `chair` then five `armchair` detections on the same box give two co-located ACTIVE instances, `{chair: 1.0}` and `{armchair: 1.0}`. Knock-on effects: label confidence is always 1.0, `min_label_confidence` never applies, AP ranks by instance ID (D17), and a "W/o Semantic Fusion" ablation would change nothing. | Reproduced |
| D2 | `geometric_consistency.py:116-129, 197-212` (7ce1887) | Eq. 9 uses the depth D(u) at the projected pixel. | A point is only Disappeared when the *nearest* reading in a 3×3 window is also behind it (`contradiction_window_px: 1`). Pruning needs two contradicting frames: the threshold is −3.30, not −1.5 (`prune_min_contradictions: 2`). Log-odds are clamped to ±8. These guard against review item C3 (a single glitched frame deleting a point, and edge erosion at silhouettes), but they delay disappearance, especially for thin objects. The paper-literal behaviour is available with `contradiction_window_px: 0` and `prune_min_contradictions: 1`. | Checked |
| D3 | `pipeline.py:731-769` | Association leverages 3D spatial consistency; Fig. 2 shows "2D→3D Validation". | Stage 1 and stage 2 matches are fused whatever depth the detection lifts to. The only guard is the opt-in `_refuse_oversized` (`class_size_limits`, empty by default). A chair at 2 m replaced by a chair at 6 m in the same image box keeps its ID, and its box stretches over 2–6 m. The replacement is never reported as one disappearance and one appearance. | Checked |
| D4 | `evaluation.py:149-152, 209-225` | Table IV scores the change-detection recall of *appeared* objects separately from their detection recall (Cart: 0.262 detection, 0.622 change). | Frames before `appear_frame` are never scored, so for an appeared object change recall equals detection recall. Change recall pools per-frame hits over the appearance and absence intervals, which is a reasonable per-frame reading of "detection in the former and zero false positives in the latter". The module docstring claims that never-detected objects get no absence credit, but after a disappearance they do. | Checked |
| D5 | `config/semantic_mapping.yaml:75, 91` | Sec. V-B detects with Grounding DINO and refines with SAM2. | The default detector is YOLOE with no SAM2. Grounding DINO + SAM2 is opt-in. Scores, labels and masks differ, which affects D1 and D20. | Checked |

## 2. Reinterpretations, on by default

| ID | Location | Paper | Code | Status |
|---|---|---|---|---|
| D6 | `tracking.py:24-57, 111-133`, `pipeline.py:695-704` | Eq. 5–6: the KF prior is the projected 3D centroid π(K·P⁻¹·X_i), and the transition is Ŝ = F·S + w. State is R⁶, or R⁸ with scale velocity. | The prior is the centre of the image-clipped 2D envelope of the projected 3D box. Eq. 5 is used only as a fallback when a box corner lies behind the camera or the box is off-image. The prior then overwrites the predicted position with weight 1.0, so the image-plane velocity never affects a prediction, only the covariance. X_i is the box centre, not the point centroid. Q, R and the gate are hand-tuned constants, not derived from pose or centroid uncertainty. The state is 6-D; the R⁸ variant is not implemented. The KF is predicted only on detector frames, over the gap since the last match. | Checked |
| D7 | `geometric_consistency.py:54-57, 151-163` | Eq. 8–9 with Gaussian noise p(Δd) ~ N(0, σ²). | P(o_k\|Q_t) is three constants: 0.9 for Observable, 0.1 for Disappeared, 0.5 (no update) otherwise. σ is not used; `gaussian_likelihood` has no callers. | Checked |
| D8 | `node.py:1473-1497`, `geometry_utils.py:207-303` (splatting 269600a) | D(u) is "the raw sensor depth". | With LiDAR, D(u) is the cloud rasterized with a per-pixel z-buffer. Occlusion splatting is on by default (`pointcloud_splat_radius_m: 0.05`) and drops returns hidden behind a nearer point's footprint. Scan accumulation, depth fill, depth-range limits and the platform mask are opt-in modifiers. | Checked |
| D9 | `object_map.py:508-514, 645-656`, `geometric_consistency.py:220-230` | The paper does not say how per-point states become an object status. | The rule is the fraction of points with log-odds ≥ 0, counted against alive plus contradicted points: ≤ 0.2 is DISAPPEARED, < 0.6 or unseen for 30 frames is OCCLUDED. Points never re-observed count as occupied. A matched instance is forced ACTIVE regardless of its geometry. | Checked |
| D10 | `semantic_fusion.py:97-120`, `object_map.py:401-414, 536-539, 658-667` | "For observable points in (9), we update the semantic label using (10), and consequently remove object points whose posterior belief is too small." | Points are removed by a separate per-point log-odds of being inside or outside the 2D mask, which is gated on observability. Eq. 10 runs once per matched detection and is not gated on observability; observability only chooses the sharper likelihood. The instance-level cut `should_discard_instance` needs 5 hits while an instance confirms at 2, so it almost never applies. | Checked |
| D11 | `semantic_fusion.py:41-87` | P(z_t\|L=c) is "the detector's confusion matrix". | A two-level stand-in: the observed label gets `0.05 + score·(0.85 − 0.05)` (0.95 when corroborated), and every other label gets 0.05. It is normalised over labels the instance has seen, and a new label enters with prior 1e-3. | Checked |
| D12 | `scene_graph.py:36-135`, yaml:218 | Candidate pairs come from clustering centroid distances. On(A,B) ⟺ z_min(A) ≈ z_max(B) ∧ IoU_xy > γ. Predicates are class-dependent (on, beside, under). | Candidate pairs come from a box gap ≤ 2 m; the yaml comment still says "centroid distance". `On` is the literal IoU, so a small object on a large support never qualifies (a cup on a 4×1 m table has IoU 0.0025). This is faithful to the paper's formula, and review item C15 remains in effect. `under` is the inverse of `on`. `beside` is not class-dependent. There is no `near`. | Checked |
| D13 | `scene_graph.py:30-33, 178-198`, `vln/serialize_prompt.py:106-147` | Temporal edges E_t link nodes across time. | There is no E_t. Each node keeps its own trajectory samples. The prompt receives only disappeared / reappeared / moved summaries, not timestamped positions. | Checked |
| D14 | `node.py:175-229, 1242-1256` (7ce1887) | The parser retrieves "their 3D centroids X_j ... used as navigation waypoints". | ROS goals are approach poses 0.6 m outside the object's footprint, at robot height. The offline `Grounder` still returns centroids. Disappeared instances can be chosen as goals. | Checked |
| D15 | yaml:261, 268-269, `vln/clients.py:62-90` | "We serialize a local subgraph"; the reasoning uses Gemini 2.0 Flash. | By default the whole graph is serialized (`vlm.local_radius_m` and `max_objects` are 0). The default client is `keyword`, a label substring match. | Checked |
| D16 | `node.py:1597-1635`, `pipeline.py:852`, `datasets.py:119-121` | Sec. V-H: segmentation 1 Hz, 3D mapping 3 Hz, scene graph 5 Hz. | Mapping runs on every synchronized frame (sensor rate), and the graph is rebuilt every mapped frame; 5 Hz only throttles publishing. Offline runs (`examples/evaluate.py`) detect on every frame, not at the paper's 1:3 ratio that the hit and miss thresholds assume. | Checked |
| D17 | `segmentation_metrics.py:101-136, 157-158, 235-262` | Tables II–III. | Labels transfer to GT points by nearest neighbour within 0.1 m. mIoU averages over the GT classes present. AP ranks predictions by `label_confidence`, which D1 pins at 1.0. | Checked |
| D18 | `evaluation.py:58, 211` | Detection recall is measured over the appearance interval. | Only GT `visible_frames` count. OCCLUDED and TENTATIVE instances count as detections. | Checked |

## 3. Extensions, on by default

| ID | Location | Code | Effect | Status |
|---|---|---|---|---|
| D19 | `pipeline.py:426-477, 541-549` (269600a) | `ground_exclusion`: masked readings less than 0.15 m above a locally fitted ground plane are dropped. | The bottom 15 cm of objects is never mapped or checked. mIoU 1.000 → 0.988 on the synthetic scene. | Checked |
| D20 | `pipeline.py:728-729, 811-818`, yaml:194 | ByteTrack score split: only detections scoring ≥ 0.5 may create an instance. | The paper does not mention ByteTrack's split. Grounding DINO's own `box_threshold` is 0.35, so detections between 0.35 and 0.5 can only extend existing tracks. That likely costs recall. | Checked |
| D21 | `association.py:202-307`, `pipeline.py:790-809`, `object_map.py:739-843` | Re-identification brings retired instances back by place, or anywhere by a chromaticity histogram (similarity ≥ 0.85, no age limit). Reconciliation and merging also run every frame. | Chromaticity ignores brightness, so grey, white and black objects share a signature. A newly introduced object of the same label can inherit a retired ID, but Fig. 3 says new objects get new IDs. | Checked |
| D22 | `node.py:1498-1501` (ac159e9) | Platform mask: pixels of the robot's own links are unknown depth, and detections on them are dropped. Opt-in (`platform_mask.enabled`). | Points under robot pixels get no Eq. 9 update. The paper's robot does not see itself. | Checked |

## 4. Opt-in options that contradict the paper when enabled

| ID | Option | Code | Status |
|---|---|---|---|
| D23 | `existence_hit_gain > 0` (f1a0a67) | Repeated detector misses mark an instance DISAPPEARED whatever its geometry, bypassing Eq. 9 (`object_map.py:624-633`). | Checked |
| D24 | `bbox_support_miss > 0` (f1a0a67) | Culls points a matched frame "looked at" but did not re-hit. Pixels with no depth reading count as looked at (`object_map.py:228-240`); in Eq. 9 they carry no evidence. | Checked |
| D25 | `dynamic_geometry_enabled` (4e1137d, 554b473) | Each matched frame replaces the class's points and log-odds instead of updating them recursively (Eq. 8). A compact body is retired as a whole (`object_map.py:406-434, 520-527`). | Checked |
| D26 | `mask_completion_labels`, `ground_contact_depth_labels` (f1a0a67) | Synthetic points are placed at a median or ground-ray range. They then become map points X_k judged by Eq. 7–9 (`pipeline.py:479-525, 567-590`). | Checked |

## 5. Not implemented

| ID | Item |
|---|---|
| D27 | There are no switches for the Table V ablations (W/o 2D Tracker, W/o Semantic Fusion, W/o Geometric Consistency Update). The Sec. V-E precision/recall/F1 is computed on the final frame only. |
| D28 | The R⁸ tracklet state (scale velocity). |

## Resolution

Fixes are made on `dev`, one item per commit where practical.

Baseline on the synthetic scene at `ac159e9` (`examples/evaluate.py`): mean
detection recall 0.971, mean change recall 0.954, final-map F1 0.941,
12 instance IDs, 2 / 2 identities kept, mIoU without background 0.988,
mAP50 1.0.

| Metric | Before | After D1, D3, D4 |
|---|---|---|
| mean detection recall | 0.971 | 0.971 |
| mean change recall | 0.954 | 0.960 |
| final-map F1 | 0.941 | 0.941 |
| instance IDs created | 12 | 12 |
| identities kept | 2 | 2 |
| mIoU without background | 0.988 | 0.988 |
| mAP50 | 1.0 | 1.0 |

Change recall rises only because of the D4 metric fix: the appeared objects
(bucket, cart, safety sign, the moved box's new place) now have their
pre-appearance absence scored. The synthetic scene has no label flicker and no
depth-ambiguous replacement, so D1 and D3 do not move its numbers. Regression
tests for each item are in `test/test_paper_review_2026_09_25.py`.

- **D1 fixed.**
  - A new association stage, *relabelling* (`association.associate_relabel`), admits a high-confidence detection with any label to a visible, unmatched track. It requires the 2D boxes to overlap (`association_iou_threshold`, plus the motion gate) and at least `relabel_min_overlap` (0.5) of the smaller 3D box to lie inside the other.
  - Eq. 10 then weighs the new label. A flickering label moves the belief over about three observations, and a single wrong label only dents it.
  - A person standing in front of a chair lifts to a box in front of it and still spawns its own instance (C2).
  - The C2 regression test now places the person 1 m in front; before, it sat exactly on the chair's geometry.
- **D3 fixed.**
  - Every 2D match from stages 1 and 2 is validated in 3D (`association.split_depth_inconsistent`). A pair whose detection lifts more than `match_max_gap_m` (1.0 m) from the instance's box is split: the detection re-activates or spawns elsewhere, and the instance gets this frame's evidence as unmatched.
  - A chair replaced by another 4 m behind it in the same image box is now one disappearance and one appearance.
  - Diagnostics count `depth_split_matches` and `relabel_matches`.
- **D4 fixed.**
  - Change recall scores both absence intervals: before an object appears and after it is removed.
  - Frames between two phases of the same object at the same place are scored once, as the earlier phase's post-removal interval.
  - The docstring no longer claims that never-detected objects get no absence credit.
- **D2 kept as a deliberate deviation.**
  - The 3×3 contradiction window and the two-contradiction prune threshold guard against review item C3: single glitched frames deleting points, and erosion at silhouettes.
  - On the synthetic scene the paper-literal settings (`contradiction_window_px: 0`, `prune_min_contradictions: 1`) give identical metrics. That scene has perfect poses and depth, so it cannot show the failure the guards address.
  - Deciding this needs a real recording with ground truth.
- **Test fix for ac159e9.** The dense-launch test compared only top-level YAML keys with declared parameters. It now flattens nested keys such as `platform_mask.enabled`, as ROS does.
- **D21 fixed**, in two parts:
  - **Descriptor.** The colour histogram now bins neutral pixels by intensity (3 soft, log-spaced bins) instead of putting them all at one chromaticity. A pixel is neutral when its channel spread is below 15% of its brightest channel, or below 16 levels, so dark sensor noise stays neutral.

    | Pair | Before | After |
    |---|---|---|
    | black vs white | 0.83 | 0.19 |
    | black vs grey | 0.83 | 0.48 |
    | grey vs white | 1.00 | 0.85 |

    Coloured objects stay shading-invariant: beige at half the light scores 0.94.

    The cost is that a *neutral* object at half the light drops to 0.70. Its relocation is then not claimed, so it gets a new ID, which is the safe direction (Fig. 3). The descriptor grows from 64 to 67 dimensions. Descriptors of different dimension are now treated as not comparable, so older saved maps still re-identify by place; before, they would have been vetoed as dissimilar.
  - **Plausibility.** A relocation claimed on appearance alone must also be plausible: within `reconcile_max_distance_m` (10 m) of where the instance was, and within `reconcile_max_gap_sec` (120 s) of when it was last seen. These are the bounds `reconcile_retired` already applied. Re-identification in the old place is unaffected. A lookalike that turns up far away or much later is a new object.
  - Synthetic-scene metrics are unchanged, and both identities (moved and returned) are still kept.
- **Open:** D5–D20 and D22–D28.
