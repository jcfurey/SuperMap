"""Lifting a detection works on the crop around its pixels (plus the ground-fit
and depth-fill margin), and gives exactly the points the whole image gives."""
import numpy as np
import pytest

from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import Detection2D
from test.test_outdoor_geometry import BAG, FOREGROUND, H, K, W, _bag_detection, _dense_mask, _pose, _render

CONFIGS = {
    "default": {},
    "no_ground_exclusion": {"ground_exclusion": False},
    "ground_removal_largest_layer": dict(FOREGROUND, ground_removal_labels=["bulk bag"],
                                         foreground_depth_largest_labels=["bulk bag"],
                                         foreground_depth_min_image_span=0.3),
    "completion": dict(FOREGROUND, ground_removal_labels=["bulk bag"], mask_completion_labels=["bulk bag"],
                       mask_completion_stride_px=3),
    "ground_contact": {"ground_context_px": 40, "mask_completion_labels": ["bulk bag"],
                       "ground_contact_depth_labels": ["bulk bag"]},
    # No margin: the crop ends at the mask's bottom row, above the image border.
    "ground_contact_no_margin": {"ground_context_px": 0, "ground_removal_labels": ["bulk bag"],
                                 "mask_completion_labels": ["bulk bag"], "ground_contact_depth_labels": ["bulk bag"]},
    "depth_fill": {"depth_fill_radius_px": 3, "ground_context_px": 2},
    "depth_gate_and_cap": {"mask_depth_mad_factor": 3.0, "max_points_per_detection": 25},
}


def _box(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)


def _wide_bleed(hit):
    """The bag's mask bled far down and sideways onto the ground: enough ground
    inside its own box to fit a plane without any context margin."""
    mask = _dense_mask(hit, 0, grow_down=20)
    ys, xs = np.nonzero(mask)
    mask[ys.min():ys.max() + 1, max(xs.min() - 40, 0):xs.max() + 40] = True
    return Detection2D(bbox=_box(mask), label="bulk bag", score=0.9, mask=mask)


def _detections():
    """Masks bled onto the ground, cut off by the image border, empty, and box-only."""
    _, hit = _render([BAG])
    bag = _bag_detection(hit)
    at_bottom = _bag_detection(hit, grow_down=H)  # reaches the last row: no ground contact
    corner = np.zeros((H, W), bool)
    corner[:15, :20] = True
    right = np.zeros((H, W), bool)
    right[50:90, W - 12:] = True
    return {
        "bag": bag,
        "bag_no_bleed": _bag_detection(hit, grow_down=0),
        "at_bottom": at_bottom,
        "wide_bleed": _wide_bleed(hit),
        "corner": Detection2D(bbox=_box(corner), label="bulk bag", score=0.9, mask=corner),
        "right_edge": Detection2D(bbox=_box(right), label="bulk bag", score=0.9, mask=right),
        "empty": Detection2D(bbox=np.array([10.0, 10.0, 20.0, 20.0]), label="bulk bag", score=0.9,
                             mask=np.zeros((H, W), bool)),
        "box_only": Detection2D(bbox=bag.bbox.copy(), label="bulk bag", score=0.9),
        "box_past_the_border": Detection2D(bbox=np.array([-30.0, 40.0, W + 30.0, H + 30.0]), label="bulk bag", score=0.9),
    }


def _scenes():
    return {
        "rings": _render([BAG], ring_step=4),
        "sparse_rings": _render([BAG], ring_step=8),
        "no_returns_on_the_object": _render([BAG], ring_step=16, ring_offset=12),
    }


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_cropped_lifting_equals_whole_image_lifting(config_name):
    whole_image = (0, H, 0, W)
    for scene, (depth, _) in _scenes().items():
        original = depth.copy()
        for name, detection in _detections().items():
            cropped = SemanticMappingPipeline(PipelineConfig(**CONFIGS[config_name]))
            reference = SemanticMappingPipeline(PipelineConfig(**CONFIGS[config_name]))
            points = cropped._detection_points_world(detection, depth, K, _pose())
            expected = reference._detection_points_world(detection, depth, K, _pose(), region=whole_image)
            assert points.shape == expected.shape, (scene, name)
            np.testing.assert_array_equal(points, expected, err_msg=f"{scene} / {name}")
            assert cropped.object_map.stats == reference.object_map.stats, (scene, name)
        np.testing.assert_array_equal(depth, original)  # the frame's depth is never written


def test_crop_comparison_reaches_every_lifting_branch():
    """Guards the equality test above against passing vacuously."""
    depth, hit = _render([BAG], ring_step=16, ring_offset=12)
    contact = SemanticMappingPipeline(PipelineConfig(**CONFIGS["ground_contact"]))
    assert len(contact._detection_points_world(_bag_detection(hit, grow_down=0), depth, K, _pose()))
    assert contact.object_map.stats["ground_contact_completions"] == 1
    no_margin = SemanticMappingPipeline(PipelineConfig(**CONFIGS["ground_contact_no_margin"]))
    detection = _wide_bleed(hit)
    assert np.flatnonzero(detection.mask.any(axis=1))[-1] < H - 1
    assert len(no_margin._detection_points_world(detection, depth, K, _pose()))
    assert no_margin.object_map.stats["ground_contact_completions"] == 1

    depth, hit = _render([BAG], ring_step=8)
    completion = SemanticMappingPipeline(PipelineConfig(**CONFIGS["completion"]))
    assert len(completion._detection_points_world(_bag_detection(hit), depth, K, _pose()))
    assert completion.object_map.stats["mask_completions"] == 1

    region = SemanticMappingPipeline(PipelineConfig(**CONFIGS["depth_fill"]))._detection_region(
        _bag_detection(hit), (H, W))
    mask_rows = np.flatnonzero(_bag_detection(hit).mask.any(axis=1))
    assert region[0] == max(mask_rows[0] - 3, 0) and region[1] == min(mask_rows[-1] + 4, H)
    assert region[3] - region[2] < W  # an actual crop, not the whole image


def test_ground_contact_ray_on_a_crop_uses_the_full_image():
    pipeline = SemanticMappingPipeline(PipelineConfig())
    plane = np.zeros(3)  # level ground at z = 0
    mask = np.zeros((H, W), bool)
    mask[70:85, 60:95] = True
    full = pipeline._ground_contact_range(mask, K, _pose(), plane)
    assert full is not None
    # The crop ends at the mask's bottom row, which is not the image's.
    assert pipeline._ground_contact_range(mask[70:85, 60:95], K, _pose(), plane, (60, 70), H) == full
    mask[85:, 60:95] = True  # now the image border cuts the mask off
    assert pipeline._ground_contact_range(mask, K, _pose(), plane) is None
    assert pipeline._ground_contact_range(mask[70:, 60:95], K, _pose(), plane, (60, 70), H) is None


def test_box_outside_the_image_has_no_pixels():
    depth, _ = _render([BAG])
    pipeline = SemanticMappingPipeline(PipelineConfig())
    for bbox in ([-50.0, 10.0, -5.0, 40.0], [W + 1.0, 10.0, W + 20.0, 40.0], [10.0, -40.0, 50.0, -1.0]):
        detection = Detection2D(bbox=np.array(bbox), label="bulk bag", score=0.9)
        assert pipeline._detection_region(detection, (H, W))[1] == 0
        assert len(pipeline._detection_points_world(detection, depth, K, _pose())) == 0
