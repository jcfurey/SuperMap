import numpy as np
import pytest

from semantic_mapping import appearance
from semantic_mapping.types import Detection2D


def _frame_with_patches():
    rgb = np.full((40, 60, 3), 240, dtype=np.uint8)
    rgb[5:20, 5:20] = (200, 40, 40)      # red
    rgb[5:20, 30:45] = (120, 24, 24)     # the same red, seen darker (uniform shading)
    rgb[25:38, 5:20] = (40, 40, 200)     # blue
    return rgb


def test_color_histogram_embeddings_are_unit_norm_and_discriminate_colours():
    rgb = _frame_with_patches()
    dets = [Detection2D(bbox=np.array([5.0, 5.0, 20.0, 20.0]), label="box", score=0.9),
            Detection2D(bbox=np.array([30.0, 5.0, 45.0, 20.0]), label="box", score=0.9),
            Detection2D(bbox=np.array([5.0, 25.0, 20.0, 38.0]), label="box", score=0.9)]
    embedder = appearance.ColorHistogramEmbedder(bins=8)
    red_a, red_dark, blue = embedder.embed(rgb, dets)
    assert red_a.shape == (64,) and red_a.dtype == np.float32
    assert np.isclose(np.linalg.norm(red_a), 1.0, atol=1e-5)
    assert appearance.cosine_similarity(red_a, red_dark) > 0.99   # shading-invariant
    assert appearance.cosine_similarity(red_a, blue) < 0.3

    raw = appearance.ColorHistogramEmbedder(bins=8, space="rgb")
    raw_a, raw_dark, _ = raw.embed(rgb, dets)
    assert raw_a.shape == (512,) and appearance.cosine_similarity(raw_a, raw_dark) < 0.5  # brightness-sensitive


def test_mask_pixels_take_precedence_over_the_box_and_tiny_regions_give_none():
    rgb = _frame_with_patches()
    mask = np.zeros((40, 60), dtype=bool)
    mask[25:38, 5:20] = True  # the blue patch, while the box covers the red one
    masked = Detection2D(bbox=np.array([5.0, 5.0, 20.0, 20.0]), label="box", score=0.9, mask=mask)
    boxed = Detection2D(bbox=np.array([5.0, 25.0, 20.0, 38.0]), label="box", score=0.9)
    tiny = Detection2D(bbox=np.array([0.0, 0.0, 2.0, 2.0]), label="box", score=0.9)
    embedder = appearance.ColorHistogramEmbedder(bins=8, min_pixels=16)
    a, b, none = embedder.embed(rgb, [masked, boxed, tiny])
    assert appearance.cosine_similarity(a, b) > 0.99
    assert none is None


def test_running_embedding_stays_unit_norm_and_tracks_the_mean():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mean, count = appearance.update_running_embedding(None, 0, a)
    assert count == 1 and np.allclose(mean, a)
    mean, count = appearance.update_running_embedding(mean, count, b)
    assert count == 2 and np.isclose(np.linalg.norm(mean), 1.0) and np.allclose(mean, [2 ** -0.5, 2 ** -0.5, 0.0])
    mean, count = appearance.update_running_embedding(mean, count, np.zeros(5, dtype=np.float32) + 1)  # shape change: restart
    assert count == 1 and mean.shape == (5,)


def test_build_embedder_names():
    assert appearance.build_embedder("none") is None
    assert isinstance(appearance.build_embedder("color_histogram", bins=4), appearance.ColorHistogramEmbedder)
    assert appearance.build_embedder("color_histogram", bins=4).dim == 16
    assert appearance.build_embedder("color_histogram", bins=4, space="rgb").dim == 64
    with pytest.raises(ValueError):
        appearance.ColorHistogramEmbedder(space="lab")
    with pytest.raises(ValueError):
        appearance.build_embedder("hog")
    assert appearance.cosine_similarity(np.ones(3), np.ones(4)) == 0.0
