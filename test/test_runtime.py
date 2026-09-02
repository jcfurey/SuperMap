import numpy as np
import pytest

from semantic_mapping import runtime
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from test.helpers import make_object
from test.test_pipeline import _observation

STAGES = ("predict", "backproject", "associate", "map_update", "scene_graph", "total")


def test_process_frame_reports_stage_timings_that_add_up():
    pipeline = SemanticMappingPipeline(PipelineConfig(min_hits_to_confirm=2))
    result = None
    for i in range(3):
        result = pipeline.process_frame(_observation(stamp=i * 0.1, distance=2.0, with_detection=True))
    assert tuple(result.timings) == STAGES
    assert all(v >= 0.0 for v in result.timings.values())
    parts = sum(result.timings[s] for s in STAGES if s != "total")
    assert result.timings["total"] == pytest.approx(parts, rel=1e-6, abs=1e-9)


def test_runtime_stats_summary_math():
    stats = runtime.RuntimeStats()
    for seconds in (0.010, 0.020, 0.030, 0.040):
        stats.add("stage", seconds)
    stats.add_timings({"other": 0.5})
    summary = stats.summary()
    assert summary["stage"]["count"] == 4
    assert summary["stage"]["mean_ms"] == pytest.approx(25.0)
    assert summary["stage"]["p50_ms"] == pytest.approx(25.0)
    assert summary["stage"]["max_ms"] == pytest.approx(40.0)
    assert summary["stage"]["hz"] == pytest.approx(40.0)
    assert summary["other"]["hz"] == pytest.approx(2.0)
    assert stats.stages() == ["stage", "other"]


def test_map_memory_and_peak_rss():
    obj = make_object(1, "chair", [0, 0, 0, 1, 1, 1])
    obj.points_world = np.zeros((100, 3))
    memory = runtime.map_memory([obj, make_object(2, "table", [0, 0, 0, 1, 1, 1])])
    assert memory == {"instances": 2, "points": 100, "bytes": 100 * runtime.BYTES_PER_MAP_POINT,
                      "mb": 100 * runtime.BYTES_PER_MAP_POINT / 2 ** 20}
    assert runtime.peak_rss_mb() > 1.0


def test_format_runtime_summary_lists_every_stage():
    stats = runtime.RuntimeStats()
    stats.add("detector", 0.2)
    stats.add("total", 0.05)
    text = runtime.format_runtime_summary(stats.summary(), {"instances": 1, "points": 5, "mb": 0.0, "peak_rss_mb": 50.0},
                                          notes="note")
    assert "detector" in text and "total" in text and "peak RSS" in text and text.endswith("note")
