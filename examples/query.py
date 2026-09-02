#!/usr/bin/env python3
"""Ground a natural-language instruction in the 4D scene graph (Sec. IV-D), offline.

Builds the map by running the pipeline over the sequence, then serializes the
scene graph, queries a VLM, and resolves the answered instance IDs to 3D
waypoints:

    python examples/query.py "go to the chair next to the table"
    python examples/query.py --client openai_compatible --model gpt-4o --base_url https://api.openai.com/v1 \\
        "return to where the trash can used to be"
    python examples/query.py --client anthropic --model claude-opus-5 "..."

The default ``keyword`` client is a deterministic stand-in with no network
access (it matches labels mentioned in the instruction); use a real backend
for relational or temporal instructions. API keys are read from OPENAI_API_KEY
/ ANTHROPIC_API_KEY (override with --api_key_env).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_mapping.datasets import SequenceDataset, load_prompts, load_yaml_params, run_sequence  # noqa: E402
from semantic_mapping.detectors import build_detector  # noqa: E402
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline  # noqa: E402
from semantic_mapping.vln.clients import build_vlm_client  # noqa: E402
from semantic_mapping.vln.grounding import Grounder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instructions", nargs="+", help="One or more natural-language instructions.")
    parser.add_argument("--client", choices=["keyword", "openai_compatible", "anthropic"], default="keyword")
    parser.add_argument("--model", default=None, help="Model name for the chosen backend.")
    parser.add_argument("--base_url", default=None, help="Endpoint base URL for the chosen backend.")
    parser.add_argument("--api_key_env", default=None, help="Environment variable holding the API key.")
    parser.add_argument("--data_dir", default="data/example_scene")
    parser.add_argument("--config", default="config/semantic_mapping.yaml")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    parser.add_argument("--show_prompt", action="store_true", help="Print the serialized scene-graph prompt.")
    args = parser.parse_args()

    dataset = SequenceDataset(args.data_dir)
    params = load_yaml_params(args.config)
    detector = build_detector("offline", detections_dir=dataset.detections_dir)
    pipeline = SemanticMappingPipeline(PipelineConfig.from_dict(params))
    result = None
    for _frame, _dets, result in run_sequence(dataset, pipeline, detector, load_prompts(args.prompts)):
        pass
    if result is None:
        raise SystemExit("empty sequence")

    client_kwargs = {k: v for k, v in (("model", args.model), ("base_url", args.base_url),
                                       ("api_key_env", args.api_key_env)) if v is not None}
    grounder = Grounder(build_vlm_client(args.client, **client_kwargs), coordinate_frame="map")

    for instruction in args.instructions:
        grounding = grounder.ground(instruction, result.objects, result.scene_graph)
        if args.show_prompt:
            print(grounding.prompt)
        print(f"\nInstruction: {instruction}")
        print(f"Response   : {grounding.response.strip() or '(none)'}")
        if grounding.error:
            print(f"Error      : {grounding.error}")
        for instance_id, waypoint in zip(grounding.target_ids, grounding.waypoints):
            print(f"Waypoint   : instance {instance_id} -> {json.dumps([round(float(v), 2) for v in waypoint])}")


if __name__ == "__main__":
    main()
