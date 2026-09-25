#!/usr/bin/env python3
"""Segment full XYZ clouds offline, with optional manual partial-camera masks.

Usage: python examples/dense_cloud.py cloud.npz --out_dir results/dense --input-mode snapshot
Multiple files may be supplied; --input-mode scan accumulates registered scans.
NPZ files contain xyz and optional stamp/T_world_from_cloud. NPY files contain Nx3 XYZ.
No range cropping, model inference, camera requirement, or automatic downloads.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_mapping.dense_cloud import DenseCloudConfig, DenseCloudPipeline  # noqa: E402
from semantic_mapping.dense_cloud_io import camera_labels_from_dict, dumps_json, save_result, sha256_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clouds", type=Path, nargs="+")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-mode", choices=["snapshot", "scan"])
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--xyz-key", default="xyz")
    parser.add_argument("--annotations", type=Path, help="JSON list: each manual camera observation includes cloud filename")
    args = parser.parse_args()
    config = {}
    if args.config:
        payload = yaml.safe_load(args.config.read_text())
        for key in ("/**", "dense_cloud_mapping"):
            if isinstance(payload.get(key), dict) and "ros__parameters" in payload[key]:
                payload = payload[key]["ros__parameters"]
                break
        config = payload
        config = {key: value for key, value in config.items() if key in {item.name for item in fields(DenseCloudConfig)}}
    if args.input_mode:
        config["input_mode"] = args.input_mode
    if args.voxel_size is not None:
        config["voxel_size"] = args.voxel_size
    pipeline = DenseCloudPipeline(DenseCloudConfig(**config))
    annotations = json.loads(args.annotations.read_text()) if args.annotations else []
    if not isinstance(annotations, list):
        raise ValueError("annotation file must contain a list")
    if annotations and len({path.name for path in args.clouds}) != len(args.clouds):
        raise ValueError("annotation matching requires unique input filenames")
    names = {path.name for path in args.clouds}
    for annotation in annotations:
        if annotation.get("cloud") not in names:
            raise ValueError("annotation references an input cloud that was not supplied")
        camera_labels_from_dict(annotation).validate()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for index, path in enumerate(args.clouds):
        if path.suffix == ".npy":
            xyz = np.load(path, allow_pickle=False)
            stamp, transform = float(index), np.eye(4)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                xyz = data[args.xyz_key]
                stamp = float(data["stamp"]) if "stamp" in data else float(index)
                transform = data["T_world_from_cloud"] if "T_world_from_cloud" in data else np.eye(4)
        else:
            raise ValueError(f"unsupported cloud file: {path}")
        if xyz.ndim == 3 and xyz.shape[-1] == 3:
            xyz = xyz.reshape(-1, 3)
        started = time.perf_counter()
        result = pipeline.update(xyz, stamp, transform)
        for annotation in annotations:
            if annotation["cloud"] == path.name:
                result = pipeline.annotate(camera_labels_from_dict(annotation))
        seconds = time.perf_counter()-started
        output = args.out_dir/f"{index:06d}_{path.stem}.npz"
        save_result(result, output)
        record = {"input": str(path.resolve()), "source_sha256": sha256_file(path),
                  "output": output.name, "processing_seconds": seconds, **result.stats}
        records.append(record)
        print(dumps_json(record), flush=True)
    (args.out_dir/"run.json").write_text(dumps_json({
        "config": vars(pipeline.config), "models": [], "frames": records,
        "limitations": ["Regions are geometric surfaces, not recognized object instances.",
                        "Scan accumulation does not remove moved objects or integrate loop-closure corrections.",
                        "Point labels inherit voxel evidence; full input order and count are preserved."]}, indent=2)+"\n")


if __name__ == "__main__":
    main()
