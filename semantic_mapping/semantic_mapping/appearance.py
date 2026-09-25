"""Appearance descriptors for instance re-identification.

The paper's identities are meant to survive not only occlusion but
relocation: the same physical object carries the same instance ID when it
turns up somewhere else (Sec. IV-B, Sec. V-C). Geometry cannot decide that
case, so each detection gets an appearance embedding and each instance keeps
a running mean of the embeddings it was built from. The default embedder is
a colour histogram (no model, no GPU, deterministic), which separates the
synthetic scene's objects cleanly and is a reasonable stand-in indoors; a
CLIP image embedding is available for deployments that can afford it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from semantic_mapping.types import Detection2D


class Embedder(ABC):
    """Maps detections on one RGB frame to unit-norm float32 descriptors (or None)."""

    name = "embedder"

    @abstractmethod
    def embed(self, rgb: np.ndarray, detections: list[Detection2D]) -> list[np.ndarray | None]:
        raise NotImplementedError


def _detection_pixels(rgb: np.ndarray, detection: Detection2D, max_pixels: int | None = None) -> np.ndarray:
    """(N, 3) pixels inside the detection's mask, or its box when there is no mask."""
    if detection.mask is not None and detection.mask.shape == rgb.shape[:2]:
        if max_pixels is None:
            return rgb[detection.mask]
        indices = np.flatnonzero(detection.mask)
        stride = max(1, int(np.ceil(indices.size / max_pixels)))
        indices = indices[::stride]
        return rgb[indices // rgb.shape[1], indices % rgb.shape[1]]
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    x1, y1 = int(max(np.floor(x1), 0)), int(max(np.floor(y1), 0))
    x2, y2 = int(min(np.ceil(x2), w)), int(min(np.ceil(y2), h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 3), dtype=rgb.dtype)
    return rgb[y1:y2, x1:x2].reshape(-1, rgb.shape[2] if rgb.ndim == 3 else 1)[:, :3]


class ColorHistogramEmbedder(Embedder):
    """Hellinger-normalized colour histogram of the detection's pixels.

    The default ``space`` is chromaticity, (r, g) / (r + g + b) binned
    ``bins`` x ``bins`` (8 -> 64 dimensions): dividing out intensity makes
    the descriptor invariant to uniform shading, so the same object seen
    closer, farther, or under dimmer light keeps its signature, which is what
    re-identification across time needs. ``space="rgb"`` bins raw RGB
    (``bins`` cubed dimensions) and is brightness-sensitive. Taking the square
    root of the normalized histogram makes the cosine similarity between two
    descriptors the Bhattacharyya coefficient of their colour distributions.

    Chromaticity is undefined for black and meaningless for near-black
    pixels (sensor noise decides their hue), and mapping them to (0, 0) put
    them in the bin of saturated blue. Pixels whose mean intensity is below
    ``min_intensity`` (0-255 scale) are therefore left out of the chroma
    histogram; a detection with fewer than ``min_pixels`` lit pixels gets no
    descriptor at all rather than a misleading one.

    Neutral pixels (channel spread below ``achromatic_saturation`` times the
    brightest channel, or below ``achromatic_spread`` for sensor noise in dark
    pixels) all share one
    chromaticity, so black, grey and white objects had identical descriptors
    and re-identification could hand a new white chair a retired grey chair's
    ID (doc/paper-review-2026-09-25.md, D21). They go instead into
    ``achromatic_bins`` intensity bins, spaced logarithmically from
    ``min_intensity`` to 255 with linear (soft) assignment between adjacent
    bins, so a halving of the light moves a grey object's mass less than one
    bin while black and white stay apart. ``achromatic_bins=0`` restores the
    pure chromaticity descriptor.
    """

    name = "color_histogram"

    def __init__(self, bins: int = 8, min_pixels: int = 16, space: str = "chromaticity",
                 max_pixels: int = 4096, min_intensity: float = 20.0, achromatic_bins: int = 3,
                 achromatic_saturation: float = 0.15, achromatic_spread: float = 16.0) -> None:
        if space not in ("chromaticity", "rgb"):
            raise ValueError(f"unknown colour space {space!r} (chromaticity | rgb)")
        self.bins = int(bins)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        """A 64-bin histogram is estimated well from a few thousand pixels;
        larger regions are sampled with a fixed stride (doc/audit-2026-09.md, P3)."""
        self.space = space
        self.min_intensity = float(min_intensity)
        self.achromatic_bins = int(achromatic_bins) if space == "chromaticity" else 0
        self.achromatic_saturation = float(achromatic_saturation)
        self.achromatic_spread = float(achromatic_spread)
        if self.achromatic_bins < 0 or not 0 < self.min_intensity < 255:
            raise ValueError("achromatic_bins must be >= 0 and min_intensity in (0, 255)")
        self.dim = self.bins ** (2 if space == "chromaticity" else 3) + self.achromatic_bins

    def _histogram(self, pixels: np.ndarray) -> np.ndarray | None:
        """Raw bin counts, or None when too few pixels carry a colour."""
        pixels = pixels.astype(np.float64)
        if self.space == "rgb":
            edges = np.linspace(0.0, 256.0, self.bins + 1)
            hist, _ = np.histogramdd(pixels, bins=(edges, edges, edges))
            return hist.ravel()
        total = pixels.sum(axis=1)
        lit = (total > 0) & (total >= 3.0 * self.min_intensity)
        if int(lit.sum()) < self.min_pixels:
            return None
        pixels, total = pixels[lit], total[lit]
        neutral = np.zeros(len(pixels), dtype=bool)
        if self.achromatic_bins:
            neutral = np.ptp(pixels, axis=1) < np.maximum(self.achromatic_spread,
                                                          self.achromatic_saturation * pixels.max(axis=1))
        chroma = pixels[~neutral, :2] / total[~neutral, None]
        edges = np.linspace(0.0, 1.0 + 1e-9, self.bins + 1)
        hist, _, _ = np.histogram2d(chroma[:, 0], chroma[:, 1], bins=(edges, edges))
        if not self.achromatic_bins:
            return hist.ravel()
        return np.concatenate([hist.ravel(), self._intensity_histogram(total[neutral] / 3.0)])

    def _intensity_histogram(self, intensity: np.ndarray) -> np.ndarray:
        """Soft histogram of neutral pixels over log-spaced intensity bin centres."""
        n = self.achromatic_bins
        hist = np.zeros(n)
        if n == 1 or not len(intensity):
            hist[0] = len(intensity)
            return hist
        position = (np.log(np.clip(intensity, self.min_intensity, 255.0)) - np.log(self.min_intensity)) \
            / (np.log(255.0) - np.log(self.min_intensity)) * (n - 1)
        low = np.minimum(np.floor(position).astype(np.int64), n - 2)
        upper = position - low
        np.add.at(hist, low, 1.0 - upper)
        np.add.at(hist, low + 1, upper)
        return hist

    def embed(self, rgb: np.ndarray, detections: list[Detection2D]) -> list[np.ndarray | None]:
        out: list[np.ndarray | None] = []
        for detection in detections:
            # Subsample mask locations before copying RGB, rather than copying
            # every foreground pixel and immediately discarding almost all.
            sample_limit = self.max_pixels if self.max_pixels >= 2 * self.min_pixels else None
            pixels = _detection_pixels(rgb, detection, sample_limit)
            if pixels.shape[0] < self.min_pixels:
                out.append(None)
                continue
            if pixels.shape[0] > self.max_pixels:
                pixels = pixels[:: int(np.ceil(pixels.shape[0] / self.max_pixels))]
            hist = self._histogram(pixels)
            if hist is None:
                out.append(None)
                continue
            hist = np.sqrt(hist / max(float(hist.sum()), 1.0))
            norm = np.linalg.norm(hist)
            out.append((hist / norm).astype(np.float32) if norm > 0 else None)
        return out


class CLIPEmbedder(Embedder):
    """CLIP image embeddings of the detection crops (``open_clip``; optional dependency)."""

    name = "clip"

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda",
                 crop_margin: float = 0.1) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError("CLIPEmbedder requires 'open_clip_torch' and 'torch' (pip install open_clip_torch).") from exc
        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self.crop_margin = crop_margin

    def embed(self, rgb: np.ndarray, detections: list[Detection2D]) -> list[np.ndarray | None]:
        from PIL import Image

        h, w = rgb.shape[:2]
        crops, index = [], []
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            mx, my = self.crop_margin * (x2 - x1), self.crop_margin * (y2 - y1)
            x1, y1 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
            x2, y2 = int(min(x2 + mx, w)), int(min(y2 + my, h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crops.append(self.preprocess(Image.fromarray(np.ascontiguousarray(rgb[y1:y2, x1:x2]))))
            index.append(i)
        out: list[np.ndarray | None] = [None] * len(detections)
        if not crops:
            return out
        with self.torch.no_grad():
            features = self.model.encode_image(self.torch.stack(crops).to(self.device)).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        for i, feature in zip(index, features.cpu().numpy()):
            out[i] = feature.astype(np.float32)
        return out


def build_embedder(name: str | None, **kwargs) -> Embedder | None:
    key = (name or "none").lower()
    if key in ("none", "", "off"):
        return None
    if key in ("color_histogram", "histogram", "color"):
        return ColorHistogramEmbedder(
            **{k: v for k, v in kwargs.items()
               if k in ("bins", "min_pixels", "space", "max_pixels", "min_intensity", "achromatic_bins",
                        "achromatic_saturation", "achromatic_spread")})
    if key == "clip":
        return CLIPEmbedder(**{k: v for k, v in kwargs.items() if k in ("model_name", "pretrained", "device", "crop_margin")})
    raise ValueError(f"Unknown appearance embedder: {name!r}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else 0.0


def update_running_embedding(
    current: np.ndarray | None, count: int, new: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Unit-norm running mean of the descriptors an instance was observed with."""
    new = np.asarray(new, dtype=np.float32)
    if current is None or count <= 0 or current.shape != new.shape:
        return new / max(float(np.linalg.norm(new)), 1e-8), 1
    mean = (current * count + new) / (count + 1)
    return (mean / max(float(np.linalg.norm(mean)), 1e-8)).astype(np.float32), count + 1
