"""Dither pipeline implementation."""

from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .effects import ordered_dither
from .schema import DitherConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class DitherPipeline(Pipeline):
    """GPU-accelerated black and white dithering pipeline."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return DitherConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def prepare(self, **kwargs) -> Requirements:
        """We need exactly one input frame per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Apply dithering effect to input video frames."""
        video = kwargs.get("video")
        if video is None:
            raise ValueError("DitherPipeline requires video input")

        # Stack input frames -> (T, H, W, C) and normalise to [0, 1]
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # Apply dithering effect
        frames = ordered_dither(
            frames,
            threshold=kwargs.get("threshold", 0.5),
            dither_size=kwargs.get("dither_size", 8),
            spacing=kwargs.get("spacing", 1.0),
            contrast=kwargs.get("contrast", 1.0),
        )

        return {"video": frames.clamp(0, 1)}
