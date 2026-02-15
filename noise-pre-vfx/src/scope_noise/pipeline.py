"""Noise preprocessor pipeline implementation."""

from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .effects import simplex_noise
from .schema import NoiseConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class NoisePipeline(Pipeline):
    """GPU-accelerated 3D simplex noise preprocessor."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return NoiseConfig

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
        """Apply noise effect to input video frames."""
        video = kwargs.get("video")
        if video is None:
            raise ValueError("NoisePipeline requires video input")

        # Stack input frames -> (T, H, W, C) and normalise to [0, 1]
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # Apply noise effect
        frames = simplex_noise(
            frames,
            animation=kwargs.get("animation", 0.0),
            seed=kwargs.get("seed", 0),
            period=kwargs.get("period", 1.0),
            harmonics=kwargs.get("harmonics", 1),
            amplitude=kwargs.get("amplitude", 1.0),
            offset=kwargs.get("offset", 0.0),
            exponent=kwargs.get("exponent", 1.0),
            mix=kwargs.get("mix", 1.0),
        )

        return {"video": frames.clamp(0, 1)}
