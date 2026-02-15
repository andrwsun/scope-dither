"""Configuration schema for the Noise preprocessor."""

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from scope.core.pipelines.usage import UsageType


class NoiseConfig(BasePipelineConfig):
    """Configuration for the Noise preprocessor."""

    pipeline_id = "noise"
    pipeline_name = "Simplex Noise"
    pipeline_description = (
        "3D Simplex noise generator with animation and fractal controls"
    )

    supports_prompts = False
    usage = [UsageType.PREPROCESSOR]
    modes = {"video": ModeDefaults(default=True)}

    # --- Noise Generation Controls ---

    animation: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Animation phase - evolves noise over time",
        json_schema_extra=ui_field_config(order=1, label="Animation"),
    )

    seed: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="Random seed for noise pattern",
        json_schema_extra=ui_field_config(order=2, label="Seed"),
    )

    period: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Spatial frequency/scale of noise (lower = larger features)",
        json_schema_extra=ui_field_config(order=3, label="Period"),
    )

    harmonics: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of octaves for fractal noise (more = finer detail)",
        json_schema_extra=ui_field_config(order=4, label="Harmonics"),
    )

    amplitude: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Overall strength of noise effect",
        json_schema_extra=ui_field_config(order=5, label="Amplitude"),
    )

    offset: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Baseline offset added to noise values",
        json_schema_extra=ui_field_config(order=6, label="Offset"),
    )

    exponent: float = Field(
        default=1.0,
        ge=0.1,
        le=4.0,
        description="Power curve applied to noise (1.0 = linear, >1 = more contrast)",
        json_schema_extra=ui_field_config(order=7, label="Exponent"),
    )

    # --- Mixing Controls ---

    mix: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend between input video (0.0) and noise (1.0)",
        json_schema_extra=ui_field_config(order=8, label="Mix"),
    )
