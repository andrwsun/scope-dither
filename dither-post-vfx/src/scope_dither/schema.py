"""Configuration schema for the Dither pipeline."""

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


class DitherConfig(BasePipelineConfig):
    """Configuration for the Dither pipeline."""

    pipeline_id = "dither"
    pipeline_name = "Dither Effect"
    pipeline_description = (
        "Classic black and white dithering effect with GPU acceleration"
    )

    supports_prompts = False

    modes = {"video": ModeDefaults(default=True)}

    # --- Dithering Controls ---

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Brightness threshold for dithering (0 = black, 1 = white)",
        json_schema_extra=ui_field_config(order=1, label="Threshold"),
    )

    dither_size: int = Field(
        default=8,
        ge=2,
        le=16,
        description="Size of the dither pattern matrix (2, 4, 8, or 16)",
        json_schema_extra=ui_field_config(order=2, label="Dither Size"),
    )

    spacing: float = Field(
        default=1.0,
        ge=0.5,
        le=4.0,
        description="Spacing multiplier for dither pattern (1.0 = normal, higher = more spaced out)",
        json_schema_extra=ui_field_config(order=3, label="Spacing"),
    )

    contrast: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Contrast adjustment before dithering (1.0 = normal)",
        json_schema_extra=ui_field_config(order=4, label="Contrast"),
    )
