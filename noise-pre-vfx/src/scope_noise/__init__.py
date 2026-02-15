"""Scope Noise - 3D Simplex noise preprocessor."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """Register the Noise pipeline with Scope."""
    from .pipeline import NoisePipeline

    register(NoisePipeline)
