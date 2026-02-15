"""Scope Dither - Classic black and white dithering effect."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """Register the Dither pipeline with Scope."""
    from .pipeline import DitherPipeline

    register(DitherPipeline)
