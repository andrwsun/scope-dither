# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **collection of Daydream Scope plugins** organized as separate subprojects. Each plugin is a standalone visual effect or processor for Daydream Scope's real-time video AI platform.

**Key architectural principle:** Each plugin lives in its own subdirectory with a complete, independent structure. The root directory serves only as an organizational wrapper.

## Plugin Development Commands

### Installing a Plugin Locally
```bash
# From within a plugin directory (e.g., dither-post-vfx/)
uv run daydream-scope install -e .
```

### Testing Changes
After modifying plugin code, reinstall it:
```bash
cd <plugin-directory>
uv run daydream-scope install -e .
```

Then test in Daydream Scope's UI (typically at `localhost:8000`).

## Plugin Architecture

Every Scope plugin follows a **three-component pattern**:

### 1. Schema (`schema.py`)
- Inherits from `BasePipelineConfig`
- Defines pipeline metadata (`pipeline_id`, `pipeline_name`, `pipeline_description`)
- Declares `modes` (typically `{"video": ModeDefaults(default=True)}` for post/pre-processors)
- Uses Pydantic `Field` with `ui_field_config()` to create UI controls
- `order` parameter controls UI display order
- `ge`/`le` constraints create sliders; `description` provides tooltips

### 2. Pipeline (`pipeline.py`)
- Inherits from `Pipeline` interface
- Implements `get_config_class()` to link the schema
- Implements `prepare()` to declare input requirements:
  - Post/pre-processors: `return Requirements(input_size=N)` where N is frames needed
  - Generators: return `None` or omit `prepare()`
- Implements `__call__(**kwargs)` to process/generate video:
  - Input: `kwargs.get("video")` - list of frame tensors `(1, H, W, C)` in `[0, 255]`
  - Output: `{"video": frames}` - tensor in `[0, 1]` range
- Handles device management (GPU/CPU)

### 3. Effects (`effects/`)
- Pure functions that apply transformations to PyTorch tensors
- Operate on shape `(T, H, W, C)` where T=time, H=height, W=width, C=channels
- GPU-accelerated using PyTorch operations (avoid CPU-bound loops)
- Imported and called from pipeline's `__call__()`

### Registration (`__init__.py`)
```python
from scope.core.plugins.hookspecs import hookimpl

@hookimpl
def register_pipelines(register):
    from .pipeline import YourPipeline
    register(YourPipeline)
```

## Pipeline Types

Set pipeline type via the `usage` field in schema and `modes`:

**Post-Processor** (processes existing video):
```python
usage = [UsageType.POSTPROCESSOR]  # Optional, default behavior
modes = {"video": ModeDefaults(default=True)}
```

**Pre-Processor** (modifies input before main pipeline):
```python
usage = [UsageType.PREPROCESSOR]
modes = {"video": ModeDefaults(default=True)}
```

**Generator** (creates video from scratch):
```python
modes = {"text": ModeDefaults(default=True)}  # No video input
supports_prompts = True  # If using text prompts
```

## Creating a New Plugin

1. Create a new subdirectory: `mkdir my-new-effect`
2. Set up structure:
   ```
   my-new-effect/
   ├── pyproject.toml          # Entry point: [project.entry-points."scope"]
   ├── README.md
   └── src/
       └── scope_my_effect/
           ├── __init__.py     # hookimpl registration
           ├── schema.py       # Config + UI controls
           ├── pipeline.py     # Orchestration logic
           └── effects/
               ├── __init__.py
               └── effect.py   # PyTorch implementations
   ```
3. In `pyproject.toml`, the entry point name must match the package path:
   ```toml
   [project.entry-points."scope"]
   scope_my_effect = "scope_my_effect"
   ```

## Important Patterns

- **Tensor shapes**: Input frames come as list of `(1, H, W, C)`, stack to `(T, H, W, C)` for processing
- **Normalization**: Input is `[0, 255]` uint8, convert to `[0, 1]` float32 for effects, return as `[0, 1]`
- **Device handling**: Always check `torch.cuda.is_available()`, support both GPU and CPU
- **Parameter passing**: Runtime parameters come through `kwargs` in `__call__()`, use `.get()` with defaults
- **Dependencies**: PyTorch and Pydantic are provided by Scope; don't declare them in `pyproject.toml`

## Repository Structure

```
scope vfx/                    # Root - organizational only
├── CLAUDE.md                 # This file
├── README.md                 # Collection overview
├── dither-post-vfx/          # Complete plugin #1
│   ├── pyproject.toml
│   └── src/scope_dither/
├── future-plugin-1/          # Complete plugin #2
└── future-plugin-2/          # Complete plugin #3
```

Each plugin is independent and can be installed/distributed separately.
