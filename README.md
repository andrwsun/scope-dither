# Scope VFX Collection

A collection of visual effects plugins for Daydream Scope.

## Plugins

### 🎨 [dither-post-vfx](./dither-post-vfx)
Classic black and white dithering effect (post-processor)
- GPU-accelerated ordered dithering using Bayer matrix
- Adjustable threshold, dither size, spacing, and contrast

## Installation

Each plugin can be installed independently:

```bash
# Install dither effect
cd dither-post-vfx
uv run daydream-scope install -e .
```

Or install from GitHub (when available):
```bash
uv run daydream-scope install https://github.com/andrwsun/scope-vfx/dither-post-vfx
```

## Creating New Plugins

Each subfolder contains a complete Scope plugin with its own:
- `pyproject.toml` - Plugin configuration
- `src/` - Source code
- `README.md` - Plugin-specific documentation

## Structure

```
scope vfx/
├── dither-post-vfx/          # Dithering post-processor
├── test-pre-vfx1/            # Your pre-processor plugins
├── test-post-vfx1/           # More post-processor plugins
└── test-post-vfx2/           # And more...
```
