# Scope VFX Collection

A collection of visual effects plugins for Daydream Scope.

## Plugins

### 🔊 [noise-pre-vfx](./noise-pre-vfx)
3D Simplex noise generator (pre-processor)
- GPU-accelerated fractal noise with harmonics
- Animation, seed, period, amplitude, offset, and exponent controls
- Mix control for blending with input video
- Inspired by TouchDesigner's Noise TOP

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
├── noise-pre-vfx/            # 3D Simplex noise pre-processor
├── dither-post-vfx/          # Dithering post-processor
└── [your plugins here]/      # Add more plugins...
```
