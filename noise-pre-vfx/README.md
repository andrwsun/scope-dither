# Scope Noise

3D Simplex noise preprocessor plugin for Daydream Scope, inspired by TouchDesigner's Noise TOP.

## Features

- **GPU-accelerated** 3D simplex noise generation
- **Fractal/Harmonic layering** for rich detail (1-8 octaves)
- **Animation control** for evolving noise patterns
- **Full parameter control** - seed, period, amplitude, offset, exponent
- **Alpha blending** - mix noise with input video

## Installation

```bash
# From the plugin directory
cd noise-pre-vfx
uv run daydream-scope install -e .
```

## Parameters

### Generation Controls

- **Animation** (0.0 - 100.0): Time phase for evolving noise. Increase to animate the pattern over time.
- **Seed** (0 - 1000): Random seed for different noise patterns.
- **Period** (0.1 - 10.0): Spatial frequency/scale. Lower values = larger features.
- **Harmonics** (1 - 8): Number of fractal octaves. More harmonics = finer detail and complexity.

### Shaping Controls

- **Amplitude** (0.0 - 2.0): Overall strength of the noise effect.
- **Offset** (-1.0 - 1.0): Baseline offset added to noise values.
- **Exponent** (0.1 - 4.0): Power curve for contrast. 1.0 = linear, >1 = more contrast.

### Mixing

- **Mix** (0.0 - 1.0): Blend between input video (0.0) and pure noise (1.0).

## Use Cases

- **Pre-processing** - Add noise texture before generation
- **Grain/Texture** - Film grain, texture overlays
- **Displacement** - Use as a displacement map input
- **Animation** - Animate the Animation parameter for evolving textures
- **Layering** - Combine with other preprocessors for complex effects

## TouchDesigner Similarity

This plugin is inspired by TouchDesigner's Noise TOP and includes similar controls:

| TouchDesigner | Scope Noise |
|---------------|-------------|
| Type: Simplex | 3D Simplex (fixed) |
| Period | Period |
| Harmonics | Harmonics |
| Amplitude | Amplitude |
| Offset | Offset |
| Exponent | Exponent |
| Time (animating) | Animation |
| Seed | Seed |

## Requirements

- Python 3.12+
- Daydream Scope
- PyTorch (provided by Scope)

## License

MIT
