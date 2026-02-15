# Scope Dither

Classic black and white dithering effect plugin for Daydream Scope.

## Features

- **GPU-accelerated** ordered dithering using Bayer matrix
- **Adjustable threshold** for brightness control
- **Configurable dither size** (2x2, 4x4, 8x8, 16x16)
- **Spacing control** for pattern density
- **Contrast adjustment** for punchier results

## Installation

```bash
# From GitHub
uv run daydream-scope install https://github.com/YOUR_USERNAME/scope-dither

# Local development (editable mode)
uv run daydream-scope install -e .
```

## Parameters

- **Threshold** (0.0 - 1.0): Brightness cutoff between black and white
- **Dither Size** (2 - 16): Size of the dithering pattern matrix
- **Spacing** (0.5 - 4.0): Pattern spacing multiplier
- **Contrast** (0.5 - 2.0): Contrast adjustment before dithering

## Requirements

- Python 3.12+
- Daydream Scope
- PyTorch (provided by Scope)

## License

MIT
