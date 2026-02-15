"""GPU-accelerated ordered dithering implementation using Bayer matrix."""

import torch


def create_bayer_matrix(size: int, device: torch.device) -> torch.Tensor:
    """
    Create a Bayer matrix of the given size (must be power of 2).

    Args:
        size: Matrix size (2, 4, 8, or 16)
        device: Torch device to create the matrix on

    Returns:
        Normalized Bayer matrix with values in [0, 1]
    """
    if size == 2:
        matrix = torch.tensor([[0, 2], [3, 1]], dtype=torch.float32, device=device)
    elif size == 4:
        # Build 4x4 from 2x2
        m2 = torch.tensor([[0, 2], [3, 1]], dtype=torch.float32, device=device)
        matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
        matrix[0::2, 0::2] = 4 * m2 + 0
        matrix[0::2, 1::2] = 4 * m2 + 2
        matrix[1::2, 0::2] = 4 * m2 + 3
        matrix[1::2, 1::2] = 4 * m2 + 1
    elif size == 8:
        # Build 8x8 from 4x4
        m2 = torch.tensor([[0, 2], [3, 1]], dtype=torch.float32, device=device)
        m4 = torch.zeros((4, 4), dtype=torch.float32, device=device)
        m4[0::2, 0::2] = 4 * m2 + 0
        m4[0::2, 1::2] = 4 * m2 + 2
        m4[1::2, 0::2] = 4 * m2 + 3
        m4[1::2, 1::2] = 4 * m2 + 1

        matrix = torch.zeros((8, 8), dtype=torch.float32, device=device)
        matrix[0::2, 0::2] = 4 * m4 + 0
        matrix[0::2, 1::2] = 4 * m4 + 2
        matrix[1::2, 0::2] = 4 * m4 + 3
        matrix[1::2, 1::2] = 4 * m4 + 1
    elif size == 16:
        # Build 16x16 from 8x8
        m2 = torch.tensor([[0, 2], [3, 1]], dtype=torch.float32, device=device)
        m4 = torch.zeros((4, 4), dtype=torch.float32, device=device)
        m4[0::2, 0::2] = 4 * m2 + 0
        m4[0::2, 1::2] = 4 * m2 + 2
        m4[1::2, 0::2] = 4 * m2 + 3
        m4[1::2, 1::2] = 4 * m2 + 1

        m8 = torch.zeros((8, 8), dtype=torch.float32, device=device)
        m8[0::2, 0::2] = 4 * m4 + 0
        m8[0::2, 1::2] = 4 * m4 + 2
        m8[1::2, 0::2] = 4 * m4 + 3
        m8[1::2, 1::2] = 4 * m4 + 1

        matrix = torch.zeros((16, 16), dtype=torch.float32, device=device)
        matrix[0::2, 0::2] = 4 * m8 + 0
        matrix[0::2, 1::2] = 4 * m8 + 2
        matrix[1::2, 0::2] = 4 * m8 + 3
        matrix[1::2, 1::2] = 4 * m8 + 1
    else:
        # Default to 8x8 if invalid size
        return create_bayer_matrix(8, device)

    # Normalize to [0, 1]
    return matrix / (size * size)


def ordered_dither(
    frames: torch.Tensor,
    threshold: float = 0.5,
    dither_size: int = 8,
    spacing: float = 1.0,
    contrast: float = 1.0,
) -> torch.Tensor:
    """
    Apply ordered dithering (Bayer matrix) to create classic black and white effect.

    Args:
        frames: Input tensor of shape (T, H, W, C) in range [0, 1]
        threshold: Brightness threshold (0.0 - 1.0)
        dither_size: Size of Bayer matrix (2, 4, 8, or 16)
        spacing: Spacing multiplier for the dither pattern
        contrast: Contrast adjustment before dithering

    Returns:
        Dithered frames in range [0, 1]
    """
    T, H, W, C = frames.shape
    device = frames.device

    # Convert to grayscale (weighted average for better perceptual results)
    # Using standard luminance weights: R=0.299, G=0.587, B=0.114
    gray = (
        frames[..., 0] * 0.299 +
        frames[..., 1] * 0.587 +
        frames[..., 2] * 0.114
    )  # Shape: (T, H, W)

    # Apply contrast adjustment
    if contrast != 1.0:
        # Contrast around midpoint (0.5)
        gray = (gray - 0.5) * contrast + 0.5
        gray = gray.clamp(0, 1)

    # Create Bayer matrix
    # Ensure dither_size is valid (round to nearest power of 2)
    valid_sizes = [2, 4, 8, 16]
    dither_size = min(valid_sizes, key=lambda x: abs(x - dither_size))

    bayer = create_bayer_matrix(dither_size, device)

    # Apply spacing by scaling the effective dither size
    effective_size = int(dither_size * spacing)
    if effective_size < 1:
        effective_size = 1

    # Tile the Bayer matrix to cover the entire frame
    # Calculate how many tiles we need
    tiles_h = (H + effective_size - 1) // effective_size
    tiles_w = (W + effective_size - 1) // effective_size

    # Repeat and interpolate the Bayer matrix if spacing != 1.0
    if spacing != 1.0:
        # Use nearest neighbor interpolation to scale the pattern
        bayer_tiled = bayer.repeat(tiles_h, tiles_w)[:H, :W]
    else:
        # Simple tiling
        bayer_tiled = bayer.repeat(tiles_h, tiles_w)[:H, :W]

    # Add batch dimension for broadcasting: (1, H, W)
    bayer_tiled = bayer_tiled.unsqueeze(0)

    # Apply threshold with Bayer matrix
    # The Bayer matrix adds a spatial varying threshold
    threshold_map = threshold + (bayer_tiled - 0.5) * 0.5

    # Compare grayscale values to threshold map
    dithered = (gray > threshold_map).float()

    # Convert back to RGB (all channels the same for B&W)
    result = dithered.unsqueeze(-1).repeat(1, 1, 1, 3)

    return result
