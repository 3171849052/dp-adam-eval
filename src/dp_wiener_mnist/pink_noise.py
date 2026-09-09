"""Original complex FFT pink-noise generator (no extra dependencies)."""

from typing import Tuple
import torch
from torch import Tensor


def generate_pink_noise(
    batch_size: int,
    input_shape: Tuple[int, ...],
    device: torch.device,
    alpha: float = 1.0,
) -> Tensor:
    if len(input_shape) == 1:
        # Flat features (e.g. TF-IDF): use 1D pink noise via FFT
        n_features = input_shape[0]
        white_noise = torch.randn(
            batch_size, n_features, dtype=torch.cfloat, device=device
        )
        freqs = torch.fft.fftfreq(n_features, device=device)
        f = freqs.abs()
        f[0] = 1.0
        scale = 1.0 / (f**alpha)
        scale[0] = 0.0
        pink_freq = white_noise * scale.unsqueeze(0)
        pink_noise = torch.fft.ifft(pink_freq).real
        std = pink_noise.std(dim=1, keepdim=True)
        pink_noise = pink_noise / (std + 1e-8) * 0.5
        return pink_noise

    if len(input_shape) == 3:
        channels, height, width = input_shape
    else:
        channels, height, width = 3, input_shape[0], input_shape[0]

    white_noise = torch.randn(
        batch_size, channels, height, width, dtype=torch.cfloat, device=device
    )

    freqs = torch.fft.fftfreq(height, device=device)
    fx, fy = torch.meshgrid(freqs, freqs, indexing="ij")
    f = torch.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0

    scale = 1.0 / (f**alpha)
    scale[0, 0] = 0.0
    scale = scale.view(1, 1, height, width)

    pink_freq = white_noise * scale
    pink_noise = torch.fft.ifft2(pink_freq).real

    std = pink_noise.view(batch_size, -1).std(dim=1, keepdim=True)
    pink_noise = pink_noise / (std.view(batch_size, 1, 1, 1) + 1e-8) * 0.5

    return pink_noise
