"""Small dependency-free loss helpers used by the ablation training loop."""

import torch
import torch.nn.functional as F


def _gaussian_kernel_3d(kernel_size=5, sigma=1.0, device=None, dtype=None):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return kernel


def _conv3d_gaussian(x, kernel):
    channels = x.shape[1]
    weight = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1, 1)
    return F.conv3d(x, weight, padding=tuple(k // 2 for k in kernel.shape), groups=channels)


def ssim3d(pred, target, data_range=1.0, kernel_size=5, sigma=1.0):
    """Mean 3D SSIM with a Gaussian window."""
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {pred.shape} vs {target.shape}")
    kernel = _gaussian_kernel_3d(kernel_size, sigma, pred.device, pred.dtype)
    mu_x = _conv3d_gaussian(pred, kernel)
    mu_y = _conv3d_gaussian(target, kernel)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = _conv3d_gaussian(pred * pred, kernel) - mu_x2
    sigma_y2 = _conv3d_gaussian(target * target, kernel) - mu_y2
    sigma_xy = _conv3d_gaussian(pred * target, kernel) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def ssim_loss(pred, target, data_range=1.0):
    return 1.0 - ssim3d(pred, target, data_range=data_range)
