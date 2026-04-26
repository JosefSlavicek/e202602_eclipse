"""Geometric and polar-coordinate primitives shared across pipeline stages."""
import math

import torch
import torch.nn.functional as F


def circumcenter(i1, j1, i2, j2, i3, j3, img_size):
    """Return (center_i, center_j, radius) for the circle through three points, or None.

    Returns None when the three points are collinear, or when the circumradius exceeds
    img_size (used to reject wildly off-image solutions).
    """
    x1, y1 = float(j1), float(i1)
    x2, y2 = float(j2), float(i2)
    x3, y3 = float(j3), float(i3)
    D = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-10:
        return None
    ox = (
        (x1 * x1 + y1 * y1) * (y2 - y3)
        + (x2 * x2 + y2 * y2) * (y3 - y1)
        + (x3 * x3 + y3 * y3) * (y1 - y2)
    ) / D
    oy = (
        (x1 * x1 + y1 * y1) * (x3 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x3)
        + (x3 * x3 + y3 * y3) * (x2 - x1)
    ) / D
    oi, oj = oy, ox
    d1 = math.hypot(i1 - oi, j1 - oj)
    d2 = math.hypot(i2 - oi, j2 - oj)
    d3 = math.hypot(i3 - oi, j3 - oj)
    if d1 > img_size or d2 > img_size or d3 > img_size:
        return None
    radius = (d1 + d2 + d3) / 3.0
    return (oi, oj, radius)


def cartesian_to_polar(image, center, radius_min, radius_max, n_r, n_theta, mask_margin=0):
    """Sample a 2-D image into (radius × angle) polar coordinates centred on `center`.

    Row 0 of the output corresponds to r=radius_max; row n_r-1 to r=radius_min.
    Column 0 corresponds to θ=0 (rightward); columns increase counter-clockwise.
    Device and dtype are inherited from `image`.

    Args:
        image:        (H, W) float tensor
        center:       (center_i, center_j) in pixel coordinates
        radius_min:   inner radius of the annulus sampled
        radius_max:   outer radius of the annulus sampled
        n_r:          number of radial rows in the output
        n_theta:      number of angular columns in the output
        mask_margin:  source pixels within this many pixels of any border are
                      marked False in the returned mask (0 = no margin)

    Returns:
        polar  – (n_r, n_theta) tensor
        mask   – (n_r, n_theta) bool tensor; True where the source pixel lay
                 at least mask_margin pixels from any image border
    """
    assert image.ndim == 2
    assert len(center) == 2
    assert 0 <= radius_min < radius_max
    device = image.device
    dtype = image.dtype
    H, W = image.shape
    ci, cj = float(center[0]), float(center[1])
    y = torch.arange(n_r, device=device, dtype=dtype).view(-1, 1)
    x = torch.arange(n_theta, device=device, dtype=dtype).view(1, -1)
    r = radius_max - y * (radius_max - radius_min) / max(n_r - 1, 1)
    theta = 2 * math.pi * x / (n_theta - 1) if n_theta > 1 else torch.zeros_like(x)
    i_src = ci + r * torch.sin(theta)
    j_src = cj + r * torch.cos(theta)
    j_norm = 2.0 * j_src / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_src)
    i_norm = 2.0 * i_src / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_src)
    grid = torch.stack([j_norm, i_norm], dim=-1).unsqueeze(0)
    polar = F.grid_sample(
        image.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    m = mask_margin
    mask = (i_src >= m) & (i_src < H - m) & (j_src >= m) & (j_src < W - m)
    return polar, mask


def polar_to_cartesian(polar, center, radius_min, radius_max, H, W):
    """Inverse of `cartesian_to_polar`: map a polar image back to Cartesian space.

    The polar grid dimensions (n_r, n_theta) are inferred from `polar.shape`.
    Device and dtype are inherited from `polar`.

    Returns a (H, W) tensor with zeros outside the [radius_min, radius_max] annulus.
    """
    assert polar.ndim == 2
    assert len(center) == 2
    device = polar.device
    dtype = polar.dtype
    n_r, n_theta = polar.shape
    ci, cj = float(center[0]), float(center[1])
    i = torch.arange(H, device=device, dtype=dtype).view(-1, 1)
    j = torch.arange(W, device=device, dtype=dtype).view(1, -1)
    r = torch.sqrt((i - ci) ** 2 + (j - cj) ** 2)
    theta = torch.atan2(i - ci, j - cj)
    theta = torch.where(theta < 0, theta + 2 * math.pi, theta)
    y_polar = (radius_max - r) * (n_r - 1) / max(radius_max - radius_min, 1e-9)
    x_polar = theta / (2 * math.pi) * (n_theta - 1) if n_theta > 1 else torch.zeros_like(theta)
    x_norm = 2.0 * x_polar / (n_theta - 1) - 1.0 if n_theta > 1 else torch.zeros_like(x_polar)
    y_norm = 2.0 * y_polar / (n_r - 1) - 1.0 if n_r > 1 else torch.zeros_like(y_polar)
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    out = F.grid_sample(
        polar.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    valid = (r >= radius_min) & (r <= radius_max)
    return torch.where(valid, out, torch.zeros_like(out))
