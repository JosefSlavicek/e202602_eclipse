# Stage 0: same pipeline as eda00.py / eda00.ipynb (no notebook visualizations).
from dataclasses import dataclass
import math
import random
import collections
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pickle
import re
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from datetime import datetime
import exiftool
import enum
import torchvision
import itertools
import tqdm

BRIGHTNESS_MIN = 0.0
BRIGHTNESS_MAX = 1.0
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")


class MoonInfoOrigin(enum.Enum):
    DIRECT = 0
    INTERPOLATED = 1


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int
    avg_brightness: float
    timestamp: float
    exposure_time: float
    moon: tuple[float, float, float] = None
    moon_info_origin: MoonInfoOrigin = None
    moon_pos_std_px: float = None


def get_info_from_exif(img_path: Path):
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(str(img_path))[0]
    exposure_time = metadata.get("EXIF:ExposureTime")
    assert exposure_time is not None
    assert isinstance(exposure_time, (int, float)), type(exposure_time)
    subsec_date_time_original = metadata.get("Composite:SubSecDateTimeOriginal")
    assert subsec_date_time_original is not None
    assert re.match(
        r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}\.\d{2}-\d{2}:\d{2}",
        subsec_date_time_original,
    ), subsec_date_time_original
    expected_format = "%Y:%m:%d %H:%M:%S.%f%z"
    try:
        dt = datetime.strptime(subsec_date_time_original, expected_format)
        timestamp = dt.timestamp()
    except ValueError as e:
        raise ValueError(
            f"Failed to parse DateTime '{subsec_date_time_original}' in {img_path}: {e}"
        )
    return float(exposure_time), timestamp


def get_image_infos(data_root: str | Path):
    root = Path(data_root)
    jpg_files = list(root.rglob("*.jpg")) + list(root.rglob("*.JPG"))
    image_infos = []
    for jpg_file in tqdm.tqdm(jpg_files, desc="First scan of images"):
        with Image.open(jpg_file) as img:
            width, height = img.size
            avg_brightness = np.array(img).astype(np.float32).mean() / 255.0
            if BRIGHTNESS_MIN <= avg_brightness <= BRIGHTNESS_MAX:
                exposure_time, timestamp = get_info_from_exif(jpg_file)
                image_infos.append(
                    ImageInfo(
                        path=jpg_file,
                        width=width,
                        height=height,
                        avg_brightness=avg_brightness,
                        timestamp=timestamp,
                        exposure_time=exposure_time,
                    )
                )
    assert len(image_infos) > 0
    for ii in image_infos:
        assert ii.width == image_infos[0].width
        assert ii.height == image_infos[0].height
    image_infos.sort(key=lambda x: x.avg_brightness)
    return image_infos


N_SECTORS = 360
N_TRIPLETS = 1024
N_CLUSTER = 256
REFINE_ITERATIONS = 3
MIN_TRIPLET_DEGREES = 30


def _indices_within_degrees(center: int, deg: int) -> set:
    return {(center + d) % N_SECTORS for d in range(-(deg - 1), deg)}


def sample_triplet_indices(n_pts: int, min_degrees: int = MIN_TRIPLET_DEGREES):
    available = set(range(n_pts))
    a = random.sample(list(available), 1)[0]
    available -= _indices_within_degrees(a, min_degrees) & available
    assert len(available) >= 2
    b = random.sample(list(available), 1)[0]
    available -= _indices_within_degrees(b, min_degrees) & available
    assert len(available) >= 1
    c = random.sample(list(available), 1)[0]
    return (a, b, c)


def refine_moon(img: torch.Tensor, center_i: float, center_j: float):
    assert img.ndim == 3 and img.shape[2] == 3
    H, W = img.shape[0], img.shape[1]
    dev = img.device
    img_size = float(max(H, W))
    gray = img.mean(dim=2)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=dev
    ).view(1, 1, 3, 3)
    g = gray.unsqueeze(0).unsqueeze(0)
    grad_x = torch.nn.functional.conv2d(g, sobel_x, padding=1).squeeze()
    grad_y = torch.nn.functional.conv2d(g, sobel_y, padding=1).squeeze()
    dy = torch.arange(H, device=dev, dtype=torch.float32).view(-1, 1) - center_i
    dx = torch.arange(W, device=dev, dtype=torch.float32).view(1, -1) - center_j
    norm = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
    u_x, u_y = dx / norm, dy / norm
    angle = torch.atan2(dy, dx)
    sector_id = (
        torch.floor((angle + math.pi) / (2 * math.pi) * N_SECTORS).long() % N_SECTORS
    )
    dot_product = grad_x * u_x + grad_y * u_y
    dot_product_flat = dot_product.reshape(-1)
    sector_flat = sector_id.reshape(-1)
    W_t = W
    points_list = []
    for s in range(N_SECTORS):
        mask = sector_flat == s
        if mask.any():
            masked = torch.where(
                mask, dot_product_flat, torch.tensor(-1e9, device=dev, dtype=torch.float32)
            )
            idx = masked.argmax().item()
            i, j = idx // W_t, idx % W_t
            points_list.append((i, j))
    n_pts = len(points_list)
    if n_pts < 3:
        return (center_i, center_j, 0.0)

    def circumcenter(i1, j1, i2, j2, i3, j3):
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

    circumcenters, radii = [], []
    while len(circumcenters) < N_TRIPLETS:
        a, b, c = sample_triplet_indices(n_pts)
        i1, j1 = points_list[a]
        i2, j2 = points_list[b]
        i3, j3 = points_list[c]
        cc_result = circumcenter(i1, j1, i2, j2, i3, j3)
        if cc_result is not None:
            oi, oj, radius = cc_result
            circumcenters.append((oi, oj))
            radii.append(radius)
    pts = np.array(circumcenters, dtype=np.float64)
    Z = linkage(pts, method="complete")
    t_lo, t_hi = 0.0, float(Z[-1, 2])
    for _ in range(60):
        t = (t_lo + t_hi) / 2
        labels = fcluster(Z, t, criterion="distance")
        sizes = np.bincount(labels)
        max_size = int(sizes.max())
        if max_size >= N_CLUSTER:
            t_hi = t
        else:
            t_lo = t
    labels = fcluster(Z, t_hi, criterion="distance")
    sizes = np.bincount(labels)
    which = int(np.argmax(sizes))
    cluster_mask = labels == which
    cluster_pts = pts[cluster_mask]
    cluster_radii = np.array(radii)[cluster_mask]
    ci = float(cluster_pts[:, 0].mean())
    cj = float(cluster_pts[:, 1].mean())
    radius = float(cluster_radii.mean())
    return (ci, cj, radius)


def find_moon(img: torch.Tensor, i0: float, j0: float):
    assert img.ndim == 3 and img.shape[2] == 3
    center_i, center_j = i0, j0
    radius = 0.0
    for _ in range(REFINE_ITERATIONS):
        center_i, center_j, radius = refine_moon(img, center_i, center_j)
    return (center_i, center_j, radius)


class ApproxMoonFinder:
    _kernels = {}
    _min_radius = 3
    _target_size = 256
    _max_radius = _target_size // 2 - 3

    @classmethod
    def _create_circle_kernel(cls, radius: int, device: torch.device):
        kernel_size = 2 * radius + 1
        center = radius
        y = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        kernel = (torch.abs(dist - radius) < 0.5).float()
        return kernel.unsqueeze(0).unsqueeze(0)

    @classmethod
    def _get_kernel(cls, radius: int, device: torch.device):
        if radius not in cls._kernels:
            cls._kernels[radius] = cls._create_circle_kernel(radius, device)
        kernel = cls._kernels[radius]
        if kernel.device != device:
            kernel = kernel.to(device)
            cls._kernels[radius] = kernel
        return kernel

    @classmethod
    def find_moon_approx(cls, img: torch.Tensor):
        assert img.ndim == 3 and img.shape[2] == 3
        original_H, original_W = img.shape[0], img.shape[1]
        device = img.device
        gray = img.mean(dim=2)
        scale = min(cls._target_size / original_H, cls._target_size / original_W)
        H_scaled = int(round(original_H * scale))
        W_scaled = int(round(original_W * scale))
        gray_4d = gray.unsqueeze(0).unsqueeze(0)
        gray_scaled = torch.nn.functional.interpolate(
            gray_4d, size=(H_scaled, W_scaled), mode="bilinear", align_corners=False
        ).squeeze()
        pad_h = (cls._target_size - H_scaled) // 2
        pad_w = (cls._target_size - W_scaled) // 2
        gray_downscaled = torch.nn.functional.pad(
            gray_scaled,
            (
                pad_w,
                cls._target_size - W_scaled - pad_w,
                pad_h,
                cls._target_size - H_scaled - pad_h,
            ),
            mode="constant",
            value=0.0,
        )
        H_down, W_down = gray_downscaled.shape
        best_diff = torch.full(
            (H_down, W_down), float("-inf"), device=device, dtype=torch.float32
        )
        sum_prev = sum_prev2 = None
        for radius in range(cls._min_radius, cls._max_radius + 1):
            kernel = cls._get_kernel(radius, device)
            padding = radius
            gray_input = gray_downscaled.unsqueeze(0).unsqueeze(0)
            sum_along_circle = torch.nn.functional.conv2d(
                gray_input, kernel, padding=padding
            ).squeeze()
            if sum_prev2 is not None:
                diff = sum_along_circle - sum_prev2
                best_diff = torch.maximum(best_diff, diff)
            sum_prev2, sum_prev = sum_prev, sum_along_circle
        flat_idx = best_diff.argmax().item()
        i_down = flat_idx // W_down
        j_down = flat_idx % W_down
        i_scaled = i_down - pad_h
        j_scaled = j_down - pad_w
        i = int(round((i_scaled + 0.5) * (original_H / H_scaled) - 0.5))
        j = int(round((j_scaled + 0.5) * (original_W / W_scaled) - 0.5))
        return (i, j)


def compose_transforms(s1_i, s1_j, rot1_deg, s2_i, s2_j, rot2_deg):
    theta1_rad = math.radians(rot1_deg)
    cos1, sin1 = math.cos(theta1_rad), math.sin(theta1_rad)
    s_rot_i = cos1 * s2_i - sin1 * s2_j
    s_rot_j = sin1 * s2_i + cos1 * s2_j
    return (s1_i + s_rot_i, s1_j + s_rot_j, rot1_deg + rot2_deg)


def transform_moon_center_batched(
    moon_center_i, moon_center_j, ci, cj, shift_i_t, shift_j_t, cos_a_t, sin_a_t
):
    di_src = moon_center_i - ci
    dj_src = moon_center_j - cj
    cos_a_flat = cos_a_t.flatten()
    sin_a_flat = sin_a_t.flatten()
    shift_i_flat = shift_i_t.flatten()
    shift_j_flat = shift_j_t.flatten()
    di_rot = di_src * cos_a_flat - dj_src * sin_a_flat
    dj_rot = di_src * sin_a_flat + dj_src * cos_a_flat
    i_out = di_rot + shift_i_flat + ci
    j_out = dj_rot + shift_j_flat + cj
    return tuple((float(i_out[k]), float(j_out[k])) for k in range(len(i_out)))


def find_common_centers_and_radii(
    moon_center_target, moon_radius_target, moon_centers_warped, moon_radius_warped
):
    centers = []
    radii = []
    for center2 in moon_centers_warped:
        vector = (
            center2[0] - moon_center_target[0],
            center2[1] - moon_center_target[1],
        )
        distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if distance < 0.001:
            center = center2
            radius = max(moon_radius_target, moon_radius_warped)
        else:
            vector = (vector[0] / distance, vector[1] / distance)
            pt_target = (
                moon_center_target[0] - vector[0] * moon_radius_target,
                moon_center_target[1] - vector[1] * moon_radius_target,
            )
            pt_warped = (
                center2[0] + vector[0] * moon_radius_warped,
                center2[1] + vector[1] * moon_radius_warped,
            )
            center = (
                0.5 * (pt_target[0] + pt_warped[0]),
                0.5 * (pt_target[1] + pt_warped[1]),
            )
            radius = 0.5 * (distance + moon_radius_target + moon_radius_warped)
        centers.append(center)
        radii.append(radius)
    return centers, radii


def find_max_radii(common_centers, height: int, width: int):
    radii = []
    for center in common_centers:
        di = np.abs(center[0] - height / 2)
        dj = np.abs(center[1] - width / 2)
        ri = height / 2 - di
        rj = width / 2 - dj
        radius = min(ri, rj)
        assert radius > 10, (center, height, width)
        radii.append(radius)
    return radii


def image_to_polars(image, center, radius_min: float, radius_max: float):
    assert image.ndim == 2
    assert len(center) == 2
    assert 0 < radius_min < radius_max
    device = image.device
    H, W = image.shape
    center_i, center_j = center
    output_height = int(radius_max - radius_min + 1)
    output_width = int(2 * math.pi * radius_max)
    y = torch.arange(
        output_height, device=device, dtype=torch.float32
    ).view(-1, 1)
    x = torch.arange(output_width, device=device, dtype=torch.float32).view(
        1, -1
    )
    r = radius_max - y
    if output_width > 1:
        theta = 2 * math.pi * x / (output_width - 1)
    else:
        theta = torch.zeros_like(x)
    i_src = center_i + r * torch.sin(theta)
    j_src = center_j + r * torch.cos(theta)
    j_norm = 2.0 * j_src / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_src)
    i_norm = 2.0 * i_src / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_src)
    grid = torch.stack([j_norm, i_norm], dim=-1)
    img_4d = image.unsqueeze(0).unsqueeze(0)
    polar_image = torch.nn.functional.grid_sample(
        img_4d, grid.unsqueeze(0), mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(0).squeeze(0)
    mask = (i_src >= 0) & (i_src < H) & (j_src >= 0) & (j_src < W)
    return (polar_image, mask)


def remove_lowfeq(polar_image, n_remove: int):
    assert polar_image.ndim == 2
    W = polar_image.shape[1]
    spec = torch.fft.rfft(polar_image, dim=-1)
    spec[:, :n_remove] = 0.0
    out = torch.fft.irfft(spec, n=W, dim=-1)
    return out.to(polar_image.dtype)


def polar_to_cartesian(polar_image, center, radius_min, radius_max, output_size):
    assert polar_image.ndim == 2
    assert len(center) == 2
    assert len(output_size) == 2
    device = polar_image.device
    H_polar, W_polar = polar_image.shape
    H, W = output_size
    center_i, center_j = center
    i = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1)
    j = torch.arange(W, device=device, dtype=torch.float32).view(1, -1)
    di = i - center_i
    dj = j - center_j
    r = torch.sqrt(di * di + dj * dj)
    theta = torch.atan2(di, dj)
    theta = torch.where(theta < 0, theta + 2 * math.pi, theta)
    y_polar = radius_max - r
    if W_polar > 1:
        x_polar = theta / (2 * math.pi) * (W_polar - 1)
    else:
        x_polar = torch.zeros_like(theta)
    x_norm = (
        2.0 * x_polar / (W_polar - 1) - 1.0
        if W_polar > 1
        else torch.zeros_like(x_polar)
    )
    y_norm = (
        2.0 * y_polar / (H_polar - 1) - 1.0
        if H_polar > 1
        else torch.zeros_like(y_polar)
    )
    grid = torch.stack([x_norm, y_norm], dim=-1)
    polar_4d = polar_image.unsqueeze(0).unsqueeze(0)
    cartesian_image = torch.nn.functional.grid_sample(
        polar_4d, grid.unsqueeze(0), mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(0).squeeze(0)
    mask_valid = (r >= radius_min) & (r <= radius_max)
    cartesian_image = torch.where(
        mask_valid, cartesian_image, torch.zeros_like(cartesian_image)
    )
    return cartesian_image


def fill_bottom(polar_img, n_step: int):
    assert polar_img.ndim == 2
    assert 0 <= n_step < polar_img.shape[0]
    if n_step == 0:
        return polar_img
    retval = polar_img.clone()
    mx = polar_img[-n_step:, :].amax(dim=0)
    apply = torch.ones(retval.shape[1], dtype=torch.bool, device=retval.device)
    for i in range(n_step):
        apply = apply & (retval[-i - 1, :] >= mx)
        retval[-i - 1, apply] = mx[apply]
    return retval


def fill_moon(img, moon_center, moon_radius: float, value: float):
    assert img.ndim == 2
    assert len(moon_center) == 2
    assert moon_radius > 0
    out = img.clone()
    ci, cj = moon_center
    H, W = img.shape
    i = torch.arange(H, dtype=torch.float32, device=img.device)[:, None]
    j = torch.arange(W, dtype=torch.float32, device=img.device)[None, :]
    dist = torch.sqrt((i - ci) ** 2 + (j - cj) ** 2)
    mask = dist <= moon_radius
    out[mask] = value
    return out


def gaussian_blur(img, sigma: float):
    assert img.ndim == 2
    assert sigma > 0
    sigma = int(np.ceil(sigma))
    kernel_size = min(17, 4 * sigma + 1)
    gblur = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size, sigma=sigma
    ).to(img.device)
    return gblur(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def discrepancy_batched_fourier3(
    target,
    warped,
    moon_center_target,
    moon_radius_target,
    moon_centers_warped,
    moon_radius_warped,
    batch,
    blur_sigma,
    best_setup,
):
    assert target.ndim == 2
    assert warped.ndim == 3
    assert len(moon_center_target) == 2
    assert len(moon_centers_warped) == len(warped)
    assert moon_radius_target > 0
    assert moon_radius_warped > 0
    assert len(batch) == len(warped)
    list_of_all = [(target, moon_center_target, moon_radius_target)]
    for i in range(len(moon_centers_warped)):
        list_of_all.append(
            (warped[i], moon_centers_warped[i], moon_radius_warped)
        )
    list_of_all_processed = []
    antiprotuberance_threshold = None
    for img, moon_center, moon_radius in list_of_all:
        radius_max = min(
            moon_center[0],
            img.shape[0] - moon_center[0],
            moon_center[1],
            img.shape[1] - moon_center[1],
        )
        assert 32 < moon_radius < radius_max - 32
        polar_img, _ = image_to_polars(
            img, moon_center, moon_radius, radius_max
        )
        if antiprotuberance_threshold is None:
            maxidx = polar_img.sum(dim=1).argmax()
            maxrow = polar_img[maxidx, :]
            maxrow = maxrow[maxrow > 0]
            antiprotuberance_threshold = maxrow.quantile(0.9)
        polar_img[polar_img > antiprotuberance_threshold] = antiprotuberance_threshold
        polar_img = fill_bottom(polar_img, 4)
        polar_img = remove_lowfeq(polar_img, 16)
        img = polar_to_cartesian(
            polar_img, moon_center, moon_radius, radius_max, (img.shape[0], img.shape[1])
        )
        if blur_sigma > 0:
            img = gaussian_blur(img, blur_sigma)
        mask = torch.ones_like(polar_img)
        mask = polar_to_cartesian(
            mask, moon_center, moon_radius, radius_max, (img.shape[0], img.shape[1])
        )
        mask = fill_moon(mask, moon_center, moon_radius, 0.0)
        list_of_all_processed.append((img, mask))
    target_img, target_mask = list_of_all_processed.pop(0)
    if best_setup is not None:
        bs_shift_i, bs_shift_j, bs_angle, bs_img, bs_mask = best_setup
        batch.append((bs_shift_i, bs_shift_j, bs_angle))
        list_of_all_processed.append((bs_img, bs_mask))
    assert len(list_of_all_processed) == len(batch)
    while len(list_of_all_processed) >= 2:
        shift_i_0, shift_j_0, angle_0 = batch.pop()
        shift_i_1, shift_j_1, angle_1 = batch.pop()
        warped_img_0, warped_mask_0 = list_of_all_processed.pop()
        warped_img_1, warped_mask_1 = list_of_all_processed.pop()
        mask = target_mask * warped_mask_0 * warped_mask_1
        diff_0 = torch.abs(target_img - warped_img_0)
        diff_0 = ((diff_0 * mask).sum() / mask.sum()).item()
        diff_1 = torch.abs(target_img - warped_img_1)
        diff_1 = ((diff_1 * mask).sum() / mask.sum()).item()
        if diff_0 < diff_1:
            batch.append((shift_i_0, shift_j_0, angle_0))
            list_of_all_processed.append((warped_img_0, warped_mask_0))
        else:
            batch.append((shift_i_1, shift_j_1, angle_1))
            list_of_all_processed.append((warped_img_1, warped_mask_1))
    shift_i, shift_j, angle = batch.pop()
    warped_img, warped_mask = list_of_all_processed.pop()
    return shift_i, shift_j, angle, warped_img, warped_mask


def register_equal_exposure(image_info0, image_info1):
    device = torch.device("cuda")

    def load_grayscale(ii):
        with Image.open(ii.path) as img:
            arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    g0 = load_grayscale(image_info0)
    g1 = load_grayscale(image_info1)
    H, W = g0.shape
    assert g1.shape == (H, W)
    moon0 = image_info0.moon
    moon1 = image_info1.moon
    r0, r1 = moon0[2], moon1[2]
    moon_radius_avg = (r0 + r1) / 2.0
    u0 = (
        image_info0.moon_pos_std_px
        if image_info0.moon_pos_std_px is not None
        else 2.0
    )
    u1 = (
        image_info1.moon_pos_std_px
        if image_info1.moon_pos_std_px is not None
        else 2.0
    )
    sun_drift_per_sec = 0.001 * moon_radius_avg
    dt_sec = abs(image_info1.timestamp - image_info0.timestamp)
    possible_sun_drift = dt_sec * sun_drift_per_sec
    initial_shift_half = 2.0 * (5.0 * (u0 + u1) + possible_sun_drift + 3.0)
    ci, cj = H / 2.0, W / 2.0
    r_border = max(H, W)
    ii = (
        torch.arange(H, device=device, dtype=torch.float32)
        .view(-1, 1)
        .expand(H, W)
        .unsqueeze(0)
    )
    jj = (
        torch.arange(W, device=device, dtype=torch.float32)
        .view(1, -1)
        .expand(H, W)
        .unsqueeze(0)
    )

    def apply_transform_batched(img, shift_i_t, shift_j_t, cos_a_t, sin_a_t):
        di = ii - ci - shift_i_t
        dj = jj - cj - shift_j_t
        i_src = di * cos_a_t + dj * sin_a_t + ci
        j_src = -di * sin_a_t + dj * cos_a_t + cj
        j_norm = (
            2.0 * j_src / (W - 1) - 1.0
            if W > 1
            else torch.zeros_like(j_src)
        )
        i_norm = (
            2.0 * i_src / (H - 1) - 1.0
            if H > 1
            else torch.zeros_like(i_src)
        )
        grid = torch.stack([j_norm, i_norm], dim=-1)
        img_4d = img.unsqueeze(0).unsqueeze(1).expand(grid.shape[0], 1, H, W)
        out = torch.nn.functional.grid_sample(
            img_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return out.squeeze(1)

    best_shift_i, best_shift_j, best_angle = 0.0, 0.0, 0.0
    step_shift = initial_shift_half / 2.0
    step_angle = 5.0
    refine_shift = refine_angle = True
    best_setup = None
    while refine_shift or refine_angle:
        if step_shift < 0.1:
            refine_shift = False
        step_angle_in_px = step_angle * r_border * (math.pi / 180.0)
        if step_angle_in_px < 0.1:
            refine_angle = False
        if step_shift < 2 and step_angle_in_px < 2:
            blur_sigma = 0.0
        else:
            blur_sigma = min(8, max(step_angle_in_px, step_shift))
        shift_i_vals = (
            [best_shift_i]
            if not refine_shift
            else [best_shift_i + step_shift * (k - 2) for k in range(5)]
        )
        shift_j_vals = (
            [best_shift_j]
            if not refine_shift
            else [best_shift_j + step_shift * (k - 2) for k in range(5)]
        )
        angle_vals = (
            [best_angle]
            if not refine_angle
            else [best_angle + step_angle * (k - 2) for k in range(5)]
        )
        triples = [
            (si, sj, a)
            for si in shift_i_vals
            for sj in shift_j_vals
            for a in angle_vals
        ]
        GRID_BATCH_SIZE = 8
        for start in range(0, len(triples), GRID_BATCH_SIZE):
            batch = triples[start : start + GRID_BATCH_SIZE]
            N = len(batch)
            shift_i_t = torch.tensor(
                [t[0] for t in batch], device=device, dtype=torch.float32
            ).view(N, 1, 1)
            shift_j_t = torch.tensor(
                [t[1] for t in batch], device=device, dtype=torch.float32
            ).view(N, 1, 1)
            angles_rad = torch.tensor(
                [math.radians(-t[2]) for t in batch],
                device=device,
                dtype=torch.float32,
            )
            cos_a_t = torch.cos(angles_rad).view(N, 1, 1)
            sin_a_t = torch.sin(angles_rad).view(N, 1, 1)
            warped = apply_transform_batched(
                g1.clone(), shift_i_t, shift_j_t, cos_a_t, sin_a_t
            )
            moon_centers_warped = transform_moon_center_batched(
                moon1[0], moon1[1], ci, cj, shift_i_t, shift_j_t, cos_a_t, sin_a_t
            )
            best_setup = discrepancy_batched_fourier3(
                g0.clone(),
                warped,
                moon0[:2],
                r0,
                moon_centers_warped,
                r1,
                list(batch),
                blur_sigma,
                best_setup,
            )
        best_shift_i, best_shift_j, best_angle, _, _ = best_setup
        is_corner = len(shift_i_vals) > 2 and (
            best_shift_i in [shift_i_vals[0], shift_i_vals[-1]]
            or best_shift_j in [shift_j_vals[0], shift_j_vals[-1]]
        )
        step_shift = step_shift / 2.0 if refine_shift and not is_corner else step_shift
        is_corner = (
            len(angle_vals) > 2
            and best_angle in [angle_vals[0], angle_vals[-1]]
        )
        step_angle = (
            step_angle / 2.0 if refine_angle and not is_corner else step_angle
        )
    return (float(best_shift_i), float(best_shift_j), float(best_angle))


def stage0_detect_moons(image_infos: list) -> None:
    """GPU: approximate disk finder + gradient/triplet refine; sets moon on each ImageInfo."""
    for ii in tqdm.tqdm(image_infos, desc="Finding moon"):
        img = Image.open(ii.path)
        img_arr = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).cuda()
        if img_arr.ndim == 2:
            img_arr = img_arr.unsqueeze(-1).expand(-1, -1, 3)
        elif img_arr.shape[2] == 1:
            img_arr = img_arr.expand(-1, -1, 3)
        i0, j0 = ApproxMoonFinder.find_moon_approx(img_arr)
        i, j, radius = find_moon(img_arr, i0, j0)
        ii.moon = (i, j, radius)
        ii.moon_info_origin = MoonInfoOrigin.DIRECT


def stage0_print_exposure_groups_stats(exposure_groups: dict) -> None:
    for exposure_time in sorted(exposure_groups.keys()):
        group = exposure_groups[exposure_time]
        radii = [ii.moon[2] for ii in group if ii.moon is not None]
        print(
            f"exposure_time={exposure_time:.5f} len(group)={len(group)} "
            f"np.mean(radii)={np.mean(radii):.2f} np.std(radii)={np.std(radii):.2f}"
        )


def stage0_group_by_exposure(image_infos: list) -> dict:
    """Build exposure_time -> [ImageInfo, ...] and print per-group radius stats."""
    exposure_groups = collections.defaultdict(list)
    for ii in image_infos:
        exposure_groups[ii.exposure_time].append(ii)
    return exposure_groups


def stage0_prune_moon_info_for_radius_outliers(exposure_groups: dict) -> None:
    """Drop moons that disagree with rolling reference radius (in-place)."""
    RADIUS_STD_THRESHOLD = 1.0
    MIN_GROUP_ELMS = 3

    def radius(ii):
        return ii.moon[2]

    prev_avg_radius = None
    for exposure_time in sorted(exposure_groups.keys()):
        group = list(exposure_groups[exposure_time])
        assert len(group) >= MIN_GROUP_ELMS
        radii = [radius(ii) for ii in group]
        mean_r = np.mean(radii)
        std_r = np.std(radii)
        mean_r_ok = (
            prev_avg_radius is None
            or abs(mean_r - prev_avg_radius) <= 0.1 * prev_avg_radius
        )
        if std_r <= RADIUS_STD_THRESHOLD and mean_r_ok:
            prev_avg_radius = mean_r
        else:
            assert prev_avg_radius is not None
            subgroup = [
                ii
                for ii in group
                if abs(radius(ii) - prev_avg_radius) <= 0.1 * prev_avg_radius
            ]
            if len(subgroup) < MIN_GROUP_ELMS:
                subgroup = []
            else:
                mean_r = np.mean([radius(ii) for ii in subgroup])
                std_r = np.std([radius(ii) for ii in subgroup])
                if std_r > RADIUS_STD_THRESHOLD:
                    subgroup = []
                else:
                    prev_avg_radius = mean_r
            for ii in group:
                if ii not in subgroup:
                    ii.moon = None
                    ii.moon_info_origin = None


def stage0_interpolate_missing_moons(image_infos: list, exposure_groups: dict) -> tuple:
    """Linear fit moon (i,j) vs time for direct detections; fill missing + radii (in-place)."""
    pts_with_moon = [
        (ii.timestamp, ii.moon[0], ii.moon[1])
        for ii in image_infos
        if ii.moon is not None
    ]
    assert len(pts_with_moon) >= 2
    t_arr = np.array([p[0] for p in pts_with_moon], dtype=np.float64)
    i_arr = np.array([p[1] for p in pts_with_moon], dtype=np.float64)
    j_arr = np.array([p[2] for p in pts_with_moon], dtype=np.float64)
    (a_i, b_i) = np.polyfit(t_arr, i_arr, 1)
    (a_j, b_j) = np.polyfit(t_arr, j_arr, 1)

    def interpolate_moon_at_time(t: float):
        return (float(a_i * t + b_i), float(a_j * t + b_j))

    last_avg_radius = None
    for exposure_time in sorted(exposure_groups.keys()):
        group = exposure_groups[exposure_time]
        with_moon = [ii for ii in group if ii.moon is not None]
        if with_moon:
            avg_radius = float(np.mean([ii.moon[2] for ii in with_moon]))
            last_avg_radius = avg_radius
        else:
            assert last_avg_radius is not None
            avg_radius = last_avg_radius
        for ii in group:
            if ii.moon is None:
                i_pred, j_pred = interpolate_moon_at_time(ii.timestamp)
                ii.moon = (i_pred, j_pred, avg_radius)
                ii.moon_info_origin = MoonInfoOrigin.INTERPOLATED

    return interpolate_moon_at_time


def stage0_set_moon_position_std(
    image_infos: list, exposure_groups: dict, interpolate_moon_at_time
) -> None:
    """Residual std on direct points -> moon_pos_std_px per image (in-place)."""
    MIN_GROUP_ELMS = 3
    residuals_i = []
    residuals_j = []
    for ii in image_infos:
        if ii.moon_info_origin != MoonInfoOrigin.DIRECT:
            continue
        i_pred, j_pred = interpolate_moon_at_time(ii.timestamp)
        residuals_i.append(ii.moon[0] - i_pred)
        residuals_j.append(ii.moon[1] - j_pred)
    residuals_i = np.array(residuals_i)
    residuals_j = np.array(residuals_j)
    std_i = float(np.std(residuals_i))
    std_j = float(np.std(residuals_j))
    moon_position_uncertainty_px = max(std_i, std_j)
    print(f"moon_position_uncertainty_px={moon_position_uncertainty_px:.2f}")

    for exposure_time in sorted(exposure_groups.keys()):
        group = exposure_groups[exposure_time]
        direct_in_group = [
            ii for ii in group if ii.moon_info_origin == MoonInfoOrigin.DIRECT
        ]
        std_direct = None
        if len(direct_in_group) >= MIN_GROUP_ELMS:
            std_direct = float(np.std([ii.moon[2] for ii in direct_in_group]))
        for ii in group:
            if (
                ii.moon_info_origin == MoonInfoOrigin.DIRECT
                and std_direct is not None
            ):
                ii.moon_pos_std_px = std_direct
            else:
                ii.moon_pos_std_px = moon_position_uncertainty_px


def stage0_register_intra_exposure_pairs(exposure_groups: dict) -> dict:
    """For each exposure, all ordered pairs: Fourier-style registration on GPU (slow loop)."""
    reg = {}
    for exposure_time in sorted(exposure_groups.keys()):
        print(f"Doing check for exposure time {exposure_time}")
        group = list(exposure_groups[exposure_time])
        n = len(group)
        if n < 2:
            print(f"  Skip (group size {n} < 2)")
            continue
        tasks = [(i, j) for i, j in itertools.permutations(range(n), 2)]
        for i, j in tqdm.tqdm(tasks, desc="Registration pairs"):
            key = (exposure_time, i, j)
            shift_i, shift_j, rotation = register_equal_exposure(group[i], group[j])
            reg[key] = (shift_i, shift_j, rotation)
        # registration done, now just compute some debuging statistics
        res_i, res_j, res_rot = [], [], []
        for a, b in itertools.combinations(range(n), 2):
            rab = reg[(exposure_time, a, b)]
            rba = reg[(exposure_time, b, a)]
            res_i.append(rab[0] + rba[0])
            res_j.append(rab[1] + rba[1])
            res_rot.append(rab[2] + rba[2])
        res_i = np.array(res_i)
        res_j = np.array(res_j)
        res_rot = np.array(res_rot)
        ijmean = 0.5 * (np.mean(np.abs(res_i)) + np.mean(np.abs(res_j)))
        ijmax = float(max(np.max(np.abs(res_i)), np.max(np.abs(res_j))))
        rotmean = float(np.mean(np.abs(res_rot)))
        rotmax = float(np.max(np.abs(res_rot)))
        print(
            f"  Check 1: ij mean={ijmean:.4f} max={ijmax:.4f}  rot mean={rotmean:.4f} max={rotmax:.4f}"
        )
        if n >= 3:
            tri_i, tri_j, tri_rot = [], [], []
            for a, b, c in itertools.permutations(range(n), 3):
                rab = reg[(exposure_time, a, b)]
                rbc = reg[(exposure_time, b, c)]
                rac = reg[(exposure_time, a, c)]
                composed = compose_transforms(
                    rab[0], rab[1], rab[2], rbc[0], rbc[1], rbc[2]
                )
                tri_i.append(composed[0] - rac[0])
                tri_j.append(composed[1] - rac[1])
                tri_rot.append(composed[2] - rac[2])
            tri_i, tri_j, tri_rot = np.array(tri_i), np.array(tri_j), np.array(tri_rot)
            ijmean = 0.5 * (np.mean(np.abs(tri_i)) + np.mean(np.abs(tri_j)))
            ijmax = float(max(np.max(np.abs(tri_i)), np.max(np.abs(tri_j))))
            rotmean = float(np.mean(np.abs(tri_rot)))
            rotmax = float(np.max(np.abs(tri_rot)))
            print(
                f"  Check 2: ij mean={ijmean:.4f} max={ijmax:.4f}  rot mean={rotmean:.4f} max={rotmax:.4f}"
            )
    return reg


def stage0_save_pickle(exposure_groups: dict, reg: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    with open(out_path, "wb") as fd:
        pickle.dump(exposure_groups, fd)
        pickle.dump(reg, fd)
    print(f"Saved {out_path} (exposure_groups, reg)")
    return out_path


def run_stage0(data_root: str | Path, out_path: str | Path) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage 0.")
    out_path = Path(out_path)
    image_infos = get_image_infos(data_root)
    stage0_detect_moons(image_infos)
    exposure_groups = stage0_group_by_exposure(image_infos)
    stage0_prune_radius_outliers(exposure_groups)
    interp = stage0_interpolate_missing_moons(image_infos, exposure_groups)
    stage0_set_moon_position_std(image_infos, exposure_groups, interp)
    reg = stage0_register_intra_exposure_pairs(exposure_groups)
    return stage0_save_pickle(exposure_groups, reg, out_path)
