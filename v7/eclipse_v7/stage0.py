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
import itertools
import tqdm

from eclipse_v7.coords import cartesian_to_polar, polar_to_cartesian, circumcenter
from eclipse_v7.utils import (
    load_grayscale,
    compose_transforms,
    remove_lowfeq,
    fill_bottom,
    fill_moon,
    gaussian_blur,
    transform_moon_center_batched,
    clean_polar_fft,
    stage1_grid_search,
    stage2_finetune,
)

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
    source: object = None       # active NefSource; re-attached per stage, never pickled
    link: tuple = None          # free-form per-frame linkage (e.g. test fixtures); pickled, frozen

    def __getstate__(self):
        # Do not pickle `source` (it may hold a large radiance map and is reconstructed
        # cheaply via inputs.attach_source after each load).
        state = self.__dict__.copy()
        state["source"] = None
        return state


def get_info_from_exif(img_path: Path):
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(str(img_path))[0]
    exposure_time = metadata.get("EXIF:ExposureTime")
    assert exposure_time is not None
    assert isinstance(exposure_time, (int, float)), type(exposure_time)
    subsec_date_time_original = metadata.get("Composite:SubSecDateTimeOriginal")
    assert subsec_date_time_original is not None
    assert re.match(
        r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}\.\d{2}[-+]\d{2}:\d{2}",
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


def get_image_infos(source):
    """Enumerate frames via the active NefSource."""
    image_infos = list(tqdm.tqdm(source.scan(), desc="First scan of images"))
    for ii in image_infos:
        assert BRIGHTNESS_MIN <= ii.avg_brightness <= BRIGHTNESS_MAX, (ii.path, ii.avg_brightness)
    assert len(image_infos) > 0
    for ii in image_infos:
        assert ii.width == image_infos[0].width
        assert ii.height == image_infos[0].height
    image_infos.sort(key=lambda x: x.avg_brightness)
    assert image_infos[-1].avg_brightness > source.min_peak_brightness, (
        'Suspiciously low brightness ... are we interpreting data correctly?',
        [ii.avg_brightness for ii in image_infos])
    return image_infos


N_SECTORS = 360
N_TRIPLETS = 1024
N_CLUSTER = 256
REFINE_ITERATIONS = 3
MIN_TRIPLET_DEGREES = 30
RADIUS_STD_THRESHOLD = 1.0
MIN_GROUP_ELMS = 3
SUN_DRIFT_RATE = 0.001


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


def _refine_moon_find_sector_edge_points(img: torch.Tensor, center_i: float, center_j: float):
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
    points_list = []
    for s in range(N_SECTORS):
        mask = sector_flat == s
        if mask.any():
            masked = torch.where(
                mask, dot_product_flat, torch.tensor(-1e9, device=dev, dtype=torch.float32)
            )
            idx = masked.argmax().item()
            points_list.append((idx // W, idx % W))
    return points_list, img_size


def _refine_moon_cluster_circumcenters(points_list, img_size):
    n_pts = len(points_list)
    circumcenters, radii = [], []
    while len(circumcenters) < N_TRIPLETS:
        a, b, c = sample_triplet_indices(n_pts)
        i1, j1 = points_list[a]
        i2, j2 = points_list[b]
        i3, j3 = points_list[c]
        cc_result = circumcenter(i1, j1, i2, j2, i3, j3, img_size)
        if cc_result is not None:
            oi, oj, radius = cc_result
            circumcenters.append((oi, oj))
            radii.append(radius)
    pts = np.array(circumcenters, dtype=np.float32)
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


def refine_moon(img: torch.Tensor, center_i: float, center_j: float):
    assert img.ndim == 3 and img.shape[2] == 3
    points_list, img_size = _refine_moon_find_sector_edge_points(img, center_i, center_j)
    if len(points_list) < 3:
        return (center_i, center_j, 0.0)
    return _refine_moon_cluster_circumcenters(points_list, img_size)


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
        max_radius = min(H_scaled, W_scaled) // 2 - 3
        best_diff = torch.full(
            (H_scaled, W_scaled), float("-inf"), device=device, dtype=torch.float32
        )
        sum_prev = sum_prev2 = None
        gray_input = gray_scaled.unsqueeze(0).unsqueeze(0)
        for radius in range(cls._min_radius, max_radius + 1):
            kernel = cls._get_kernel(radius, device)
            sum_along_circle = torch.nn.functional.conv2d(
                gray_input, kernel, padding=radius
            ).squeeze()
            if sum_prev2 is not None:
                diff = sum_along_circle - sum_prev2
                best_diff = torch.maximum(best_diff, diff)
            sum_prev2, sum_prev = sum_prev, sum_along_circle
        flat_idx = best_diff.argmax().item()
        i_down = flat_idx // W_scaled
        j_down = flat_idx % W_scaled
        i = int(round((i_down + 0.5) * (original_H / H_scaled) - 0.5))
        j = int(round((j_down + 0.5) * (original_W / W_scaled) - 0.5))
        return (i, j)


def _initial_shift_half(ii_a, ii_b):
    moon_radius_avg = (ii_a.moon[2] + ii_b.moon[2]) / 2.0
    u_a = ii_a.moon_pos_std_px if ii_a.moon_pos_std_px is not None else 2.0
    u_b = ii_b.moon_pos_std_px if ii_b.moon_pos_std_px is not None else 2.0
    sun_drift_per_sec = SUN_DRIFT_RATE * moon_radius_avg
    dt_sec = abs(ii_b.timestamp - ii_a.timestamp)
    possible_sun_drift = dt_sec * sun_drift_per_sec
    return 2.0 * (5.0 * (u_a + u_b) + possible_sun_drift + 3.0)


def detect_moons(image_infos: list) -> None:
    """GPU: approximate disk finder + gradient/triplet refine; sets moon on each ImageInfo."""
    device = torch.device("cuda")
    for ii in tqdm.tqdm(image_infos, desc="Finding moon"):
        img_arr = ii.source.load_rgb(ii, device)
        assert img_arr.ndim == 3 and img_arr.shape[-1] == 3, img_arr.shape
        i0, j0 = ApproxMoonFinder.find_moon_approx(img_arr)
        i, j, radius = find_moon(img_arr, i0, j0)
        ii.moon = (i, j, radius)
        ii.moon_info_origin = MoonInfoOrigin.DIRECT


def print_exposure_groups_stats(exposure_groups: dict) -> None:
    for exposure_time in sorted(exposure_groups.keys()):
        group = exposure_groups[exposure_time]
        radii = [ii.moon[2] for ii in group if ii.moon is not None]
        print(
            f"exposure_time={exposure_time:.5f} len(group)={len(group)} "
            f"np.mean(radii)={np.mean(radii):.2f} np.std(radii)={np.std(radii):.2f}"
        )


def group_by_exposure(image_infos: list) -> dict:
    """Build exposure_time -> [ImageInfo, ...] and print per-group radius stats."""
    exposure_groups = collections.defaultdict(list)
    for ii in image_infos:
        exposure_groups[ii.exposure_time].append(ii)
    return exposure_groups


def subsample_exposure_groups(exposure_groups: dict, factor: int) -> dict:
    """Keep ~1/`factor` of the exposure groups, evenly spaced across the exposure range.

    `factor` is a positive integer (1 = keep every group). For factor > 1 roughly
    (factor-1)/factor of the groups are dropped to speed up smoke / iteration runs
    (factor=2 -> ~half, factor=3 -> ~a third, ...). The shortest and longest exposures
    are always preserved so the full dynamic range is still spanned, and the survivors
    are picked evenly in exposure-sorted order. Whole groups are dropped (never individual
    frames), so downstream per-group invariants (MIN_GROUP_ELMS, pruning) are untouched.

    NOTE: dropping middle groups widens the log-exposure gap between surviving neighbours,
    which degrades the stage2 cross-exposure brightness/gamma fit. This is a runtime knob
    for faster iteration, not a "same result, faster" switch.
    """
    if not isinstance(factor, int) or factor < 1:
        raise ValueError(f"exposure-group subsample factor must be an int >= 1, got {factor!r}")
    keys = sorted(exposure_groups.keys())
    n = len(keys)
    if factor == 1 or n <= 2:
        return dict(exposure_groups)
    keep_count = max(2, math.ceil(n / factor))  # >= 2 so both endpoints survive
    idx = sorted({int(round(x)) for x in np.linspace(0, n - 1, keep_count)})
    kept = {keys[i]: exposure_groups[keys[i]] for i in idx}
    print(
        f"Exposure-group subsample factor={factor}: kept {len(kept)}/{n} groups "
        f"(exposures {[round(keys[i], 5) for i in idx]})"
    )
    return kept


def prune_moon_info_for_radius_outliers(exposure_groups: dict) -> None:
    """Drop moons that disagree with rolling reference radius (in-place).

    The rolling reference starts seeded, not empty: scanning exposure times ascending, the
    first group whose own radii already agree with each other (std <= RADIUS_STD_THRESHOLD)
    supplies the initial `prev_avg_radius`. The shortest exposure is often the noisiest for
    moon-radius detection and may not qualify itself -- seeding from the first group that does
    means the main loop below (unchanged) can prune that noisy group against a real reference
    like any other outlier group, instead of having no reference at all to prune it against.
    """
    def radius(ii):
        return ii.moon[2]

    sorted_times = sorted(exposure_groups.keys())

    prev_avg_radius = None
    for exposure_time in sorted_times:
        radii = [radius(ii) for ii in exposure_groups[exposure_time]]
        if np.std(radii) <= RADIUS_STD_THRESHOLD:
            prev_avg_radius = float(np.mean(radii))
            break
    assert prev_avg_radius is not None, "no exposure group has a self-consistent moon radius"

    for exposure_time in sorted_times:
        group = list(exposure_groups[exposure_time])
        assert len(group) >= MIN_GROUP_ELMS
        radii = [radius(ii) for ii in group]
        mean_r = np.mean(radii)
        std_r = np.std(radii)
        mean_r_ok = abs(mean_r - prev_avg_radius) <= 0.1 * prev_avg_radius
        if std_r <= RADIUS_STD_THRESHOLD and mean_r_ok:
            prev_avg_radius = mean_r
        else:
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


def interpolate_missing_moons(image_infos: list, exposure_groups: dict) -> tuple:
    """Linear fit moon (i,j) vs time for direct detections; fill missing + radii (in-place)."""
    pts_with_moon = [
        (ii.timestamp, ii.moon[0], ii.moon[1])
        for ii in image_infos
        if ii.moon is not None
    ]
    assert len(pts_with_moon) >= 2
    t_arr = np.array([p[0] for p in pts_with_moon], dtype=np.float32)
    i_arr = np.array([p[1] for p in pts_with_moon], dtype=np.float32)
    j_arr = np.array([p[2] for p in pts_with_moon], dtype=np.float32)
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


def set_moon_position_std(
    image_infos: list, exposure_groups: dict, interpolate_moon_at_time
) -> None:
    """Residual std on direct points -> moon_pos_std_px per image (in-place)."""
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


def _print_pair_consistency(reg, exposure_time, group):
    n = len(group)
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
    print(f"  Check 1: ij mean={ijmean:.4f} max={ijmax:.4f}  rot mean={rotmean:.4f} max={rotmax:.4f}")


def _print_triplet_consistency(reg, exposure_time, group):
    n = len(group)
    tri_i, tri_j, tri_rot = [], [], []
    for a, b, c in itertools.permutations(range(n), 3):
        rab = reg[(exposure_time, a, b)]
        rbc = reg[(exposure_time, b, c)]
        rac = reg[(exposure_time, a, c)]
        composed = compose_transforms(rab[0], rab[1], rab[2], rbc[0], rbc[1], rbc[2])
        tri_i.append(composed[0] - rac[0])
        tri_j.append(composed[1] - rac[1])
        tri_rot.append(composed[2] - rac[2])
    tri_i, tri_j, tri_rot = np.array(tri_i), np.array(tri_j), np.array(tri_rot)
    ijmean = 0.5 * (np.mean(np.abs(tri_i)) + np.mean(np.abs(tri_j)))
    ijmax = float(max(np.max(np.abs(tri_i)), np.max(np.abs(tri_j))))
    rotmean = float(np.mean(np.abs(tri_rot)))
    rotmax = float(np.max(np.abs(tri_rot)))
    print(f"  Check 2: ij mean={ijmean:.4f} max={ijmax:.4f}  rot mean={rotmean:.4f} max={rotmax:.4f}")


ENABLE_STAGE_2 = False

# Cap on images registered per exposure group. Groups with more frames are reduced to a
# self-consistent MAX_IMG-subset (see _select_consistent_subset); groups with <= MAX_IMG
# frames are registered in full, exactly as before.
MAX_IMG = 5
# A subset is "good enough" once its worst self-consistency discrepancy is <= this many px.
SUBSET_THRESHOLD_PX = 0.5


def _clean_group(group: list, device) -> tuple[list, list]:
    """Load + polar/FFT-clean every image in a group once, sharing one anti-protuberance
    threshold across the group (so the moon-limb step cancels consistently in every pair).
    Returns (raws, cleans) where cleans[i] = (cleaned_cart, cart_mask)."""
    raws = [load_grayscale(ii, device) for ii in group]
    antiprot = None
    cleans = []
    for raw, ii in zip(raws, group):
        cleaned_cart, cart_mask, antiprot = clean_polar_fft(
            raw, ii.moon[:2], ii.moon[2], antiprot=antiprot
        )
        cleans.append((cleaned_cart, cart_mask))
    return raws, cleans


def _ensure_pair(pair_cache, group, cleans, a, b, device):
    """Stage-1 grid-search transform mapping image b onto image a, memoised by (a, b)
    index into `group`. A subset swap therefore only ever computes the genuinely new pairs."""
    if (a, b) not in pair_cache:
        initial_shift_half = _initial_shift_half(group[a], group[b])
        target_cart, target_mask = cleans[a]
        source_cart, source_mask = cleans[b]
        pair_cache[(a, b)] = stage1_grid_search(
            target_cart, target_mask, source_cart, source_mask, initial_shift_half, device
        )
    return pair_cache[(a, b)]


def _residual_px(res_i, res_j, res_rot_deg, lever_px) -> float:
    """Combine a translation+rotation residual into one pixel displacement. The rotation is
    about the moon centre (the polar transform centre), so a rotation residual dtheta displaces
    a point at radius `lever_px` by lever_px*dtheta. We use the moon radius as that lever: the
    inner limb of the registered corona annulus (a lower bound on rotation-induced misalignment)."""
    return max(abs(res_i), abs(res_j), lever_px * math.radians(abs(res_rot_deg)))


def _subset_discrepancy(subset, pair_cache, lever_px) -> tuple[float, dict]:
    """Self-consistency of a subset, in pixels. Discrepancy = max over (a) pair symmetry
    (T(a,b) + T(b,a) should vanish) and (b) triplet closure (T(a,b)∘T(b,c) should equal
    T(a,c)), each folded to px via _residual_px. Also returns a per-image score (sum of the
    residual px of every check the image takes part in) used to pick the worst image."""
    per_image = {i: 0.0 for i in subset}
    max_disc = 0.0
    for a, b in itertools.combinations(subset, 2):
        rab, rba = pair_cache[(a, b)], pair_cache[(b, a)]
        px = _residual_px(rab[0] + rba[0], rab[1] + rba[1], rab[2] + rba[2], lever_px)
        per_image[a] += px
        per_image[b] += px
        max_disc = max(max_disc, px)
    for a, b, c in itertools.permutations(subset, 3):
        rab, rbc, rac = pair_cache[(a, b)], pair_cache[(b, c)], pair_cache[(a, c)]
        comp = compose_transforms(rab[0], rab[1], rab[2], rbc[0], rbc[1], rbc[2])
        px = _residual_px(comp[0] - rac[0], comp[1] - rac[1], comp[2] - rac[2], lever_px)
        per_image[a] += px
        per_image[b] += px
        per_image[c] += px
        max_disc = max(max_disc, px)
    return max_disc, per_image


def _initial_subset_indices(timestamps, k) -> list:
    """k group indices spread evenly across the timestamp-sorted order (endpoints included),
    so the starting subset is well distributed in time. Rounding collisions are filled in."""
    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    n = len(order)
    picks = []
    for t in range(k):
        idx = order[round(t * (n - 1) / (k - 1))]
        if idx not in picks:
            picks.append(idx)
    for idx in order:  # fill if rounding produced fewer than k distinct picks
        if len(picks) >= k:
            break
        if idx not in picks:
            picks.append(idx)
    return picks


def _select_consistent_subset(group, cleans, pair_cache, device, lever_px) -> list:
    """Greedily find MAX_IMG group indices whose pairwise registration is self-consistent
    to <= SUBSET_THRESHOLD_PX. Start from a time-spread subset; while it fails the gate,
    drop the worst-scoring image and pull in the untried image closest in time to it. An
    image is only ever pulled in once, so this terminates after at most (n - MAX_IMG) swaps;
    if nothing passes, the smallest-discrepancy subset seen is returned."""
    n = len(group)
    timestamps = [ii.timestamp for ii in group]
    subset = _initial_subset_indices(timestamps, MAX_IMG)
    tried = set(subset)
    best_subset, best_disc = None, float("inf")
    while True:
        for a, b in itertools.permutations(subset, 2):
            _ensure_pair(pair_cache, group, cleans, a, b, device)
        disc, per_image = _subset_discrepancy(subset, pair_cache, lever_px)
        if disc < best_disc:
            best_disc, best_subset = disc, list(subset)
        if disc <= SUBSET_THRESHOLD_PX:
            print(f"  subset {sorted(subset)} disc={disc:.4f}px <= {SUBSET_THRESHOLD_PX} (pass)")
            return subset
        untried = [i for i in range(n) if i not in tried]
        if not untried:
            print(f"  subset search exhausted; best {sorted(best_subset)} disc={best_disc:.4f}px")
            return best_subset
        worst = max(subset, key=lambda i: per_image[i])
        repl = min(untried, key=lambda i: abs(timestamps[i] - timestamps[worst]))
        print(f"  subset {sorted(subset)} disc={disc:.4f}px: drop {worst}, add {repl}")
        subset = [i for i in subset if i != worst] + [repl]
        tried.add(repl)


def register_intra_exposure_pairs(exposure_groups: dict) -> dict:
    """For each exposure, all ordered pairs: two-stage Fourier-style registration on GPU.

    Stage 1 (always): per-image polar+FFT cleanup is hoisted out of the per-pair grid search
    (Fix 3 in v1/perf_analysis_register_intra_exposure_pairs.md). Each image is loaded and
    cleaned once per group; the grid search then compares pre-cleaned cartesian tensors.

    Subset capping: groups larger than MAX_IMG are reduced to a self-consistent MAX_IMG-subset
    (_select_consistent_subset) before the reg dict is written; exposure_groups is mutated in
    place to the kept frames so downstream sees a clean MAX_IMG-image group reindexed 0..k-1.

    Stage 2 (gated by ENABLE_STAGE_2): per-pair narrow-bracket finetune around Stage 1's
    result, using a common polar center C midway between the two moons. Cleanup re-runs
    per candidate around C; the moon-limb feature lands at near-identical (r, theta) in
    both images so it cancels in the L1 diff. Applied only to the final kept pairs.
    """
    device = torch.device("cuda")
    reg = {}
    for exposure_time in sorted(exposure_groups.keys()):
        print(f"Doing check for exposure time {exposure_time}")
        group = list(exposure_groups[exposure_time])
        n = len(group)
        if n < 2:
            print(f"  Skip (group size {n} < 2)")
            continue
        raws, cleans = _clean_group(group, device)
        # Lever arm for folding a rotation residual into pixels: the moon radius (the polar/rotation
        # centre is the moon centre, and moon_radius is the inner limb of the registered annulus).
        # Near-constant across a group; average it so one outlier moon fit doesn't skew the gate.
        lever_px = sum(ii.moon[2] for ii in group) / n
        pair_cache = {}  # (a, b) index pair -> stage-1 transform, shared across selection + finalize

        if n > MAX_IMG:
            chosen = _select_consistent_subset(group, cleans, pair_cache, device, lever_px)
        else:
            chosen = list(range(n))
        # Reindex kept frames in timestamp order so printed indices stay interpretable.
        chosen = sorted(chosen, key=lambda a: group[a].timestamp)

        final_group = [group[a] for a in chosen]
        exposure_groups[exposure_time] = final_group
        tasks = [
            (new_i, new_j, chosen[new_i], chosen[new_j])
            for new_i, new_j in itertools.permutations(range(len(chosen)), 2)
        ]
        for new_i, new_j, a, b in tqdm.tqdm(tasks, desc="Registration pairs"):
            T1 = _ensure_pair(pair_cache, group, cleans, a, b, device)
            if ENABLE_STAGE_2:
                T = stage2_finetune(raws[a], raws[b], group[a].moon, group[b].moon, T1, device)
            else:
                T = T1
            reg[(exposure_time, new_i, new_j)] = T
        _print_pair_consistency(reg, exposure_time, final_group)
        if len(final_group) >= 3:
            _print_triplet_consistency(reg, exposure_time, final_group)
    return reg


def save_pickle(exposure_groups: dict, reg: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    with open(out_path, "wb") as fd:
        pickle.dump(exposure_groups, fd)
        pickle.dump(reg, fd)
    print(f"Saved {out_path} (exposure_groups, reg)")
    return out_path


def run(data_root: str | Path, out_path: str | Path) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage 0.")
    out_path = Path(out_path)
    image_infos = get_image_infos(data_root)
    detect_moons(image_infos)
    exposure_groups = group_by_exposure(image_infos)
    prune_moon_info_for_radius_outliers(exposure_groups)
    interp = interpolate_missing_moons(image_infos, exposure_groups)
    set_moon_position_std(image_infos, exposure_groups, interp)
    reg = register_intra_exposure_pairs(exposure_groups)
    return save_pickle(exposure_groups, reg, out_path)
