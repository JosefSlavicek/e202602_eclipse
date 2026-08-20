"""Image-processing helpers shared across pipeline stages."""
import math

import numpy as np
import torch
import torchvision
from PIL import Image

from eclipse_v7.coords import cartesian_to_polar, polar_to_cartesian

GRID_BATCH_SIZE = 4


def load_grayscale(ii, device):
    """Load an image as a float32 [0,1] grayscale tensor, via the frame's input source.

    Routes through `ii.source` (NefSource). Falls back to a plain JPG/PNG-style decode when
    no source is attached, so a stray unpickle without `attach_source` still loads something.
    """
    if getattr(ii, "source", None) is not None:
        return ii.source.load_gray(ii, device)
    with Image.open(ii.path) as img:
        arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def apply_transform_single(img, shift_i, shift_j, angle_deg, device):
    """Warp a 2-D tensor by (shift_i, shift_j, angle_deg) around the image centre."""
    H, W = img.shape
    ci, cj = H / 2.0, W / 2.0
    angle_rad = math.radians(-angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
    jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
    di = ii - ci - shift_i
    dj = jj - cj - shift_j
    i_src = di * cos_a + dj * sin_a + ci
    j_src = -di * sin_a + dj * cos_a + cj
    j_norm = 2.0 * j_src / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_src)
    i_norm = 2.0 * i_src / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_src)
    grid = torch.stack([j_norm, i_norm], dim=-1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(
        img.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.squeeze(0).squeeze(0)


def compute_weighted_average(group, abs_xy, abs_angle_t, device, epsilon=1e-6, keep_warped=True):
    """Mask (moon blacked out), warp, and average a group of images.

    `keep_warped=False` returns an empty `warped_list`, so peak memory is two full-resolution
    tensors instead of one per frame in the group — pass it from callers that only want the
    average (the calibration and the radiometric merge do).

    Returns:
        avg_img      – weighted average image
        avg_mask     – coverage mask normalised by group size (sum_mask / n)
        warped_list  – per-image warped tensors, in group order (useful for debug GIFs)
    """
    n = len(group)
    sum_img = None
    sum_mask = None
    warped_list = []
    for j in range(n):
        img_j = load_grayscale(group[j], device)
        mj_i, mj_j, r_j = group[j].moon[0], group[j].moon[1], group[j].moon[2]
        H, W = img_j.shape
        ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
        jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
        dist = torch.sqrt((ii - mj_i) ** 2 + (jj - mj_j) ** 2)
        mask_j = (dist > r_j + 2.0).to(torch.float32)
        x_j = float(abs_xy[j, 0])
        y_j = float(abs_xy[j, 1])
        theta_j_deg = -math.degrees(float(abs_angle_t[j]))
        w_img = apply_transform_single(img_j, x_j, y_j, theta_j_deg, device)
        w_mask = apply_transform_single(mask_j, x_j, y_j, theta_j_deg, device)
        if keep_warped:
            warped_list.append(w_img)
        if sum_img is None:
            sum_img = w_img * w_mask
            sum_mask = w_mask.clone()
        else:
            sum_img = sum_img + w_img * w_mask
            sum_mask = sum_mask + w_mask
    avg_img = sum_img / (sum_mask + epsilon)
    avg_mask = sum_mask / n
    return avg_img, avg_mask, warped_list


def moon_median(group):
    """Median moon centre (i, j) and radius across a group of ImageInfo objects."""
    centers_i = [ii.moon[0] for ii in group]
    centers_j = [ii.moon[1] for ii in group]
    radii = [ii.moon[2] for ii in group]
    return (float(np.median(centers_i)), float(np.median(centers_j)), float(np.median(radii)))


def compose_transforms(s1_i, s1_j, rot1_deg, s2_i, s2_j, rot2_deg):
    """Compose transform (s1, rot1) then (s2, rot2). Returns (s_i, s_j, rot_deg)."""
    theta1_rad = math.radians(rot1_deg)
    cos1, sin1 = math.cos(theta1_rad), math.sin(theta1_rad)
    s_rot_i = cos1 * s2_i - sin1 * s2_j
    s_rot_j = sin1 * s2_i + cos1 * s2_j
    return (s1_i + s_rot_i, s1_j + s_rot_j, rot1_deg + rot2_deg)


def remove_lowfeq(polar_image, n_remove: int):
    """Zero the lowest n_remove Fourier frequencies along the angular axis of a polar image."""
    assert polar_image.ndim == 2
    W = polar_image.shape[1]
    spec = torch.fft.rfft(polar_image, dim=-1)
    spec[:, :n_remove] = 0.0
    out = torch.fft.irfft(spec, n=W, dim=-1)
    return out.to(polar_image.dtype)


def fill_bottom(polar_img, n_step: int):
    """Clip the innermost n_step radial rows of a polar image to their column-wise maximum."""
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
    """Fill the moon disk in `img` with `value`."""
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
    """Apply Gaussian blur with the given sigma to a 2-D tensor."""
    assert img.ndim == 2
    assert sigma > 0
    sigma = int(np.ceil(sigma))
    kernel_size = min(17, 4 * sigma + 1)
    gblur = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size, sigma=sigma
    ).to(img.device)
    return gblur(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def transform_moon_center_batched(
    moon_center_i, moon_center_j, ci, cj, shift_i_t, shift_j_t, cos_a_t, sin_a_t
):
    """Apply a batch of N transforms to a single moon centre point.

    Returns a tuple of N (i, j) float pairs.
    """
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


def apply_transform_batched(img, shift_i_t, shift_j_t, cos_a_t, sin_a_t):
    """Warp a (H, W) tensor by a batch of N transforms; returns (N, H, W).

    shift_i_t, shift_j_t, cos_a_t, sin_a_t must be (N, 1, 1) tensors on the same device as img.
    """
    H, W = img.shape
    device = img.device
    ci, cj = H / 2.0, W / 2.0
    ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W).unsqueeze(0)
    jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W).unsqueeze(0)
    di = ii - ci - shift_i_t
    dj = jj - cj - shift_j_t
    i_src = di * cos_a_t + dj * sin_a_t + ci
    j_src = -di * sin_a_t + dj * cos_a_t + cj
    j_norm = 2.0 * j_src / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_src)
    i_norm = 2.0 * i_src / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_src)
    grid = torch.stack([j_norm, i_norm], dim=-1)
    img_4d = img.unsqueeze(0).unsqueeze(1).expand(grid.shape[0], 1, H, W)
    return torch.nn.functional.grid_sample(
        img_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(1)


def clean_polar_fft(
    img,
    moon_center,
    moon_radius,
    *,
    polar_center=None,
    valid_row_mask=None,
    antiprot=None,
):
    """Polar+FFT cleanup pipeline. Returns (cleaned_cart, cart_mask, antiprot_used).

    Unwraps the corona around `polar_center` (defaults to `moon_center`), clips
    protuberance peaks at `antiprot`, suppresses the moon-limb sharp step via
    `fill_bottom`, strips low angular frequencies, and maps back to cartesian.
    The cartesian mask zeros the moon disk via `fill_moon(moon_center, moon_radius)`.

    `polar_center` overrides `moon_center` for the polar transform itself; this is
    what Stage 2's common-center pipeline uses. `fill_moon` always uses the image's
    actual moon, not `polar_center`.

    `valid_row_mask` is an optional (n_r,) float tensor multiplied row-wise into both
    the polar image and its companion ones-mask before `polar_to_cartesian` — used
    by Stage 2 to exclude radii whose circle crosses either moon.

    `antiprot` shares a single anti-protuberance threshold across multiple images of
    the same group/pair: pass `None` for the first call, then re-use the returned value.
    """
    assert img.ndim == 2
    polar_ctr = polar_center if polar_center is not None else moon_center
    radius_max = min(
        polar_ctr[0],
        img.shape[0] - polar_ctr[0],
        polar_ctr[1],
        img.shape[1] - polar_ctr[1],
    )
    assert 32 < moon_radius < radius_max - 32
    H_img, W_img = img.shape[0], img.shape[1]
    n_r = int(radius_max - moon_radius + 1)
    n_theta = int(2 * math.pi * radius_max)
    polar_img, _ = cartesian_to_polar(img, polar_ctr, moon_radius, radius_max, n_r, n_theta)
    if antiprot is None:
        maxidx = polar_img.sum(dim=1).argmax()
        maxrow = polar_img[maxidx, :]
        maxrow = maxrow[maxrow > 0]
        antiprot = maxrow.quantile(0.9)
    polar_img[polar_img > antiprot] = antiprot
    polar_img = fill_bottom(polar_img, 4)
    polar_img = remove_lowfeq(polar_img, 16)
    polar_mask = torch.ones_like(polar_img)
    if valid_row_mask is not None:
        polar_img = polar_img * valid_row_mask.view(-1, 1)
        polar_mask = polar_mask * valid_row_mask.view(-1, 1)
    cleaned_cart = polar_to_cartesian(polar_img, polar_ctr, moon_radius, radius_max, H_img, W_img)
    cart_mask = polar_to_cartesian(polar_mask, polar_ctr, moon_radius, radius_max, H_img, W_img)
    cart_mask = fill_moon(cart_mask, moon_center, moon_radius, 0.0)
    return cleaned_cart, cart_mask, antiprot


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
    apriori_valid=None,
):
    """Score a batch of candidate warps against a target image using Fourier-filtered polar residuals.

    apriori_valid: optional (H, W) float mask applied to the target before comparison
                   (use this to down-weight saturated pixels in cross-exposure registration).
                   Defaults to all-ones (no masking).

    Mutates `batch` in-place (tournament-style pruning). Returns (shift_i, shift_j, angle, img, mask)
    for the best candidate.
    """
    assert target.ndim == 2
    assert warped.ndim == 3
    assert len(moon_center_target) == 2
    assert len(moon_centers_warped) == len(warped)
    assert moon_radius_target > 0
    assert moon_radius_warped > 0
    assert len(batch) == len(warped)
    list_of_all = [(target, moon_center_target, moon_radius_target)]
    for i in range(len(moon_centers_warped)):
        list_of_all.append((warped[i], moon_centers_warped[i], moon_radius_warped))
    list_of_all_processed = []
    antiprot = None
    for img, moon_center, moon_radius in list_of_all:
        cleaned_cart, cart_mask, antiprot = clean_polar_fft(
            img, moon_center, moon_radius, antiprot=antiprot
        )
        if blur_sigma > 0:
            cleaned_cart = gaussian_blur(cleaned_cart, blur_sigma)
        list_of_all_processed.append((cleaned_cart, cart_mask))
    target_img, target_mask = list_of_all_processed.pop(0)
    if apriori_valid is not None:
        target_mask = target_mask * apriori_valid.to(target_mask.device)
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
        diff_0 = ((torch.abs(target_img - warped_img_0) * mask).sum() / (mask.sum() + 1e-9)).item()
        diff_1 = ((torch.abs(target_img - warped_img_1) * mask).sum() / (mask.sum() + 1e-9)).item()
        if diff_0 < diff_1:
            batch.append((shift_i_0, shift_j_0, angle_0))
            list_of_all_processed.append((warped_img_0, warped_mask_0))
        else:
            batch.append((shift_i_1, shift_j_1, angle_1))
            list_of_all_processed.append((warped_img_1, warped_mask_1))
    shift_i, shift_j, angle = batch.pop()
    warped_img, warped_mask = list_of_all_processed.pop()
    return shift_i, shift_j, angle, warped_img, warped_mask


def grid_search_registration(g0, g1, moon0, moon1, initial_shift_half, device, apriori_valid=None):
    """Grid-search (shift_i, shift_j, angle_deg) minimising Fourier discrepancy between g0 and g1.

    Iteratively refines a 5×5×5 grid, halving step sizes until both shift and angular steps
    fall below their pixel-resolution thresholds. Returns (shift_i, shift_j, angle) as floats.
    """
    H, W = g0.shape
    r_border = max(H, W)
    ci, cj = H / 2.0, W / 2.0
    r0, r1 = moon0[2], moon1[2]
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
        blur_sigma = 0.0 if (step_shift < 2 and step_angle_in_px < 2) else min(8, max(step_angle_in_px, step_shift))
        shift_i_vals = [best_shift_i] if not refine_shift else [best_shift_i + step_shift * (k - 2) for k in range(5)]
        shift_j_vals = [best_shift_j] if not refine_shift else [best_shift_j + step_shift * (k - 2) for k in range(5)]
        angle_vals = [best_angle] if not refine_angle else [best_angle + step_angle * (k - 2) for k in range(5)]
        triples = [(si, sj, a) for si in shift_i_vals for sj in shift_j_vals for a in angle_vals]
        for start in range(0, len(triples), GRID_BATCH_SIZE):
            batch = triples[start : start + GRID_BATCH_SIZE]
            N = len(batch)
            shift_i_t = torch.tensor([t[0] for t in batch], device=device, dtype=torch.float32).view(N, 1, 1)
            shift_j_t = torch.tensor([t[1] for t in batch], device=device, dtype=torch.float32).view(N, 1, 1)
            cos_a_t = torch.cos(
                torch.tensor([math.radians(-t[2]) for t in batch], device=device, dtype=torch.float32)
            ).view(N, 1, 1)
            sin_a_t = torch.sin(
                torch.tensor([math.radians(-t[2]) for t in batch], device=device, dtype=torch.float32)
            ).view(N, 1, 1)
            warped = apply_transform_batched(g1, shift_i_t, shift_j_t, cos_a_t, sin_a_t)
            moon_centers_warped = transform_moon_center_batched(
                moon1[0], moon1[1], ci, cj, shift_i_t, shift_j_t, cos_a_t, sin_a_t
            )
            best_setup = discrepancy_batched_fourier3(
                g0,
                warped,
                moon0[:2],
                r0,
                moon_centers_warped,
                r1,
                list(batch),
                blur_sigma,
                best_setup,
                apriori_valid,
            )
        best_shift_i, best_shift_j, best_angle, _, _ = best_setup
        is_corner = len(shift_i_vals) > 2 and (
            best_shift_i in [shift_i_vals[0], shift_i_vals[-1]] or best_shift_j in [shift_j_vals[0], shift_j_vals[-1]]
        )
        step_shift = step_shift / 2.0 if (refine_shift and not is_corner) else step_shift
        is_corner = len(angle_vals) > 2 and best_angle in [angle_vals[0], angle_vals[-1]]
        step_angle = step_angle / 2.0 if (refine_angle and not is_corner) else step_angle
    return (float(best_shift_i), float(best_shift_j), float(best_angle))


def compare_pre_cleaned_batch(
    target_cart,
    target_mask,
    source_cart,
    source_mask,
    shift_i_t,
    shift_j_t,
    cos_a_t,
    sin_a_t,
    batch,
    blur_sigma,
    best_setup,
):
    """Warp pre-cleaned source by a batch of candidates, masked-L1 vs cleaned target, tournament reduce.

    The source and target have been cleaned around their own native moons by `clean_polar_fft`
    (see Fix 3 in perf_analysis_register_intra_exposure_pairs.md): polar-around-moon commutes
    with cartesian translation of the source (the moon center moves with the image), and the
    angular FFT cleanup is theta-shift invariant, so the per-candidate polar+FFT pipeline
    reduces to a single per-image cleanup followed by per-candidate `grid_sample`.

    Mutates `batch` in-place. Returns (shift_i, shift_j, angle, img, mask) of the best candidate.
    """
    assert target_cart.ndim == 2 and target_mask.ndim == 2
    assert source_cart.ndim == 2 and source_mask.ndim == 2
    assert len(batch) == shift_i_t.shape[0]
    warped_imgs = apply_transform_batched(source_cart, shift_i_t, shift_j_t, cos_a_t, sin_a_t)
    warped_masks = apply_transform_batched(source_mask, shift_i_t, shift_j_t, cos_a_t, sin_a_t)
    if blur_sigma > 0:
        target_cart_use = gaussian_blur(target_cart, blur_sigma)
    else:
        target_cart_use = target_cart
    list_of_all_processed = []
    for k in range(len(batch)):
        wi = gaussian_blur(warped_imgs[k], blur_sigma) if blur_sigma > 0 else warped_imgs[k]
        list_of_all_processed.append((wi, warped_masks[k]))
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
        diff_0 = ((torch.abs(target_cart_use - warped_img_0) * mask).sum() / (mask.sum() + 1e-9)).item()
        diff_1 = ((torch.abs(target_cart_use - warped_img_1) * mask).sum() / (mask.sum() + 1e-9)).item()
        if diff_0 < diff_1:
            batch.append((shift_i_0, shift_j_0, angle_0))
            list_of_all_processed.append((warped_img_0, warped_mask_0))
        else:
            batch.append((shift_i_1, shift_j_1, angle_1))
            list_of_all_processed.append((warped_img_1, warped_mask_1))
    shift_i, shift_j, angle = batch.pop()
    warped_img, warped_mask = list_of_all_processed.pop()
    return shift_i, shift_j, angle, warped_img, warped_mask


def stage1_grid_search(
    target_cart,
    target_mask,
    source_cart,
    source_mask,
    initial_shift_half,
    device,
):
    """Stage 1: refinement-loop grid search on pre-cleaned tensors.

    Same iterative 5x5x5 / halving schedule as `grid_search_registration`, but operates on
    pre-cleaned cartesian tensors (per-image cleanup hoisted upstream — see Fix 3 in
    perf_analysis_register_intra_exposure_pairs.md). Initial angle step is 0.5 deg
    (vs the legacy 5.0 deg): intra-exposure rotation between consecutive frames is small,
    so the wide initial bracket was waste.
    """
    H, W = target_cart.shape
    r_border = max(H, W)
    best_shift_i, best_shift_j, best_angle = 0.0, 0.0, 0.0
    step_shift = initial_shift_half / 2.0
    step_angle = 0.5
    refine_shift = refine_angle = True
    best_setup = None
    while refine_shift or refine_angle:
        if step_shift < 0.1:
            refine_shift = False
        step_angle_in_px = step_angle * r_border * (math.pi / 180.0)
        if step_angle_in_px < 0.1:
            refine_angle = False
        blur_sigma = 0.0 if (step_shift < 2 and step_angle_in_px < 2) else min(8, max(step_angle_in_px, step_shift))
        shift_i_vals = [best_shift_i] if not refine_shift else [best_shift_i + step_shift * (k - 2) for k in range(5)]
        shift_j_vals = [best_shift_j] if not refine_shift else [best_shift_j + step_shift * (k - 2) for k in range(5)]
        angle_vals = [best_angle] if not refine_angle else [best_angle + step_angle * (k - 2) for k in range(5)]
        triples = [(si, sj, a) for si in shift_i_vals for sj in shift_j_vals for a in angle_vals]
        for start in range(0, len(triples), GRID_BATCH_SIZE):
            batch = triples[start : start + GRID_BATCH_SIZE]
            N = len(batch)
            shift_i_t = torch.tensor([t[0] for t in batch], device=device, dtype=torch.float32).view(N, 1, 1)
            shift_j_t = torch.tensor([t[1] for t in batch], device=device, dtype=torch.float32).view(N, 1, 1)
            cos_a_t = torch.cos(
                torch.tensor([math.radians(-t[2]) for t in batch], device=device, dtype=torch.float32)
            ).view(N, 1, 1)
            sin_a_t = torch.sin(
                torch.tensor([math.radians(-t[2]) for t in batch], device=device, dtype=torch.float32)
            ).view(N, 1, 1)
            best_setup = compare_pre_cleaned_batch(
                target_cart,
                target_mask,
                source_cart,
                source_mask,
                shift_i_t, shift_j_t, cos_a_t, sin_a_t,
                list(batch),
                blur_sigma,
                best_setup,
            )
        best_shift_i, best_shift_j, best_angle, _, _ = best_setup
        is_corner = len(shift_i_vals) > 2 and (
            best_shift_i in [shift_i_vals[0], shift_i_vals[-1]] or best_shift_j in [shift_j_vals[0], shift_j_vals[-1]]
        )
        step_shift = step_shift / 2.0 if (refine_shift and not is_corner) else step_shift
        is_corner = len(angle_vals) > 2 and best_angle in [angle_vals[0], angle_vals[-1]]
        step_angle = step_angle / 2.0 if (refine_angle and not is_corner) else step_angle
    return (float(best_shift_i), float(best_shift_j), float(best_angle))


def _apply_transform_to_point(pi, pj, shift_i, shift_j, angle_deg, ci, cj):
    """Forward transform of a single (i, j) point, matching apply_transform_batched's convention."""
    di = pi - ci
    dj = pj - cj
    cos_a = math.cos(math.radians(-angle_deg))
    sin_a = math.sin(math.radians(-angle_deg))
    di_rot = di * cos_a - dj * sin_a
    dj_rot = di * sin_a + dj * cos_a
    return (di_rot + shift_i + ci, dj_rot + shift_j + cj)


def _compare_batch_common_C(
    warped,
    moon_centers_warped,
    moon_radius_warped,
    target_cart,
    target_mask,
    polar_center,
    valid_row_mask,
    antiprot,
    batch,
    blur_sigma,
    best_setup,
):
    """Stage 2 per-candidate clean+compare: re-clean each warped image around the common C, masked L1 vs once-cleaned target.

    The target was cleaned once per pair (does not change with the candidate). The warped
    source is re-cleaned per candidate because polar-around-C does not commute with
    cartesian translation of the source (unlike Stage 1's moon-locked polar), so the warped
    moon position changing within the search bound matters.
    """
    target_cart_use = gaussian_blur(target_cart, blur_sigma) if blur_sigma > 0 else target_cart
    list_of_all_processed = []
    for k in range(len(warped)):
        cleaned_cart, cart_mask, _ = clean_polar_fft(
            warped[k],
            moon_centers_warped[k],
            moon_radius_warped,
            polar_center=polar_center,
            valid_row_mask=valid_row_mask,
            antiprot=antiprot,
        )
        if blur_sigma > 0:
            cleaned_cart = gaussian_blur(cleaned_cart, blur_sigma)
        list_of_all_processed.append((cleaned_cart, cart_mask))
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
        diff_0 = ((torch.abs(target_cart_use - warped_img_0) * mask).sum() / (mask.sum() + 1e-9)).item()
        diff_1 = ((torch.abs(target_cart_use - warped_img_1) * mask).sum() / (mask.sum() + 1e-9)).item()
        if diff_0 < diff_1:
            batch.append((shift_i_0, shift_j_0, angle_0))
            list_of_all_processed.append((warped_img_0, warped_mask_0))
        else:
            batch.append((shift_i_1, shift_j_1, angle_1))
            list_of_all_processed.append((warped_img_1, warped_mask_1))
    shift_i, shift_j, angle = batch.pop()
    warped_img, warped_mask = list_of_all_processed.pop()
    return shift_i, shift_j, angle, warped_img, warped_mask


def stage2_finetune(g0, g1, moon0, moon1, T1, device):
    """Stage 2: narrow-bracket finetune of T1 with a common per-pair polar center C.

    C is the midpoint between moon_i and the Stage-1-aligned moon_j position. Cleanup
    re-runs per candidate (warp first, then polar around C, then FFT clean) because
    polar-around-C is not translation-invariant in the source — the warped moon position
    drifts within the search band.

    Symmetric placement of C makes the moon-limb feature land at near-identical (r, theta)
    in both images so it cancels in the L1 diff. The 1-D row mask additionally zeros
    out polar rows whose circle around C would cross either moon disk, plus a 2 px
    margin for the candidate band.

    Returns the refined (shift_i, shift_j, angle), replacing T1.
    """
    shift_i_1, shift_j_1, angle_1 = T1
    H, W = g0.shape
    ci, cj = H / 2.0, W / 2.0
    moon_j_aligned = _apply_transform_to_point(
        moon1[0], moon1[1], shift_i_1, shift_j_1, angle_1, ci, cj
    )
    C = (0.5 * (moon0[0] + moon_j_aligned[0]), 0.5 * (moon0[1] + moon_j_aligned[1]))
    R_i, R_j = moon0[2], moon1[2]
    R_moon = 0.5 * (R_i + R_j)
    moon_radius_common = max(R_i, R_j)
    d_i = math.hypot(C[0] - moon0[0], C[1] - moon0[1])
    d_j = math.hypot(C[0] - moon_j_aligned[0], C[1] - moon_j_aligned[1])
    candidate_shift_margin = 2.0
    r_min_valid = max(d_i + R_i, d_j + R_j) + candidate_shift_margin
    radius_max = min(C[0], H - C[0], C[1], W - C[1])
    n_r = int(radius_max - moon_radius_common + 1)
    y = torch.arange(n_r, device=device, dtype=torch.float32)
    r_per_row = radius_max - y * (radius_max - moon_radius_common) / max(n_r - 1, 1)
    valid_row_mask = (r_per_row >= r_min_valid).to(torch.float32)
    target_cart, target_mask, antiprot = clean_polar_fft(
        g0, moon0[:2], moon_radius_common,
        polar_center=C,
        valid_row_mask=valid_row_mask,
    )
    best_shift_i, best_shift_j, best_angle = float(shift_i_1), float(shift_j_1), float(angle_1)
    step_shift = 0.5
    step_angle = 0.5 * math.degrees(1.0 / R_moon)
    refine_shift = refine_angle = True
    best_setup = None
    r_border = max(H, W)
    while refine_shift or refine_angle:
        if step_shift < 0.1:
            refine_shift = False
        step_angle_in_px = step_angle * r_border * (math.pi / 180.0)
        if step_angle_in_px < 0.1:
            refine_angle = False
        blur_sigma = 0.0 if (step_shift < 2 and step_angle_in_px < 2) else min(8, max(step_angle_in_px, step_shift))
        shift_i_vals = [best_shift_i] if not refine_shift else [best_shift_i + step_shift * (k - 2) for k in range(5)]
        shift_j_vals = [best_shift_j] if not refine_shift else [best_shift_j + step_shift * (k - 2) for k in range(5)]
        angle_vals = [best_angle] if not refine_angle else [best_angle + step_angle * (k - 2) for k in range(5)]
        triples = [(si, sj, a) for si in shift_i_vals for sj in shift_j_vals for a in angle_vals]
        for start in range(0, len(triples), GRID_BATCH_SIZE):
            batch = triples[start : start + GRID_BATCH_SIZE]
            N = len(batch)
            shift_i_t = torch.tensor([t[0] for t in batch], device=device, dtype=torch.float32).view(N, 1, 1)
            shift_j_t = torch.tensor([t[1] for t in batch], device=device, dtype=torch.float32).view(N, 1, 1)
            cos_a_t = torch.cos(
                torch.tensor([math.radians(-t[2]) for t in batch], device=device, dtype=torch.float32)
            ).view(N, 1, 1)
            sin_a_t = torch.sin(
                torch.tensor([math.radians(-t[2]) for t in batch], device=device, dtype=torch.float32)
            ).view(N, 1, 1)
            warped = apply_transform_batched(g1, shift_i_t, shift_j_t, cos_a_t, sin_a_t)
            moon_centers_warped = transform_moon_center_batched(
                moon1[0], moon1[1], ci, cj, shift_i_t, shift_j_t, cos_a_t, sin_a_t
            )
            best_setup = _compare_batch_common_C(
                warped,
                moon_centers_warped,
                moon_radius_common,
                target_cart,
                target_mask,
                C,
                valid_row_mask,
                antiprot,
                list(batch),
                blur_sigma,
                best_setup,
            )
        best_shift_i, best_shift_j, best_angle, _, _ = best_setup
        is_corner = len(shift_i_vals) > 2 and (
            best_shift_i in [shift_i_vals[0], shift_i_vals[-1]] or best_shift_j in [shift_j_vals[0], shift_j_vals[-1]]
        )
        step_shift = step_shift / 2.0 if (refine_shift and not is_corner) else step_shift
        is_corner = len(angle_vals) > 2 and best_angle in [angle_vals[0], angle_vals[-1]]
        step_angle = step_angle / 2.0 if (refine_angle and not is_corner) else step_angle
    return (float(best_shift_i), float(best_shift_j), float(best_angle))
