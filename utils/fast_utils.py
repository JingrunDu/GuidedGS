import torch
import numpy as np
from PIL import ImageFilter
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random

def sampling_cameras(my_viewpoint_stack):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    num_cams = 10
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    
    return camlist


def sampling_cameras_stratified(my_viewpoint_stack):
    """Stratified viewpoint sampling for multi-view consistency scoring.

    Instead of purely random sampling, cameras are grouped into 4 elevation
    strata and at least 2 cameras are drawn from each stratum.  This ensures
    that under-represented viewing angles (e.g. oblique views with only 1-2%
    of total frames) still participate in every densification scoring round.

    The remaining budget (10 - 4*2 = 2) is filled randomly from any stratum.

    Elevation boundaries are chosen based on the v3 dataset analysis:
      [-90, -15): up-looking (27.7%)
      [-15, +15): level (44.5%)
      [+15, +45): oblique-down (1.2% — the critical gap)
      [+45, +90): down-looking (26.6%)
    """
    num_cams = 10
    per_stratum = 2
    elevation_bins = [-90.0, -15.0, 15.0, 45.0, 90.0]
    num_strata = len(elevation_bins) - 1

    # Partition cameras into elevation strata
    strata = [[] for _ in range(num_strata)]
    for idx, cam in enumerate(my_viewpoint_stack):
        elevation = getattr(cam, 'elevation', 0.0)
        for stratum_idx in range(num_strata):
            if elevation_bins[stratum_idx] <= elevation < elevation_bins[stratum_idx + 1]:
                strata[stratum_idx].append(idx)
                break
        else:
            # Edge case: elevation == 90.0 exactly → last stratum
            strata[-1].append(idx)

    camlist = []
    used_indices = set()
    extra_budget = 0

    # Draw per_stratum cameras from each stratum
    for stratum_idx in range(num_strata):
        pool = strata[stratum_idx]
        available = min(per_stratum, len(pool))
        if available > 0:
            chosen = random.sample(pool, available)
            for idx in chosen:
                camlist.append(my_viewpoint_stack[idx])
                used_indices.add(idx)
        # Shortfall from empty/tiny strata goes to random budget
        extra_budget += per_stratum - available

    # Fill remaining budget randomly from unused cameras
    remaining_indices = [i for i in range(len(my_viewpoint_stack)) if i not in used_indices]
    num_random = num_cams - len(camlist)
    if remaining_indices and num_random > 0:
        extra = random.sample(remaining_indices, min(num_random, len(remaining_indices)))
        for idx in extra:
            camlist.append(my_viewpoint_stack[idx])

    return camlist


def build_camera_clusters(cameras, num_clusters=20):
    """Pre-compute spatial clusters of cameras using KMeans on camera positions.

    Groups cameras by their 3D world position so that cameras in the same
    cluster observe roughly the same region of the scene.  Called once at the
    start of training; the returned labels are reused for every densification
    round.

    Args:
        cameras: list of Camera objects (must have `.camera_center` attribute,
            a [3] tensor on CPU or CUDA).
        num_clusters: number of clusters (K).

    Returns:
        cluster_labels: np.ndarray of shape [N] with cluster id per camera.
        cluster_centers: np.ndarray of shape [K, 3] with cluster centroids.
    """
    from sklearn.cluster import KMeans

    positions = np.stack([
        cam.camera_center.cpu().numpy() if torch.is_tensor(cam.camera_center)
        else np.array(cam.camera_center)
        for cam in cameras
    ])  # [N, 3]

    effective_k = min(num_clusters, len(cameras))
    kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
    kmeans.fit(positions)

    print(f"[Camera clustering] {len(cameras)} cameras -> {effective_k} clusters, "
          f"sizes: {np.bincount(kmeans.labels_).tolist()}")

    return kmeans.labels_, kmeans.cluster_centers_


def sampling_cameras_clustered(viewpoint_stack, cluster_labels, cluster_id, num_cams=10):
    """Sample cameras from a specific spatial cluster for focused densification.

    Strategy A: pick all cameras from the given cluster (up to num_cams).
    If the cluster has fewer than num_cams cameras, fill the rest from
    the nearest clusters.

    Args:
        viewpoint_stack: list of all training Camera objects (NOT modified).
        cluster_labels: np.ndarray of cluster ids per camera (same order as
            the original training camera list).
        cluster_id: which cluster to focus on this round.
        num_cams: total number of cameras to sample.

    Returns:
        camlist: list of num_cams Camera objects.
        sampled_indices: list of integer indices into viewpoint_stack.
    """
    # Indices belonging to the target cluster
    target_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

    if len(target_indices) >= num_cams:
        chosen = random.sample(target_indices, num_cams)
    else:
        # Take all from target cluster, fill remainder from other cameras
        chosen = list(target_indices)
        other_indices = [i for i, label in enumerate(cluster_labels) if label != cluster_id]
        remaining = num_cams - len(chosen)
        if len(other_indices) >= remaining:
            chosen += random.sample(other_indices, remaining)
        else:
            chosen += other_indices

    camlist = [viewpoint_stack[i] for i in chosen]
    return camlist, chosen


def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    # Guard against constant L1 maps (max==min) which would yield nan and
    # propagate into pruning_score -> torch.multinomial crash (e.g. the
    # W_equip=1.0 case). Clamp the denominator to a small positive value.
    l1_min = torch.min(l1_loss)
    denom = (torch.max(l1_loss) - l1_min).clamp(min=1e-6)
    l1_loss_norm = (l1_loss - l1_min) / denom

    return l1_loss_norm

def _blur_gt(gt_image, sigma):
    """Curriculum Gaussian blur for the GT used in importance_score evaluation.

    Mirrors train.curriculum_blur so the densification score sees the same
    coarse-to-fine GT as the photometric loss (paper §3.4). sigma<=1e-3 -> no
    blur (cheap early-exit; honors the full linear-to-0 Phase-2 decay).
    """
    if sigma <= 1e-3:
        return gt_image
    import math
    import torch.nn.functional as F
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    x = torch.arange(kernel_size, dtype=torch.float32, device=gt_image.device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(3, 1, 1, 1)
    kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(3, 1, 1, 1)
    pad = kernel_size // 2
    img = gt_image.unsqueeze(0)
    img = F.pad(img, [pad, pad, 0, 0], mode='reflect')
    img = F.conv2d(img, kernel_h, groups=3)
    img = F.pad(img, [0, 0, pad, pad], mode='reflect')
    img = F.conv2d(img, kernel_v, groups=3)
    return img.squeeze(0)

def compute_photometric_loss(viewpoint_cam, image, gt_image_override=None):
    gt_image = gt_image_override if gt_image_override is not None else viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_value)
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY = False, curriculum_sigma=0.0):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]

        gt_image = my_viewpoint_cam.original_image.cuda()
        # Curriculum (§3.4): blur the GT used for importance_score evaluation
        # too, so early densification responds only to low-frequency signal
        # (matching the photometric loss path). _blur_gt is a no-op for
        # sigma<=1e-3 (curriculum off / Phase 3), so call it unconditionally.
        gt_image = _blur_gt(gt_image, curriculum_sigma)

        # Apply mask to render_image so masked regions don't produce false high-error signals
        if hasattr(my_viewpoint_cam, 'gt_alpha_mask') and my_viewpoint_cam.gt_alpha_mask is not None:
            cam_mask = my_viewpoint_cam.gt_alpha_mask.cuda()
            masked_render = render_image * cam_mask
        else:
            cam_mask = None
            masked_render = render_image

        photometric_loss = compute_photometric_loss(my_viewpoint_cam, masked_render, gt_image_override=gt_image)

        get_flag = True
        l1_loss_norm = get_loss(masked_render, gt_image)
        
        if getattr(args, 'use_residual_energy_gate', False):
            # Residual-energy: quantize L1 to integer energy so importance_score
            # accumulates residual ENERGY (magnitude) not just high-error pixel COUNT.
            _Q = getattr(args, 'energy_Q', 32)
            metric_map = (l1_loss_norm * _Q).floor().clamp(max=_Q).int()
        else:
            metric_map = (l1_loss_norm > args.loss_thresh).int()
        # Zero out metric_map in masked regions to prevent false prune signals
        if cam_mask is not None:
            # cam_mask is [1,H,W] or [3,H,W]; metric_map is [H,W]
            mask_2d = cam_mask[0] if cam_mask.dim() == 3 else cam_mask
            metric_map = metric_map * mask_2d.int()

        # Semantic densification bias: amplify the high-error count in equipment
        # regions so equipment Gaussians get a higher importance_score and are
        # densified more aggressively (and pruned less). This is the geometric
        # half of semantic guidance, complementing the loss weighting. The
        # presence of a semantic_weight on the camera indicates semantic mode is
        # on, so no separate flag lookup is needed here.
        if (hasattr(my_viewpoint_cam, 'semantic_weight')
                and my_viewpoint_cam.semantic_weight is not None):
            sem = my_viewpoint_cam.semantic_weight.cuda()
            sem_2d = sem[0] if sem.dim() == 3 else sem
            # Use an integer factor so metric_map stays Int; the CUDA rasterizer
            # requires metric_map to be of scalar type Int (float -> runtime error).
            equip_factor = int(round(getattr(args, 'densify_equipment_factor', 2.0)))
            # Equipment region from the raw class mask (equipment_mask), NOT from
            # (sem > bg_w): the old threshold coupled region detection to loss-weight
            # magnitudes, breaking for W_equip <= W_bg (no-op / reverse experiments).
            if hasattr(my_viewpoint_cam, 'equipment_mask') and my_viewpoint_cam.equipment_mask is not None:
                em = my_viewpoint_cam.equipment_mask.cuda()
                equip_region = (em[0] if em.dim() == 3 else em).int()
            else:
                bg_w = getattr(args, 'semantic_background_weight', 1.0)
                equip_region = (sem_2d > bg_w).int()  # legacy fallback
            metric_map = (metric_map * (1 + (equip_factor - 1) * equip_region)).int()

        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score
