#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import numpy as np
import os, random, time, csv, json
from random import randint
from lpipsPyTorch import lpips, LPIPS
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import (compute_gaussian_score_fastgs, sampling_cameras,
                              sampling_cameras_stratified, build_camera_clusters,
                              sampling_cameras_clustered)


def curriculum_blur(gt_image, sigma):
    """Apply Gaussian blur to GT image for curriculum training.

    During early training, a blurred GT guides Gaussians to establish
    coarse geometry first.  As sigma decreases to 0 the original GT is
    recovered, allowing high-frequency refinement.

    Args:
        gt_image: [3, H, W] tensor on CUDA
        sigma: current blur sigma (<=1e-3 means skip)

    Returns:
        blurred (or original) [3, H, W] tensor
    """
    if sigma <= 1e-3:
        return gt_image
    import math
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    # Build 1-D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=gt_image.device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    # Separable 2-D convolution (depthwise, per-channel)
    kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(3, 1, 1, 1)  # [3,1,1,K]
    kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(3, 1, 1, 1)  # [3,1,K,1]
    pad = kernel_size // 2
    img = gt_image.unsqueeze(0)  # [1,3,H,W]
    img = F.pad(img, [pad, pad, 0, 0], mode='reflect')
    img = F.conv2d(img, kernel_h, groups=3)
    img = F.pad(img, [0, 0, pad, pad], mode='reflect')
    img = F.conv2d(img, kernel_v, groups=3)
    return img.squeeze(0)


def compute_curriculum_sigma(iteration, sigma_0, T1, T2, schedule="linear"):
    """Compute the current blur sigma based on training phase.

    Args:
        iteration: current training iteration
        sigma_0: initial blur sigma
        T1: end of Phase 1 (constant blur)
        T2: end of Phase 2 (blur decays to 0)
        schedule: "linear", "cosine", or "exponential"

    Returns:
        sigma: current blur sigma (0 means no blur)
    """
    if sigma_0 <= 0 or T1 <= 0 or T2 <= T1:
        return 0.0
    if iteration <= T1:
        return sigma_0
    if iteration > T2:
        return 0.0
    progress = (iteration - T1) / (T2 - T1)
    if schedule == "cosine":
        import math
        return sigma_0 * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == "exponential":
        import math
        return sigma_0 * math.exp(-5.0 * progress)
    else:  # linear
        return sigma_0 * (1.0 - progress)


def compute_adaptive_thresholds(iteration, total_iterations, gaussian_count, args):
    """Compute adaptive clone/split threshold factors for densification.

    Returns (clone_factor, split_factor) where:
      - factor < 1.0 -> threshold lowered -> easier to densify
      - factor > 1.0 -> threshold raised -> harder to densify

    Three modes:
      - iteration: cosine schedule based on training progress p = t/T
      - count: based on current Gaussian count vs target
      - stage: fixed factors per stage defined by absolute iteration numbers
    """
    if not getattr(args, 'use_adaptive_thresh', False):
        return 1.0, 1.0

    mode = getattr(args, 'adaptive_mode', 'iteration')

    if mode == "iteration":
        p = iteration / total_iterations
        p1 = getattr(args, 'adaptive_p1', 0.3)
        p2 = getattr(args, 'adaptive_p2', 0.7)
        cs = getattr(args, 'adaptive_clone_start_factor', 0.5)
        ce = getattr(args, 'adaptive_clone_end_factor', 2.0)
        ss = getattr(args, 'adaptive_split_start_factor', 0.5)
        se = getattr(args, 'adaptive_split_end_factor', 1.5)
        if p <= p1:
            clone_factor = cs
            split_factor = ss
        elif p >= p2:
            clone_factor = ce
            split_factor = se
        else:
            import math
            progress = (p - p1) / (p2 - p1)
            clone_factor = cs + (ce - cs) * 0.5 * (1.0 - math.cos(math.pi * progress))
            split_factor = ss + (se - ss) * 0.5 * (1.0 - math.cos(math.pi * progress))
        return clone_factor, split_factor

    elif mode == "stage":
        T1 = getattr(args, 'stage_T1', 5000)
        T2 = getattr(args, 'stage_T2', 50000)
        cf1 = getattr(args, 'stage_clone_factor_1', 2.0)
        cf3 = getattr(args, 'stage_clone_factor_3', 0.5)
        sf1 = getattr(args, 'stage_split_factor_1', 0.5)
        sf3 = getattr(args, 'stage_split_factor_3', 1.0)
        if iteration <= T1:
            return cf1, sf1
        elif iteration <= T2:
            return 1.0, 1.0
        else:
            return cf3, sf3

    return 1.0, 1.0


def compute_frequency_weight(gt_image, alpha=1.0):
    """Compute per-pixel frequency weight from GT image using Laplacian.

    High-frequency regions (edges, textures) get higher weight so that the
    loss gradient is amplified there, indirectly driving more Gaussian
    densification in detail-rich areas.

    Args:
        gt_image: [3, H, W] tensor on CUDA
        alpha: strength of frequency weighting (0=disabled, 1=normal)

    Returns:
        weight_map: [1, H, W] tensor with values in [1.0, 1.0+alpha],
            where 1.0 means flat region (no boost) and 1.0+alpha means
            maximum frequency (full boost).
    """
    if alpha <= 0:
        return None
    gray = gt_image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32, device=gt_image.device
    ).view(1, 1, 3, 3)
    freq_map = F.conv2d(gray, laplacian_kernel, padding=1).abs().squeeze(0)  # [1, H, W]
    # Normalize to [0, 1]
    freq_max = freq_map.max().clamp(min=1e-6)
    freq_norm = freq_map / freq_max
    # Weight: 1.0 (flat) to 1.0+alpha (high frequency)
    weight_map = 1.0 + alpha * freq_norm
    return weight_map


def compute_ms_ssim(img1, img2, levels=3):
    """Compute multi-scale SSIM between two images.

    Args:
        img1, img2: [1, 3, H, W] tensors
        levels: number of scales (default 3)

    Returns:
        ms_ssim value (scalar tensor)
    """
    weights = torch.tensor([0.3, 0.35, 0.35], device=img1.device)[:levels]
    weights = weights / weights.sum()

    ssim_values = []
    current_img1 = img1
    current_img2 = img2

    for level in range(levels):
        ssim_val = fast_ssim(current_img1, current_img2)
        ssim_values.append(ssim_val)
        if level < levels - 1:
            current_img1 = F.avg_pool2d(current_img1, kernel_size=2)
            current_img2 = F.avg_pool2d(current_img2, kernel_size=2)

    ms_ssim_val = torch.zeros(1, device=img1.device)
    for level in range(levels):
        ms_ssim_val = ms_ssim_val + weights[level] * ssim_values[level]
    return ms_ssim_val



def compute_freq_map_normalized(gt_image):
    """Compute normalized frequency map from a single GT image.
    Args: gt_image: [3, H, W] tensor on CUDA
    Returns: freq_map: [H, W] tensor with values in [0, 1]
    """
    gray = gt_image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32, device=gt_image.device
    ).view(1, 1, 3, 3)
    freq_map = F.conv2d(gray, laplacian_kernel, padding=1).abs().squeeze()  # [H, W]
    # Robust normalization by the 99th percentile instead of max: the
    # Laplacian max is an outlier edge pixel, so max-normalization pushes
    # almost all F~0 and the frequency modulation (§3.5) becomes a near-no-op.
    # A high percentile spreads F across [0,1] so most gaussians get a
    # meaningful per-Gaussian frequency signal.
    freq_max = torch.quantile(freq_map.float().flatten(), 0.99).clamp(min=1e-6)
    return (freq_map / freq_max).clamp(0.0, 1.0)


def compute_per_gaussian_freq_signal(camlist, gaussians):
    """Project Gaussian centers onto GT frequency maps and compute per-Gaussian R_high.
    Uses Method B (forward projection + center query).
    Returns: R_high: [N] tensor in [0, 1], or None if no valid observations.
    """
    import math
    xyz = gaussians.get_xyz.detach()  # [N, 3]
    num_gaussians = xyz.shape[0]
    freq_accum = torch.zeros(num_gaussians, device="cuda")
    freq_count = torch.zeros(num_gaussians, device="cuda")
    xyz_homo = torch.cat([xyz, torch.ones(num_gaussians, 1, device="cuda")], dim=1)  # [N, 4]

    for cam in camlist:
        H = cam.image_height
        W = cam.image_width
        focal_x = W / (2.0 * math.tan(cam.FoVx / 2.0))
        focal_y = H / (2.0 * math.tan(cam.FoVy / 2.0))
        p_cam = (xyz_homo @ cam.world_view_transform)[:, :3]  # [N, 3]
        in_front = p_cam[:, 2] > 0.1
        px = (focal_x * p_cam[:, 0] / p_cam[:, 2].clamp(min=0.1) + W / 2.0).long()
        py = (focal_y * p_cam[:, 1] / p_cam[:, 2].clamp(min=0.1) + H / 2.0).long()
        valid = in_front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not valid.any():
            continue
        gt_image = cam.original_image.cuda()  # [3, H, W]
        freq_map = compute_freq_map_normalized(gt_image)  # [H, W]
        freq_accum[valid] += freq_map[py[valid], px[valid]]
        freq_count[valid] += 1.0

    has_obs = freq_count > 0
    if not has_obs.any():
        return None
    R_high = torch.zeros(num_gaussians, device="cuda")
    R_high[has_obs] = freq_accum[has_obs] / freq_count[has_obs]
    return R_high


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    global _MASK_SOURCE_PATH
    _MASK_SOURCE_PATH = dataset.source_path

    # Set random seed for reproducibility
    seed = getattr(opt, 'seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All random seeds set to {seed}")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    # Structure-Aware Gaussian Grouping: pre-compute prototype clusters
    if getattr(opt, 'use_structure_grouping', False):
        from utils.structure_utils import cluster_into_prototypes
        xyz_np = gaussians._xyz.detach().cpu().numpy()
        num_protos = getattr(opt, 'num_prototypes', 12)
        proto_ids, proto_centers, _ = cluster_into_prototypes(
            xyz_np, num_prototypes=num_protos, k_neighbors=30, random_state=seed)
        gaussians.setup_structure_grouping(proto_ids, num_protos)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # Pre-compute spatial camera clusters for clustered sampling (v10)
    cluster_labels = None
    cluster_centers = None
    cluster_round_robin = [0]  # mutable counter for round-robin cluster selection
    if getattr(opt, 'use_clustered_sampling', False):
        all_train_cameras = scene.getTrainCameras()
        num_clusters = getattr(opt, 'num_camera_clusters', 20)
        cluster_labels, cluster_centers = build_camera_clusters(all_train_cameras, num_clusters)

    # Prepare densification log directory
    densify_log_dir = None
    if getattr(opt, 'save_densify_logs', False):
        densify_log_dir = os.path.join(dataset.model_path, "densify_logs")
        os.makedirs(densify_log_dir, exist_ok=True)

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    # Disable tqdm's background monitor thread to avoid a multi-thread deadlock
    # that occurs when set_postfix/refresh contends with the monitor thread,
    # especially when stdout is redirected through tee to a non-tty file.
    tqdm.monitor_interval = 0
    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Training progress",
        mininterval=1.0,
        miniters=100,
    )
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # Initialize loss CSV logger
    loss_csv_path = os.path.join(dataset.model_path, "loss_log.csv")
    loss_csv_file = open(loss_csv_path, mode='w', newline='')
    loss_csv_writer = csv.writer(loss_csv_file)
    loss_csv_writer.writerow(["iteration", "l1_loss", "ssim", "total_loss", "ema_loss", "num_gaussians"])

    # Set total iterations for Gumbel-Softmax temperature annealing
    gaussians._total_iterations = opt.iterations

    # Persistent VGG/LPIPS net for perceptual loss (frozen, grad flows to input only)
    lpips_net = None
    if getattr(opt, 'lambda_lpips', 0.0) > 0:
        lpips_net = LPIPS('vgg', '0.1').to("cuda").eval()
        for p in lpips_net.parameters():
            p.requires_grad = False

    for iteration in range(first_iter, opt.iterations + 1):
        gaussians._current_iteration = iteration

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Skip iteration if no gaussians are visible (avoids gradient shape mismatch)
        if not visibility_filter.any():
            continue

        # Loss computation with frequency-aware weighting and MS-SSIM
        gt_image = viewpoint_cam.original_image.cuda()

        # Curriculum training: blur GT in early phases for coarse-to-fine optimization
        curriculum_sigma_val = getattr(opt, 'curriculum_sigma', 0.0)
        if curriculum_sigma_val > 0:
            curriculum_T1 = getattr(opt, 'curriculum_T1', 15000)
            curriculum_T2 = getattr(opt, 'curriculum_T2', 80000)
            curriculum_schedule = getattr(opt, 'curriculum_schedule', 'linear')
            current_sigma = compute_curriculum_sigma(
                iteration, curriculum_sigma_val, curriculum_T1, curriculum_T2, curriculum_schedule)
            gt_image = curriculum_blur(gt_image, current_sigma)

        # Compute frequency weight map from GT (amplifies loss in detail regions)
        freq_alpha = getattr(opt, 'freq_weight_alpha', 0.0)
        freq_weight = compute_frequency_weight(gt_image, alpha=freq_alpha)

        # Semantic weight map (equipment > background, person excluded). When
        # enabled it reweights the per-pixel loss; when disabled it is None and
        # behavior is identical to before.
        semantic_weight = None
        if (getattr(dataset, 'use_semantic_weight', False)
                and hasattr(viewpoint_cam, 'semantic_weight')
                and viewpoint_cam.semantic_weight is not None):
            semantic_weight = viewpoint_cam.semantic_weight.cuda()

        if hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
            mask = viewpoint_cam.gt_alpha_mask.cuda()
            masked_image = image * mask
            # Frequency-aware weighted L1
            pixel_error = torch.abs(masked_image - gt_image) * mask
            # Normalization denominator. Default is the person-mask pixel count.
            # With semantic weighting we use a *weighted average*: numerator and
            # denominator are both scaled by semantic_weight, so equipment-region
            # errors get a larger share of the loss WITHOUT shifting the overall
            # loss magnitude (which would otherwise unbalance it against SSIM and
            # the regularizers).
            use_additive = (semantic_weight is not None and getattr(dataset, 'use_additive_equip_loss', False))
            if use_additive:
                # v4 §5: Decoupled Additive Equipment-Bonus Loss.
                # base = uniform L1 over person-masked image (background gradient
                #   == baseline, never starved by equipment weights);
                # bonus = lambda_equip * mean(|render-gt| over equipment pixels).
                # Replaces the multiplicative semantic weighting which diluted
                # background gradient to ~1/3 under W_equip=3.
                bg_w = getattr(dataset, 'semantic_background_weight', 1.0)
                sem_2d = semantic_weight[0] if semantic_weight.dim() == 3 else semantic_weight
                equip_region = (sem_2d > bg_w).float()
                equip_count = (equip_region * mask).sum().clamp(min=1.0)
                base_err = pixel_error
                if freq_weight is not None:
                    base_err = base_err * freq_weight
                    Ll1_base = base_err.sum() / (mask.sum() * 3 * freq_weight.mean()).clamp(min=1.0)
                else:
                    Ll1_base = base_err.sum() / (mask.sum() * 3).clamp(min=1.0)
                equip_bonus = (pixel_error * equip_region).sum() / (equip_count * 3)
                Ll1 = Ll1_base + getattr(opt, 'lambda_equip', 0.3) * equip_bonus
            else:
                norm_weight = mask
                if semantic_weight is not None:
                    pixel_error = pixel_error * semantic_weight
                    norm_weight = mask * semantic_weight
                if freq_weight is not None:
                    pixel_error = pixel_error * freq_weight
                    Ll1 = pixel_error.sum() / (norm_weight.sum() * 3 * freq_weight.mean()).clamp(min=1.0)
                else:
                    Ll1 = pixel_error.sum() / (norm_weight.sum() * 3).clamp(min=1.0)
            # MS-SSIM if enabled, otherwise single-scale SSIM
            use_ms_ssim = getattr(opt, 'use_ms_ssim', False)
            if use_ms_ssim:
                ssim_value = compute_ms_ssim(masked_image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = fast_ssim(masked_image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            # No person-exclusion mask path
            pixel_error = torch.abs(image - gt_image)
            if semantic_weight is not None:
                # Weighted average: divide by sum of weights, not pixel count,
                # so the loss magnitude is preserved (see masked branch above).
                pixel_error = pixel_error * semantic_weight
                weight_sum = semantic_weight.sum()
                if freq_weight is not None:
                    pixel_error = pixel_error * freq_weight
                    Ll1 = pixel_error.sum() / (weight_sum * 3 * freq_weight.mean()).clamp(min=1.0)
                else:
                    Ll1 = pixel_error.sum() / (weight_sum * 3).clamp(min=1.0)
            elif freq_weight is not None:
                pixel_error = pixel_error * freq_weight
                Ll1 = pixel_error.mean() / freq_weight.mean()
            else:
                Ll1 = l1_loss(image, gt_image)
            use_ms_ssim = getattr(opt, 'use_ms_ssim', False)
            if use_ms_ssim:
                ssim_value = compute_ms_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if getattr(opt, 'lambda_opa', 0.0) > 0 or getattr(opt, 'lambda_scale', 0.0) > 0:
            loss = loss + getattr(opt, 'lambda_opa', 0.0) * gaussians.get_opacity.mean() + getattr(opt, 'lambda_scale', 0.0) * gaussians.get_scaling.max(dim=1).values.mean()
        if lpips_net is not None:
            loss = loss + opt.lambda_lpips * lpips_net(image.unsqueeze(0), gt_image.unsqueeze(0))

        # Structure-Aware Gaussian Grouping: additional regularization losses
        # Computed every 10 steps to minimize overhead
        if gaussians.use_structure_grouping and iteration > 500 and iteration % 10 == 0:
            from utils.structure_utils import compute_group_consistency_loss, compute_residual_sparsity_loss
            lambda_group = getattr(opt, 'lambda_group_consistency', 0.01)
            lambda_residual = getattr(opt, 'lambda_residual_sparsity', 0.001)
            
            group_loss = compute_group_consistency_loss(
                gaussians, gaussians.prototype_ids, gaussians.num_prototypes)
            residual_loss = compute_residual_sparsity_loss(
                gaussians._sh_residual, gaussians._residual_scale)
            
            loss = loss + lambda_group * group_loss + lambda_residual * residual_loss

        # V8.5: STE prune regularization loss with lambda(t) scheduling
        if getattr(opt, 'use_ste_prune', False):
            warmup_steps = getattr(opt, 'ste_warmup', 10000)
            if iteration > warmup_steps:
                ste_lambda_max = getattr(opt, 'ste_prune_lambda', 1e-3)
                ste_lambda_min = getattr(opt, 'ste_prune_lambda_min', 0.0)
                ste_lambda_power = getattr(opt, 'ste_prune_lambda_power', 2.0)

                # Lambda scheduling: small early (allow growth) -> large late (force compression)
                if ste_lambda_min > 0:
                    progress = (iteration - warmup_steps) / max(opt.iterations - warmup_steps, 1)
                    progress = min(1.0, progress)
                    ste_lambda = ste_lambda_min + (ste_lambda_max - ste_lambda_min) * (progress ** ste_lambda_power)
                else:
                    ste_lambda = ste_lambda_max

                # Min-gaussian protection: only apply reg when total gaussians > K_min
                # and alive count >= K_min; prevents over-pruning in early/large stages
                _, ste_prob = gaussians.get_ste_mask(
                    threshold=getattr(opt, 'ste_prune_threshold', 0.5),
                    iteration=iteration,
                    total_iterations=opt.iterations)
                total_count = ste_prob.shape[0]
                alive_count = int((ste_prob > 0.5).float().sum().item())
                # Dynamic K_min: combine absolute floor with ratio-based protection
                abs_floor = getattr(opt, 'ste_prune_min_alive', 5000000)
                min_ratio = getattr(opt, 'ste_prune_min_alive_ratio', 0.0)
                if min_ratio > 0:
                    min_alive_count = max(abs_floor, int(total_count * min_ratio))
                else:
                    min_alive_count = abs_floor
                if total_count > min_alive_count and alive_count >= min_alive_count:
                    prune_reg_loss = ste_lambda * ste_prob.sum()
                    loss = loss + prune_reg_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Write loss to CSV every 1000 iterations
            if iteration % 1000 == 0:
                loss_csv_writer.writerow([
                    iteration,
                    f"{Ll1.item():.7f}",
                    f"{ssim_value.item():.7f}",
                    f"{loss.item():.7f}",
                    f"{ema_loss_for_log:.7f}",
                    gaussians._xyz.shape[0],
                ])
                loss_csv_file.flush()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
            
            # Accumulate per-Gaussian visibility count (used by coverage-aware
            # densification and pruning; nearly zero overhead).
            gaussians.visibility_count[visibility_filter] += 1

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    if getattr(opt, 'vanilla_densify', False):
                        # Vanilla 3DGS: no multi-view scoring, densify all gradient-qualified candidates
                        num_gaussians = gaussians._xyz.shape[0]
                        importance_score = torch.full((num_gaussians,), 10.0, device="cuda")
                        pruning_score = torch.zeros((num_gaussians,), device="cuda")
                        sampled_cam_indices = []
                        current_cluster_id = -1
                        freq_signal = None
                    else:
                        my_viewpoint_stack = scene.getTrainCameras().copy()
                        sampled_cam_indices = []
                        current_cluster_id = -1

                        if getattr(opt, 'use_clustered_sampling', False) and cluster_labels is not None:
                            # Strategy A: round-robin through spatial clusters
                            num_k = int(cluster_labels.max()) + 1
                            current_cluster_id = cluster_round_robin[0] % num_k
                            cluster_round_robin[0] += 1
                            camlist, sampled_cam_indices = sampling_cameras_clustered(
                                my_viewpoint_stack, cluster_labels, current_cluster_id)
                        elif getattr(opt, 'use_stratified_sampling', False):
                            camlist = sampling_cameras_stratified(my_viewpoint_stack)
                        else:
                            camlist = sampling_cameras(my_viewpoint_stack)

                        # The multiview consistent densification of fastgs.
                        # Curriculum (§3.4): pass the current blur sigma so the
                        # importance_score (densification gate) is computed on the
                        # same blurred GT as the photometric loss, making early
                        # densification respond only to low-frequency signal.
                        cur_sigma = 0.0
                        if getattr(opt, 'curriculum_sigma', 0.0) > 0:
                            cur_sigma = compute_curriculum_sigma(
                                iteration, opt.curriculum_sigma,
                                opt.curriculum_T1, opt.curriculum_T2,
                                getattr(opt, 'curriculum_schedule', 'linear'))
                        importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True, curriculum_sigma=cur_sigma)

                        # GuidedGS: compute per-Gaussian frequency signal for spatial guidance
                        freq_signal = None
                        if getattr(opt, 'use_freq_guided_densify', False):
                            freq_signal = compute_per_gaussian_freq_signal(camlist, gaussians)

                    num_before_densify = gaussians._xyz.shape[0]

                    # V8.4 方案三：自适应阈值调度
                    if getattr(opt, 'use_adaptive_thresh', False):
                        clone_factor, split_factor = compute_adaptive_thresholds(
                            iteration, opt.densify_until_iter, num_before_densify, opt)
                        opt._adaptive_clone_factor = clone_factor
                        opt._adaptive_split_factor = split_factor

                    use_progressive_prune = getattr(opt, 'use_progressive_prune', False)
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score,
                                                skip_prune = use_progressive_prune,
                                                freq_signal = freq_signal,
                                                iteration = iteration)

                    num_after_densify = gaussians._xyz.shape[0]

                    # Save densification diagnostics
                    if densify_log_dir is not None and importance_score is not None:
                        log_path = os.path.join(densify_log_dir, f"iter_{iteration}.npz")
                        vis_count_cpu = gaussians.visibility_count[:num_before_densify].cpu().numpy()
                        imp_score_cpu = importance_score[:num_before_densify].cpu().numpy()

                        # Per-cluster coverage statistics (assign each Gaussian
                        # to the nearest camera-cluster center in batches to
                        # avoid OOM on large scenes with millions of Gaussians).
                        cluster_coverage_stats = {}
                        if cluster_labels is not None and cluster_centers is not None and len(vis_count_cpu) > 0:
                            all_positions = gaussians.get_xyz[:num_before_densify].detach().cpu().numpy()
                            centers = cluster_centers  # [K, 3]
                            batch_size = 500_000
                            gaussian_cluster_ids = np.empty(len(all_positions), dtype=np.int32)
                            for start in range(0, len(all_positions), batch_size):
                                end = min(start + batch_size, len(all_positions))
                                dists = np.linalg.norm(
                                    all_positions[start:end, None, :] - centers[None, :, :], axis=2)
                                gaussian_cluster_ids[start:end] = dists.argmin(axis=1)
                            for cid in range(len(centers)):
                                mask = gaussian_cluster_ids == cid
                                if mask.sum() > 0:
                                    vis_in_cluster = vis_count_cpu[mask]
                                    cluster_coverage_stats[f"cluster_{cid}_count"] = int(mask.sum())
                                    cluster_coverage_stats[f"cluster_{cid}_vis_mean"] = float(vis_in_cluster.mean())
                                    cluster_coverage_stats[f"cluster_{cid}_vis_median"] = float(np.median(vis_in_cluster))
                                    cluster_coverage_stats[f"cluster_{cid}_vis_min"] = float(vis_in_cluster.min())

                        np.savez_compressed(
                            log_path,
                            iteration=iteration,
                            cluster_id=current_cluster_id,
                            sampled_cam_indices=np.array(sampled_cam_indices),
                            num_gaussians_before=num_before_densify,
                            num_gaussians_after=num_after_densify,
                            importance_score=imp_score_cpu,
                            visibility_count=vis_count_cpu,
                            imp_mean=float(imp_score_cpu.mean()),
                            imp_median=float(np.median(imp_score_cpu)),
                            imp_p90=float(np.percentile(imp_score_cpu, 90)),
                            imp_max=float(imp_score_cpu.max()),
                            **cluster_coverage_stats,
                        )
                        print(f"  [Densify log] saved to {log_path} "
                              f"(cluster={current_cluster_id}, "
                              f"{num_before_densify:,}->{num_after_densify:,})")

                    # Progressive pruning: bound to densification step,
                    # runs right after densify_and_prune_fastgs so they are
                    # always in sync (like FastGS native pruning).
                    use_progressive_prune = getattr(opt, 'use_progressive_prune', False)
                    progressive_prune_ratio = getattr(opt, 'progressive_prune_ratio', 0.05)
                    if use_progressive_prune:
                        num_before = gaussians._xyz.shape[0]
                        opacity_vals = gaussians.get_opacity.squeeze()

                        # Coverage-aware pruning: under-observed Gaussians get a
                        # protection factor that raises their effective opacity in
                        # the ranking so they are less likely to be pruned.
                        use_cov_prune = getattr(opt, 'use_coverage_aware_prune', False)
                        if use_cov_prune and gaussians.visibility_count.shape[0] == num_before:
                            vis_median = gaussians.visibility_count.median().clamp(min=1.0)
                            rel_coverage = gaussians.visibility_count / vis_median
                            max_protection = getattr(opt, 'coverage_prune_max_protection', 2.0)
                            protection_factor = (1.0 / rel_coverage.clamp(min=0.01).sqrt()).clamp(min=1.0, max=max_protection)
                            effective_opacity = opacity_vals * protection_factor
                        else:
                            effective_opacity = opacity_vals

                        # Remove the bottom X% by effective opacity
                        num_to_remove = int(num_before * progressive_prune_ratio)
                        if num_to_remove > 0 and num_before > num_to_remove:
                            threshold_val = torch.kthvalue(effective_opacity, num_to_remove).values
                            prune_mask = effective_opacity <= threshold_val
                            # Log coverage stats of pruned Gaussians for diagnostics
                            if use_cov_prune and gaussians.visibility_count.shape[0] == num_before:
                                pruned_cov = rel_coverage[prune_mask]
                                cov_info = (f", pruned_cov: mean={pruned_cov.mean():.2f} "
                                            f"median={pruned_cov.median():.2f}")
                            else:
                                cov_info = ""
                            gaussians.prune_points(prune_mask)
                            num_after = gaussians._xyz.shape[0]
                            print(f"\n[ITER {iteration}] Progressive prune: "
                                  f"{num_before:,} -> {num_after:,} "
                                  f"(-{num_before - num_after:,}, "
                                  f"eff_thresh={threshold_val.item():.4f}"
                                  f"{cov_info})")

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
        
            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    # scene.save(iteration)
    loss_csv_file.close()
    if getattr(gaussians, '_use_ste_prune', False):
        alive_mask = gaussians.get_alive_mask()
        print(f"Gaussian number (total): {gaussians._xyz.shape[0]}, alive: {int(alive_mask.sum().item())}")
    else:
        print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    train_time_hours = total_time / 3600.0
    print(f"Training time: {total_time:.1f}s ({train_time_hours:.1f}h)")
    print(f"Loss log saved to: {loss_csv_path}")

    # Append training time to train_results.json and test_results.json
    for name in ['train', 'test']:
        results_path = os.path.join(dataset.model_path, f"{name}_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as fp:
                results = json.load(fp)
            results["Train Time (h)"] = round(train_time_hours, 1)
            with open(results_path, 'w') as fp:
                json.dump(results, fp, indent=True)
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # Apply person exclusion mask during evaluation (consistent with training)
                    if hasattr(viewpoint, 'gt_alpha_mask') and viewpoint.gt_alpha_mask is not None:
                        eval_mask = viewpoint.gt_alpha_mask.cuda()
                        masked_image = image * eval_mask
                        masked_gt = gt_image * eval_mask
                    else:
                        masked_image = image
                        masked_gt = gt_image
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(masked_image, masked_gt).mean().double()
                    psnr_test += psnr(masked_image, masked_gt).mean().double()
                    ssim_test += fast_ssim(masked_image.unsqueeze(0), masked_gt.unsqueeze(0)).mean().double()
                    lpips_test += lpips(masked_image, masked_gt, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.4f} SSIM {:.6f} LPIPS {:.6f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # V8.5: STE mask distribution statistics (for Gumbel-Softmax analysis)
        if getattr(scene.gaussians, '_use_ste_prune', False) and hasattr(scene.gaussians, '_prune_logit'):
            with torch.no_grad():
                ste_mask, ste_prob = scene.gaussians.get_ste_mask(
                    threshold=getattr(scene.gaussians, '_ste_threshold', 0.5),
                    iteration=iteration,
                    total_iterations=getattr(scene.gaussians, '_total_iterations', None))
                total_count = ste_prob.shape[0]
                alive_count = int((ste_prob > 0.5).sum().item())
                p_mean = ste_prob.mean().item()
                p_std = ste_prob.std().item()
                p_near_0 = (ste_prob < 0.1).float().mean().item()
                p_near_1 = (ste_prob > 0.9).float().mean().item()
                p_fuzzy = ((ste_prob > 0.3) & (ste_prob < 0.7)).float().mean().item()
                tau = getattr(scene.gaussians, '_current_tau', None)
                tau_str = f"{tau:.4f}" if tau is not None else "N/A"
                print("[ITER {}] STE stats: alive={}/{} ({:.1f}%) | p_mean={:.4f} p_std={:.4f} | near_0={:.1f}% near_1={:.1f}% fuzzy={:.1f}% | tau={}".format(
                    iteration, alive_count, total_count, 100.0 * alive_count / max(total_count, 1),
                    p_mean, p_std,
                    100.0 * p_near_0, 100.0 * p_near_1, 100.0 * p_fuzzy,
                    tau_str))
                if tb_writer:
                    tb_writer.add_scalar('ste/alive_count', alive_count, iteration)
                    tb_writer.add_scalar('ste/p_mean', p_mean, iteration)
                    tb_writer.add_scalar('ste/p_std', p_std, iteration)
                    tb_writer.add_scalar('ste/p_near_0', p_near_0, iteration)
                    tb_writer.add_scalar('ste/p_near_1', p_near_1, iteration)
                    tb_writer.add_scalar('ste/p_fuzzy', p_fuzzy, iteration)
                    if tau is not None:
                        tb_writer.add_scalar('ste/tau', tau, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets
    )

    # All done
    print("\nTraining complete.")
