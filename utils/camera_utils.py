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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import os
import torch
import torch.nn.functional as F
from PIL import Image

WARNED = False

def load_external_mask(image_path, resolution, source_path, soft_mask_sigma=0):
    """Load a binary mask from the external mask_sam3/ directory.

    Mask convention: white(255)=occluded(person), black(0)=valid scene
    Returns alpha mask: 1.0=valid, 0.0=occluded. The final-paper protocol
    uses nearest-neighbour resize followed by a fixed 0.5 threshold, so the
    training mask is strictly binary. A positive ``soft_mask_sigma`` remains
    only for explicit legacy experiments.

    Args:
        image_path: path to the source image
        resolution: target (W, H) tuple
        source_path: dataset root containing mask_sam3/
        soft_mask_sigma: Gaussian blur sigma for explicit legacy soft-mask
            experiments. 0 = hard binary mask (final-paper default).
    """
    rel_path = os.path.relpath(image_path, os.path.join(source_path, "images"))
    mask_rel = os.path.splitext(rel_path)[0] + ".png"
    mask_path = os.path.join(source_path, "mask_sam3", mask_rel)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            "Mask file not found: %s\n"
            "Expected masks in: %s/mask_sam3/\n"
            "Please ensure the mask_sam3 directory exists and contains a mask for every image."
            % (mask_path, source_path))

    mask_img = Image.open(mask_path).convert("L")
    resample = Image.NEAREST if soft_mask_sigma <= 0 else Image.BILINEAR
    mask_img = mask_img.resize(resolution, resample)
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    # Invert: white(1.0)=person -> alpha=0, black(0.0)=valid -> alpha=1
    alpha_np = 1.0 - mask_np
    if soft_mask_sigma <= 0:
        alpha_np = (alpha_np >= 0.5).astype(np.float32)
    alpha_tensor = torch.from_numpy(alpha_np).unsqueeze(0)  # [1, H, W]

    # Apply Gaussian blur for soft mask edges
    if soft_mask_sigma > 0:
        kernel_size = int(6 * soft_mask_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        alpha_tensor = alpha_tensor.unsqueeze(0)  # [1, 1, H, W]
        alpha_tensor = _gaussian_blur_2d(alpha_tensor, kernel_size, soft_mask_sigma)
        alpha_tensor = alpha_tensor.squeeze(0)  # [1, H, W]
        alpha_tensor = alpha_tensor.clamp(0.0, 1.0)

    return alpha_tensor


def load_semantic_weight(image_path, resolution, source_path,
                         equipment_weight=2.0, background_weight=1.0):
    """Load the semantic weight map from semantic_weights/ directory.

    The weight map is a single-channel PNG encoded as pixel_value = weight * 100
    (see generate_semantic_weights.py). The stored three classes are:
        0   -> person / excluded
        100 -> building background
        200 -> equipment

    The stored background/equipment weights (1.0 / 2.0) are remapped at load time
    to the configurable `background_weight` / `equipment_weight`, so the weight
    contrast can be tuned without regenerating the maps. Person stays at 0.

    Returns a [1, H, W] float tensor of per-pixel weights, or None if missing.
    """
    rel_path = os.path.relpath(image_path, os.path.join(source_path, "images"))
    weight_rel = os.path.splitext(rel_path)[0] + ".png"
    weight_path = os.path.join(source_path, "semantic_weights", weight_rel)

    if not os.path.exists(weight_path):
        return None, None

    weight_img = Image.open(weight_path).convert("L")
    weight_img = weight_img.resize(resolution, Image.NEAREST)
    raw = np.array(weight_img, dtype=np.float32)  # 0 / 100 / 200

    # Remap stored classes to the configurable weights (for loss).
    weight_np = np.zeros_like(raw)
    weight_np[raw >= 150] = equipment_weight              # equipment (200)
    weight_np[(raw >= 50) & (raw < 150)] = background_weight  # background (100)
    # person (0) stays 0.0

    # Binary equipment-region mask from raw class (independent of W_equip/W_bg).
    equip_np = (raw >= 150).astype(np.float32)  # 1=equipment, 0=else

    weight_tensor = torch.from_numpy(weight_np).unsqueeze(0)  # [1, H, W]
    equip_tensor = torch.from_numpy(equip_np).unsqueeze(0)    # [1, H, W]
    return weight_tensor, equip_tensor

def _gaussian_blur_2d(tensor, kernel_size, sigma):
    """Apply 2D Gaussian blur to a [B, C, H, W] tensor."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.to(tensor.device)
    padding = kernel_size // 2
    return F.conv2d(tensor, kernel_2d, padding=padding)


def loadCam(args, id, cam_info, resolution_scale):
    # Support lazy loading: if image is None, open from image_path
    if cam_info.image is None:
        pil_image = Image.open(cam_info.image_path)
    else:
        pil_image = cam_info.image

    orig_w, orig_h = pil_image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(pil_image, resolution)

    # Release PIL image immediately to save memory
    del pil_image

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    source_path = getattr(args, 'source_path', None)

    # Try loading external person-exclusion mask if no embedded alpha
    if loaded_mask is None and not getattr(args, 'no_mask', False):
        mask_dir = os.path.join(source_path, "mask_sam3")
        # Follow symlinks - the directory itself may be a symlink
        mask_dir_real = os.path.realpath(mask_dir)
        if source_path and os.path.isdir(mask_dir_real):
            loaded_mask = load_external_mask(cam_info.image_path, resolution, source_path)
            # Verify masks were actually loaded (could be broken symlinks)
            if loaded_mask is None and not getattr(args, 'no_mask', False):
                raise FileNotFoundError(
                    "Mask files found in mask_sam3/ but could not be loaded for image: %s\n"
                    "This usually means broken symlinks. Check that mask files point to valid targets."
                    % cam_info.image_path)
        elif source_path and not getattr(args, 'no_mask', False):
            raise FileNotFoundError(
                "Person mask is enabled (no_mask=False) but mask_sam3/ directory not found under: %s\n"
                "Either create the mask_sam3/ directory with mask files, or set --no_mask to disable masking."
                % source_path)

    # Try loading the semantic weight map for semantic-weighted training
    semantic_weight = None
    equipment_mask = None
    if (getattr(args, 'use_semantic_weight', False)
            and source_path
            and os.path.isdir(os.path.join(source_path, "semantic_weights"))):
        equipment_weight = getattr(args, 'semantic_equipment_weight', 3.0)
        background_weight = getattr(args, 'semantic_background_weight', 1.0)
        semantic_weight, equipment_mask = load_semantic_weight(
            cam_info.image_path, resolution, source_path,
            equipment_weight=equipment_weight,
            background_weight=background_weight)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  semantic_weight=semantic_weight,
                  equipment_mask=equipment_mask,
                  image_name=cam_info.image_name, uid=id,
                  image_path=cam_info.image_path,
                  data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
