#!/usr/bin/env python3
"""Evaluate trained 3DGS-style point clouds on one explicit test split.

The historical FastGS/GuidedGS and external-baseline metrics were produced by
each repository's own test-view construction. This script bypasses that source
of mismatch: it reads the COLMAP cameras named in a shared test.txt, loads an
existing point_cloud.ply, renders those exact views, and computes one consistent
set of quality metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified quality evaluator for exp_6650.")
    parser.add_argument("--repo-dir", required=True, help="Repository to import renderer/scene code from.")
    parser.add_argument("--repo-type", choices=("3dgs", "fastgs", "sparegs", "steepgs"), required=True)
    parser.add_argument("--model-path", required=True, help="Trained model output directory.")
    parser.add_argument("--source-path", required=True, help="COLMAP scene root.")
    parser.add_argument("--split-file", required=True, help="Explicit test.txt with COLMAP image names.")
    parser.add_argument("--images", default="images")
    parser.add_argument("--iteration", type=int, default=665000)
    parser.add_argument("--label", default="")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--data-device", default="cpu")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--mult", type=float, default=0.6, help="FastGS render_fastgs compact-box multiplier.")
    parser.add_argument("--s-estimator", default="inv_cov", help="SteepGS splitting estimator.")
    parser.add_argument("--mask-mode", choices=("sam3", "none"), default="sam3")
    parser.add_argument(
        "--mask-type",
        choices=("hard", "soft"),
        default="hard",
        help="SAM3 mask processing. Final-paper protocol uses hard.",
    )
    parser.add_argument(
        "--soft-mask-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for legacy --mask-type soft only; final hard-mask protocol requires 0.",
    )
    parser.add_argument("--max-views", type=int, default=0, help="Evaluate only the first N split views; 0 means all.")
    parser.add_argument("--view-indices", default="", help="Comma-separated 0-based indices into the split file.")
    parser.add_argument("--save-renders-dir", default="", help="Optional directory for rendered PNGs.")
    parser.add_argument("--save-gt-dir", default="", help="Optional directory for masked GT PNGs.")
    parser.add_argument("--skip-metrics", action="store_true", help="Render/save views without computing quality metrics.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--per-view-json", default="")
    parser.add_argument("--list-only", action="store_true", help="Validate the split/images and exit before CUDA imports.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def add_repo_to_path(repo_dir: str) -> None:
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)


def read_split(path: Path) -> List[str]:
    names = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    if not names:
        raise ValueError(f"No image names found in split file: {path}")
    return names


def select_split_names(split_names: List[str], args: argparse.Namespace) -> Tuple[List[int], List[str]]:
    if args.view_indices:
        indices = [int(value.strip()) for value in args.view_indices.split(",") if value.strip()]
        bad = [idx for idx in indices if idx < 0 or idx >= len(split_names)]
        if bad:
            raise IndexError(f"Split indices out of range, first bad index: {bad[0]}")
        return indices, [split_names[idx] for idx in indices]
    if args.max_views > 0:
        return list(range(min(args.max_views, len(split_names)))), split_names[: args.max_views]
    return list(range(len(split_names))), split_names


def resolve_existing_file(base_dir: Path, rel_name: str) -> Path:
    candidates = [
        base_dir / rel_name,
        base_dir / Path(rel_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve '{rel_name}' under {base_dir}")


def resolve_mask_path(source_path: Path, image_path: Path) -> Path:
    images_root = source_path / "images"
    try:
        rel_path = image_path.relative_to(images_root)
    except ValueError:
        rel_path = Path(image_path.name)
    mask_path = source_path / "mask_sam3" / rel_path.with_suffix(".png")
    if mask_path.exists():
        return mask_path
    fallback = source_path / "mask_sam3" / image_path.with_suffix(".png").name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Mask not found for {image_path}: expected {mask_path}")


def load_mask(mask_path: Path, resolution: Tuple[int, int], mask_type: str, sigma: float):
    import torch
    import torch.nn.functional as F

    mask_img = Image.open(mask_path).convert("L")
    resample = Image.NEAREST if mask_type == "hard" else Image.BILINEAR
    mask_img = mask_img.resize(resolution, resample)
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    alpha = torch.from_numpy(1.0 - mask_np).unsqueeze(0)

    if mask_type == "hard":
        # The SAM3 files encode invalid/background as white. Threshold after
        # nearest-neighbour resize so the evaluation mask is strictly binary.
        alpha = (alpha >= 0.5).to(dtype=torch.float32)
    elif sigma > 0:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        alpha = F.conv2d(alpha.unsqueeze(0), kernel_2d, padding=kernel_size // 2).squeeze(0)
        alpha = alpha.clamp(0.0, 1.0)

    return alpha.cuda()


def image_name_for_repo(repo_type: str, split_name: str) -> str:
    if repo_type == "fastgs":
        return Path(split_name).name.rsplit(".", 1)[0]
    if repo_type == "steepgs":
        return Path(split_name).with_suffix("").as_posix().replace("/", "__")
    return split_name


def load_colmap(repo_type: str, source_path: Path, images: str, split_names: Iterable[str]):
    from scene.colmap_loader import (
        qvec2rotmat,
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from scene.dataset_readers import CameraInfo
    from utils.graphics_utils import focal2fov

    sparse_dir = source_path / "sparse" / "0"
    try:
        cam_extrinsics = read_extrinsics_binary(str(sparse_dir / "images.bin"))
        cam_intrinsics = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(str(sparse_dir / "images.txt"))
        cam_intrinsics = read_intrinsics_text(str(sparse_dir / "cameras.txt"))

    extr_by_name = {extr.name: extr for extr in cam_extrinsics.values()}
    images_folder = source_path / images

    cam_infos = []
    split_to_image_path = {}
    missing = []
    for split_idx, split_name in enumerate(split_names):
        extr = extr_by_name.get(split_name)
        if extr is None:
            missing.append(split_name)
            continue

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        r = np.transpose(qvec2rotmat(extr.qvec))
        t = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            fov_y = focal2fov(intr.params[0], height)
            fov_x = focal2fov(intr.params[0], width)
        elif intr.model == "PINHOLE":
            fov_y = focal2fov(intr.params[1], height)
            fov_x = focal2fov(intr.params[0], width)
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

        image_path = resolve_existing_file(images_folder, extr.name)
        split_to_image_path[split_name] = str(image_path)
        image_name = image_name_for_repo(repo_type, split_name)

        if repo_type == "sparegs":
            cam_info = CameraInfo(
                uid=intr.id,
                R=r,
                T=t,
                FovY=fov_y,
                FovX=fov_x,
                depth_params=None,
                image_path=str(image_path),
                image_name=image_name,
                depth_path="",
                width=width,
                height=height,
                is_test=True,
            )
        elif repo_type == "3dgs":
            cam_info = CameraInfo(
                uid=intr.id,
                R=r,
                T=t,
                FovY=fov_y,
                FovX=fov_x,
                depth_params=None,
                image_path=str(image_path),
                image_name=image_name,
                depth_path="",
                width=width,
                height=height,
                is_test=True,
            )
        else:
            if repo_type == "fastgs":
                pil_image = None
            else:
                with Image.open(image_path) as image:
                    pil_image = image.copy()
            cam_info = CameraInfo(
                uid=intr.id,
                R=r,
                T=t,
                FovY=fov_y,
                FovX=fov_x,
                image=pil_image,
                image_path=str(image_path),
                image_name=image_name,
                width=width,
                height=height,
            )
        cam_infos.append(cam_info)

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} split images were not found in COLMAP extrinsics: {preview}")
    if not cam_infos:
        raise RuntimeError("No camera infos were built from the split.")
    return cam_infos, split_to_image_path


def make_camera_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        resolution=args.resolution,
        data_device=args.data_device,
        source_path=os.path.abspath(args.source_path),
        images=args.images,
        train_test_exp=False,
        no_mask=True,
        use_semantic_weight=False,
        semantic_equipment_weight=3.0,
        semantic_background_weight=1.0,
        use_additive_equip_loss=False,
    )


def build_cameras(repo_type: str, cam_infos: list, camera_args: SimpleNamespace) -> list:
    from utils.camera_utils import cameraList_from_camInfos

    if repo_type in ("3dgs", "sparegs"):
        return cameraList_from_camInfos(cam_infos, 1.0, camera_args, False, True)
    return cameraList_from_camInfos(cam_infos, 1.0, camera_args)


def load_gaussians(repo_type: str, model_path: Path, iteration: int, s_estimator: str):
    from scene.gaussian_model import GaussianModel

    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing point cloud: {ply_path}")

    try:
        gaussians = GaussianModel(3, optimizer_type="default")
    except TypeError:
        gaussians = GaussianModel(3)

    if repo_type == "sparegs":
        gaussians.load_ply(str(ply_path), False)
    else:
        gaussians.load_ply(str(ply_path))

    if repo_type == "steepgs":
        gaussians.S_estimator = s_estimator

    return gaussians, ply_path


def make_pipeline_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
        separate_sh=True,
        S_estimator=args.s_estimator,
    )


def render_one(repo_type: str, view, gaussians, pipeline, background, mult: float):
    if repo_type == "fastgs":
        from gaussian_renderer import render_fastgs

        return render_fastgs(view, gaussians, pipeline, background, mult)["render"]

    from gaussian_renderer import render

    if repo_type == "sparegs":
        return render(view, gaussians, pipeline, background)["render"]
    return render(view, gaussians, pipeline, background)["render"]


def save_tensor_png(image_tensor, path: Path) -> None:
    image = image_tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.uint8(np.clip(image * 255.0 + 0.5, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    import torch

    if not args.skip_metrics:
        from lpipsPyTorch import lpips
        from utils.image_utils import psnr
        from utils.loss_utils import ssim

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for unified evaluation.")

    all_split_names = read_split(Path(args.split_file))
    split_indices, split_names = select_split_names(all_split_names, args)

    source_path = Path(args.source_path).resolve()
    model_path = Path(args.model_path).resolve()
    cam_infos, split_to_image_path = load_colmap(args.repo_type, source_path, args.images, split_names)
    cameras = build_cameras(args.repo_type, cam_infos, make_camera_args(args))
    gaussians, ply_path = load_gaussians(args.repo_type, model_path, args.iteration, args.s_estimator)
    pipeline = make_pipeline_args(args)

    bg_color = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    psnrs, ssims, lpipss, l1s = [], [], [], []
    per_view = {}
    save_renders_dir = Path(args.save_renders_dir).resolve() if args.save_renders_dir else None
    save_gt_dir = Path(args.save_gt_dir).resolve() if args.save_gt_dir else None
    if save_renders_dir:
        save_renders_dir.mkdir(parents=True, exist_ok=True)
    if save_gt_dir:
        save_gt_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    with torch.no_grad():
        for split_idx, split_name, view in tqdm(
            zip(split_indices, split_names, cameras),
            total=len(cameras),
            desc=f"Unified eval {args.label or model_path.name}",
            disable=args.quiet,
        ):
            rendering = torch.clamp(
                render_one(args.repo_type, view, gaussians, pipeline, background, args.mult),
                0.0,
                1.0,
            )
            gt = torch.clamp(view.original_image[0:3, :, :].cuda(), 0.0, 1.0)

            mask_used = False
            if args.mask_mode == "sam3":
                image_path = Path(split_to_image_path[split_name])
                mask = load_mask(
                    resolve_mask_path(source_path, image_path),
                    (int(view.image_width), int(view.image_height)),
                    args.mask_type,
                    args.soft_mask_sigma,
                )
                rendering = rendering * mask
                gt = gt * mask
                mask_used = True

            render_path = None
            gt_path = None
            if save_renders_dir:
                render_path = save_renders_dir / f"{split_idx:05d}.png"
                save_tensor_png(rendering, render_path)
            if save_gt_dir:
                gt_path = save_gt_dir / f"{split_idx:05d}.png"
                save_tensor_png(gt, gt_path)

            view_result = {
                "split_index": split_idx,
                "split_name": split_name,
                "repo_image_name": view.image_name,
                "image_path": split_to_image_path[split_name],
                "mask_used": mask_used,
                "render_path": str(render_path) if render_path else None,
                "gt_path": str(gt_path) if gt_path else None,
            }
            if not args.skip_metrics:
                rendering_b = rendering.unsqueeze(0)
                gt_b = gt.unsqueeze(0)
                l1_val = torch.abs(rendering - gt).mean().item()
                psnr_val = psnr(rendering_b, gt_b).mean().item()
                ssim_val = ssim(rendering_b, gt_b).item()
                lpips_val = lpips(rendering_b, gt_b, net_type="vgg").item()

                l1s.append(l1_val)
                psnrs.append(psnr_val)
                ssims.append(ssim_val)
                lpipss.append(lpips_val)
                view_result.update(
                    {
                        "L1": l1_val,
                        "PSNR": psnr_val,
                        "SSIM": ssim_val,
                        "LPIPS": lpips_val,
                    }
                )
            per_view[split_name] = view_result

    elapsed_s = time.perf_counter() - started_at
    metrics = {
        "L1": None,
        "PSNR": None,
        "SSIM": None,
        "LPIPS": None,
    }
    if not args.skip_metrics:
        metrics = {
            "L1": float(torch.tensor(l1s).mean().item()),
            "PSNR": float(torch.tensor(psnrs).mean().item()),
            "SSIM": float(torch.tensor(ssims).mean().item()),
            "LPIPS": float(torch.tensor(lpipss).mean().item()),
        }

    result = {
        "label": args.label or model_path.name,
        "repo_type": args.repo_type,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "model_path": str(model_path),
        "point_cloud": str(ply_path),
        "source_path": str(source_path),
        "split_file": str(Path(args.split_file).resolve()),
        "iteration": args.iteration,
        "num_source_split_views": len(all_split_names),
        "num_test_views": len(cameras),
        "selected_split_indices": split_indices,
        "mask_mode": args.mask_mode,
        "mask_type": args.mask_type if args.mask_mode == "sam3" else None,
        "soft_mask_sigma": args.soft_mask_sigma if args.mask_mode == "sam3" and args.mask_type == "soft" else 0.0,
        "mult": args.mult if args.repo_type == "fastgs" else None,
        "s_estimator": args.s_estimator if args.repo_type == "steepgs" else None,
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "elapsed_s": elapsed_s,
        "save_renders_dir": str(save_renders_dir) if save_renders_dir else None,
        "save_gt_dir": str(save_gt_dir) if save_gt_dir else None,
        **metrics,
    }

    output_json = args.output_json
    if not output_json:
        output_json = str(model_path / "results_unified_canonical.json")
    per_view_json = args.per_view_json
    if not per_view_json:
        per_view_json = str(Path(output_json).with_name(Path(output_json).stem + "_per_view.json"))

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    Path(per_view_json).parent.mkdir(parents=True, exist_ok=True)
    with open(per_view_json, "w", encoding="utf-8") as f:
        json.dump(per_view, f, indent=2)
        f.write("\n")

    print(json.dumps(result, indent=2))
    print(f"[unified-eval] Saved {output_json}")
    print(f"[unified-eval] Saved {per_view_json}")
    return result


def list_only(args: argparse.Namespace) -> int:
    all_split_names = read_split(Path(args.split_file))
    split_indices, split_names = select_split_names(all_split_names, args)

    source_path = Path(args.source_path).resolve()
    images_folder = source_path / args.images
    missing = []
    missing_masks = []
    for name in split_names:
        try:
            image_path = resolve_existing_file(images_folder, name)
        except FileNotFoundError:
            missing.append(name)
            continue
        if args.mask_mode == "sam3":
            try:
                resolve_mask_path(source_path, image_path)
            except FileNotFoundError:
                missing_masks.append(name)

    summary = {
        "label": args.label,
        "repo_type": args.repo_type,
        "split_file": str(Path(args.split_file).resolve()),
        "num_source_split_views": len(all_split_names),
        "num_split_views": len(split_names),
        "selected_split_indices": split_indices,
        "missing_images": len(missing),
        "missing_masks": len(missing_masks),
        "mask_mode": args.mask_mode,
        "preview": split_names[:5],
    }
    print(json.dumps(summary, indent=2))
    if missing:
        print("Missing preview:", ", ".join(missing[:5]), file=sys.stderr)
        return 2
    if missing_masks:
        print("Missing mask preview:", ", ".join(missing_masks[:5]), file=sys.stderr)
        return 2
    return 0


def absolutize_path_args(args: argparse.Namespace) -> None:
    for attr in (
        "repo_dir",
        "model_path",
        "source_path",
        "split_file",
        "save_renders_dir",
        "save_gt_dir",
        "output_json",
        "per_view_json",
    ):
        value = getattr(args, attr, "")
        if value:
            setattr(args, attr, str(Path(value).resolve()))


def main() -> int:
    args = parse_args()
    if args.mask_type == "hard" and args.soft_mask_sigma != 0.0:
        raise ValueError("--mask-type hard requires --soft-mask-sigma 0.")
    absolutize_path_args(args)
    if args.list_only:
        return list_only(args)

    add_repo_to_path(args.repo_dir)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
