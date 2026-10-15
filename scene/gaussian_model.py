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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._prune_logit = torch.empty(0)  # STE learnable mask logit
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.visibility_count = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # Structure-Aware Gaussian Grouping
        self.use_structure_grouping = False
        self.num_prototypes = 0
        self.prototype_ids = None  # [N] long tensor, non-learnable
        self._prototype_sh_templates = None  # [K, C, 3] learnable per-prototype SH template
        self._sh_residual = None  # [N, C, 3] learnable per-gaussian residual
        self._residual_scale = None  # [N, 1] learnable scale factor
        self.setup_functions()

    def capture(self, optimizer_type):
        if optimizer_type == "default":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
        else:
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        # _prune_logit restoration is handled inside training_setup() based on use_ste_prune flag
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    def get_ellipticity(self):
        """Compute ellipticity E = max(s) / min(s) for each Gaussian.
        E >> 1 -> flattened/elongated -> high-frequency texture.
        E ~ 1 -> spherical -> low-frequency region."""
        scaling = self.get_scaling
        s_max = scaling.max(dim=1).values
        s_min = scaling.min(dim=1).values
        s_min = s_min.clamp(min=1e-8)
        return s_max / s_min
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    def sync_structure_sh(self):
        """Write template + residual back into _features_rest (in-place, preserving grad).
        
        Called after optimizer step so that the rasterizer always sees up-to-date SH
        without any runtime overhead in get_features.
        """
        if not self.use_structure_grouping or self._prototype_sh_templates is None:
            return
        with torch.no_grad():
            template_rest = self._prototype_sh_templates[self.prototype_ids]  # [N, C, 3]
            effective = template_rest + self._residual_scale.unsqueeze(-1) * self._sh_residual
            self._features_rest.data.copy_(effective)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_ste_mask(self, threshold=0.5, iteration=None, total_iterations=None):
        """Compute STE binary mask M_k from learnable logit m_k.
        
        Supports two modes:
        - Standard Sigmoid (default): p = sigmoid(m)
        - Gumbel-Softmax + temperature annealing: p = sigmoid(m / tau(t))
          where tau(t) = max(tau_min, tau_init * (1 - t/T_total))
        
        Forward: M_k = 1 if p > threshold, else 0
        Backward: gradient flows through p (STE trick)
        
        Returns:
            mask: [N, 1] tensor with values in {0, 1} (forward) but differentiable (backward)
            prob: [N, 1] tensor with p values (for regularization loss)
        """
        use_gumbel = getattr(self, '_use_gumbel', False)
        
        if use_gumbel and iteration is not None and total_iterations is not None:
            tau_init = getattr(self, '_tau_init', 1.0)
            tau_min = 0.01
            progress = min(1.0, iteration / max(total_iterations, 1))
            tau = max(tau_min, tau_init * (1.0 - progress))
            self._current_tau = tau
            prob = torch.sigmoid(self._prune_logit / tau)
        else:
            self._current_tau = None
            prob = torch.sigmoid(self._prune_logit)
        
        # STE: forward is hard binary, backward uses sigmoid gradient
        binary = (prob > threshold).float()
        mask = binary - prob.detach() + prob  # sg(binary - prob) + prob
        return mask, prob

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # _prune_logit is initialized lazily in training_setup() only when use_ste_prune=True
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def setup_structure_grouping(self, prototype_ids_np, num_prototypes):
        """Initialize structure-aware gaussian grouping after create_from_pcd.
        
        Design: _features_rest remains the "effective SH" used by the rasterizer
        (no runtime overhead). The learnable parameters are _sh_residual,
        _prototype_sh_templates, and _residual_scale. After each optimizer step,
        sync_structure_sh() writes template + scale*residual back into
        _features_rest.data, keeping it non-learnable (detached buffer).
        
        Args:
            prototype_ids_np: numpy array of shape [N], cluster assignments
            num_prototypes: number of prototype clusters
        """
        self.use_structure_grouping = True
        self.num_prototypes = num_prototypes
        num_points = self._xyz.shape[0]
        num_sh_rest = self._features_rest.shape[1]  # number of SH coeffs (excluding DC)
        
        # Non-learnable prototype assignment
        self.prototype_ids = torch.tensor(prototype_ids_np, dtype=torch.long, device="cuda")
        
        # Initialize prototype SH templates by averaging current SH rest per group
        templates = torch.zeros((num_prototypes, num_sh_rest, 3), device="cuda")
        current_rest = self._features_rest.detach()  # [N, C, 3]
        for proto_id in range(num_prototypes):
            mask = (self.prototype_ids == proto_id)
            if mask.sum() > 0:
                templates[proto_id] = current_rest[mask].mean(dim=0)
        
        self._prototype_sh_templates = nn.Parameter(templates.requires_grad_(True))
        
        # Residual = current_rest - template[proto_id] (so initial output is unchanged)
        initial_residual = current_rest - templates[self.prototype_ids]
        self._sh_residual = nn.Parameter(initial_residual.requires_grad_(True))
        
        # Residual scale starts at 1.0 (so initial output = template + 1.0 * residual = original)
        self._residual_scale = nn.Parameter(torch.ones((num_points, 1), device="cuda").requires_grad_(True))
        
        # Convert _features_rest to a plain buffer (not optimized directly)
        # It will be kept in sync via sync_structure_sh()
        self._features_rest = nn.Parameter(self._features_rest.data.clone(), requires_grad=False)
        
        print(f"[Structure] Initialized grouping: {num_prototypes} prototypes, "
              f"{num_sh_rest} SH rest coeffs, {num_points} gaussians")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.visibility_count = torch.zeros((self.get_xyz.shape[0],), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.use_structure_grouping:
            sh_l = [
                {'params': [self._sh_residual], 'lr': training_args.highfeature_lr / 20.0, "name": "sh_residual"},
                {'params': [self._prototype_sh_templates], 'lr': training_args.highfeature_lr / 10.0, "name": "sh_templates"},
                {'params': [self._residual_scale], 'lr': 0.001, "name": "residual_scale"},
            ]
        else:
            sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        # V8.5: STE prune logit optimizer group (only when enabled)
        self._use_ste_prune = getattr(training_args, 'use_ste_prune', False)
        if self._use_ste_prune:
            # Initialize _prune_logit if not already present (e.g. from checkpoint)
            if not hasattr(self, '_prune_logit') or self._prune_logit.numel() == 0:
                self._prune_logit = nn.Parameter(
                    torch.full((self._xyz.shape[0], 1), 3.0, dtype=torch.float, device="cuda").requires_grad_(True))
            self._ste_threshold = getattr(training_args, 'ste_prune_threshold', 0.5)
            self._use_gumbel = getattr(training_args, 'ste_prune_use_gumbel', False)
            self._tau_init = getattr(training_args, 'ste_prune_tau_init', 1.0)
            l.append({'params': [self._prune_logit], 'lr': training_args.opacity_lr, "name": "prune_logit"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def get_alive_mask(self):
        """Return boolean mask of alive gaussians (STE mask > 0.5 or STE not enabled)."""
        if getattr(self, '_use_ste_prune', False) and hasattr(self, '_prune_logit') and self._prune_logit.numel() > 0:
            prob = torch.sigmoid(self._prune_logit)
            return (prob > 0.5).squeeze(-1)
        return torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # Filter out pruned gaussians only when STE pruning is active
        if getattr(self, '_use_ste_prune', False):
            alive = self.get_alive_mask()
            xyz = self._xyz[alive].detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc[alive].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest[alive].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity[alive].detach().cpu().numpy()
            scale = self._scaling[alive].detach().cpu().numpy()
            rotation = self._rotation[alive].detach().cpu().numpy()
        else:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        # Names of global (non-per-gaussian) parameters to skip during pruning
        global_param_names = {"sh_templates"}

        for opt in optimizers:
            for group in opt.param_groups:
                if group["name"] in global_param_names:
                    continue
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        if self.use_structure_grouping:
            self._sh_residual = optimizable_tensors["f_rest"]
            self._residual_scale = optimizable_tensors["residual_scale"]
            # prototype_ids is non-learnable, filter directly
            self.prototype_ids = self.prototype_ids[valid_points_mask]
        else:
            self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "prune_logit" in optimizable_tensors:
            self._prune_logit = optimizable_tensors["prune_logit"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.visibility_count = self.visibility_count[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                # Skip global parameters that are not per-gaussian (e.g. sh_templates)
                if group["name"] not in tensors_dict:
                    continue
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_prototype_ids=None):
        if self.use_structure_grouping:
            # For structure grouping: new_features_rest is the residual
            new_residual_scale = torch.ones((new_xyz.shape[0], 1), device="cuda")
            d = {"xyz": new_xyz,
                 "f_dc": new_features_dc,
                 "f_rest": new_features_rest,  # this is sh_residual
                 "opacity": new_opacities,
                 "scaling": new_scaling,
                 "rotation": new_rotation,
                 "residual_scale": new_residual_scale}
        else:
            d = {"xyz": new_xyz,
                 "f_dc": new_features_dc,
                 "f_rest": new_features_rest,
                 "opacity": new_opacities,
                 "scaling": new_scaling,
                 "rotation": new_rotation}

        # STE prune logit for new Gaussians: init to 3.0 (visible, sigmoid(3)≈0.95)
        if hasattr(self, '_prune_logit') and self._prune_logit.numel() > 0:
            new_prune_logit = torch.full((new_xyz.shape[0], 1), 3.0, device="cuda")
            d["prune_logit"] = new_prune_logit

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        if self.use_structure_grouping:
            self._sh_residual = optimizable_tensors["f_rest"]
            self._residual_scale = optimizable_tensors["residual_scale"]
            # Extend prototype_ids (non-learnable)
            if new_prototype_ids is not None:
                self.prototype_ids = torch.cat((self.prototype_ids, new_prototype_ids))
        else:
            self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "prune_logit" in optimizable_tensors:
            self._prune_logit = optimizable_tensors["prune_logit"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Inherit parent visibility_count for cloned/split Gaussians, new ones start at 0
        new_vis_count = torch.zeros(new_tmp_radii.shape[0], device="cuda")
        self.visibility_count = torch.cat((self.visibility_count, new_vis_count))

    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        if self.use_structure_grouping:
            new_features_rest = self._sh_residual[selected_pts_mask].repeat(N,1,1)
            new_proto_ids = self.prototype_ids[selected_pts_mask].repeat(N)
        else:
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_proto_ids = None
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_proto_ids)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        if self.use_structure_grouping:
            new_features_rest = self._sh_residual[selected_pts_mask]
            new_proto_ids = self.prototype_ids[selected_pts_mask]
        else:
            new_features_rest = self._features_rest[selected_pts_mask]
            new_proto_ids = None
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_proto_ids)

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score=None, pruning_score=None, skip_prune=False, freq_signal=None, iteration=None):
        
        ''' 
            Densification and Pruning based on FastGS criteria:
            1.  The gaussians candidate for densification are selected based on the gradient of their position first.
            2.  Then, based on their average metric score (computed over multiple sampled views), they are either densified (cloned) or split.
                This is our main contribution compared to the vanilla 3DGS.
            3.  Finally, gaussians with low opacity or very large size are pruned (skipped when skip_prune=True,
                e.g. when progressive pruning handles pruning separately).
        '''
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        # V8.4 方案三：自适应阈值调度 — 在椭圆度之前应用
        clone_factor = getattr(args, '_adaptive_clone_factor', 1.0)
        split_factor = getattr(args, '_adaptive_split_factor', 1.0)
        eff_grad_thresh = args.grad_thresh * clone_factor
        eff_grad_abs_thresh = args.grad_abs_thresh * split_factor

        # GuidedGS: frequency-guided threshold modulation (spatial guidance)
        # High-frequency Gaussians -> lower clone threshold (easier to clone)
        # Low-frequency Gaussians -> lower split threshold (easier to split)
        if freq_signal is not None and getattr(args, 'use_freq_guided_densify', False):
            beta = getattr(args, 'freq_guided_beta', 0.3)
            eps_min = getattr(args, 'freq_epsilon_min', 0.3)
            # GuidedGS §3.5 (new formula): both clone and split thresholds
            # DECREASE with spatial frequency, so high-frequency regions can
            # densify freely (small, dense gaussians for detail) while low-
            # frequency regions keep the base threshold. The eps_min floor
            # prevents runaway densification in the highest-frequency areas.
            # (Replaces the old formula that raised the split threshold in
            # high-frequency regions, which suppressed split and caused
            # overlap; unified lowering removes that asymmetry.)
            freq_mod = torch.clamp(1.0 - beta * freq_signal, min=eps_min)
            grad_norms = torch.norm(grad_vars, dim=-1)
            grad_norms_abs = torch.norm(grads_abs, dim=-1)
            grad_qualifiers = grad_norms >= (eff_grad_thresh * freq_mod)
            grad_qualifiers_abs = grad_norms_abs >= (eff_grad_abs_thresh * freq_mod)
        else:
            grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= eff_grad_thresh, True, False)
            grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= eff_grad_abs_thresh, True, False)

        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        # V8.4 方案二：椭圆度频率推断
        # 设计原则：椭圆度作为"加法"额外通道，不修改原有 qualifiers
        # 高频区（扁长高斯）→ 额外降低 clone 阈值，给更多 clone 机会
        if getattr(args, 'use_ellipticity_freq', False):
            ellip = self.get_ellipticity()
            if args.ellipticity_mode == "hard":
                # 高频高斯：降分录取 — 梯度接近阈值也给 clone 机会
                is_high_freq = ellip > args.ellipticity_thresh
                # 原有 qualifiers 不动，额外给高频高斯降低 clone 门槛
                extra_clone_thresh = eff_grad_thresh * 0.5  # 高频区 clone 阈值减半
                extra_clone_qualifiers = is_high_freq & (
                    torch.norm(grad_vars, dim=-1) >= extra_clone_thresh)
                # 合并：原有 clone_qualifiers 或 高频额外通道
                clone_qualifiers = clone_qualifiers | extra_clone_qualifiers
                # split_qualifiers 完全不动
            elif args.ellipticity_mode == "soft":
                ellip_norm = (ellip - 1.0) / (ellip.max() - 1.0).clamp(min=1e-6)
                w = args.ellipticity_weight
                grad_thresh_mod = args.grad_thresh * (1.0 - w * ellip_norm)
                grad_abs_thresh_mod = args.grad_abs_thresh * (1.0 + w * ellip_norm)
                grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= grad_thresh_mod, True, False)
                grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_thresh_mod, True, False)

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # Multi-view consistent metric for densification filtering.
        # When coverage-aware densification is enabled, the hard threshold is
        # replaced by an adaptive per-Gaussian threshold that is lower for
        # under-observed Gaussians (giving them a fair chance to densify) and
        # higher for well-observed ones (avoiding over-densification).
        if getattr(args, 'use_budgeted_growth', False):
            # v4 §6: Budgeted Importance-Gated Growth. Replace the hard
            # importance>5 gate (densifies ALL qualified -> ~5M late) with a
            # per-step topk growth budget. importance_score already carries
            # densify_equipment_factor amplification, so equipment gaussians
            # win budget share even under a tight global budget.
            qualified = torch.logical_or(all_clones, all_splits)
            n_qualified = int(qualified.sum().item())
            cur_count = self.get_xyz.shape[0]
            g = getattr(args, 'growth_rate', 0.02)
            d = getattr(args, 'growth_decay', 1.0)
            eff_rate = g
            if d < 1.0 and iteration is not None:
                t0 = getattr(args, 'densify_from_iter', 0)
                t1 = getattr(args, 'densify_until_iter', 1)
                prog = min(1.0, max(0.0, (iteration - t0) / max(1, t1 - t0)))
                eff_rate = g * (1.0 - (1.0 - d) * prog)
            budget = int(cur_count * eff_rate)
            metric_mask = torch.zeros((cur_count,), dtype=torch.bool, device="cuda")
            if n_qualified > 0 and budget > 0:
                budget = min(budget, n_qualified)
                alpha = getattr(args, 'growth_combine_prune', 0.0)
                q_idx = torch.nonzero(qualified, as_tuple=False).squeeze(-1)
                imp = importance_score[q_idx].float()
                if alpha > 0.0 and pruning_score is not None:
                    pr = pruning_score[q_idx].float()
                    pr = (pr - pr.min()) / (pr.max() - pr.min()).clamp(min=1e-6)
                    rank_score = imp + alpha * pr
                else:
                    rank_score = imp
                rank_score = torch.nan_to_num(rank_score, nan=-1.0)
                k = min(budget, q_idx.shape[0])
                _, top_local = torch.topk(rank_score, k, largest=True)
                win_idx = q_idx[top_local]
                metric_mask[win_idx] = True
        elif getattr(args, 'use_coverage_aware_densify', False) and self.visibility_count.shape[0] > 0:
            vis_median = self.visibility_count.median().clamp(min=1.0)
            rel_coverage = self.visibility_count / vis_median
            power = getattr(args, 'coverage_densify_power', 0.5)
            adaptive_threshold = 5.0 * rel_coverage.pow(power)
            adaptive_threshold = adaptive_threshold.clamp(min=1.0, max=10.0)
            metric_mask = importance_score > adaptive_threshold
        else:
            if getattr(args, 'use_residual_energy_gate', False):
                metric_mask = importance_score > getattr(args, 'tau_energy', 100)
            else:
                metric_mask = importance_score > 5

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        if not skip_prune:
            if getattr(args, 'use_ste_prune', False):
                # V8.5: STE learnable mask — skip traditional hard pruning.
                # Pruning is driven by gradient through the STE mask during rendering.
                # Only prune oversized Gaussians (geometric safety check).
                if max_screen_size:
                    big_points_vs = self.max_radii2D > max_screen_size
                    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                    oversize_mask = torch.logical_or(big_points_vs, big_points_ws)
                    if oversize_mask.any():
                        self.prune_points(oversize_mask)

                # STE hybrid hard pruning: periodically remove low-mask Gaussians
                # to control memory usage. The STE mask still drives fine-grained
                # selection via gradients; this only physically deletes Gaussians
                # that the mask has already learned to suppress.
                ste_hard_prune = getattr(args, 'ste_hard_prune', False)
                hard_warmup = getattr(args, 'ste_warmup', 10000)
                if ste_hard_prune and iteration >= hard_warmup and hasattr(self, '_prune_logit') and self._prune_logit.numel() > 0:
                    mask_prob = torch.sigmoid(self._prune_logit).squeeze(-1)
                    hard_thresh = getattr(args, 'ste_hard_prune_thresh', 0.1)
                    hard_ratio = getattr(args, 'ste_hard_prune_ratio', 0.05)
                    min_alive = getattr(args, 'ste_prune_min_alive', 5000000)

                    # Candidates: mask probability below threshold
                    candidates = mask_prob < hard_thresh
                    num_candidates = candidates.sum().item()
                    total_gaussians = mask_prob.shape[0]

                    if num_candidates > 0 and total_gaussians > min_alive:
                        # Budget: at most hard_ratio of total, but never go below min_alive
                        budget = int(total_gaussians * hard_ratio)
                        max_removable = total_gaussians - min_alive
                        budget = min(budget, max_removable)

                        if budget > 0:
                            # Among candidates, remove those with lowest mask probability
                            candidate_indices = torch.where(candidates)[0]
                            candidate_probs = mask_prob[candidate_indices]
                            num_to_remove = min(budget, candidate_indices.shape[0])
                            _, worst_indices = torch.topk(candidate_probs, num_to_remove, largest=False)
                            remove_indices = candidate_indices[worst_indices]

                            hard_prune_mask = torch.zeros(total_gaussians, dtype=torch.bool, device="cuda")
                            hard_prune_mask[remove_indices] = True
                            self.prune_points(hard_prune_mask)
            elif getattr(args, 'use_redundancy_prune', False):
                # Redundancy prune: remove low-importance + low-opacity + well-observed
                # redundant overlap (importance already equip-amplified => implicitly protects equip).
                n_cur = self.get_xyz.shape[0]
                prune_mask = (self.get_opacity < min_opacity).squeeze()
                if max_screen_size:
                    big_points_vs = self.max_radii2D > max_screen_size
                    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                redundancy_mask = torch.zeros(n_cur, dtype=torch.bool, device="cuda")
                if (iteration is not None and iteration >= getattr(args, 'redundancy_start_iter', 0)
                        and self.visibility_count.shape[0] == n_cur and importance_score is not None):
                    imp = torch.full((n_cur,), float('inf'), device=importance_score.device, dtype=torch.float32)
                    imp[:importance_score.shape[0]] = importance_score.float()
                    vis = self.visibility_count
                    vis_med = vis.median().clamp(min=1.0)
                    opa = self.get_opacity.squeeze()
                    cand = ((imp < getattr(args, 'redundancy_thresh', 2.0)) &
                            (opa < getattr(args, 'redundancy_opacity', 0.05)) &
                            (vis > vis_med * getattr(args, 'redundancy_vis_mult', 1.5)))
                    if cand.any():
                        cand_idx = torch.where(cand)[0]
                        imp_n = imp[cand_idx] / (imp[cand_idx].max().clamp(min=1e-6))
                        vis_n = 1.0 - (vis[cand_idx] / (vis[cand_idx].max().clamp(min=1e-6)))
                        opa_n = opa[cand_idx]
                        combined = 0.4 * imp_n + 0.4 * vis_n + 0.2 * opa_n
                        budget = int(getattr(args, 'redundancy_budget_frac', 0.5) * cand_idx.shape[0])
                        if budget > 0:
                            _, worst = torch.topk(combined, min(budget, cand_idx.shape[0]), largest=False)
                            redundancy_mask[cand_idx[worst]] = True
                final_prune = torch.logical_or(prune_mask, redundancy_mask)
                if final_prune.any():
                    self.prune_points(final_prune)
            else:
                # Original hard pruning logic
                prune_mask = (self.get_opacity < min_opacity).squeeze()
                if max_screen_size:
                    big_points_vs = self.max_radii2D > max_screen_size
                    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

                scores = 1 - pruning_score 
                to_remove = torch.sum(prune_mask)
                remove_budget = int(0.5 * to_remove)

                if remove_budget:
                    n_init_points = self.get_xyz.shape[0]
                    padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
                    padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
                    # Sanitize: non-finite / negative probabilities crash
                    # torch.multinomial ("probability tensor contains inf/nan/
                    # element < 0"). This was the C3 (W_equip=1.0) crash path;
                    # zero out offending entries and clamp to >= 0.
                    padded_importance = torch.nan_to_num(padded_importance, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
                    selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
                    max_multinomial_categories = 2 ** 24
                    # Fall back to topk when the category count is huge OR the
                    # probability mass is degenerate (all-zero / non-finite),
                    # which would otherwise raise in torch.multinomial.
                    if n_init_points > max_multinomial_categories or padded_importance.sum() <= 0:
                        _, topk_indices = torch.topk(padded_importance, remove_budget)
                        selected_pts_mask[topk_indices] = True
                    else:
                        sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
                        selected_pts_mask[sampled_indices] = True
                    final_prune = torch.logical_and(prune_mask, selected_pts_mask)
                    self.prune_points(final_prune)

        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def final_prune_fastgs(self, min_opacity, pruning_score = None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze() 
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune)
