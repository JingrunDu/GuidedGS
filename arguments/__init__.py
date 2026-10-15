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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.llffhold = 10
        self._test_file = ""
        # IndustrialGS: semantic weighting params are consumed during camera
        # loading (loadCam receives ModelParams), so they must live here.
        self.no_mask = False                  # skip loading external SAM3 masks
        self.use_semantic_weight = False      # enable per-pixel semantic loss weighting
        self.semantic_equipment_weight = 3.0  # loss weight for equipment region
        self.semantic_background_weight = 1.0 # loss weight for building background
        self.use_additive_equip_loss = False  # v4 §5: decoupled additive equip-bonus loss (replaces multiplicative semantic; bg gradient == baseline, not starved)
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        if g.source_path.startswith(("oss://", "http://", "https://")):
            raise ValueError("source_path must be a local directory. Mount or download remote data first.")
        g.source_path = os.path.abspath(os.path.expanduser(os.path.expandvars(g.source_path)))
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025 
        self.shfeature_lr = 0.005 
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.lambda_equip = 0.3  # v4 §5: weight of additive equipment-bonus L1 (sweep {0.1,0.3,0.5})
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012  
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat

        self.random_background = False
        self.optimizer_type = "default"
        self.vanilla_densify = False   # use vanilla 3DGS densification (no multi-view consistency)

        # IndustrialGS: frequency-aware loss weighting
        self.freq_weight_alpha = 0.0   # 0=disabled, >0 amplifies loss in high-freq regions
        self.use_ms_ssim = False       # use multi-scale SSIM instead of single-scale

        # Curriculum training: frequency-adaptive coarse-to-fine
        self.curriculum_sigma = 0.0        # initial blur sigma (0=disabled)
        self.curriculum_T1 = 49875         # end of Phase 1 (constant blur); 49.9K for the 6650-image scene (=unit*0.15, unit=N*50)
        self.curriculum_T2 = 266000        # end of Phase 2 (blur decays to 0); 266K (=unit*0.80)
        self.curriculum_schedule = "linear"  # sigma decay: linear / cosine / exponential
        self.curriculum_lr_decay = 1.0     # Phase 3 position_lr extra decay (1.0=disabled)

        # IndustrialGS: semantic densification bias (used by compute_gaussian_score_fastgs
        # which receives OptimizationParams). The loss-weighting params live in ModelParams.
        self.densify_equipment_factor = 2.0   # densification bias for equipment region

        # IndustrialGS: progressive pruning (bound to densification step)
        self.use_progressive_prune = False    # enable progressive pruning during densification
        self.progressive_prune_ratio = 0.05   # fraction of Gaussians to remove each prune step

        # IndustrialGS v9: coverage-aware densification & pruning
        self.use_stratified_sampling = False     # stratified viewpoint sampling for densification scoring
        self.use_coverage_aware_densify = False  # adaptive importance_score threshold by visibility count
        self.coverage_densify_power = 0.5        # exponent for threshold scaling (sqrt by default)
        self.use_coverage_aware_prune = False    # protect under-observed Gaussians during progressive pruning
        self.coverage_prune_max_protection = 2.0 # max opacity multiplier for low-coverage Gaussians

        # IndustrialGS v10: clustered spatial sampling for densification
        self.use_clustered_sampling = False      # KMeans spatial clustering of cameras, round-robin per cluster
        self.num_camera_clusters = 20            # number of spatial clusters (K)
        self.save_densify_logs = False           # save per-step importance_score/visibility_count tensors to NPZ

        # V8.4 方案二：椭圆度频率推断
        self.use_ellipticity_freq = False        # enable ellipticity-based frequency inference
        self.ellipticity_mode = "hard"           # hard / soft
        self.ellipticity_thresh = 3.0            # hard threshold for marking high-freq Gaussians
        self.ellipticity_weight = 0.3            # soft modulation strength (0=disabled, 1=full)

        # V8.4 方案三：自适应阈值调度
        self.use_adaptive_thresh = False         # enable adaptive densification threshold
        self.adaptive_mode = "iteration"          # iteration / count / stage
        self.adaptive_p1 = 0.3                   # Phase 1 end ratio (iteration mode)
        self.adaptive_p2 = 0.7                   # Phase 2 end ratio (iteration mode)
        self.adaptive_clone_start_factor = 0.5   # initial clone threshold factor
        self.adaptive_clone_end_factor = 2.0     # final clone threshold factor
        self.adaptive_split_start_factor = 0.5   # initial split threshold factor
        self.adaptive_split_end_factor = 1.5     # final split threshold factor
        self.target_gaussian_count = 9_600_000   # target Gaussian count (count mode)
        self.stage_T1 = 5000                     # Stage 1 end iteration (stage mode)
        self.stage_T2 = 50000                    # Stage 2 end iteration (stage mode)
        self.stage_clone_factor_1 = 2.0          # Stage 1 clone threshold multiplier
        self.stage_clone_factor_3 = 0.5          # Stage 3 clone threshold multiplier
        self.stage_split_factor_1 = 0.5          # Stage 1 split threshold multiplier
        self.stage_split_factor_3 = 1.0          # Stage 3 split threshold multiplier

        # Structure-Aware Gaussian Grouping
        self.use_structure_grouping = False      # enable prototype-based SH template sharing
        self.num_prototypes = 12                 # number of structure prototype clusters
        self.lambda_group_consistency = 0.01     # weight for group consistency loss
        self.lambda_residual_sparsity = 0.001    # weight for residual sparsity loss

        # V8.5: STE learnable mask pruning with lambda(t) scheduling
        self.use_ste_prune = False               # enable STE learnable mask pruning
        self.ste_prune_lambda = 1e-3             # L1 regularization max coefficient (lambda_max)
        self.ste_prune_lambda_min = 1e-5         # L1 regularization min coefficient (lambda_min); 0=constant lambda
        self.ste_prune_lambda_power = 2.0        # scheduling power: lambda(t) = min + (max-min)*(progress^power)
        self.ste_prune_threshold = 0.5           # binarization threshold (p_k > threshold -> M_k=1)
        self.ste_warmup = 10000                  # unified warmup: both L1 regularization and hard pruning start after this
        self.ste_prune_min_alive = 8000000       # absolute floor for min alive gaussians (safety net)
        self.ste_prune_min_alive_ratio = 0.0     # ratio-based protection: min_alive = max(abs_floor, total * ratio); 0 = disabled
        self.ste_hard_prune = False              # enable periodic hard pruning of low-mask Gaussians (memory control)
        self.ste_hard_prune_ratio = 0.05         # max fraction of Gaussians to hard-prune per densify step
        self.ste_hard_prune_thresh = 0.1         # mask probability threshold (below this = candidate for removal)
        self.ste_prune_use_gumbel = False        # use Gumbel-Softmax + temperature annealing instead of sigmoid
        self.ste_prune_tau_init = 1.0            # initial temperature for Gumbel-Softmax (anneals to ~0)

        # GuidedGS: frequency-guided densification (spatial guidance)
        self.use_freq_guided_densify = False     # enable GT frequency map -> per-Gaussian threshold modulation
        self.freq_guided_beta = 0.3              # modulation strength: higher = stronger differentiation
        self.freq_epsilon_min = 0.3              # safety floor on threshold modulation (paper §3.5); prevents runaway densification in high-freq regions

        # Sparsity regularization (loss-side, pushes low-value gaussians to prune)
        self.lambda_opa = 0.0                    # opacity L1 reg (-> existing min_opacity prune); 0=off
        self.lambda_scale = 0.0                   # scale L1 reg (discourages bloated gaussians); 0=off

        # Redundancy prune (remove low-importance + low-opacity + well-observed redundant gaussians)
        self.use_redundancy_prune = False
        self.redundancy_thresh = 2.0              # importance_score < this => redundant candidate
        self.redundancy_opacity = 0.05            # opacity < this (low-contribution)
        self.redundancy_vis_mult = 1.5            # visibility_count > vis_median * this (well-observed)
        self.redundancy_budget_frac = 0.5         # fraction of redundancy candidates actually pruned
        self.redundancy_start_iter = 49875        # = curriculum T1; don't prune on blurred importance

        # GuidedGS v4 §6: Budgeted Importance-Gated Growth (count control)
        self.use_budgeted_growth = False         # replace importance>5 gate with per-step topk growth budget
        self.growth_rate = 0.02                   # per-step growth fraction of cur_count (sweep {0.01,0.015,0.02,0.03})
        self.growth_decay = 1.0                  # late-phase rate multiplier (1.0=off; 0.5 halves late growth)
        self.growth_combine_prune = 0.0          # blend pruning_score into ranking (0.0=pure importance; 0.5/1.0 favor worst-reconstructed)

        # Reproducibility
        self.seed = 0                            # random seed for reproducibility (0=default)

        # VGG/LPIPS perceptual loss (targets the LPIPS metric directly)
        self.lambda_lpips = 0.0                  # perceptual loss weight; 0=off (sweep {0.02,0.05,0.1})

        # Residual-Energy importance gate (magnitude, not count)
        self.use_residual_energy_gate = False
        self.energy_Q = 32                       # L1 quantization levels (importance => residual energy)
        self.tau_energy = 100                    # energy gate threshold (sweep {50,100,200})

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
