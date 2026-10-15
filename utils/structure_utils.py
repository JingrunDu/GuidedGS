"""
Structure-Aware Gaussian Grouping utilities.

Provides functions for:
1. Computing local geometry descriptors from point clouds
2. Clustering points into structure prototypes
3. Assigning prototype IDs to new gaussians during densification
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def compute_local_geometry_descriptors(xyz: np.ndarray, k_neighbors: int = 30) -> np.ndarray:
    """Compute local geometry descriptors for each point based on k-nearest neighbors.
    
    For each point, computes:
    - local density (mean distance to k neighbors)
    - linearity, planarity, sphericity (from PCA of local neighborhood)
    - height (z coordinate, normalized)
    
    Uses batched numpy operations for speed on large point clouds.
    
    Args:
        xyz: Point positions, shape [N, 3]
        k_neighbors: Number of neighbors to consider
        
    Returns:
        descriptors: shape [N, 5] (density, linearity, planarity, sphericity, height)
    """
    num_points = len(xyz)
    
    # Find k nearest neighbors (use kd-tree for speed)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='kd_tree', n_jobs=-1)
    nn.fit(xyz)
    distances, indices = nn.kneighbors(xyz)
    
    # Skip self (index 0)
    distances = distances[:, 1:]  # [N, k]
    indices = indices[:, 1:]      # [N, k]
    
    # Local density: inverse of mean neighbor distance
    mean_dist = distances.mean(axis=1)  # [N]
    local_density = 1.0 / (mean_dist + 1e-8)
    
    # Batched PCA-based shape descriptors
    # Gather all neighborhoods: [N, k, 3]
    all_neighbors = xyz[indices]  # [N, k, 3]
    # Center each neighborhood
    centroids = all_neighbors.mean(axis=1, keepdims=True)  # [N, 1, 3]
    centered = all_neighbors - centroids  # [N, k, 3]
    
    # Covariance matrices: [N, 3, 3] = [N, 3, k] @ [N, k, 3] / k
    covs = np.einsum('nki,nkj->nij', centered, centered) / k_neighbors
    
    # Batch eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(covs)  # [N, 3], ascending order
    eigenvalues = eigenvalues[:, ::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    total = eigenvalues.sum(axis=1)  # [N]
    linearity = (eigenvalues[:, 0] - eigenvalues[:, 1]) / total
    planarity = (eigenvalues[:, 1] - eigenvalues[:, 2]) / total
    sphericity = eigenvalues[:, 2] / total
    
    # Normalized height
    height_norm = (xyz[:, 2] - xyz[:, 2].min()) / (xyz[:, 2].ptp() + 1e-8)
    
    # Stack descriptors
    descriptors = np.stack([
        local_density / (local_density.max() + 1e-8),  # normalize to [0, 1]
        linearity,
        planarity,
        sphericity,
        height_norm,
    ], axis=-1)  # [N, 5]
    
    return descriptors


def cluster_into_prototypes(xyz: np.ndarray, num_prototypes: int = 12,
                            k_neighbors: int = 30, random_state: int = 42) -> tuple:
    """Cluster initial point cloud into structure prototypes.
    
    Args:
        xyz: Point positions, shape [N, 3]
        num_prototypes: Number of prototype clusters
        k_neighbors: Neighbors for local descriptor computation
        random_state: Random seed for reproducibility
        
    Returns:
        prototype_ids: shape [N], int array of cluster assignments
        cluster_centers: shape [num_prototypes, 5], descriptor space centers
        descriptors: shape [N, 5], computed descriptors
    """
    print(f"[Structure] Computing local geometry descriptors (k={k_neighbors})...")
    descriptors = compute_local_geometry_descriptors(xyz, k_neighbors=k_neighbors)
    
    print(f"[Structure] Clustering {len(xyz)} points into {num_prototypes} prototypes...")
    kmeans = KMeans(n_clusters=num_prototypes, random_state=random_state, n_init=3, max_iter=100)
    prototype_ids = kmeans.fit_predict(descriptors)
    cluster_centers = kmeans.cluster_centers_
    
    # Print cluster statistics
    for proto_id in range(num_prototypes):
        count = (prototype_ids == proto_id).sum()
        center = cluster_centers[proto_id]
        print(f"  Prototype {proto_id:2d}: {count:6d} points "
              f"(density={center[0]:.2f}, linear={center[1]:.2f}, "
              f"planar={center[2]:.2f}, spheric={center[3]:.2f}, height={center[4]:.2f})")
    
    return prototype_ids, cluster_centers, descriptors


def assign_prototype_for_new_gaussians(parent_prototype_ids: torch.Tensor,
                                       split_mask: torch.Tensor) -> torch.Tensor:
    """Assign prototype IDs to newly created gaussians during densification.
    
    New gaussians inherit their parent's prototype ID.
    
    Args:
        parent_prototype_ids: Current prototype IDs, shape [N]
        split_mask: Boolean mask indicating which gaussians are being split, shape [N]
        
    Returns:
        new_prototype_ids: Prototype IDs for the new gaussians
    """
    return parent_prototype_ids[split_mask]


def compute_group_consistency_loss(gaussians, prototype_ids: torch.Tensor,
                                   num_prototypes: int) -> torch.Tensor:
    """Compute group consistency loss encouraging similar properties within each prototype.
    
    Uses vectorized scatter operations to avoid per-group for-loops.
    Penalizes variance of opacity and scale within each group.
    
    Args:
        gaussians: GaussianModel instance
        prototype_ids: shape [N], prototype assignment per gaussian
        num_prototypes: total number of prototypes
        
    Returns:
        loss: scalar tensor
    """
    opacity = gaussians.get_opacity.squeeze(-1)  # [N]
    scaling = gaussians.get_scaling  # [N, 3]
    
    # Vectorized group mean and variance using scatter
    # Opacity variance per group
    ids = prototype_ids.long()
    group_counts = torch.zeros(num_prototypes, device=opacity.device)
    group_counts.scatter_add_(0, ids, torch.ones_like(opacity))
    group_counts = group_counts.clamp(min=1.0)
    
    # Mean opacity per group
    group_opacity_sum = torch.zeros(num_prototypes, device=opacity.device)
    group_opacity_sum.scatter_add_(0, ids, opacity)
    group_opacity_mean = group_opacity_sum / group_counts  # [K]
    
    # Variance = E[(x - mean)^2]
    opacity_diff_sq = (opacity - group_opacity_mean[ids]) ** 2
    group_opacity_var_sum = torch.zeros(num_prototypes, device=opacity.device)
    group_opacity_var_sum.scatter_add_(0, ids, opacity_diff_sq)
    opacity_var = (group_opacity_var_sum / group_counts).mean()
    
    # Scale variance per group (average across 3 dims)
    ids_expanded = ids.unsqueeze(-1).expand_as(scaling)  # [N, 3]
    group_scale_sum = torch.zeros(num_prototypes, 3, device=scaling.device)
    group_scale_sum.scatter_add_(0, ids_expanded, scaling)
    group_scale_mean = group_scale_sum / group_counts.unsqueeze(-1)  # [K, 3]
    
    scale_diff_sq = (scaling - group_scale_mean[ids]) ** 2
    group_scale_var_sum = torch.zeros(num_prototypes, 3, device=scaling.device)
    group_scale_var_sum.scatter_add_(0, ids_expanded, scale_diff_sq)
    scale_var = (group_scale_var_sum / group_counts.unsqueeze(-1)).mean()
    
    return opacity_var + scale_var


def compute_residual_sparsity_loss(sh_residual: torch.Tensor,
                                   residual_scale: torch.Tensor) -> torch.Tensor:
    """Compute L1 sparsity loss on SH residuals to encourage using templates.
    
    Args:
        sh_residual: shape [N, num_sh_coeffs, 3]
        residual_scale: shape [N, 1]
        
    Returns:
        loss: scalar tensor
    """
    # Effective residual magnitude; unsqueeze scale for broadcast [N,1,1] * [N,C,3]
    effective_residual = residual_scale.unsqueeze(-1) * sh_residual  # [N, C, 3]
    return torch.abs(effective_residual).mean()
