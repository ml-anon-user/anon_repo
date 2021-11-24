import torch
import numpy as np

def regression_loss_fn(alpha, beta, delta, pts_pred, normals_pred, gt_patch_pts, gt_patch_normals, power1, power2):

    normals_pred = normals_pred.unsqueeze(1)
    pts_pred = pts_pred.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)

    # weighting for positions
    dist_square_from_center = (pts_pred - gt_patch_pts).pow(2).sum(2)

    dist_closest_points, cp_idx = torch.min(dist_square_from_center, 1)
    normals_of_closest_points = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(cp_idx)])
    normals_of_closest_points = normals_of_closest_points.view(-1, 3)
    normals_of_closest_points = normals_of_closest_points.unsqueeze(1)

    dist_furthest_points, fp_idx = torch.max(dist_square_from_center, 1)

    # cosine similarity loss
    cosTheta = (normals_pred * normals_of_closest_points).sum(2)
    cosine_similarity_from_pred_normal_cp = (1 - (delta * (cosTheta) ** power1 + (1 - delta) * (cosTheta) ** power2)).squeeze()

    full_position_loss = (1 - beta) * dist_closest_points + beta * dist_furthest_points

    # final loss
    return torch.mean(alpha * full_position_loss + (1 - alpha) * cosine_similarity_from_pred_normal_cp)