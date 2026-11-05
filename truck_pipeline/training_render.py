"""Training render dispatch, independently testable without a CUDA extension.

The pinned pose rasterizer computes alpha in forward but does not implement its
backward gradient. Use the base rasterizer's *native accumulated opacity* for
silhouette optimization. RGB/depth keep pose derivatives while pose refinement
is active. Once refinement ends, one base render provides every output.
"""
import torch.nn.functional as F


def render_training_view(camera, gaussians, pipe, background, *, refine_pose,
                         need_silhouette, render_fixed, render_pose):
    if not refine_pose:
        return render_fixed(camera, gaussians, pipe, background)
    result = render_pose(camera, gaussians, pipe, background)
    if need_silhouette:
        fixed = render_fixed(camera, gaussians, pipe, background)
        result['silhouette_alpha'] = fixed['rendered_alpha']
    return result


def silhouette_loss(render_package, mask, loss_type='bce'):
    alpha = render_package.get('silhouette_alpha', render_package['rendered_alpha'])
    if loss_type == 'bce':
        return F.binary_cross_entropy(alpha, mask)
    if loss_type == 'mse':
        return F.mse_loss(alpha, mask)
    raise ValueError(f'Unsupported silhouette loss: {loss_type}')
