from __future__ import division

import torch


def mm_pre(A, x):
    # Premultiply vector `x` by `A` (batched).
    return torch.matmul(x, A.transpose(2, 1))


def projection(vertices, K, R, t, dist_coeffs, image_size, left_hand, eps=1e-9):
    assert torch.all(dist_coeffs == 0.0), dist_coeffs
    return simple_projection(
        p_WV=vertices,
        K=K,
        R_CW=R,
        t_CW=t,
        image_size=image_size,
        left_hand=left_hand,
        eps=eps,
    )


def simple_projection(p_WV, K, R_CW, t_CW, image_size, left_hand, eps):
    # Project from world coordiantes to camera coordinates.
    #   V - vertices
    #   W - world (scene)
    #   C - camera
    p_CV = mm_pre(R_CW, p_WV) + t_CW
    x, y, z = p_CV[:, :, 0], p_CV[:, :, 1], p_CV[:, :, 2]
    # Normalize.
    x_ = x / (z + eps)
    y_ = y / (z + eps)
    z_ = torch.ones_like(z)

    # Homogeneous, normalized coordinatges.
    ph_CV = torch.stack([x_, y_, z_], dim=-1)
    # Pinhole projection (pixels).
    uvh = mm_pre(K, ph_CV)
    u, v = uvh[:, :, 0], uvh[:, :, 1]
    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    sh = image_size / 2.0
    u = (u - sh) / sh
    v = (v - sh) / sh
    if not left_hand:
        # Flip y.
        v = -v
    uvz = torch.stack([u, v, z], dim=-1)
    return uvz
