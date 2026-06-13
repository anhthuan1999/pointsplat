import torch
import gin 
import gsplat 
import math
import numpy as np
import os, cv2
from collections import OrderedDict
from plyfile import PlyData, PlyElement
import json
from argparse import Namespace
import torch_scatter
BLOCK_WIDTH = 16 

C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5
def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def rasterize_gaussians_to_multiimgs(gs_params, cameras, gt_imgs):
    camera_to_worlds = cameras['camera_to_worlds']
    rgbs, alphas = [], []
    idx = 0
    for camera_to_world in camera_to_worlds:
        rgb, alpha = rasterize_gaussians_to_singleimg(gs_params, camera_to_world, gt_imgs[idx], **cameras)
        rgbs.append(rgb)
        alphas.append(alpha)
        idx+=1
    return rgbs, alphas

def rasterize_gaussians_to_singleimg(gs_params, camera_to_world, gt_img, cx, cy, fx, fy, width, height, background_color, **kwargs):
    #Turn half to float
    gs_params = {k:v.float() if v.dtype==torch.half else v for k,v in gs_params.items()}
    R = camera_to_world[:3, :3]
    T = camera_to_world[:3, 3:4]
    # flip the z and y axes to align with gsplat conventions (opengl/blender to opencv/colmap)
    R_edit = torch.diag(torch.tensor([1, -1, -1], device='cuda', dtype=R.dtype))
    R = R @ R_edit
    R_additional = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device='cuda', dtype=R.dtype)
    R = R @ R_additional
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv

    means = gs_params['means']
    scales = torch.exp(gs_params['scales'])
    quats = gs_params['quats']/torch.norm(gs_params['quats'], dim=-1, keepdim=True)
    mask = (quats.norm(dim=-1) - 1)<1e-6
    inv_mask = ~mask
    if inv_mask.sum() > 0:
        print(f"Warning: {mask.sum()} quaternions are not normalized\n, This quaternions are ")
        quats[inv_mask] = torch.tensor([0, 0, 0, 1.], device=quats.device)

    if 'opacities' in gs_params:
        opacities = torch.sigmoid(gs_params['opacities'])
    elif 'opacities_sigmoid' in gs_params:
        opacities = gs_params['opacities_sigmoid']
    else:
        raise ValueError("No opacities found in gs_params")
    if 'features_rest' in gs_params:
        colors = torch.cat([gs_params['features_dc'].unsqueeze(1), gs_params['features_rest']], dim=1)
    else:
        colors = gs_params['features_dc'].unsqueeze(1)
    n = int(math.sqrt(colors.shape[1])-1)
    if n==0:
        rgbs = torch.sigmoid(colors[:,0,:])
    else:
        viewdirs_ = means.detach() - camera_to_world.detach()[:3, 3]  # (N, 3)
        viewdirs_norm = viewdirs_.norm(dim=-1, keepdim=True)
        viewdirs = viewdirs_ / viewdirs_norm
        ## In some extremely rare case, the gs mean can be the same as the camera position
        # In this case, we set viewdirs randomly
        if torch.isnan(viewdirs).any():
            mask_ = (viewdirs_norm==0).squeeze() #N,
            newviewdir = torch.randn_like(viewdirs_[mask_]) #N,3
            newviewdir_norm = newviewdir.norm(dim=-1, keepdim=True)
            viewdirs[mask_] = newviewdir/newviewdir_norm
            
        rgbs = gsplat.spherical_harmonics(n, viewdirs, colors)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
    
    actual_height, actual_width = gt_img.shape[:2]
    scale_x = actual_width / width.item()
    scale_y = actual_height / height.item()

    # Scale only the principal point to match actual image resolution; keep focal lengths as calibrated
    cx = cx * scale_x - 0.5
    cy = cy * scale_y - 0.5

    H = actual_height
    W = actual_width

    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = gsplat.project_gaussians(  # type: ignore
        means,
        scales,
        1,
        quats,
        viewmat.squeeze()[:3, :].float(),
        fx.item(),
        fy.item(),
        cx.item(),
        cy.item(),
        H,
        W,
        BLOCK_WIDTH,
    ) 
    rgb, alpha = gsplat.rasterize_gaussians(  
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,  
        rgbs,
        opacities,
        H,
        W,
        BLOCK_WIDTH,
        background = background_color,
        return_alpha=True,
    )  

    rgb = torch.clamp(rgb, max=1.0)  
    alpha = alpha.unsqueeze(-1)

    return rgb, alpha


@gin.configurable(allowlist=['opacities', 'take_from_input','scales'])
def create_pseudo_target(sh_degree, input_gs, take_from_input, N, opacities, scales):
    target = {k:input_gs[k] for k in take_from_input}
    assert 'means' in target and 'features_dc'in target
    if 'scales' not in target:
        target['scales'] = torch.ones_like(target['means'])*torch.log(torch.tensor(scales))
    if 'opacities' in input_gs: #As logit
        if 'opacities' not in target:
            target['opacities'] = torch.logit(torch.tensor(opacities))*torch.ones(N, 1)
    if 'features_rest' not in target and sh_degree>0:
        sh_dim = (sh_degree+1)**2-1
        target['features_rest'] = torch.zeros(N, sh_dim, 3)
    if 'quats' not in target:
        target['quats'] = torch.tensor([0, 0, 0, 1.]).repeat(N, 1)
    return target


    

