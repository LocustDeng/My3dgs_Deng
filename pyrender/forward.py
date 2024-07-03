from math import sqrt, ceil
from typing import NamedTuple
import numpy as np
import torch.nn as nn
import torch
from pyrender.auxiliary import *

#### forward中的preprocess函数，对每个高斯球进行预处理
def preprocess(
    P, 
    D, 
    M, 
    background, 
    width,
    height,
    means3D,
    shs,
    opacities,
    scales,
    scale_modifier,
    rotations,
    viewmatrix,
    projmatrix,
    cam_pos,
    tanfovx,
    tanfovy,
    focal_y,
    focal_x,
    tile_grid,
):
    ### 定义参数
    radii = torch.zeros(P, device="cuda")
    points_xyz_camera = torch.zeros((P, 3), device="cuda", dtype=torch.float)
    points_xyz_proj = torch.zeros((P, 3), device="cuda", dtype=torch.float)
    points_xy_image = torch.zeros((P, 2), device="cuda", dtype=torch.float)
    depths = torch.zeros(P, device="cuda", dtype=torch.float)
    cov3Ds = torch.zeros((P, 3, 3), device="cuda", dtype=torch.float)
    cov2Ds = torch.zeros((P, 2, 2), device="cuda", dtype=torch.float)
    cov2Ds_re = torch.zeros((P, 2, 2), device="cuda", dtype=torch.float)
    rgb = torch.zeros((P, 3), device="cuda", dtype=torch.float)
    conic_opacity = torch.zeros((P, 4), device="cuda", dtype=torch.float)
    tiles_touched = torch.zeros(P, device="cuda", dtype=torch.int)

    ### 计算高斯在相机空间中的坐标
    points_xyz_camera = means3D @ viewmatrix.t()
    #print(points_xyz_camera.shape[0])

    ### 进行近平面裁剪
    ## 标记不可见的点
    visibility = torch.ones(P, device="cuda", dtype=torch.float)
    visibility[points_xyz_camera[:, 2] <= 0.2] = 0
    ## 剔除不可见的点
    # 创建掩码标记可见点的索引
    mask = visibility == 1
    # 依次剔除涉及参数的相应维度
    means3D = means3D[mask]
    shs = shs[mask]
    opacities = opacities[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    radii = radii[mask]
    points_xyz_camera = points_xyz_camera[mask]
    points_xyz_proj = points_xyz_proj[mask]
    points_xy_image = points_xy_image[mask]
    depths = depths[mask]
    cov3Ds = cov3Ds[mask]
    cov2Ds = cov2Ds[mask]
    cov2Ds_re = cov2Ds_re[mask]
    rgb = rgb[mask]
    conic_opacity = conic_opacity[mask]
    tiles_touched = tiles_touched[mask]
    # 更新点的个数
    P = means3D.shape[0]

    ### 计算高斯在投影空间中的坐标
    ## 计算齐次化坐标
    points_xyzw_proj_hom = torch.zeros((P, 4), device="cuda", dtype=torch.float)
    points_xyzw_proj_hom = means3D @ projmatrix.t()
    ## 计算非齐次化坐标
    p_w = 1 / (points_xyzw_proj_hom[:, 3] + 0.0000001)
    points_xyz_proj[:, 0] = points_xyzw_proj_hom[:, 0] * p_w
    points_xyz_proj[:, 1] = points_xyzw_proj_hom[:, 1] * p_w
    points_xyz_proj[:, 2] = points_xyzw_proj_hom[:, 2] * p_w
    
    ### 计算高斯的3D协方差
    cov3Ds = computeCov3D(P, scales, scale_modifier, rotations)
    
    ### 进行投影，计算高斯的2D协方差，EWA
    cov2Ds = computeCov2D(P, points_xyz_camera, focal_x, focal_y, tanfovx, tanfovy, cov3Ds, viewmatrix)
    
    ### 计算2D协方差的逆矩阵
    # ## 计算行列式的值
    # det = cov2Ds[:, 0, 0] * cov2Ds[:, 1, 1] - cov2Ds[:, 0, 1] * cov2Ds[:, 1, 0]
    # det_inv = 1 / det
    # ## 【待实现】对行列式值为0的高斯进行处理
    # ## 构造逆矩阵
    # cov2Ds_re[:, 0, 0] = cov2Ds[:, 1, 1] * det_inv[:]
    # cov2Ds_re[:, 0, 1] = -cov2Ds[:, 0, 1] * det_inv[:]
    # cov2Ds_re[:, 1, 0] = -cov2Ds[:, 1, 0] * det_inv[:]
    # cov2Ds_re[:, 1, 1] = cov2Ds[:, 0, 0] * det_inv[:]
    cov2Ds_re = torch.inverse(cov2Ds)
    
    ### 计算3D高斯的2D投影在图像中的覆盖范围
    ## 计算行列式的值
    det = cov2Ds[:, 0, 0] * cov2Ds[:, 1, 1] - cov2Ds[:, 0, 1] * cov2Ds[:, 1, 0]
    ## 计算2D投影的包围盒
    mid = 0.5 * (cov2Ds[:, 0, 0] + cov2Ds[:, 1, 1])
    lambda1 = mid + torch.sqrt(torch.where((mid * mid - det) > 0.1, (mid * mid - det), torch.tensor(0.1, device="cuda", dtype=torch.float)))
    lambda2 = mid - torch.sqrt(torch.where((mid * mid - det) > 0.1, (mid * mid - det), torch.tensor(0.1, device="cuda", dtype=torch.float)))
    temp_radius = torch.ceil(3.0 * torch.sqrt(torch.where((lambda1 > lambda2), lambda1, lambda2)))
    my_radius = torch.cat((temp_radius.unsqueeze(1), temp_radius.unsqueeze(1)), dim=1)
    ## 计算高斯在图像平面上的坐标，ndc -> pix
    points_xy_image[:, 0] = ((points_xyz_proj[:, 0] + 1.0) * width - 1.0) * 0.5
    points_xy_image[:, 1] = ((points_xyz_proj[:, 1] + 1.0) * height - 1.0) * 0.5
    ## 计算包围盒边界所处的tile
    rect_min = torch.zeros((P, 2), device="cuda", dtype=torch.int)
    rect_max = torch.zeros((P, 2), device="cuda", dtype=torch.int)
    rect_min = torch.where((0 > (points_xy_image - my_radius) / 16), 0, (points_xy_image - my_radius) / 16)
    rect_min = torch.where((tile_grid < rect_min), tile_grid, rect_min)
    rect_max = torch.where((0 > (points_xy_image + my_radius + 15) / 16), 0, (points_xy_image + my_radius + 15) / 16)
    rect_max = torch.where((tile_grid < rect_max), tile_grid, rect_max)
    rect_min = rect_min.int()
    rect_max = rect_max.int()
    ## 【待实现】包围盒为0时不继续处理
    ## 计算高斯影响的tile个数
    tiles_touched[:] = (rect_max[:, 0] - rect_min[:, 0]) * (rect_max[:, 1] - rect_min[:, 1])

    ### 计算高斯的RGB颜色（球谐）
    rgb = computeColorFromSH(P, D, M, means3D, cam_pos, shs)
    
    ### 存储深度
    depths = points_xyz_camera[:, 2]
    
    ### 存储半径
    radii = temp_radius
    
    ### 存储conic_opacity
    conic_opacity[:, 0] = cov2Ds[:, 0, 0]
    conic_opacity[:, 1] = cov2Ds[:, 0, 1]
    conic_opacity[:, 2] = cov2Ds[:, 1, 1]
    conic_opacity[:, 3] = opacities[:, 0]

    ### 返回参数
    return (
        radii,
        points_xyz_camera,
        points_xyz_proj,
        points_xy_image,
        depths,
        cov3Ds,
        cov2Ds,
        cov2Ds_re,
        rgb,
        conic_opacity,
        tiles_touched
    )
    


    

