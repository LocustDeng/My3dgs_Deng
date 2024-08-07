from math import sqrt, ceil
from typing import NamedTuple
import numpy as np
import torch.nn as nn
import torch
from pyrender.auxiliary import *
from tqdm import tqdm

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
    points_xyz_camera = torch.zeros((P, 4), device="cuda", requires_grad=True, dtype=torch.float)
    points_xyz_proj = torch.zeros((P, 3), device="cuda", dtype=torch.float)
    points_xy_image = torch.zeros((P, 2), device="cuda", dtype=torch.float)
    depths = torch.zeros(P, device="cuda", dtype=torch.float)
    cov3Ds = torch.zeros((P, 3, 3), device="cuda", dtype=torch.float)
    cov2Ds = torch.zeros((P, 2, 2), device="cuda", dtype=torch.float)
    cov2Ds_re = torch.zeros((P, 2, 2), device="cuda", dtype=torch.float)
    rgb = torch.zeros((P, 3), device="cuda", dtype=torch.float)
    conic_opacity = torch.zeros((P, 4), device="cuda", dtype=torch.float)
    tiles_touched = torch.zeros(P, device="cuda", dtype=torch.int)
    rect_min = torch.zeros((P, 2), device="cuda", dtype=torch.int)
    rect_max = torch.zeros((P, 2), device="cuda", dtype=torch.int)
    rect_pix_min = torch.zeros((P, 2), device="cuda", dtype=torch.float)
    rect_pix_max = torch.zeros((P, 2), device="cuda", dtype=torch.float)

    ### 计算高斯在相机空间中的坐标
    points_xyz_camera = means3D @ viewmatrix.t()
    points_xyz_camera.retain_grad

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
    ## 计算2D投影的包围盒
    # 写法1
    # 计算行列式的值
    # det = cov2Ds[:, 0, 0] * cov2Ds[:, 1, 1] - cov2Ds[:, 0, 1] * cov2Ds[:, 1, 0]
    # mid = 0.5 * (cov2Ds[:, 0, 0] + cov2Ds[:, 1, 1])
    # lambda1 = mid + torch.sqrt(torch.where((mid * mid - det) > 0.1, (mid * mid - det), torch.tensor(0.1, device="cuda", dtype=torch.float)))
    # lambda2 = mid - torch.sqrt(torch.where((mid * mid - det) > 0.1, (mid * mid - det), torch.tensor(0.1, device="cuda", dtype=torch.float)))
    # temp_radius = torch.ceil(3.0 * torch.sqrt(torch.where((lambda1 > lambda2), lambda1, lambda2)))
    # 写法2
    a = cov2Ds[:, 0, 0]
    b = cov2Ds[:, 0, 1]
    c = cov2Ds[:, 1, 0]
    d = cov2Ds[:, 1, 1]
    lambda1 = 0.5 * (a + d) + 0.5 * torch.sqrt((a - d) * (a - d) + 4 * b * c)
    temp_radius = 3.0 * torch.sqrt(lambda1).ceil()
    my_radius = torch.cat((temp_radius.unsqueeze(1), temp_radius.unsqueeze(1)), dim=1)
    ## 计算高斯在图像平面上的坐标，ndc -> pix
    points_xy_image[:, 0] = ((points_xyz_proj[:, 0] + 1.0) * width - 1.0) * 0.5
    points_xy_image[:, 1] = ((points_xyz_proj[:, 1] + 1.0) * height - 1.0) * 0.5
    ## 计算包围盒边界所处的tile
    rect_min = torch.where((0 > (points_xy_image - my_radius) / 16), 0, (points_xy_image - my_radius) / 16)
    rect_min = torch.where((tile_grid < rect_min), tile_grid, rect_min)
    rect_max = torch.where((0 > (points_xy_image + my_radius + 15) / 16), 0, (points_xy_image + my_radius + 15) / 16)
    rect_max = torch.where((tile_grid < rect_max), tile_grid, rect_max)
    rect_min = rect_min.int()
    rect_max = rect_max.int()
    ## 【待实现】包围盒为0时不继续处理
    ## 计算高斯影响的tile个数
    tiles_touched[:] = (rect_max[:, 0] - rect_min[:, 0]) * (rect_max[:, 1] - rect_min[:, 1])
    ## 计算包围盒边界所处的tile
    size = torch.zeros(2, device="cuda", dtype=torch.float)
    size[0] = width
    size[1] = height
    rect_pix_min = torch.where((0 > (points_xy_image - my_radius)), 0, (points_xy_image - my_radius))
    rect_pix_min = torch.where((size < rect_pix_min), size - 1, rect_pix_min)
    rect_pix_max = torch.where((0 > (points_xy_image + my_radius)), 0, (points_xy_image + my_radius))
    rect_pix_max = torch.where((size < rect_pix_max), size - 1, rect_pix_max)

    ### 计算高斯的RGB颜色（球谐）
    rgb = computeColorFromSH(P, D, M, means3D, cam_pos, shs)
    
    ### 存储深度
    depths = points_xyz_camera[:, 2]
    
    ### 存储半径
    radii = temp_radius
    
    ### 存储conic_opacity
    opacities.clip(min=0, max=1)
    conic_opacity[..., 0] = cov2Ds_re[..., 0, 0]
    conic_opacity[..., 1] = cov2Ds_re[..., 0, 1]
    conic_opacity[..., 2] = cov2Ds_re[..., 1, 1]
    conic_opacity[..., 3] = opacities[..., 0]

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
        tiles_touched,
        rect_min,
        rect_max,
        rect_pix_min,
        rect_pix_max,
        visibility
    )


#### forward中的sort_render函数， 遍历每一tile，在遍历的过程中完成对tile所涉及的高斯排序，同时进行渲染，计算像素颜色
def sort_render(
    P,
    tile_grid,
    width,
    height,
    depths,
    points_xy_image,
    rgb,
    conic_opacity,
    rect_min,
    rect_max,
    rect_pix_min,
    rect_pix_max,
    background
):
    ### 初始化表示渲染结果的张量
    out_color = torch.zeros((width, height, 3), device="cuda", dtype=torch.float) # 每一像素点的颜色
    out_depth = torch.zeros((width, height, 1), device="cuda", dtype=torch.float) # 每一像素点的深度
    final_T = torch.zeros((width, height, 1), device="cuda", dtype=torch.float) # 每一像素点的透明度
    n_contrib = torch.zeros((width, height, 1), device="cuda", dtype=torch.int) # 对每一像素点产生影响的高斯个数

    ### 定义命令行进度条，显示每一轮的渲染进度
    # progress_bar = tqdm(range(0, (tile_grid[0] * tile_grid[1]).item()), desc="Rendering progress")
    ### 遍历每一tile，完成排序和渲染
    for t in range(0, (tile_grid[0] * tile_grid[1]).item()):
    # for t in range(0, 1):
        ## 计算当前tile的坐标范围，横x竖y
        # tile对应的序号
        tile_x = int(t % tile_grid[0])
        tile_y = int(t / tile_grid[0])
        # 当前tile对应的像素范围，并对图像边界进行检查，tile对应的像素范围不能超过图像边界
        pix_min = torch.zeros(2, device="cuda", dtype=torch.int)
        pix_max = torch.zeros(2, device="cuda", dtype=torch.int)
        pix_min[0] = tile_x * 16
        pix_min[1] = tile_y * 16
        pix_max[0] = tile_x * 16 + 15
        pix_max[0] = torch.where((pix_max[0] > width), (width - 1), pix_max[0])
        pix_max[1] = tile_y * 16 + 15
        pix_max[1] = torch.where((pix_max[1] > height), (height - 1), pix_max[1])
        # 当前tile对应的像素宽高
        tile_width = (pix_max[0] - pix_min[0]).item() + 1
        tile_height = (pix_max[1] - pix_min[1]).item() + 1

        ## 找出影响当前tile的所有高斯
        # 设置高斯影响当前tile的条件
        condition = torch.zeros(P, device="cuda", dtype=torch.bool)
        # 等价条件
        # condition[:] = ((tile_x >= rect_min[:, 0]) & (tile_x < rect_max[:, 0]) & (tile_y >= rect_min[:, 1]) & (tile_y < rect_max[:, 1]))
        # condition[:] = (pix_min[0] >= rect_pix_min[:, 0]) & (pix_max[0] <= rect_pix_max[:, 0]) & (pix_min[1] >= rect_pix_min[:, 1]) & (pix_max[1] <= rect_pix_max[:, 1])
        # condition[:] = ~((rect_pix_min[:, 0] >= pix_max[0]) | (rect_pix_max[:, 0] <= pix_min[0]) | (rect_pix_min[:, 1] >= pix_max[1]) | (rect_pix_max[:, 1] <= pix_min[1]))
        over_l = rect_pix_min[..., 0].clip(min=pix_min[0])
        over_t = rect_pix_min[..., 1].clip(min=pix_min[1])
        over_r = rect_pix_max[..., 0].clip(max=pix_max[0])
        over_b = rect_pix_max[..., 1].clip(max=pix_max[1])
        condition[:] = (over_r > over_l) & (over_b > over_t)
        # 标记对当前tile产生影响的高斯，获取对当前tile产生影响的高斯在原列表中的下标
        gaussian_touched_indices = torch.where(condition == True)
        # 利用该下标对高斯深度进行过滤
        depths_filtered = depths[gaussian_touched_indices]

        ## 基于深度对高斯球的各参数排序
        depths_filtered, indices = torch.sort(depths_filtered)
        points_xy_image_filtered = points_xy_image[gaussian_touched_indices][indices]
        rgb_filtered = rgb[gaussian_touched_indices][indices]
        conic_opacity_filtered = conic_opacity[gaussian_touched_indices][indices]

        ## 记录影响当前tile的高斯个数
        tile_P = depths_filtered.size(0)

        ## 对同一tile内的所有像素的颜色进行批量计算
        # 初始化中间变量
        xy = torch.zeros((tile_width, tile_height, tile_P, 2), device="cuda", dtype=torch.float) # 当前2D高斯的均值坐标
        pixel_uv = torch.zeros((tile_width, tile_height, tile_P, 2), device="cuda", dtype=torch.float) # 像素网格坐标
        d = torch.zeros((tile_width, tile_height, tile_P, 2), device="cuda", dtype=torch.float) # 像素点与2D高斯中心的向量
        power = torch.zeros((tile_width, tile_height, tile_P, 1), device="cuda", dtype=torch.float) # 指数部分
        alpha = torch.zeros((tile_width, tile_height, tile_P, 1), device="cuda", dtype=torch.float) # 贡献度（不透明度）alpha
        T = torch.ones((tile_width, tile_height, tile_P, 1), device="cuda", dtype=torch.float) # 透明度
        color = torch.zeros((tile_width, tile_height, 3), device="cuda", dtype=torch.float) # 像素点最终颜色
        depth = torch.zeros((tile_width, tile_height, 1), device="cuda", dtype=torch.float) # 像素点最终深度
        # 对形状不一致的变量进行广播
        con_o = conic_opacity_filtered.unsqueeze(0).unsqueeze(0).expand(tile_width, tile_height, tile_P, 4)
        rgb_c = rgb_filtered.unsqueeze(0).unsqueeze(0).expand(tile_width, tile_height, tile_P, 3)
        depths_c = depths_filtered.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(tile_width, tile_height, tile_P, 1)
        # 计算d
        xy[:, :, :, :] = points_xy_image_filtered.unsqueeze(0).unsqueeze(0)
        pixel_uv[:, :, :, 0] = (torch.arange(tile_width, device="cuda", dtype=torch.float).unsqueeze(1).expand(tile_width, tile_height) + pix_min[0]).unsqueeze(2)
        pixel_uv[:, :, :, 1] = (torch.arange(tile_height, device="cuda", dtype=torch.float).unsqueeze(0).expand(tile_width, tile_height) + pix_min[1]).unsqueeze(2)
        d = xy - pixel_uv
        # 计算power
        power = (-0.5 * (con_o[..., 0] * d[..., 0] * d[..., 0] + con_o[..., 2] * d[..., 1] * d[..., 1]) - con_o[..., 1] * d[..., 0] * d[..., 1]).unsqueeze(-1)
        # 计算alpha
        alpha[:, :, :, 0] = con_o[:, :, :, 3] * torch.exp(power[:, :, :, 0])
        # 计算T
        one_minus_alpha = 1 - alpha
        T = torch.cumprod(one_minus_alpha, dim=2)
        final_T[pix_min[0]:pix_max[0]+1, pix_min[1]:pix_max[1]+1] = T[:, :, (tile_P - 1), 0:1] # 更新当前tile的final_T，final_T=π(i)(1-alpha)
        T = T / one_minus_alpha # 修正T，Ti对应π(i-1)(1-alpha)
        T[:, :, 0, 0] = 1 # 修正T，T1=1
        # 求和计算像素颜色
        color = torch.sum(alpha * T * rgb_c, dim=2)
        # 求和计算像素深度
        depth = torch.sum(alpha * T * depths_c, dim=2)

        ## 更新当前tile的color
        out_color[pix_min[0]:pix_max[0]+1, pix_min[1]:pix_max[1]+1] = color

        ## 更新当前tile的depth
        out_depth[pix_min[0]:pix_max[0]+1, pix_min[1]:pix_max[1]+1] = depth

        ## 更新当前tile的n_contrib
        n_contrib[pix_min[0]:pix_max[0]+1, pix_min[1]:pix_max[1]+1] = tile_P

        ## 更新进度条
        # with torch.no_grad(): # 在不计算梯度的情况下执行，节省内存并加快计算速度
        #         progress_bar.update(1)
    
    ### 关闭进度条
    # progress_bar.close()

    ### 加上背景颜色
    out_color += final_T * background

    ### 返回计算结果
    return out_color, out_depth, final_T, n_contrib
        

            








            


    


    

