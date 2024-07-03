#!/usr/bin/env python
# coding: utf-8

# In[3]:


from typing import NamedTuple
import torch.nn as nn
import torch
from pyrender.forward import *


#### 高斯光栅化器
##【待实现】考虑反向传播
# class GaussianRasterizer(nn.Module):
class GaussianRasterizer():
    ### 初始化函数
    def __init__(self):
        ## 调用父类初始化函数，进行梯度计算
        # super().__init__() 
        pass
    
    ### 前向传播
    def forward(
        self,
        P, # int，高斯的数量
        D, # int，球谐函数的度数
        M, # int，基函数的个数
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
    ):
        ## 重新构造参数，便于批量计算
        # 扩展neans3D至4维
        temp = torch.ones((P, 1), device="cuda", dtype=torch.float)
        means3D_hom = torch.cat((means3D, temp), dim=1)
        # 数据类型统一
        viewmatrix = viewmatrix.to(torch.float)
        projmatrix = projmatrix.to(torch.float)

        ## 计算必要的参数 
        # 计算图像的焦距
        focal_y = height / (2 * tanfovy)
        focal_x = width / (2 * tanfovx)
        # 计算tile块数
        tile_grid = torch.zeros(2, device="cuda", dtype=torch.int)
        tile_grid[0] = (width + 16 - 1) / 16
        tile_grid[1] = (height + 16 - 1) / 16

        ## 初始化需要计算的参数
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
        
        ## 对每个高斯点进行预处理
        (
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
        ) = preprocess(
            P, 
            D, 
            M,
            background, 
            width,
            height,
            means3D_hom,
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
        )
        print(radii)
        print(points_xyz_camera)
        print(points_xyz_proj)
        print(points_xy_image)
        print(depths)
        print(cov3Ds)
        print(cov2Ds)
        print(cov2Ds_re)
        print(rgb)
        print(conic_opacity)
        print(tiles_touched)




