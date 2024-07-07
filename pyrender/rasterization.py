#!/usr/bin/env python
# coding: utf-8

# In[3]:


import time
from typing import NamedTuple
import torch.nn as nn
import torch
from pyrender.forward import *
from PIL import Image


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
        rect_min = torch.zeros((P, 2), device="cuda", dtype=torch.int)
        rect_max = torch.zeros((P, 2), device="cuda", dtype=torch.int)
        rect_pix_min = torch.zeros((P, 2), device="cuda", dtype=torch.float)
        rect_pix_max = torch.zeros((P, 2), device="cuda", dtype=torch.float)
        
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
            tiles_touched,
            rect_min,
            rect_max,
            rect_pix_min,
            rect_pix_max
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
        # print(means3D)
        # print(viewmatrix)
        # print(radii.shape[0])
        # print(points_xyz_camera)
        # print(points_xyz_proj)
        # print(points_xy_image)
        # print(depths)
        # print(cov3Ds)
        # print(cov2Ds)
        # print(cov2Ds_re)
        # print(rgb)
        # print(conic_opacity)
        # print(tiles_touched)
        # print(rect_min)
        # print(rect_max)
        # print(rect_pix_min)
        # print(rect_pix_max)
        # print(width)
        # print(height)

        ## 重设预处理后点的个数
        P = depths.shape[0]

        ## 遍历每一tile，在遍历的过程中完成对tile所涉及的高斯排序，同时进行渲染，计算像素颜色
        # 定义表示渲染结果的张量
        out_color = torch.zeros((width, height, 3), device="cuda", dtype=torch.float) # 每一像素点的颜色
        final_T = torch.zeros((width, height, 1), device="cuda", dtype=torch.float) # 每一像素点的透明度
        n_contrib = torch.zeros((width, height, 1), device="cuda", dtype=torch.int) # 对每一像素点产生影响的高斯个数

        out_color = sort_render(
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
        )

        # 显示渲染结果
        # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
        # print(out_color.shape)
        image_np = out_color.detach().cpu().numpy()
        image_np = image_np.transpose(1, 0, 2)

        # 确保 RGB 值在 [0, 1] 的范围内，并转换为 [0, 255] 的范围内的整数
        image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)

        # 将 NumPy 数组转换为图像
        image = Image.fromarray(image_np)

        # 显示图像
        image.show()

        # 如果你希望将图像保存到文件
        # image.save("rendered_image.png")
        
        

        




