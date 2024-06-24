#!/usr/bin/env python
# coding: utf-8

# In[3]:


from typing import NamedTuple
import torch.nn as nn
import torch

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
        pass


# In[ ]:




