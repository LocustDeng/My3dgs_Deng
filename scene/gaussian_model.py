#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from scene.data_reader import BasicPointCloud

# 辅助函数
#### 利用四元数计算R
def build_r(r):
    # 计算了张量 r 中向量的欧几里得范数（大小），每行的第1-3个维度分别平方，相加后再开方
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3]) 
    # 对每个向量进行归一化，None在特定位置增加一个维度
    q = r / norm[:, None]
    # 初始化一个R
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    # 通过使用索引[:, 0]、[:, 1]、[:, 2]和[:, 3]，分别提取了q数组的第一列、第二列、第三列和第四列数据
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    # 设置R矩阵的值
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
    
#### 计算R@S
def build_s_r(s, r):
    # s.shape(0)对应点的数目，每个点的scale矩阵是3*3的
    # 利用缩放因子组装缩放矩阵
    L = torch.zeros((s.shape(0), 3, 3), dtype=torch.float, device="cuda")
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    # 利用四元数计算旋转矩阵
    R = build_r(r)
    # RS
    L = R @ L
    return L

#### 计算协方差矩阵的对称阵
def build_symm(V):
    symm = torch.zeros((V.shape[0], 6), dtype=torch.float, device="cuda")
    symm[:, 0] = V[:, 0, 0]
    symm[:, 1] = V[:, 0, 1]
    symm[:, 2] = V[:, 0, 2]
    symm[:, 3] = V[:, 1, 1]
    symm[:, 4] = V[:, 1, 2]
    symm[:, 5] = V[:, 2, 2]
    return symm

#### inverse_sigmoid
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

#### RGB2SH，将RGB转换到球谐空间
C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

# 高斯模型类
class GSModel:
    #### 初始化函数，初始化高斯球参数
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0 # 当前活跃的球谐函数阶数
        self.max_sh_degree = sh_degree # 球谐函数能达到的最大阶数
        self._xyz = torch.empty(0) # 位置
        self._features_dc = torch.empty(0) # 球协函数的0阶项
        self._features_rest = torch.empty(0) # 球谐函数的剩余阶项 
        self._scaling = torch.empty(0) # 缩放因子
        self._rotation = torch.empty(0) # 旋转矩阵，四元数
        self._opacity = torch.empty(0) # 不透明度
        self.max_radii2D = torch.empty(0) # 最大二维投影半径
        self.xyz_gradient_accum = torch.empty(0) # 位置梯度累积
        self.denom = torch.empty(0) # 
        self.optimizer = None # 优化器实例
        self.percent_dense = 0 # 当前的密度百分比
        self.spatial_lr_scale = 0 # 空间学习率缩放因子
        ### 调用setup_functions()函数，设置对部分参数进行初始化的函数
        self.setup_functions()
        
    #### 初始化高斯部分参数的设置函数
    def setup_functions(self):
        ### 通过R和S计算高斯方差
        def build_covariance_from_RS(scaling, scaling_modifier, rotation):
            V = build_s_r(scaling_modifier * scaling, rotation)
            covariance = V @ V.transpose(1,2) #RS(RS)^T
            ## 因为协方差矩阵是对称阵，所以只存储上三角就可以
            symm = build_symm(covariance)
            return symm
        
        ### 设置其它函数
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_RS
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    #### 获得高斯在世界坐标系中的坐标
    @property
    def get_xyz(self):
        return self._xyz
    
    #### 获得高斯的透明度
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    #### 获得高斯的缩放因子
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    #### 获得高斯的旋转矩阵四元数
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    #### 获得高斯的球谐系数
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    #### 获得高斯个数
    @property
    def get_P(self):
        return self._xyz.shape[0]
    
    #### 计算高斯方差
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    #### 从点云中初始化高斯
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        ### 设置空间缩放因子
        self.spatial_lr_scale = spatial_lr_scale # 只做了赋值，暂时没有使用
        ### 将点云和颜色转换为张量
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        ### 初始化球谐函数的特征张量
        # 第一维：总共有多少点/高斯，第二维：RGB，第三维：特征数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # 部分初始化
        features[:, :3, 0] = fused_color
        ### 初始化缩放比例，高斯球间无重叠
        # 计算任意两点间最小距离的平方,设置最小值1e-7
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales对应三个维度，用repeat直接复制
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1,3)
        ### 初始化旋转四元数
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        ### 初始化不透明度
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        ### 初始化可学习参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # 位置
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # 球谐系数0阶
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # 球谐系数其余阶
        self._scaling = nn.Parameter(scales.requires_grad_(True)) # 缩放比例
        self._rotation = nn.Parameter(rots.requires_grad_(True)) # 旋转参数
        self._opacity = nn.Parameter(opacities.requires_grad_(True)) # 不透明度
        # 初始化为零张量的最大半径（二维），位于 GPU
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    #### 更新球谐函数阶数
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    #### 读取GS点云
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    


# In[ ]:




