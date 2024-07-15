#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from scene.data_reader import BasicPointCloud
from utils.system_utils import mkdir_p

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
    
    @property
    def get_spatial_lr_scale(self):
        return self.spatial_lr_scale
    
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
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    # 设置模型的训练环境和参数
    def training_setup(self, training_args):
        # 初始化变量
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 定义参数组及其学习率
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        # 设置优化器，使用 Adam 优化器，lr=0.0 表示不直接使用优化器的全局学习率，而是使用每个参数组的学习率，eps=1e-15 是 Adam 优化器中的一个参数，用于数值稳定性
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 设置学习率调度器参数
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    #### 更新球谐函数阶数
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    #### 复制高斯球
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        ## 选取符合条件的高斯
        # 根据梯度的范数是否大于或等于 grad_threshold 创建一个布尔掩码 selected_pts_mask，选择梯度大于阈值的点
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 进一步选择，选择梯度大于阈值且最大缩放比例不超过 percent_dense 乘以 scene_extent
        # self.percent_dense 是一个比例值，self.get_scaling 返回点的缩放比例
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent) #选取的是小高斯球
        
        ## 复制参数
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        ## 将新提取的点添加到点云中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    #### 分裂高斯球
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0] # 获取已初始化的高斯点数
        # Extract points that satisfy the gradient condition
        ## 选择符合条件的高斯
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) # 选取的是大高斯球
        ## 计算新点的位置偏移
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds) # 从均值为零且标准差为 stds 的正态分布中采样，生成新点的位移

        ## 计算新点的参数
        rots = build_r(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # 通过矩阵乘法 torch.bmm 将旋转矩阵应用于采样得到的偏移 samples，再加上原始位置 self.get_xyz[selected_pts_mask].repeat(N, 1)，得到新点的位置 new_xyz
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 计算新点的缩放比例 new_scaling，通过对被选中点的缩放比例进行反向激活并按比例缩小
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 其他参数直接复制
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        ## 将新提取的点添加到点云中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
        ## 创建一个修剪掩码 prune_filter，将 selected_pts_mask 与相应数量的零张量拼接在一起，然后调用 prune_points 方法进行修剪操作，将原始被选中的点移除
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    #### 致密化函数，进行复制和剔除操作
    # max_grad：用于密集化操作的最大梯度阈值
    # min_opacity：用于修剪的最小不透明度阈值
    # extent：场景的范围，通常用于判断点是否超出边界
    # max_screen_size：屏幕上点的最大尺寸，用于修剪过大的点
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):    
        # 计算 grads（梯度），通过将累积的梯度 xyz_gradient_accum 除以 denom，！！！！！用xyz算梯度，结合论文
        # 然后将所有 NaN 值设置为0，以避免计算问题
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 分别进行复制和分裂
        self.densify_and_clone(grads, max_grad, extent) # 复制
        self.densify_and_split(grads, max_grad, extent) # 分裂

        # 创建 prune_mask 掩码标记不透明度低于 min_opacity 的点
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # 如果 max_screen_size 被设置，代码会进一步检查屏幕上点的尺寸
        # big_points_vs 标记那些在屏幕上尺寸超过 max_screen_size 的点
        # big_points_ws 标记那些在世界空间中尺度大于场景范围的0.1倍的点
        # 使用逻辑或操作将这些条件与之前的 prune_mask 结合，确保所有需要修剪的点都被标记。
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 将上述被prune_mask标记的点进行剔除
        self.prune_points(prune_mask)

        # 清空CUDA缓存，释放显存，确保在GPU上有足够的内存供后续操作使用
        torch.cuda.empty_cache()

    #### 更新累积梯度
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[...,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    


# In[ ]:




