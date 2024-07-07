#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
import sys
import json
from random import randint
from argparse import ArgumentParser, Namespace
from scene.data_reader import *
from scene.gaussian_model import *
from pyrender.rasterization import *

#### render函数，完成渲染过程的参数配置，初始化渲染器
def render(viewpoint_cam, pc : GSModel, scene_info, config):
    ### 计算光栅化过程中可能使用的参数
    ## 计算相机视野角度的切线值
    tanfovx = math.tan(viewpoint_cam.fovX * 0.5)
    tanfovy = math.tan(viewpoint_cam.fovY * 0.5)
    
    ### 初始化高斯光栅化器
    rasterizer = GaussianRasterizer()
    ### 进行前向传播 
    # print(viewpoint_cam.height)
    rasterizer.forward(
        P = pc.get_P,
        D = 3, # int，球谐函数的度数
        M = 16, # int，基函数的个数
        background = torch.tensor([0, 0, 0], device="cuda", dtype=torch.float), # 默认黑色
        width = viewpoint_cam.width,
        height =  viewpoint_cam.height,
        means3D = pc.get_xyz,
        shs = pc.get_features,
        opacities = pc.get_opacity,
        scales = pc.get_scaling,
        scale_modifier = 1.0,
        rotations = pc.get_rotation,
        viewmatrix = torch.from_numpy(viewpoint_cam.ViewMatrix).cuda(), # viewmatrix
        projmatrix = viewpoint_cam.ViewProjMatrix.cuda(), # projmatrix
        cam_pos = viewpoint_cam.camera_center.cuda(), # cam_pos
        tanfovx = tanfovx,
        tanfovy = tanfovy,
    )

    
#### train函数
def train(config):
    print("Train begin.\n")
    ### 初始化高斯
    gaussians = GSModel(config.sh_degree)
    ## 从colmap输出文件中读取场景相关数据
    scene_info = GSDataLoader(config.source_path, "images")
    # print(scene_info.cameras[1].ViewProjMatrix)
    # print(scene_info.points_cloud)
    ##【待实现】计算场景包围盒半径,后期进行高斯球致密化时需要使用
    ## 从点云数据中创建高斯
    # gaussians.create_from_pcd(scene_info.points_cloud, 1.0) # 空间缩放因子是随便设的
    gaussians.load_ply("D:\\Dataset\\data\\point_cloud.ply")
    # print(gaussians._rotation.shape[0])
    # print(gaussians._features_dc)
    ##【待实现】配置高斯模型训练参数
    # gaussians.training_setup(config)
    
    ### 训练循环
    ## 训练中可能用到的一些参数
    viewpoint_stack = None # 视点栈，存储相机列表
    first_iter = 0 # 初始迭代次数
    ## 开始训练
    first_iter += 1
    # for iteration in range(first_iter, config.iterations + 1):
    for iteration in range(first_iter, 2):
        ##【待实现】根据当前迭代次数更新学习率
        # gaussians.update_learning_rate(iteration)
        ## 更新球谐系数阶数，每迭代1000次就使球协函数的阶数加1，直至最大
        gaussians.oneupSHdegree()
        ## 复制相机信息列表，并随机选取一个相机视角作为训练渲染视角
        if not viewpoint_stack:
            viewpoint_stack = scene_info.cameras
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = viewpoint_stack.pop(0)
        # print(viewpoint_cam.fovX)
        # print(build_r(gaussians.get_rotation))
        ## 渲染
        render(viewpoint_cam, gaussians, scene_info, config)
    
    
#### 主函数
if __name__ == "__main__":
    ### 创建解析器
    parser = ArgumentParser(description="Begin training parameters")
    ### 直接在主函数中配置的参数，未考虑命令的缩写
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100,1_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100,1_000, 7_000, 30_000]) 
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[100, 1_000, 7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    ### 从config.json文件中配置参数，未考虑命令缩写
    with open("config.json", "r") as f:
        data = json.load(f)
    for key,value in data.items():
        t = type(value)
        if t == bool:
            parser.add_argument("--"+key, action="store_true", default=value)
        else:
            parser.add_argument("--"+key, type=t, default=value)
    
    ### 解析参数
    # 从命令行获取参数
    # config = parser.parse_args(sys.argv[1:])
    
    ### 测试
    ## config = parser.parse_args(["--detect_anomaly", "--test_iterations", "40000"])
    # config = parser.parse_args(["--source_path", "D:\\3DGS\\gaussian-splatting(note)\\data"]) # 设置source_path,后期修改为命令行获取
    config = parser.parse_args(["--source_path", "D:\\Dataset\\data"])
    # print(config.source_path) # 测试参数配置是否成功
    # print(config.sh_degree)
    ## 测试数据读取模块
    # l_data = GSDataLoader(config.source_path, "images")
    # print(l_data.cameras[1].ViewProjMatrix)
    # print(l_data.points_cloud)
    ## 测试高斯模型
    # gaussian = GSModel(3)
    # gaussian.create_from_pcd(l_data.points_cloud, 1.0) # 空间缩放因子是随便设的
    # print(gaussian.max_sh_degree)
    # print(gaussian._rotation)
    
    ### 进行训练
    train(config)
    
    ### 提示训练结束
    print("\nTrain complete.")


# In[ ]:





# In[ ]:




