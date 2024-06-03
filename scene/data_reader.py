#!/usr/bin/env python
# coding: utf-8

# 1.相机视场角：
# FOV的全称是Field of View(视场角)，根据摄像头的成像原理的情况来看，每一个摄像头的成像宽度是固定的，对于不同的焦距，视场角α的值不一样的，焦距越长，视场角越小；焦距越短，视场角越大。对于视野范围来讲，焦距越长，视野范围也远；焦距越短，视野范围越短。
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[22]:


import os
import sys
import math
import numpy as np
import torch
from PIL import Image
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

# 图片信息类，包含图片基本信息，图片对应的相机外参矩阵和投影矩阵
class ImageInfo:
    #### 初始化函数
    def __init__(self, cid, image_name, image_path, image, R, T, focalX, focalY, width, height):
        self.cid = cid
        self.image_name = image_name
        self.image_path = image_path
        self.image = image
        self.R = R
        self.T = T
        self.focalX = focalX # 相机焦距
        self.focalY = focalY
        self.width = width
        self.height = height
        self.fovX = self.processFov(self.focalX, self.width)
        self.fovY = self.processFov(self.focalY, self.height)
        self.znear = 0.01 # 近平面距离
        self.zfar = 100 # 远平面距离
        ## 
        self.ViewMatrix = self.processViewMatrix(self.R, self.T) # 外参
        self.ProjMatrix = self.processProjMatrix(self.znear, self.zfar, self.fovX, self.fovY) # 内参
        # self.ViewProjMatrix = self.ViewMatrix @ self.ProjMatrix # All
        self.ViewProjMatrix = torch.matmul(torch.tensor(self.ViewMatrix, dtype=torch.float32), self.ProjMatrix)
        
    #### 计算相机视场角
    def processFov(self, focal, distance):
        return 2*math.atan(distance/(2*focal))
    
    #### 计算ViewMatrix，外
    def processViewMatrix(self, R, T):
        Rt = np.zeros((4,4))
        Rt[:3, :3] = R.transpose() # 赋的是R的转置
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0
        return Rt
    
    #### 计算ProjMatrix, 内
    def processProjMatrix(self, znear, zfar, fovX, fovY):
        ## 简单三角形几何关系计算left, right，bottom，top
        tanHalfFovX = math.tan((fovX / 2))
        tanHalfFovY = math.tan((fovY / 2))
        right = tanHalfFovX * znear
        left = -right
        top = tanHalfFovY * znear
        bottom = -top
        ## 计算ProjMatrix
        P = torch.zeros(4, 4)
        z_sign = 1.0
        P[0, 0] = (2 * znear) / (right - left)
        P[1, 1] = (2 * znear) / (top - bottom)
        P[2, 2] = (zfar + znear) / (zfar - znear)
        P[2, 3] = (2 * znear * zfar) / (znear - zfar)
        P[3, 2] = z_sign
        ## 返回
        return P
    
# 点云类
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

# 只考虑从colmap的初始化点云中加载数据的情形
class GSDataLoader:
    #### 初始化函数，path：存储colmap输出的路径，render_dir：images文件夹
    def __init__(self, path, reading_dir):
        self.path = path
        self.reading_dir = reading_dir
        self.cameras = [] # 每张图片对应图片信息和相机位姿
        self.points_cloud = {} # 点云数据
        self.readColmap(path, reading_dir) # 在初始化时调用
        
    #### 读取点云数据文件ply的函数
    def fetchPly(self, path):
        plydata = PlyData.read(path) # 读取ply文件
        vertices = plydata['vertex'] # 获取顶点数据
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T # 使用 np.vstack 将 x、y、z 坐标堆叠并转置为 (n, 3) 形状的数组，其中 n 是顶点数目
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0 #  颜色信息同样使用 np.vstack 堆叠，并除以 255.0 将颜色值归一化到 [0, 1] 范围
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T # 法向量信息也使用 np.vstack 堆叠并转置
        return BasicPointCloud(points=positions, colors=colors, normals=normals)
    
    #### 对colmap输出做处理变换的函数
    def processColmap(self, cam_extrinsic, cam_intrinsic, images_folder):
        ### 获取相机参数和图片信息
        for idx, key in enumerate(cam_extrinsic):
            ## 读取信息
            extr = cam_extrinsic[key] # 获取cam_extrinsic中的一个Image实例，对应外参信息
            intr = cam_intrinsic[extr.camera_id] # 用图片对应的相机号获取内参信息
            width = intr.width
            height = intr.height # 尺寸
            cid = intr.id # 相机序号
            ## 求相机外参
            R = np.transpose(qvec2rotmat(extr.qvec)) # 通过四元数求R
            T = np.array(extr.tvec) # 求T
            ## 获得相机焦距
            if intr.model == "SIMPLE_PINHOLE":
                focalX = intr.params[0]
                focalY = intr.params[0]
            elif intr.model == "PINHOLE":
                focalX = intr.params[0]
                focalY = intr.params[1]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            ## 获得图片基本信息
            image_path = os.path.join(images_folder, os.path.basename(extr.name))  # 构成完整的图像文件路径，os.path.basename(extr.name)：获取 extr.name 的基本名称，即去掉路径的文件名
            image_name = os.path.basename(image_path).split(".")[0]  # 获取 image_path 的基本名称，即文件名部分
            image = Image.open(image_path)  # 使用 PIL 库中的 Image.open 函数打开 image_path 指定的图像文件，并返回一个 Image 对象
            cam_info = ImageInfo(cid=cid, image_name=image_name, image_path=image_path, image=image, R=R, T=T, 
                                 focalX=focalX, focalY=focalY, width=width, height=height)
            ## 在相机列表中追加一个新数据
            self.cameras.append(cam_info)
        ### 读取点云
        ## 设置ply文件路径
        ply_path = os.path.join(self.path, "sparse/0/points3D.ply")
        self.points_cloud = self.fetchPly(ply_path)
        '''没有计算相机中心和场景半径'''
            
    #### 从colmap中读数据的函数
    def readColmap(self, path, reading_dir):
        ### 读取colmap输出文件，只考虑bin文件类型，读取相机参数和点云
        # 从images中获取外参，从cameras中获取内参
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsic = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsic = read_intrinsics_binary(cameras_intrinsic_file)

        ### 获取相机信息，对colmap输出进行处理，变为想要的数据形式
        self.processColmap(cam_extrinsic, cam_intrinsic, os.path.join(path, reading_dir))

#### 测试
# l_data = GSDataLoader(r"D:\3DGS\gaussian-splatting(note)\data", "images")
# print(l_data.cameras[1].ViewProjMatrix)
# print(l_data.points_cloud)


# In[ ]:




