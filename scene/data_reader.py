#!/usr/bin/env python
# coding: utf-8

# 1.相机视场角：
# FOV的全称是Field of View(视场角)，根据摄像头的成像原理的情况来看，每一个摄像头的成像宽度是固定的，对于不同的焦距，视场角α的值不一样的，焦距越长，视场角越小；焦距越短，视场角越大。对于视野范围来讲，焦距越长，视野范围也远；焦距越短，视野范围越短。
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[7]:


import os
import sys
import math
import numpy as np
import torch
from PIL import Image
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

# 根据R和T计算世界坐标系到相机坐标系的变换矩阵
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0): # translate（可选）：附加的平移向量，默认为零向量。scale（可选）：缩放因子，默认为1.0。
    # 初始化4*4的变换矩阵RT
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0 # 将4行4列的元素设为1，将其变换为一个齐次矩阵，方便求逆
    # 求Rt的逆矩阵
    C2W = np.linalg.inv(Rt)
    # 提取相机中心（视图坐标系原点在世界坐标系中的位置），写公式可推，（cx，cy，cz）=（0,0,0）
    cam_center = C2W[:3, 3]
    # 对相机中心应用平移和缩放
    cam_center = (cam_center + translate) * scale
    # 获得平移和缩放后的相机中心
    C2W[:3, 3] = cam_center
    # 平移和缩放后的矩阵求逆，获得平移缩放后的变换矩阵
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


# 图片信息类，包含图片基本信息，图片对应的相机外参矩阵和投影矩阵，camera.py
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
        self.ViewProjMatrix = torch.matmul(self.ProjMatrix, torch.tensor(self.ViewMatrix, dtype=torch.float32))
        # 计算相机光心
        self.trans=np.array([0.0, 0.0, 0.0])
        self.scale=1.0
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
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
    points : np.array # 存储所有点云的位置信息（x,y,z）,n*3的矩阵，每行一个点的位置信息
    colors : np.array # 同上，每行一个点的颜色信息
    normals : np.array # 同上，每行一个点的法向量

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
    
    #### 计算相机阵列的中心和平移向量以及包围这些相机中心的最小球体的半径
    def getNerfppNorm(self, cam_info):
        ### 计算所有相机中心的平均值和最长对角线距离（用于计算半径）
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers) # np.hstack()按水平方向（列顺序）堆叠数组构成一个新的数组，堆叠的数组需要具有相同的维度 -》np.vstack()
            # 计算相机中心的平均值
            # axis不设置值，对m*n个数求平均值，返回一个实数,axis = 0：压缩行，对各列求均值，返回1*n的矩阵,axis = 1: 压缩列，对各行求均值，返回m*1的矩阵
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True) 
            center = avg_cam_center
            # 计算每个相机中心到平均中心的距离
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            # 找到最大距离，用于求半径
            diagonal = np.max(dist)
            # 返回所有相机中心的中心，和某相机中心和相机中心的中心的最大长度
            return center.flatten(), diagonal
        # 初始化相机中心列表
        cam_centers = []
        # 计算每个相机的相机中心
        for cam in cam_info:
            # 计算世界坐标系到相机坐标系的矩阵变换
            W2C = getWorld2View2(cam.R, cam.T)
            # 求W2C的逆矩阵C2W
            C2W = np.linalg.inv(W2C)
            # 获取相机的相机中心，即相机光心坐标（0,0,0）在世界坐标系中的坐标
            cam_centers.append(C2W[:3, 3:4])
        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1
        translate = -center
        return {"translate": translate, "radius": radius}
    
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
        ### 计算相机中心和场景半径
        self.cameras_box_center = self.getNerfppNorm(self.cameras)["translate"]
        self.cameras_box_radius = self.getNerfppNorm(self.cameras)["radius"]

            
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

    
    #### 返回存储路径
    def get_path(self):
        return self.path
        

#### 测试
# l_data = GSDataLoader(r"D:\3DGS\gaussian-splatting(note)\data", "images")
# print(l_data.cameras[1].ViewProjMatrix)
# print(l_data.cameras[1].camera_center)
# print(l_data.points_cloud)


# In[ ]:




