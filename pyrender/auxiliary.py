from typing import NamedTuple
import numpy as np
import torch.nn as nn
import torch


#### 计算高斯的3D协方差
def computeCov3D(P, scale, mod, q):
    ### 创建S矩阵
    S = torch.zeros((P, 3, 3), device="cuda", dtype=torch.float)
    S[:, 0, 0] = mod * scale[:, 0]
    S[:, 1, 1] = mod * scale[:, 1]
    S[:, 2, 2] = mod * scale[:, 2]
    
    ### 创建R矩阵，四元数
    R = torch.zeros((P, 3, 3), device="cuda", dtype=torch.float)
    R[:, 0, 0] = 1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    R[:, 0, 1] = 2.0 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    R[:, 0, 2] = 2.0 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    R[:, 1, 0] = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R[:, 1, 1] = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 3] * q[:, 3])
    R[:, 1, 2] = 2.0 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R[:, 2, 0] = 2.0 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    R[:, 2, 1] = 2.0 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    R[:, 2, 2] = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
    # print(q)
    # print(R)

    ### 计算3D协方差
    M = R @ S
    cov3Ds = M @ M.transpose(1, 2)
    return cov3Ds


#### 计算高斯的2D协方差
def computeCov2D(P, points_xyz_camera, focal_x, focal_y, tanfovx, tanfovy, cov3Ds, viewmatrix):
    # print(points_xyz_camera)
    ### 限制相机空间下的点坐标在视角范围内
    tx = (points_xyz_camera[..., 0] / points_xyz_camera[..., 2]).clip(min=-tanfovx*1.3, max=tanfovx*1.3) * points_xyz_camera[..., 2]
    ty = (points_xyz_camera[..., 1] / points_xyz_camera[..., 2]).clip(min=-tanfovy*1.3, max=tanfovy*1.3) * points_xyz_camera[..., 2]
    tz = points_xyz_camera[..., 2]
    
    ### 计算雅各比矩阵
    J = torch.zeros((P, 3, 3), device="cuda", dtype=torch.float).to(points_xyz_camera)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y

    ### 提取viewmatrix矩阵的旋转部分，前3行前3列
    W = viewmatrix[:3, :3]

    ### 构造变换矩阵
    T = J @ W

    ### 计算2D协方差
    cov = torch.bmm(torch.bmm(T, cov3Ds), T.permute(0, 2, 1))
    cov2Ds = cov[:, :2, :2]

    ### 进行低通滤波
    cov2Ds[:, 0, 0] += 0.3
    cov2Ds[:, 1, 1] += 0.3
    return cov2Ds


#### 计算高斯的RGB颜色
def computeColorFromSH(P, D, M, means3D, cam_pos, shs):
    ### 计算视线方向
    dir = torch.zeros((P, 3), device="cuda", dtype=torch.float)
    ## 将means3D降维
    means3D_d = means3D[:, :3]
    dir[:] = means3D_d[:] - cam_pos
    ## 将视线方向向量规范化
    length = torch.norm(dir, dim=1)
    length = torch.cat((length.unsqueeze(1), length.unsqueeze(1), length.unsqueeze(1)), dim=1)
    dir = dir / length

    ### 定义球谐系数
    SH_C0 = torch.tensor([0.28209479177387814], device="cuda", dtype=torch.float)
    SH_C1 = torch.tensor([0.4886025119029199], device="cuda", dtype=torch.float)
    SH_C2 = torch.tensor(
        [1.0925484305920792, 
        -1.0925484305920792, 
        0.31539156525252005, 
        -1.0925484305920792, 
        0.5462742152960396],
        device="cuda",
        dtype=torch.float
    )
    SH_C3 = torch.tensor(
        [-0.5900435899266435, 
        2.890611442640554, 
        -0.4570457994644658, 
        0.3731763325901154,
	    -0.4570457994644658,
	    1.445305721320277,
	    -0.5900435899266435],
        device="cuda",
        dtype=torch.float
    )

    ### 计算RGB
    result = torch.zeros((P, 3), device="cuda", dtype=torch.float)
    result[:] = SH_C0 * shs[:, 0, :]
    if D > 0:
        x, y, z = dir[:, 0:1], dir[:, 1:2], dir[:, 2:3]
        result[:] = (
            result[:] 
            - (SH_C1 * y * shs[:, 1, :]) 
            + (SH_C1 * z * shs[:, 2, :]) 
            - (SH_C1 * x * shs[:, 3, :])
        )
        if D > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[:] = (
                result[:]
                + SH_C2[0] * xy * shs[:, 4, :]
                + SH_C2[1] * yz * shs[:, 5, :]
                + SH_C2[2] * (2.0 * zz - xx - yy) * shs[:, 6, :] + 
                + SH_C2[3] * xz * shs[:, 7, :]
                + SH_C2[4] * (xx - yy) * shs[:, 8, :]
            )
            if D > 2:
                result[:] = (
                    result[:]
                    + SH_C3[0] * y * (3.0 * xx - yy) * shs[:, 9, :]
                    + SH_C3[1] * xy * z * shs[:, 10, :]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * shs[:, 11, :] +
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shs[:, 12, :]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * shs[:, 13, :]
                    + SH_C3[5] * z * (xx - yy) * shs[:, 14, :]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * shs[:, 15, :]
                )
        result[:] += 0.5
    return result
    
    

