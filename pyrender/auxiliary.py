from typing import NamedTuple
import numpy as np
import torch.nn as nn
import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image


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
        result.clip(min=0)
    return result


#### 计算Loss，L1
def l1_loss(out_image, gt_image):
    return torch.abs((out_image - gt_image)).mean()


#### 计算Loss，L2，ssim loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def show_image(out_color, out_depth, iteration):
    ## 显示渲染结果
    # 显示渲染得到的图片
    # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
    out_color = out_color.permute(1, 0, 2)
    render_image = out_color.detach().cpu().numpy()

    # 确保 RGB 值在 [0, 1] 的范围内，并转换为 [0, 255] 的范围内的整数
    render_image = (np.clip(render_image, 0, 1) * 255).astype(np.uint8)

    # 将 NumPy 数组转换为图像
    render_image = Image.fromarray(render_image)

    # 显示图像
    # render_image.show()

    # 将图像保存到文件
    render_image.save("render_image_" + str(iteration) + ".png")

    # 显示渲染得到的深度图
    # 将张量从 GPU 复制到 CPU，并转换为 numpy 数组
    out_depth = out_depth.permute(1, 0, 2)
    depth_map = out_depth.detach().cpu().squeeze().numpy()

    # 归一化深度图到 [0, 255] 范围
    depth_map = (255 * (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())).astype('uint8')

    # 使用 PIL 库将深度图转换为图像
    depth_map = Image.fromarray(depth_map)

    # 显示深度图
    # depth_map.show()

    # 将图像保存到文件
    depth_map.save("depth_map_"+ str(iteration) + ".png")

    
    

