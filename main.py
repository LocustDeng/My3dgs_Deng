import os
import sys
import json
import torch
from random import randint
import torch.optim as optim
from scene.data_reader import *
from scene.gaussian_model import *
from pyrender.rasterization import *
from argparse import ArgumentParser, Namespace
import torchvision.transforms as transforms
from pyrender.auxiliary import *


#### render函数，完成渲染过程的参数配置，初始化渲染器
def render(viewpoint_cam, pc : GSModel, scene_info, config, rasterizer):
    ### 计算光栅化过程中可能使用的参数
    ## 计算相机视野角度的切线值
    tanfovx = math.tan(viewpoint_cam.fovX * 0.5)
    tanfovy = math.tan(viewpoint_cam.fovY * 0.5)
    
    ### 进行前向传播 
    # out_color = rasterizer.forward(
    out_color, out_depth, radii, visibility = rasterizer(
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
    return out_color, out_depth, radii, visibility

    
#### train函数
def train(config):
    ### 初始化高斯/从ply文件中加载高斯数据
    gaussians = GSModel(config.sh_degree)
    ## 从colmap输出文件中读取场景相关数据
    scene_info = GSDataLoader(config.source_path, "images")

    ######【选择渲染方案：1.读入colmap数据进行训练；2.加载3dgs训练好的点云继续训练】######
    ## 从点云数据中创建高斯
    gaussians.create_from_pcd(scene_info.points_cloud, 1.0) # 空间缩放因子是随便设的
    ## 从ply文件中加载高斯数据
    # gaussians.load_ply("D:\\My3dgs\\data\\point_cloud.ply")
    #################################################################################

    ### 配置高斯模型训练参数
    gaussians.training_setup(config)

    ###【待实现】背景颜色设置
    
    ### 初始化高斯光栅化器
    rasterizer = GaussianRasterizer()

    #############【目前的迭代次数是1000轮，可在config.json文件中更改配置】###############
    #################################################################################
    ### 训练循环
    ## 训练中可能用到的一些参数
    viewpoint_stack = None # 视点栈，存储相机列表
    first_iter = 0 # 初始迭代次数
    ema_loss_for_log = 0.0 # 记录上一次迭代和当前迭代的加权平均损失
    progress_bar = tqdm(range(first_iter, config.iterations), desc="Training progress") # 训练进度条，显示训练进度
    # progress_bar = tqdm(range(first_iter, 25), desc="Training progress")

    ## 开始训练
    first_iter += 1
    viewpoint_stack = scene_info.cameras.copy()
    for iteration in range(first_iter, config.iterations + 1):
        ## 根据当前迭代次数更新学习率
        gaussians.update_learning_rate(iteration)

        ## 更新球谐系数阶数，每迭代1000次就使球协函数的阶数加1，直至最大
        gaussians.oneupSHdegree()

        ## 复制相机信息列表，并随机选取一个相机视角作为训练渲染视角
        if viewpoint_stack == []:
            viewpoint_stack = scene_info.cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        ## 前向传播，渲染，获取前向传播的返回值
        out_image, out_depth, radii, visibility = render(viewpoint_cam, gaussians, scene_info, config, rasterizer)

        ## 计算loss
        # 获得ground truth
        transform = transforms.ToTensor()
        gt_image = viewpoint_cam.image
        gt_image = transform(gt_image).cuda()
        gt_image = gt_image.permute(2, 1, 0)
        # 计算loss1
        L1 = l1_loss(out_image, gt_image)
        # 计算loss2，ssim
        L2 = ssim(out_image, gt_image)
        # 计算总loss
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * (1.0 - L2)
        # 计算ema_loss_for_log
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

        ## 反向传播
        loss.backward()

        with torch.no_grad():
            ## 如果当前迭代次数在预设的保存迭代次数列表中，打印信息并保存当前scene，场景保存
            if (iteration in config.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                point_cloud_path = os.path.join(scene_info.get_path(), "point_cloud/iteration_{}".format(iteration))
                gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            ## 高斯致密化
            # 致密化时的mask
            visibility_clip = visibility == 1
            # visibility_filter mask没有使用，应该是导致本代码渲染结果与原3dgs渲染所得ply渲染结果相比较模糊的原因
            # visibility_filter = radii > 0
            # 当前迭代次数小于致密化允许的最大次数，允许致密化
            if iteration < config.densify_until_iter:
                # 更新最大半径，用于高斯致密化时的剪枝操作
                gaussians.max_radii2D[visibility_clip] =  torch.max(gaussians.max_radii2D[visibility_clip], radii)
                # 更新累积梯度和
                gaussians.add_densification_stats(rasterizer._points_xyz_camera, visibility_clip)
                # 如果迭代次数大于 opt.densify_from_iter 并且符合密集化间隔，则执行密集化和修剪操作
                if iteration > config.densify_from_iter and iteration % config.densification_interval == 0:
                # if iteration > 0:
                        size_threshold = 20 if iteration > config.opacity_reset_interval else None
                        gaussians.densify_and_prune(config.densify_grad_threshold, 0.005, scene_info.cameras_box_radius, size_threshold)
                # 每当迭代次数符合 opt.opacity_reset_interval，或者数据集背景为白色且迭代次数等于 opt.densify_from_iter 时，重置高斯对象的不透明度
                if iteration % config.opacity_reset_interval == 0 or (config.white_background and iteration == config.densify_from_iter):
                    gaussians.reset_opacity()
            if iteration < config.iterations:
                # 在每个循环后将累积的梯度清零，并利用优化器更新参数
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            # 关闭进度条
            if iteration == config.iterations:
            # if iteration == 25:
                progress_bar.close()

    
    ## 渲染0号相机的照片，作为代码检测
    viewpoint_stack = scene_info.cameras.copy()
    viewpoint_cam = viewpoint_stack.pop(0)
    out_image, out_depth, _, _ = render(viewpoint_cam, gaussians, scene_info, config, rasterizer)
    show_image(out_image, out_depth, iteration)


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

    ############【设置读入colmap数据的路径，此处为绝对路径】############
    ### 设置source_path,后期修改为命令行获取
    config = parser.parse_args(["--source_path", "D:\\My3dgs\\data"])
    #################################################################
    
    ### 解析参数
    # 从命令行获取参数
    # config = parser.parse_args(sys.argv[1:])
    
    ### 测试
    ## 测试参数配置
    # config = parser.parse_args(["--detect_anomaly", "--test_iterations", "40000"])
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
    
    ### 提示训练开始
    print("Train begin.\n")

    ### 进行训练
    train(config)
    
    ### 提示训练结束
    print("\nTrain complete.")




