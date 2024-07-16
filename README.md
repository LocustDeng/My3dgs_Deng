# My3dgs_Deng
Rewriting of 3dgs：3dgs的Python复现，不涉及Cuda代码

# 项目结构
1. pyrender：实现Python光栅化器，包括forward前向传播过程
2. scene：实现数据读取，定义高斯模型，在高斯模型中定义参数优化器
3. utils：辅助函数
4. main.py: 实现训练主函数，训练结果保存，命令行日志打印
5. config.json：训练参数配置文件，用于设置训练模型参数
6. create_config_json.ipynb: 生成config.json文件，jupyter运行
7. 测试结果：  
    render_image_from_ply1000：使用vanilla 3dgs训练1000轮的点云数据（point_cloud.ply）渲染的结果  
    depth_map_from_ply1000：使用vanilla 3dgs训练1000轮的点云数据（point_cloud.ply）渲染的深度图  
    render_image_0：对vanilla 3dgs数据集初始化的高斯球进行渲染的渲染结果  
    render_image_1000：使用vanilla 3dgs数据集训练1000轮后的渲染结果  
    depth_map_1000：使用vanilla 3dgs数据集训练1000轮后的深度图  
    training_log：使用vanilla 3dgs数据集训练1000轮的命令行运行截图  
8. data：自定义，用于保存需要进行训练的数据（需要使用convert.py提取sfm点云）和训练后的ply文件  

# 项目环境
1. 使用vanilla 3dgs的运行环境"gaussian-splatting"
2. 为了对运行结果的图片进行处理，需额外安装Pillow库

# 项目运行
Anaconda Prompt:
1. activate gaussian-splatting
2. cd My3dgs
3. python main.py

# PS
前期部分代码用jupyter编写，所有的.ipynb文件已转换为对应的.py文件，故.ipynb文件可忽略
