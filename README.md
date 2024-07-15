# My3dgs_Deng
Rewriting of 3dgs：3dgs的Python复现，不涉及Cuda代码

1.项目结构
* pyrender：实现Python光栅化器，包括forward前向传播过程
* scene：实现数据读取，定义高斯模型，在高斯模型中定义参数优化器
* utils：辅助函数
* main.py: 实现训练主函数，训练结果保存，命令行日志打印
* config.json：训练参数配置文件，用于设置训练模型参数
* create_config_json.ipynb: 生成config.json文件，jupyter运行
* 测试结果：
  render_image_from_ply1000：使用vanilla 3dgs训练1000轮时保存的点云数据（point_cloud.ply）进行渲染的渲染结果
  depth_map_from_ply1000：使用vanilla 3dgs训练1000轮时保存的点云数据（point_cloud.ply）进行渲染的深度图
  render_image_0：vanilla 3dgs数据集初始化的高斯球的渲染结果
  render_image_1000：vanilla 3dgs数据集训练1000轮的渲染结果
  depth_map_1000：vanilla 3dgs数据集训练1000轮的深度图
  training_log：训练1000轮的命令行运行截图
* data：自定义，保存需要进行训练的数据和训练后的ply文件

2.项目环境
* vanilla 3dgs运行环境"gaussian-splatting"
* 为了对运行结果的图片进行处理，需额外安装Pillow库

3.项目运行
Anaconda Prompt:
    activate gaussian-splatting
    python main.py

4.注：前期部分代码用jupyter编写，所有的.ipynb文件已转换为对应的.py文件，.ipynb文件可忽略
