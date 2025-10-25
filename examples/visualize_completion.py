"""
GeoSpecNet Visualization Tool
可视化点云补全结果
"""

import sys
sys.path.append('..')

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

from models.GeoSpecNet import GeoSpecNet
from config_geospecnet import cfg


def visualize_point_cloud_matplotlib(point_cloud, title="Point Cloud", color='b', size=1):
    """
    使用matplotlib可视化点云
    
    Args:
        point_cloud: (N, 3) numpy array
        title: 标题
        color: 点的颜色
        size: 点的大小
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(point_cloud[:, 0], 
               point_cloud[:, 1], 
               point_cloud[:, 2],
               c=color, 
               marker='.',
               s=size)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 设置相同的坐标轴范围
    max_range = np.array([
        point_cloud[:, 0].max() - point_cloud[:, 0].min(),
        point_cloud[:, 1].max() - point_cloud[:, 1].min(),
        point_cloud[:, 2].max() - point_cloud[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (point_cloud[:, 0].max() + point_cloud[:, 0].min()) * 0.5
    mid_y = (point_cloud[:, 1].max() + point_cloud[:, 1].min()) * 0.5
    mid_z = (point_cloud[:, 2].max() + point_cloud[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def visualize_completion_stages(partial, coarse, fine1, fine2, gt=None):
    """
    可视化补全的各个阶段
    
    Args:
        partial: (N, 3) partial point cloud
        coarse: (M, 3) coarse completion
        fine1: (K, 3) first refinement
        fine2: (L, 3) final completion
        gt: (G, 3) ground truth (optional)
    """
    num_plots = 5 if gt is not None else 4
    fig = plt.figure(figsize=(20, 4))
    
    # Partial
    ax1 = fig.add_subplot(1, num_plots, 1, projection='3d')
    ax1.scatter(partial[:, 0], partial[:, 1], partial[:, 2], c='red', marker='.', s=1)
    ax1.set_title(f'Partial ({partial.shape[0]} points)')
    ax1.axis('off')
    
    # Coarse
    ax2 = fig.add_subplot(1, num_plots, 2, projection='3d')
    ax2.scatter(coarse[:, 0], coarse[:, 1], coarse[:, 2], c='orange', marker='.', s=1)
    ax2.set_title(f'Coarse ({coarse.shape[0]} points)')
    ax2.axis('off')
    
    # Fine1
    ax3 = fig.add_subplot(1, num_plots, 3, projection='3d')
    ax3.scatter(fine1[:, 0], fine1[:, 1], fine1[:, 2], c='yellow', marker='.', s=1)
    ax3.set_title(f'Fine1 ({fine1.shape[0]} points)')
    ax3.axis('off')
    
    # Fine2
    ax4 = fig.add_subplot(1, num_plots, 4, projection='3d')
    ax4.scatter(fine2[:, 0], fine2[:, 1], fine2[:, 2], c='green', marker='.', s=1)
    ax4.set_title(f'Fine2 ({fine2.shape[0]} points)')
    ax4.axis('off')
    
    # Ground Truth
    if gt is not None:
        ax5 = fig.add_subplot(1, num_plots, 5, projection='3d')
        ax5.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='blue', marker='.', s=1)
        ax5.set_title(f'Ground Truth ({gt.shape[0]} points)')
        ax5.axis('off')
    
    plt.tight_layout()
    plt.savefig('completion_stages.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存到: completion_stages.png")
    plt.show()


def visualize_with_open3d(point_cloud, window_name="Point Cloud", color=None):
    """
    使用Open3D可视化点云（更好的交互性）
    
    Args:
        point_cloud: (N, 3) numpy array
        window_name: 窗口名称
        color: (3,) RGB color, 范围 [0, 1]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    if color is not None:
        colors = np.tile(color, (point_cloud.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 根据Z坐标着色
        z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
        z_normalized = (point_cloud[:, 2] - z_min) / (z_max - z_min)
        colors = plt.cm.viridis(z_normalized)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False
    )


def compare_completion_open3d(partial, completion, gt=None):
    """
    使用Open3D并排比较部分点云和补全结果
    
    Args:
        partial: (N, 3) partial point cloud
        completion: (M, 3) completed point cloud
        gt: (K, 3) ground truth (optional)
    """
    geometries = []
    
    # Partial (红色)
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial)
    pcd_partial.paint_uniform_color([1, 0, 0])  # Red
    pcd_partial.translate([-1.5, 0, 0])
    geometries.append(pcd_partial)
    
    # Completion (绿色)
    pcd_completion = o3d.geometry.PointCloud()
    pcd_completion.points = o3d.utility.Vector3dVector(completion)
    pcd_completion.paint_uniform_color([0, 1, 0])  # Green
    geometries.append(pcd_completion)
    
    # Ground Truth (蓝色)
    if gt is not None:
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt)
        pcd_gt.paint_uniform_color([0, 0, 1])  # Blue
        pcd_gt.translate([1.5, 0, 0])
        geometries.append(pcd_gt)
    
    # 可视化
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Comparison: Partial (Red) | Completion (Green) | GT (Blue)",
        width=1920,
        height=1080,
        left=0,
        top=0
    )


def run_completion_and_visualize(model_path, partial_cloud_path):
    """
    运行补全并可视化
    
    Args:
        model_path: 模型checkpoint路径
        partial_cloud_path: 部分点云文件路径 (.npy or .pcd)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = GeoSpecNet(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载部分点云
    print(f"加载点云: {partial_cloud_path}")
    if partial_cloud_path.endswith('.npy'):
        partial = np.load(partial_cloud_path)
    elif partial_cloud_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(partial_cloud_path)
        partial = np.asarray(pcd.points)
    else:
        raise ValueError("不支持的文件格式，请使用 .npy 或 .pcd")
    
    # 补全
    print("运行补全...")
    partial_tensor = torch.from_numpy(partial).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        coarse, fine1, fine2 = model(partial_tensor)
    
    # 转换为numpy
    coarse = coarse.squeeze(0).cpu().numpy()
    fine1 = fine1.squeeze(0).cpu().numpy()
    fine2 = fine2.squeeze(0).cpu().numpy()
    
    print(f"\n补全完成!")
    print(f"  - 输入: {partial.shape[0]} 点")
    print(f"  - 粗糙: {coarse.shape[0]} 点")
    print(f"  - 精细1: {fine1.shape[0]} 点")
    print(f"  - 精细2: {fine2.shape[0]} 点")
    
    # 保存结果
    np.save('completion_result.npy', fine2)
    print(f"\n结果已保存到: completion_result.npy")
    
    # 可视化
    print("\n生成可视化...")
    visualize_completion_stages(partial, coarse, fine1, fine2)
    
    # Open3D交互式可视化
    print("\n使用Open3D可视化 (可交互)...")
    compare_completion_open3d(partial, fine2)


def demo_with_random_data():
    """
    使用随机数据演示可视化功能
    """
    print("生成演示数据...")
    
    # 生成球形点云作为ground truth
    num_points = 4096
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    radius = np.random.normal(1.0, 0.02, num_points)
    
    gt = np.stack([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi)
    ], axis=1)
    
    # 生成部分点云（只保留一半）
    partial_indices = np.where(gt[:, 0] > 0)[0]
    partial = gt[partial_indices]
    
    # 模拟补全结果（添加噪声）
    completion = gt + np.random.normal(0, 0.05, gt.shape)
    
    print(f"Ground Truth: {gt.shape[0]} points")
    print(f"Partial: {partial.shape[0]} points")
    print(f"Completion: {completion.shape[0]} points")
    
    # 可视化
    print("\n生成matplotlib可视化...")
    visualize_completion_stages(
        partial, 
        partial,  # coarse
        gt[:len(gt)//2],  # fine1
        completion,  # fine2
        gt
    )
    
    print("\n使用Open3D可视化...")
    compare_completion_open3d(partial, completion, gt)


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║              GeoSpecNet Visualization Tool                     ║
    ║                   Point Cloud Completion                       ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n选择模式:")
    print("1. 演示模式 (使用随机数据)")
    print("2. 实际补全 (需要模型和点云文件)")
    
    choice = input("\n请输入选项 (1/2): ").strip()
    
    if choice == '1':
        demo_with_random_data()
    elif choice == '2':
        model_path = input("请输入模型路径: ").strip()
        partial_path = input("请输入部分点云路径: ").strip()
        run_completion_and_visualize(model_path, partial_path)
    else:
        print("无效选项")
    
    print("\n完成!")
