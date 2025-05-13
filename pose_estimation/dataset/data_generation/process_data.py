import numpy as np
import trimesh
from trimesh import transformations
import argparse
import os, sys
from pathlib import Path
import re
from pdb import set_trace as bp
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pose_estimation.utils import farthest_point_sampling, find_partition
import torch
from sklearn.decomposition import PCA


def read_camera_pose(file_path):
    # Implement the function to read camera pose from file
    # For example, assuming the file contains a 4x4 transformation matrix
    return np.loadtxt(file_path)

def quan2rot(q):
    r = R.from_quat(q)
    return r.as_matrix()


def read_point_clouds(pts_files, mask_files):
    #! input is a list of point cloud files
    #! output is a array of point clouds [M, N, 3], 
    #! where M is the number of point clouds, N is the number of points in each point cloud
    # Implement the function to read point cloud from file
    # For example, assuming the file contains Nx3 points
    point_cloud = []
    mask = []
    assert len(pts_files) == len(mask_files)
    for i in range(len(pts_files)):
        pts = torch.tensor(np.loadtxt(pts_files[i]))
        msk = np.loadtxt(mask_files[i])
        pts_fps, indice = farthest_point_sampling(pts, 1000)
        pts_fps = np.array(pts_fps)
        msk = msk[indice]
        point_cloud.append(pts_fps)
        mask.append(msk)
    point_cloud = np.array(point_cloud)
    mask = np.array(mask).squeeze()[:,:,None]
    return point_cloud, mask



def transform_point_clouds(point_cloud, camera_pose):
    camera_pose = np.linalg.inv(camera_pose)
    M, N, _ = point_cloud.shape
    # Add a column of ones to the point cloud to make it Nx4
    ones = np.ones((M, N, 1))
    point_cloud_homogeneous = np.concatenate([point_cloud, ones], axis=2)

    # # Transform the point cloud to the camera coordinate system
    transformed_point_cloud = np.matmul(camera_pose, point_cloud_homogeneous.transpose(0, 2, 1)).transpose(0, 2, 1)
    
    # Return the transformed point cloud, dropping the homogeneous coordinate
    return transformed_point_cloud[:, :, :3]

def normalize_and_randomize_point_cloud(point_cloud, num_data):
    # Implement the function to normalize and randomize the point cloud
    # Normalize the point cloud to have a mean of 0 and standard deviation of 1
    # repeat the point cloud for num_data times
    point_cloud = np.repeat(point_cloud[None], num_data, axis=0)
    b, M, N, _ = point_cloud.shape
    point_cloud = point_cloud.reshape(b*M, N, 3)
    center = np.mean(point_cloud, axis=1).reshape(b*M, 1, 3)
    point_cloud_normalized = (point_cloud - center)# / np.std(point_cloud, axis=0)

    # 随机生成一个四元数
    random_quaternion = np.random.randn(b*M,4)  # 生成一个长度为4的随机向量
    random_quaternion /= np.linalg.norm(random_quaternion, axis=0)  # 归一化为单位四元数
    # 将四元数转换为旋转矩阵
    rotation_matrix = quan2rot(random_quaternion)

    # 随机旋转点云
    point_cloud_normalized = np.matmul(point_cloud_normalized ,rotation_matrix.transpose(0, 2, 1))

    point_cloud_normalized = point_cloud_normalized.reshape(b, M, N, 3)
    rotation_matrix = rotation_matrix.reshape(b, M, 3, 3)
    center = center.reshape(b, M, 3)
    
    return point_cloud_normalized, rotation_matrix, center

# 使用PCA对批量点云进行规范化
def canonicalize_point_cloud_batch(point_cloud_batch):
    B = point_cloud_batch.shape[0]  # 获取batch大小
    canonicalized_pcd_batch = []
    rotation_matrices = []
    centroids = []
    for b in range(B):
        # 1. 获取单个点云
        point_cloud = point_cloud_batch[b]
        # 2. 中心化点云（使其均值为0）
        centroid = np.mean(point_cloud, axis=0)
        centered_pcd = point_cloud - centroid
        # 3. 使用PCA
        pca = PCA(n_components=3)
        pca.fit(centered_pcd)
        # 4. 获取旋转矩阵（主轴方向）
        rotation_matrix = pca.components_
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix = -rotation_matrix
        # 5. 将点云投影到主轴上
        transformed_pcd = pca.transform(centered_pcd)
        # 6. 存储结果
        canonicalized_pcd_batch.append(transformed_pcd)
        rotation_matrices.append(rotation_matrix)
        centroids.append(centroid)
    # 将结果转为 NumPy 数组
    canonicalized_pcd_batch = np.array(canonicalized_pcd_batch)
    rotation_matrices = np.array(rotation_matrices)
    centroids = np.array(centroids)

    return canonicalized_pcd_batch, rotation_matrices, centroids

def canonicalize_point_cloud_batch_heu(point_cloud_batch, k=3):
    B = point_cloud_batch.shape[0]  # 获取batch大小
    canonicalized_pcd_batch = []
    rotation_matrices = []
    centroids = []
    for b in range(B):
        # 1. 获取单个点云
        point_cloud = point_cloud_batch[b]
        # 2. 中心化点云（使其均值为0）
        center = np.mean(point_cloud, axis=0)

        point_cloud = point_cloud - center
        # 计算每个点到质心的距离
        distances = np.linalg.norm(point_cloud, axis=1)
        # 找到离质心最远的 k 个点的索引
        farthest_k_indices = np.argsort(distances)[-k:]
        # 计算最远的 k 个点和质心连线的方向
        directions = point_cloud[farthest_k_indices]
        # 计算平均方向作为 x 轴
        x_axis = np.mean(directions, axis=0)
        x_axis /= np.linalg.norm(x_axis)  # 归一化 x 轴向量
        
        # 将所有点投影到与 x 轴正交的平面
        projected_cloud = point_cloud - np.outer(np.dot(point_cloud, x_axis), x_axis)
        # 找到投影后的点中离质心最远的 k 个点的索引
        projected_distances = np.linalg.norm(projected_cloud, axis=1)
        projected_farthest_k_indices = np.argsort(projected_distances)[-k:]
        # 计算投影后的 k 个点和质心连线的方向作为 y 轴
        projected_directions = projected_cloud[projected_farthest_k_indices]
        y_axis = np.mean(projected_directions, axis=0)
        y_axis /= np.linalg.norm(y_axis)  # 归一化 y 轴向量
        
        # 根据右手系规则计算 z 轴，即 z_axis = x_axis × y_axis
        z_axis = np.cross(x_axis, y_axis)
        
        # 构建旋转矩阵
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        # print(rotation_matrix)
        canonical_cloud = point_cloud  @ rotation_matrix
        canonicalized_pcd_batch.append(canonical_cloud)
        rotation_matrices.append(rotation_matrix.T)
        centroids.append(center)
    # 将结果转为 NumPy 数组
    canonicalized_pcd_batch = np.array(canonicalized_pcd_batch)
    rotation_matrices = np.array(rotation_matrices)
    centroids = np.array(centroids)

    return canonicalized_pcd_batch, rotation_matrices, centroids


def main(args):
    pts_root = Path(args.data_dir)
    pts_files = [os.path.join(pts_root,d) for d in os.listdir(pts_root) if re.match(r'points_group\d+.txt', d)]
    mask_files = [os.path.join(pts_root,d) for d in os.listdir(pts_root) if re.match(r'points_group\d+_mask.txt', d)]
    pts_files.sort()
    mask_files.sort()

    subdirectories = [d for d in os.listdir(pts_root) if os.path.isdir(os.path.join(pts_root, d))]
    for _ in subdirectories:
        cur_data_dir = os.path.join(args.data_dir, _)
        camera_pose_file = os.path.join(cur_data_dir, 'camera_pose.txt')
        camera_pose = read_camera_pose(camera_pose_file)
        point_clouds, masks = read_point_clouds(pts_files, mask_files)
        masks = masks[None]

        transformed_point_clouds = transform_point_clouds(point_clouds, camera_pose)

        # randomized_point_cloud, rotation_matrix, center = normalize_and_randomize_point_cloud(transformed_point_clouds, args.num_data)
        # randomized_point_cloud, rotation_matrix, center = canonicalize_point_cloud_batch(transformed_point_clouds)
        randomized_point_cloud, rotation_matrix, center = canonicalize_point_cloud_batch_heu(transformed_point_clouds)
        randomized_point_cloud = randomized_point_cloud[None]
        rotation_matrix = rotation_matrix[None]
        center = center[None]
        data = {
            'point_cloud': randomized_point_cloud,
            'mask': masks,
            'rotation_matrix': rotation_matrix,
            'center': center
        }
        for key, value in data.items():
            print(key, value.shape)
        np.savez(os.path.join(cur_data_dir, "data.npz"), **data)

        reconed_point_cloud = randomized_point_cloud @ rotation_matrix + center[:, :, None, :]
        if args.visualize:
        
            # 创建点云对象
            # pts1 = trimesh.points.PointCloud(randomized_point_cloud[0][0])
            # pts2 = trimesh.points.PointCloud(randomized_point_cloud[0][1])

            # pts1 = trimesh.points.PointCloud(point_clouds[0])
            # pts2 = trimesh.points.PointCloud(point_clouds[1])
            # 创建一个空的场景
            scene = trimesh.Scene()
            num_parts = reconed_point_cloud.shape[1]
            for i in range(num_parts):
                pcd = trimesh.points.PointCloud(reconed_point_cloud[0][i])
                pcd.visual.vertex_colors = np.random.randint(0, 255, size=(3))
                scene.add_geometry(pcd)

            # # 在场景中指定位置创建一个红色小球，作为红点
            # # 使用一个小球代表红点，设置球体的位置
            # red_dot_position = camera_pose[:3, 3]  # 小球的位置为相机的位置
            # red_dot = trimesh.creation.icosphere(subdivisions=2, radius=0.05)  # 创建一个半径为0.05的小球
            # red_dot.apply_translation(red_dot_position)  # 移动小球到指定位置

            # # 将红点添加到场景中
            # red_dot.visual.vertex_colors = [255, 0, 0, 255]  # 设置小球的颜色为红色
            # scene.add_geometry(red_dot)
            # 创建坐标系（原点处的三条轴）

            # 创建X轴（红色），它指向x方向
            x_axis = trimesh.creation.cylinder(radius=0.02, height=1)  # 创建一个半径为0.02的圆柱体
            # x_axis.apply_translation([0.5, 0, 0])  # 沿x轴方向平移

            # 创建Y轴（绿色），它指向y方向，绕X轴旋转90°
            y_axis = trimesh.creation.cylinder(radius=0.02, height=1)
            # y_axis.apply_translation([0, 0.5, 0])  # 沿y轴方向平移
            rotation_matrix_y = transformations.rotation_matrix(np.pi / 2, [1, 0, 0])  # 绕x轴旋转90°
            y_axis.apply_transform(rotation_matrix_y)  # 应用旋转矩阵

            # 创建Z轴（蓝色），它指向z方向，绕Y轴旋转90°
            z_axis = trimesh.creation.cylinder(radius=0.02, height=1)
            # z_axis.apply_translation([0, 0, 0.5])  # 沿z轴方向平移
            rotation_matrix_z = transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])  # 绕y轴旋转90°（负方向）
            z_axis.apply_transform(rotation_matrix_z)  # 应用旋转矩阵

            # 设置坐标轴的颜色
            x_axis.visual.vertex_colors = [255, 0, 0, 255]  # X轴为红色
            y_axis.visual.vertex_colors = [0, 255, 0, 255]  # Y轴为绿色
            z_axis.visual.vertex_colors = [0, 0, 255, 255]  # Z轴为蓝色

            # 将坐标轴添加到场景中
            scene.add_geometry(x_axis)
            scene.add_geometry(y_axis)
            scene.add_geometry(z_axis)
            # 可视化
            scene.show()

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data2/lyw/rss_data/IKEA-Manuals-at-Work/data/parts/Chair/applaro_2', help='Path to the data dir')
    parser.add_argument('--visualize',type=bool, default=False , help='Visualize the generated data by trimesh')
    args = parser.parse_args()
    main(args)