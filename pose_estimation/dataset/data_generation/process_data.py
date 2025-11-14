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

    # Generate a quaternion randomly
    random_quaternion = np.random.randn(b*M,4)  
    random_quaternion /= np.linalg.norm(random_quaternion, axis=0)  
    # Convert the quaternion into a rotation matrix
    rotation_matrix = quan2rot(random_quaternion)

    # Randomly rotate the point cloud
    point_cloud_normalized = np.matmul(point_cloud_normalized ,rotation_matrix.transpose(0, 2, 1))

    point_cloud_normalized = point_cloud_normalized.reshape(b, M, N, 3)
    rotation_matrix = rotation_matrix.reshape(b, M, 3, 3)
    center = center.reshape(b, M, 3)
    
    return point_cloud_normalized, rotation_matrix, center

# Use PCA to canonicalize the batch point clouds
def canonicalize_point_cloud_batch(point_cloud_batch):
    B = point_cloud_batch.shape[0]  # Obtain the batch size
    canonicalized_pcd_batch = []
    rotation_matrices = []
    centroids = []
    for b in range(B):
        # 1. Obtain a single point cloud
        point_cloud = point_cloud_batch[b]
        # 2. Centralized point cloud (making its mean value 0)
        centroid = np.mean(point_cloud, axis=0)
        centered_pcd = point_cloud - centroid
        # 3. use PCA
        pca = PCA(n_components=3)
        pca.fit(centered_pcd)
        # 4. Obtain the rotation matrix (principal axis direction)
        rotation_matrix = pca.components_
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix = -rotation_matrix
        # 5. Project the point cloud onto the principal axis
        transformed_pcd = pca.transform(centered_pcd)
        # 6. Store the result
        canonicalized_pcd_batch.append(transformed_pcd)
        rotation_matrices.append(rotation_matrix)
        centroids.append(centroid)
    # Convert the result to a NumPy array
    canonicalized_pcd_batch = np.array(canonicalized_pcd_batch)
    rotation_matrices = np.array(rotation_matrices)
    centroids = np.array(centroids)

    return canonicalized_pcd_batch, rotation_matrices, centroids

def canonicalize_point_cloud_batch_heu(point_cloud_batch, k=3):
    B = point_cloud_batch.shape[0]  # Obtain the batch size
    canonicalized_pcd_batch = []
    rotation_matrices = []
    centroids = []
    for b in range(B):
        # 1. Obtain a single point cloud
        point_cloud = point_cloud_batch[b]
        # 2. Centralized point cloud (making its mean value 0)
        center = np.mean(point_cloud, axis=0)

        point_cloud = point_cloud - center
        # Calculate the distance from each point to the center of mass
        distances = np.linalg.norm(point_cloud, axis=1)
        # Find the index of the k points farthest from the center of mass
        farthest_k_indices = np.argsort(distances)[-k:]
        # Calculate the direction of the line connecting the farthest k points and the center of mass
        directions = point_cloud[farthest_k_indices]
        # Calculate the average direction as the X-axis
        x_axis = np.mean(directions, axis=0)
        x_axis /= np.linalg.norm(x_axis)  # Normalize the X-axis vector
        
        # Project all the points onto the plane orthogonal to the X-axis
        projected_cloud = point_cloud - np.outer(np.dot(point_cloud, x_axis), x_axis)
        # Find the index of the k points farthest from the center of mass among the projected points
        projected_distances = np.linalg.norm(projected_cloud, axis=1)
        projected_farthest_k_indices = np.argsort(projected_distances)[-k:]
        # Calculate the direction of the line connecting the k projected points and the center of mass as the Y-axis
        projected_directions = projected_cloud[projected_farthest_k_indices]
        y_axis = np.mean(projected_directions, axis=0)
        y_axis /= np.linalg.norm(y_axis)  # Normalize the Y-axis vector
        
        # Calculate the Z-axis according to the right-hand system rule, that is, z_axis = x_axis × y_axis
        z_axis = np.cross(x_axis, y_axis)
        
        # Construct the rotation matrix
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        # print(rotation_matrix)
        canonical_cloud = point_cloud  @ rotation_matrix
        canonicalized_pcd_batch.append(canonical_cloud)
        rotation_matrices.append(rotation_matrix.T)
        centroids.append(center)
    # Convert the result to a NumPy array
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
        
            # Create an empty scene
            scene = trimesh.Scene()
            num_parts = reconed_point_cloud.shape[1]
            for i in range(num_parts):
                pcd = trimesh.points.PointCloud(reconed_point_cloud[0][i])
                pcd.visual.vertex_colors = np.random.randint(0, 255, size=(3))
                scene.add_geometry(pcd)


            x_axis = trimesh.creation.cylinder(radius=0.02, height=1)  # Create a cylinder with a radius of 0.02
            # x_axis.apply_translation([0.5, 0, 0])  # Translate along the X-axis direction

            # Create the Y-axis (green), which points in the Y direction and rotates 90° around the X-axis
            y_axis = trimesh.creation.cylinder(radius=0.02, height=1)
            # y_axis.apply_translation([0, 0.5, 0])  # Translate along the Y-axis direction
            rotation_matrix_y = transformations.rotation_matrix(np.pi / 2, [1, 0, 0])  # Rotate 90° around the X-axis
            y_axis.apply_transform(rotation_matrix_y)  

            # Create the Z-axis (blue), which points in the Z direction and rotates 90° around the Y-axis
            z_axis = trimesh.creation.cylinder(radius=0.02, height=1)
            # z_axis.apply_translation([0, 0, 0.5])  # Translate along the Z-axis direction
            rotation_matrix_z = transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])  # Rotate 90° around the Y-axis (in the negative direction)
            z_axis.apply_transform(rotation_matrix_z)  

            x_axis.visual.vertex_colors = [255, 0, 0, 255]  
            y_axis.visual.vertex_colors = [0, 255, 0, 255]  
            z_axis.visual.vertex_colors = [0, 0, 255, 255]  

            # Add the coordinate axes to the scene
            scene.add_geometry(x_axis)
            scene.add_geometry(y_axis)
            scene.add_geometry(z_axis)

            scene.show()
