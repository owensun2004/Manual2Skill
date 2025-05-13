import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from pdb import set_trace as bp
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import h5py
from pose_estimation.utils.utils import *
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from pathlib import Path
from pose_estimation.utils.utils import *
import time
from concurrent.futures import ThreadPoolExecutor

class FurnitureAssemblyDataset(Dataset):
    def __init__(self, data_dir, vis=False, partnet=True, split='train', train_ratio=0.7, seed=42, min_num_part=2, max_num_part=4):
        self.data_dir = data_dir
        self.vis = vis
        self.partnet = partnet
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part

        # Build the full dataset
        self.data = self.build_dataset()

        # Split indices using sklearn
        train_indices, val_indices = train_test_split(
            range(len(self.data)), 
            train_size=train_ratio, 
            random_state=seed
        )

        if split == 'train':
            self.data = [self.data[i] for i in train_indices]
        elif split == 'val':
            self.data = [self.data[i] for i in val_indices]
        else:
            raise ValueError("Split must be 'train' or 'val'.")
        
    def process_part_selection(self, part_selection_dir):
        data = []
        num_image = 0
        num_total_data = 0
        for image_dir in tqdm(os.listdir(part_selection_dir), desc="Processing images", leave=False):
            if not os.path.isdir(os.path.join(part_selection_dir, image_dir)):
                continue
            image_path = os.path.join(part_selection_dir, image_dir)
            image = np.array((cv2.imread(os.path.join(image_path, 'image.png'))), dtype=np.uint8)

            if not self.vis:
                # Normalize
                image = image / 255.0
            image = image.transpose(2, 0, 1)
            num_image = num_image + 1
            zip_data = np.load(os.path.join(image_path, 'data.npz'))

            point_cloud = zip_data['point_cloud']
            mask = zip_data['mask']
            rotation_matrix = zip_data['rotation_matrix']
            center = zip_data['center']

            part_ids = assign_part_ids(point_cloud, center_threshold=0.05, size_threshold=0.05)

            num_parts = point_cloud.shape[1]
            if num_parts < self.min_num_part or num_parts > self.max_num_part:
                continue
            num_data_pieces = point_cloud.shape[0]
            num_total_data = num_total_data + num_data_pieces
            
            # valids = torch.zeros(self.max_num_part, dtype=torch.float32)
            valids = np.zeros(self.max_num_part, dtype=np.float32)
            valids[:num_parts] = 1
            
            for i in range(num_data_pieces):
                data.append({
                    'image': image,
                    'mask': self._pad_data(mask[i]),
                    'point_cloud': self._pad_data(point_cloud[i]),
                    'rotation_matrix': self._pad_data(rotation_matrix[i]),
                    'center': self._pad_data(center[i]),
                    'part_valids': valids,
                    'part_ids': self._pad_data(part_ids[i])
                })
        
        return data
        
    def build_dataset(self):
        """
        Load the dataset from the data directory with a progress bar.
        """
        data = []

        # Outer loop with tqdm for part selection
        if self.partnet:
            for shape_dir in tqdm(os.listdir(self.data_dir), desc="Processing shape directories"):
                shape_dir = os.path.join(self.data_dir, shape_dir)
                for part_selection in os.listdir(shape_dir):
                    data.extend(self.process_part_selection(os.path.join(self.data_dir, shape_dir, part_selection)))

        else:
            for part_selection in tqdm(os.listdir(self.data_dir), desc="Processing part selections"):
                data.extend(self.process_part_selection(os.path.join(self.data_dir, part_selection)))

        return data

    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...] using zeros."""
        data = np.array(data)
        # data = data.clone()
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        # pad_data = torch.zeros(pad_shape, dtype=torch.float32)
        pad_data[:data.shape[0]] = data
        return pad_data
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        data_dict = {
            'image': 3 x H x W
                The rendered image.

            'mask': MAX_NUM x N
                The mask for each part, denoting the junctions.

            'point_cloud': MAX_NUM x N x 3
                The points sampled from each part.

            'center': MAX_NUM x 3
                Translation vector

            'rotation_matrix': MAX_NUM x 4
                Rotation as rotation matrix.

            'part_valids': MAX_NUM
                1 for shape parts, 0 for padded zeros.

        }
        """
        return self.data[idx]


def add_positional_embedding_to_mask(mask_batch, num_embedding_points=24):
    batch_size, num_objects, num_points, _ = mask_batch.shape
    embedding_mask = -1 * np.ones((batch_size, num_objects, num_embedding_points, 1), dtype=mask_batch.dtype)
    new_mask_batch = np.concatenate((mask_batch, embedding_mask), axis=2)  # Concatenate along the point dimension
    
    return new_mask_batch

def add_positional_embedding_to_pts(point_cloud_batch, num_embedding_points=24):
    batch_size, num_objects, num_points, _ = point_cloud_batch.shape
    new_point_cloud_batch = []

    for i in range(num_objects):
        point_cloud = point_cloud_batch[0, i]
        
        min_val = point_cloud.min(axis=0)
        max_val = point_cloud.max(axis=0)

        t = np.linspace(0, 2 * np.pi, num_embedding_points)
        
        offset = np.sin(i)
        
        embedding_points = np.zeros((num_embedding_points, 3))
        embedding_points[:, 0] = (np.cos(t + offset) + 1) / 2  # Add offset to the x coordinate
        embedding_points[:, 1] = (np.sin(t + offset) + 1) / 2  # Add offset to the y coordinate
        embedding_points[:, 2] = (np.cos(2 * t + offset) + 1) / 2  # Add offset to the z coordinate

        scaled_embedding_points = embedding_points * (max_val - min_val) + min_val

        new_point_cloud = np.vstack((point_cloud, scaled_embedding_points))
        
        new_point_cloud_batch.append(new_point_cloud)

    new_point_cloud_batch = np.array(new_point_cloud_batch)  # Shape (N, 1024, 3)
    new_point_cloud_batch = np.expand_dims(new_point_cloud_batch, axis=0)  # Shape (1, N, 1024, 3)

    return new_point_cloud_batch


def compute_bounding_boxes(point_cloud):
    """Compute the bounding box for each part"""
    num_parts = point_cloud.shape[1]
    bounding_boxes = np.zeros((num_parts, 6))  # Shape (num_parts, 6)

    for i in range(num_parts):
        part = point_cloud[0, i]  # Extract the point cloud of part i, shape (1000, 3)
        x_min, y_min, z_min = part.min(axis=0)  # Minimum values
        x_max, y_max, z_max = part.max(axis=0)  # Maximum values
        bounding_boxes[i] = [x_min, y_min, z_min, x_max, y_max, z_max]
    
    return bounding_boxes

def compute_center_and_size(bounding_boxes):
    """Compute the center and size for each part"""
    centers = (bounding_boxes[:, :3] + bounding_boxes[:, 3:]) / 2  # Centers
    sizes = bounding_boxes[:, 3:] - bounding_boxes[:, :3]  # Sizes (w, h, d)
    return centers, sizes

def are_parts_similar(centers, sizes, i, j, center_threshold, size_threshold):
    """Determine if part i and part j are similar"""
    center_distance = np.linalg.norm(centers[i] - centers[j])  # Distance between centers
    size_difference = np.abs(sizes[i] - sizes[j])  # Size difference
    return center_distance <= center_threshold and np.all(size_difference <= size_threshold)

def assign_part_ids(point_cloud, center_threshold=0.5, size_threshold=0.2):
    """Assign IDs to parts based on similarity"""
    bounding_boxes = compute_bounding_boxes(point_cloud)
    centers, sizes = compute_center_and_size(bounding_boxes)
    
    num_parts = centers.shape[0]
    part_ids = np.full((num_parts,), -1)  # Initialize IDs to -1
    current_id = 0  # Starting ID

    for i in range(num_parts):
        if part_ids[i] == -1:  # If the ID has not been assigned
            part_ids[i] = current_id
            # Traverse the remaining parts to check similarity
            for j in range(i + 1, num_parts):
                if part_ids[j] == -1 and are_parts_similar(centers, sizes, i, j, center_threshold, size_threshold):
                    part_ids[j] = current_id
            current_id += 1  # Update the ID
    
    # Convert to numpy array, shape (1, num_parts, 1)
    return part_ids.reshape(1, num_parts, 1)

if __name__ == '__main__':
    dataset = FurnitureAssemblyDataset(data_dir='/home/crtie/crtie/Furniture-Assembly/systhesis_data/mini_data')
    sample = dataset[0] 
    for key, value in sample.items():
        print(key, value.shape)
