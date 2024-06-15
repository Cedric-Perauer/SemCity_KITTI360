import os
import numpy as np
import open3d as o3d
from torch.utils import data
import yaml
import torch
import pathlib
import random

def generate_random_colors(N, seed=0):
    colors = set()  # Use a set to store unique colors
    while len(colors) < N:  # Keep generating colors until we have N unique ones
        # Generate a random color and add it to the set
        colors.add((random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)))

    return list(colors)  # Convert the set to a list before returning


colors_gen = generate_random_colors(100)


def color_point_cloud_by_labels(point_cloud, labels):
    """
    Colors a point cloud based on the provided labels and label colors.

    Args:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        labels (numpy.ndarray): An array of labels corresponding to each point in the point cloud.
        label_colors (dict): A dictionary mapping each label to its corresponding RGB color.

    Returns:
        open3d.geometry.PointCloud: The colored point cloud.
    """
    colors = np.zeros((len(labels), 3))
    for idx, label in enumerate(np.unique(labels)):
        mask_idcs = np.where(labels == label)
        colors[mask_idcs] = colors_gen[idx]
    point_cloud.colors = o3d.utility.Vector3dVector(colors/255.)
    return point_cloud

class KITTI360(data.Dataset):
    def __init__(self, imageset='train', get_query=True):
        self.base_folder = '/media/cedric/Datasets2/KITTI_360/preprocessed/'
        subfolders = os.listdir(self.base_folder)
        subfolders = [folder for folder in subfolders if os.path.isdir(
            self.base_folder + folder)]
        self.im_idx = []
        for folder in subfolders:
            fs = os.listdir(self.base_folder + folder)
            for f in fs:
                if f.endswith('aligned.npz'):
                    self.im_idx.append(self.base_folder + folder + '/' + f)

        for cur_f in self.im_idx:
            # cur_f = self.im_idx[0]
            with np.load(cur_f) as data:
                pts = data['xyz']
                colors = data['colors']
                sem = data['semantics']

            pts_median = np.median(pts[:, 0]), np.median(
                pts[:, 1]), np.median(pts[:, 2])
            inliers = np.where(
                (pts[:, 2] > pts_median[2] - 2) & (pts[:, 2] < pts_median[2] + 2))
            pts = pts[inliers]
            colors = colors[inliers]
            labels = sem[inliers]

            # Create the point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Create the voxel grid
            voxel_size = 0.15
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                pcd, voxel_size)

            # Extract voxel indices and centers
            voxel_indices = np.array(
                [voxel.grid_index for voxel in voxel_grid.get_voxels()])
            voxel_colors = np.array(
                [voxel.color for voxel in voxel_grid.get_voxels()])
            voxel_centers = np.array([voxel_grid.origin + voxel.grid_index *
                                     voxel_grid.voxel_size for voxel in voxel_grid.get_voxels()])

            # Normalize voxel centers to be between -1 and 1
            voxel_dim = np.array([256, 256, 32])
            normalized_voxel_centers = (
                voxel_centers / (voxel_dim - 1)) * 2 - 1

            # Initialize a dictionary to store semantic labels for each voxel
            voxel_semantics = {}

            # Assign each point's semantic label to the corresponding voxel
            for point, label in zip(pts, labels):
                voxel_index = tuple(
                    ((point - voxel_grid.origin) / voxel_size).astype(int))
                if voxel_index in voxel_semantics:
                    voxel_semantics[voxel_index].append(label)
                else:
                    voxel_semantics[voxel_index] = [label]

            # Aggregate the semantic labels for each voxel (e.g., using the most frequent label)
            voxel_labels = {}
            for voxel_index, label_list in voxel_semantics.items():
                voxel_labels[voxel_index] = max(
                    set(label_list), key=label_list.count)

            # Print the results
            remapped_labels = []
            voxel_label = np.zeros([256, 256, 32], dtype=int)
            for voxel_idx in voxel_indices:
                cur_label = voxel_labels[(voxel_idx[0],voxel_idx[1],voxel_idx[2])]
                remapped_labels.append(cur_label)
                voxel_label[voxel_idx[0],voxel_idx[1],voxel_idx[2]] = cur_label

            xyz_label = np.array(remapped_labels)
            xyz_center = voxel_indices
            voxel_label = voxel_label.reshape((-1, 1))
            query = normalized_voxel_centers
            colors = voxel_colors
            break
            ### query = tnd !!!

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        pass


def flip(voxel, invalid, flip_dim=0):
    voxel = np.flip(voxel, axis=flip_dim).copy()
    invalid = np.flip(invalid, axis=flip_dim).copy()
    return voxel, invalid


if __name__ == '__main__':
    dataset = KITTI360()
