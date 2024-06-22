import os
import numpy as np
import open3d as o3d
from torch.utils import data
import yaml
import torch
import pathlib
import random
from tqdm import tqdm


remapping_dict = {
    0: 255,    # 'unlabeled' -> ignored
    1: 255,    # 'ego vehicle' -> ignored
    2: 255,    # 'rectification border' -> ignored
    3: 255,    # 'out of roi' -> ignored
    4: 255,    # 'static' -> ignored
    5: 255,    # 'dynamic' -> ignored
    6: 255,    # 'ground' -> ignored
    7: 8,      # 'road' -> 'road'
    8: 10,     # 'sidewalk' -> 'sidewalk'
    9: 9,      # 'parking' -> 'parking'
    10: 11,    # 'rail track' -> 'other-ground'
    11: 12,    # 'building' -> 'building'
    12: 12,    # 'wall' -> 'building'
    13: 13,    # 'fence' -> 'fence'
    14: 13,    # 'guard rail' -> 'fence'
    15: 12,    # 'bridge' -> 'building'
    16: 12,    # 'tunnel' -> 'building'
    17: 17,    # 'pole' -> 'pole'
    18: 17,    # 'polegroup' -> 'pole'
    19: 18,    # 'traffic light' -> 'traffic-sign'
    20: 18,    # 'traffic sign' -> 'traffic-sign'
    21: 14,    # 'vegetation' -> 'vegetation'
    22: 16,    # 'terrain' -> 'terrain'
    23: 255,   # 'sky' -> ignored
    24: 5,     # 'person' -> 'person'
    25: 6,     # 'rider' -> 'bicyclist' (assuming rider is typically on a bicycle)
    26: 0,     # 'car' -> 'car'
    27: 3,     # 'truck' -> 'truck'
    28: 3,     # 'bus' -> 'truck'
    29: 4,     # 'caravan' -> 'other-vehicle'
    30: 4,     # 'trailer' -> 'other-vehicle'
    31: 4,     # 'train' -> 'other-vehicle'
    32: 2,     # 'motorcycle' -> 'motorcycle'
    33: 1,     # 'bicycle' -> 'bicycle'
    34: 12,    # 'garage' -> 'building'
    35: 13,    # 'gate' -> 'fence'
    36: 255,   # 'stop' -> ignored
    37: 17,    # 'smallpole' -> 'pole'
    38: 17,    # 'lamp' -> 'pole'
    39: 255,   # 'trash bin' -> ignored
    40: 255,   # 'vending machine' -> ignored
    41: 255,   # 'box' -> ignored
    42: 255,   # 'unknown construction' -> ignored
    43: 255,   # 'unknown vehicle' -> ignored
    44: 255,   # 'unknown object' -> ignored
    -1: 255    # 'license plate' -> ignored
}


def generate_random_colors(N, seed=0):
    colors = set()  # Use a set to store unique colors
    while len(colors) < N:  # Keep generating colors until we have N unique ones
        # Generate a random color and add it to the set
        colors.add((random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)))

    return list(colors)  # Convert the set to a list before returning




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
    colors_gen = generate_random_colors(100)
    colors = np.zeros((len(labels), 3))
    for idx, label in enumerate(np.unique(labels)):
        mask_idcs = np.where(labels == label)
        colors[mask_idcs] = colors_gen[idx]
    point_cloud.colors = o3d.utility.Vector3dVector(colors/255.)
    return point_cloud

class KITTI360(data.Dataset):
    def __init__(self, imageset='train', get_query=True,num_class=20):
        self.base_folder = '/media/cedric/Datasets2/KITTI_360/preprocessed/'
        subfolders = os.listdir(self.base_folder)
        subfolders = [folder for folder in subfolders if os.path.isdir(
            self.base_folder + folder)]
        self.im_idx = []
        self.test_samples = []
        self.num_class = num_class
        complt_num_per_class= np.asarray([1]*20)
        compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
        self.weights = torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)).cuda()
        self.min_dim = 10000000
        self.max_points = 400000
        
        
        for folder in subfolders:
            fs = os.listdir(self.base_folder + folder)
            for f in fs:
                if f.endswith('aligned.npz'):
                    self.im_idx.append(self.base_folder + folder + '/' + f)
        
        idx = 0
        for cur_f in tqdm(self.im_idx):
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
                cur_label = remapping_dict[voxel_labels[(voxel_idx[0],voxel_idx[1],voxel_idx[2])]]
                voxel_label[voxel_idx[0],voxel_idx[1],voxel_idx[2]] = cur_label

            for i in range(1, num_class):
                xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
                xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=i)
                remapped_labels.append(xyzlabel)

            pt_number = normalized_voxel_centers.shape[0]
            num_far_free = self.max_points - pt_number
            if pt_number < self.max_points:
                xyz = torch.nonzero(torch.from_numpy(voxel_label) == 0)
                xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
                idx = torch.randperm(xyzlabel.shape[0])
                xyzlabel = xyzlabel[idx][:min(xyzlabel.shape[0], num_far_free)]
                remapped_labels.append(xyzlabel)
                while len(torch.cat(remapped_labels, dim=0)) < self.max_points:
                    for i in range(1, self.num_class):
                        xyz = torch.nonzero(torch.Tensor(
                            voxel_label) == i, as_tuple=False)
                        xyzlabel = torch.nn.functional.pad(
                            xyz, (1, 0), 'constant', value=i)
                        remapped_labels.append(xyzlabel)
            remapped_labels = torch.cat(remapped_labels, dim=0)
            remapped_labels = remapped_labels[:self.max_points]
            import pdb; pdb.set_trace()
            ## need to make sure to continue here, update noirmalized voxel centers
            '''
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
            '''
            xyz_label = np.array(remapped_labels)
            xyz_center = voxel_indices
            voxel_label = voxel_label
            query = normalized_voxel_centers
            colors = voxel_colors
            invalid = torch.zeros_like(torch.from_numpy(voxel_label))
            self.test_samples.append([voxel_label,query,xyz_label,xyz_center,cur_f,invalid])
            idx += 1
            break

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.test_samples)

    def __getitem__(self, index):
        voxel_label, query,xyz_label, xyz_center,f_name,invalid = self.test_samples[1]

        return voxel_label,query,xyz_label,xyz_center,f_name,invalid

def flip(voxel, invalid, flip_dim=0):
    voxel = np.flip(voxel, axis=flip_dim).copy()
    invalid = np.flip(invalid, axis=flip_dim).copy()
    return voxel, invalid


if __name__ == '__main__':
    dataset = KITTI360()
