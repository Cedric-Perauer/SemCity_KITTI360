import os
import numpy as np
import open3d as o3d
from torch.utils import data
import yaml
import torch
import pathlib
import random
from tqdm import tqdm
import time
import h5py


remapping_dict = {
    0: 255,    # 'unlabeled' -> ignored
    1: 255,    # 'ego vehicle' -> ignored
    2: 255,    # 'rectification border' -> ignored
    3: 255,    # 'out of roi' -> ignored
    4: 255,    # 'static' -> ignored
    5: 255,    # 'dynamic' -> ignored
    6: 255,    # 'ground' -> ignored
    7: 9,      # 'road' -> 'road'
    8: 11,     # 'sidewalk' -> 'sidewalk'
    9: 10,      # 'parking' -> 'parking'
    10: 12,    # 'rail track' -> 'other-ground'
    11: 13,    # 'building' -> 'building'
    12: 13,    # 'wall' -> 'building'
    13: 14,    # 'fence' -> 'fence'
    14: 14,    # 'guard rail' -> 'fence'
    15: 8,    # 'bridge' -> 'building'
    16: 13,    # 'tunnel' -> 'building'
    17: 18,    # 'pole' -> 'pole'
    18: 18,    # 'polegroup' -> 'pole'
    19: 16,    # 'traffic light' -> 'traffic-light'
    20: 19,    # 'traffic sign' -> 'traffic-sign'
    21: 15,    # 'vegetation' -> 'vegetation'
    22: 17,    # 'terrain' -> 'terrain'
    23: 255,   # 'sky' -> ignored
    24: 6,     # 'person' -> 'person'
    25: 7,     # 'rider' -> 'bicyclist' (assuming rider is typically on a bicycle)
    26: 1,     # 'car' -> 'car'
    27: 4,     # 'truck' -> 'truck'
    28: 4,     # 'bus' -> 'truck'
    29: 5,     # 'caravan' -> 'other-vehicle'
    30: 5,     # 'trailer' -> 'other-vehicle'
    31: 5,     # 'train' -> 'other-vehicle'
    32: 3,     # 'motorcycle' -> 'motorcycle'
    33: 2,     # 'bicycle' -> 'bicycle'
    34: 13,    # 'garage' -> 'building'
    35: 14,    # 'gate' -> 'fence'
    36: 255,   # 'stop' -> ignored
    37: 18,    # 'smallpole' -> 'pole'
    38: 18,    # 'lamp' -> 'pole'
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
        self.num_class = num_class
        self.base_folder = '/media/cedric/Datasets2/KITTI_360/semcity_format2/'
        complt_num_per_class = np.load(self.base_folder + 'class_weights.npz')['class_weights']
        idcs = np.where(complt_num_per_class == 0)
        complt_num_per_class[idcs] = 1000 ### just a hack for now
        self.max_points = 400000
        self.subfolders = os.listdir(self.base_folder)
        self.subfolders = [folder for folder in self.subfolders if os.path.isdir(
            self.base_folder + folder)]
        self.im_idx = []
        for folder in self.subfolders:
            fs = os.listdir(self.base_folder + folder)
            for f in fs:
                if f.endswith('.h5'):
                    self.im_idx.append(self.base_folder + folder + '/' + f)
        print('imageset',imageset)
        
                    
        if imageset == 'train':
            compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
            self.weights = torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)).cuda()
            self.im_idx = self.im_idx[:int(0.8*len(self.im_idx))]
        elif imageset == 'val':
            self.im_idx = self.im_idx[int(0.8*len(self.im_idx)):int(0.9*len(self.im_idx))]
            self.weights = torch.Tensor(np.ones(20) * 3).cuda()
            self.weights[0] = 1
        elif imageset == 'test' : 
            self.im_idx = self.im_idx[int(0.9*len(self.im_idx)):]
            self.weights = torch.Tensor(np.ones(20) * 3).cuda()
            self.weights[0] = 1
        else : 
            raise Exception("Split must be train/val/test split")
        
        print('len',len(self.im_idx))

    def create_format(self):
        self.base_folder = '/media/cedric/Datasets2/KITTI_360/preprocessed/'
        self.subfolders = os.listdir(self.base_folder)
        self.subfolders = [folder for folder in self.subfolders if os.path.isdir(
            self.base_folder + folder)]
        self.im_idx = []
        self.test_samples = []
        self.num_class = self.num_class
        #self.weights = torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)).cuda()
        self.min_dim = 10000000
        removed = 0 
        
        complt_num_per_class= np.asarray([0]*20)
        for folder in self.subfolders:
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
    
            
            num_pts = np.asarray(pcd.points).shape[0]
            if num_pts == 0 : 
                removed += 1
                print("continue",removed)
                
                continue 
            # Extract voxel indices and centers
            voxel_indices = np.array(
                [voxel.grid_index for voxel in voxel_grid.get_voxels()])
            voxel_colors = np.array(
                [voxel.color for voxel in voxel_grid.get_voxels()])

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
            store_file = cur_f.replace('preprocessed','semcity_format2')
            voxel_label = np.zeros([256, 256, 32], dtype=int)
            voxel_colors = np.zeros([256,256,32,3],dtype=float)
            voxels = voxel_grid.get_voxels()
            start = time.time()
            
            for vox_idx,voxel_idx in enumerate(voxel_indices):
                cur_label = remapping_dict[voxel_labels[(voxel_idx[0],voxel_idx[1],voxel_idx[2])]]
                voxel_label[voxel_idx[0],voxel_idx[1],voxel_idx[2]] = cur_label
                voxel_colors[voxel_idx[0],voxel_idx[1],voxel_idx[2]] = np.array(list(voxels[vox_idx].color))

            end = time.time() - start

            
            out_pth = store_file.split('/')[:-1]
            pth = '/'.join(out_pth) + '/'
            if os.path.exists(pth) is False :
                os.makedirs(pth)
            #np.savez(store_file, voxel_label=voxel_label,voxel_colors=voxel_colors,cur_f=store_file)
            file = h5py.File(store_file.replace('npz','h5'), 'w')
            file.create_dataset('voxel_label', data=voxel_label.flatten())
            file.create_dataset('voxel_colors', data=voxel_colors.flatten())
            file.close()
            idx += 1

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        start = time.time()
        index = 0
        with h5py.File(self.im_idx[index], "r") as data:
            voxel_colors = data['voxel_colors'][:].reshape((256,256,32,3)) * 255.
            voxel_colors = voxel_colors.astype(np.uint8)
            voxel_label = data['voxel_label'][:].reshape((256,256,32))

        remapped_colors = []
        remapped_labels = []
        
        for i in range(1, self.num_class):
            xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
            xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=i)
            colors = voxel_colors[xyz[:,0],xyz[:,1],xyz[:,2]]
            remapped_labels.append(xyzlabel)
            remapped_colors.append(colors.reshape(-1,3))
        
        pt_number = voxel_colors.shape[0]
        num_far_free = self.max_points - pt_number
        if pt_number < self.max_points:
            xyz = torch.nonzero(torch.from_numpy(voxel_label) == 0)
            colors = voxel_colors[xyz[:,0],xyz[:,1],xyz[:,2]]
            xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
            idx = torch.randperm(xyzlabel.shape[0])
            xyzlabel = xyzlabel[idx][:min(xyzlabel.shape[0], num_far_free)]
            remapped_labels.append(xyzlabel)
            cols_add = colors[:min(xyzlabel.shape[0], num_far_free)]
            remapped_colors.append(cols_add.reshape(-1,3))
            while len(torch.cat(remapped_labels, dim=0)) < self.max_points:
                for i in range(1, self.num_class):
                    xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
                    colors = voxel_colors[xyz[:,0],xyz[:,1],xyz[:,2]]
                    remapped_colors.append(colors.reshape(-1,3))
                    xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=i)
                    remapped_labels.append(xyzlabel)
        
        remapped_labels = torch.cat(remapped_labels, dim=0)
        remapped_labels = remapped_labels[:self.max_points]
        remapped_colors = np.concatenate(remapped_colors, axis=0)
        remapped_colors = remapped_colors[:self.max_points]
        
        voxel_dim = np.array([256, 256, 32])
        query = (remapped_labels[:,1:] / (voxel_dim - 1)) * 2 - 1
        
        xyz_label = remapped_labels[:,0]
        xyz_center = remapped_labels[:,1:]
        invalid = torch.zeros_like(torch.from_numpy(voxel_label))
        
        #del remapped_colors, remapped_labels
        
        end = time.time() - start
        
        return voxel_label, query, xyz_label, xyz_center, self.im_idx[index], invalid, remapped_colors
     

def flip(voxel, invalid, flip_dim=0):
    voxel = np.flip(voxel, axis=flip_dim).copy()
    invalid = np.flip(invalid, axis=flip_dim).copy()
    return voxel, invalid


if __name__ == '__main__':
    dataset = KITTI360()
    dataset.create_format()
