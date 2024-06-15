import os
import numpy as np
import open3d as o3d
from torch.utils import data
import yaml 
import torch 
import pathlib

class KITTI360(data.Dataset):
    def __init__(self, imageset='train', get_query=True):
        self.base_folder = '/media/cedric/Datasets2/KITTI_360/preprocessed/'
        subfolders = os.listdir(self.base_folder)
        subfolders = [folder for folder in subfolders if os.path.isdir(self.base_folder + folder)]
        self.im_idx = []
        for folder in subfolders : 
                fs = os.listdir(self.base_folder + folder)
                for f in fs : 
                        if f.endswith('.npz'):
                                self.im_idx.append(self.base_folder + folder + '/' + f)
        
        cur_f = self.im_idx[0]
        with np.load(cur_f) as data :
                pts = data['xyz']
                colors = data['colors']
                sem = data['semantics']
        
        
        pts_extreme = np.median(pts[:,0]), np.median(pts[:,1]), np.median(pts[:,2])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.15)
        obb = pcd.get_oriented_bounding_box()
        obb.color = [1, 0, 0]  # Red color

        o3d.visualization.draw_geometries([voxel_grid,obb])
        
        
                
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        path = self.im_idx[index]
        
        if self.imageset == 'test':
            voxel_label = np.zeros([256, 256, 32], dtype=int).reshape((-1, 1))
        else:
            voxel_label = np.fromfile(path, dtype=np.uint16).reshape((-1, 1))  # voxel labels
            invalid = self.unpack(np.fromfile(path.replace('label', 'invalid').replace(self.folder, 'voxels'), dtype=np.uint8)).astype(np.float32)
            
        voxel_label = self.learning_map[voxel_label]
        voxel_label = voxel_label.reshape((256, 256, 32))
        invalid = invalid.reshape((256,256,32))
        voxel_label[invalid == 1]=255

        if self.get_query :
            if self.imageset == 'train' :
                p = torch.randint(0, 6, (1,)).item()
                if p == 0:
                    voxel_label, invalid = flip(voxel_label, invalid, flip_dim=0)
                elif p == 1:
                    voxel_label, invalid = flip(voxel_label, invalid, flip_dim=1)
                elif p == 2:
                    voxel_label, invalid = flip(voxel_label, invalid, flip_dim=0)
                    voxel_label, invalid = flip(voxel_label, invalid, flip_dim=1)
            query, xyz_label, xyz_center = get_query(voxel_label)

        else : 
            query, xyz_label, xyz_center = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        return voxel_label, query, xyz_label, xyz_center, self.im_idx[index], invalid
    
def get_query(voxel_label, num_class=20, grid_size = (256,256,32), max_points = 400000):
    xyzl = []
    for i in range(1, num_class):
        xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (1,0),'constant', value=i)
        xyzl.append(xyzlabel)
    tdf = compute_tdf(voxel_label, trunc_distance=2)
    xyz = torch.nonzero(torch.tensor(np.logical_and(tdf > 0, tdf <= 2)), as_tuple=False)
    xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
    xyzl.append(xyzlabel)
    
    num_far_free = int(max_points - len(torch.cat(xyzl, dim=0)))
    if num_far_free <= 0 :
        xyzl = torch.cat(xyzl, dim=0)
        xyzl = xyzl[:max_points]
    else : 
        xyz = torch.nonzero(torch.tensor(np.logical_and(voxel_label == 0, tdf == -1)), as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
        idx = torch.randperm(xyzlabel.shape[0])
        xyzlabel = xyzlabel[idx][:min(xyzlabel.shape[0], num_far_free)]
        xyzl.append(xyzlabel)
        while len(torch.cat(xyzl, dim=0)) < max_points:
            for i in range(1, num_class):
                xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
                xyzlabel = torch.nn.functional.pad(xyz, (1,0),'constant', value=i)
                xyzl.append(xyzlabel)
        xyzl = torch.cat(xyzl, dim=0)
        xyzl = xyzl[:max_points]
        
    xyz_label = xyzl[:, 0]
    xyz_center = xyzl[:, 1:]
    xyz = xyz_center.float()

    query = torch.zeros(xyz.shape, dtype=torch.float32, device=xyz.device)
    query[:,0] = 2*xyz[:,0].clamp(0,grid_size[0]-1)/float(grid_size[0]-1) -1
    query[:,1] = 2*xyz[:,1].clamp(0,grid_size[1]-1)/float(grid_size[1]-1) -1
    query[:,2] = 2*xyz[:,2].clamp(0,grid_size[2]-1)/float(grid_size[2]-1) -1
    
    return query, xyz_label, xyz_center

def compute_tdf(voxel_label: np.ndarray, trunc_distance: float = 3, trunc_value: float = -1) -> np.ndarray:
    """ Compute Truncated Distance Field (TDF). voxel_label -- [X, Y, Z] """
    # make TDF at free voxels.
    # distance is defined as Euclidean distance to nearest unfree voxel (occupied or unknown).
    free = voxel_label == 0
    tdf = distance_transform_edt(free)

    # Set -1 if distance is greater than truncation_distance
    tdf[tdf > trunc_distance] = trunc_value
    return tdf  # [X, Y, Z]

def flip(voxel, invalid, flip_dim=0):
    voxel = np.flip(voxel, axis=flip_dim).copy()
    invalid = np.flip(invalid, axis=flip_dim).copy()
    return voxel, invalid

if __name__ == '__main__':
	dataset = KITTI360()
	