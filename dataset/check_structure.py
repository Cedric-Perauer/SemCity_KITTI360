import os
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
import yaml
import open3d as o3d

path = '/media/cedric/Datasets2/data_odometry_voxels_all/sequences/07/voxels/'
yaml_path = 'semantic-kitti.yaml'
fs = os.listdir(path)
fs = [f for f in fs if f.endswith('label')]


def color_point_cloud_by_labels(point_cloud, labels, semkittiyaml):
    """
    Colors a point cloud based on the provided labels and label colors.

    Args:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        labels (numpy.ndarray): An array of labels corresponding to each point in the point cloud.
        label_colors (dict): A dictionary mapping each label to its corresponding RGB color.

    Returns:
        open3d.geometry.PointCloud: The colored point cloud.
    """
    # Convert labels to integers
    label_colors = semkittiyaml['color_map']
    labels = labels.numpy().astype(int)

    # Create an array of colors corresponding to each point based on the labels
    colors = np.zeros((len(labels), 3))
    for label, color in label_colors.items():
        mask = labels == label
        colors[mask] = color

    # Assign the colors to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors/255.)
    return point_cloud


def compute_tdf(voxel_label: np.ndarray, trunc_distance: float = 3, trunc_value: float = -1) -> np.ndarray:
    """ Compute Truncated Distance Field (TDF). voxel_label -- [X, Y, Z] """
    # make TDF at free voxels.
    # distance is defined as Euclidean distance to nearest unfree voxel (occupied or unknown).
    free = voxel_label == 0
    tdf = distance_transform_edt(free)

    # Set -1 if distance is greater than truncation_distance
    tdf[tdf > trunc_distance] = trunc_value
    return tdf  # [X, Y, Z]


def get_query(voxel_label, num_class=20, grid_size=(256, 256, 32), max_points=400000):
    xyzl = []
    for i in range(1, num_class):
        xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=i)
        xyzl.append(xyzlabel)
    tdf = compute_tdf(voxel_label, trunc_distance=2)
    xyz = torch.nonzero(torch.tensor(
        np.logical_and(tdf > 0, tdf <= 2)), as_tuple=False)
    xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
    xyzl.append(xyzlabel)

    num_far_free = int(max_points - len(torch.cat(xyzl, dim=0)))
    if num_far_free <= 0:
        xyzl = torch.cat(xyzl, dim=0)
        xyzl = xyzl[:max_points]
    else:
        xyz = torch.nonzero(torch.tensor(np.logical_and(
            voxel_label == 0, tdf == -1)), as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
        idx = torch.randperm(xyzlabel.shape[0])
        xyzlabel = xyzlabel[idx][:min(xyzlabel.shape[0], num_far_free)]
        xyzl.append(xyzlabel)
        while len(torch.cat(xyzl, dim=0)) < max_points:
            for i in range(1, num_class):
                xyz = torch.nonzero(torch.Tensor(
                    voxel_label) == i, as_tuple=False)
                xyzlabel = torch.nn.functional.pad(
                    xyz, (1, 0), 'constant', value=i)
                xyzl.append(xyzlabel)
        xyzl = torch.cat(xyzl, dim=0)
        xyzl = xyzl[:max_points]

    xyz_label = xyzl[:, 0]
    xyz_center = xyzl[:, 1:]
    xyz = xyz_center.float()

    query = torch.zeros(xyz.shape, dtype=torch.float32, device=xyz.device)
    query[:, 0] = 2*xyz[:, 0].clamp(0, grid_size[0]-1) / \
        float(grid_size[0]-1) - 1
    query[:, 1] = 2*xyz[:, 1].clamp(0, grid_size[1]-1) / \
        float(grid_size[1]-1) - 1
    query[:, 2] = 2*xyz[:, 2].clamp(0, grid_size[2]-1) / \
        float(grid_size[2]-1) - 1

    return query, xyz_label, xyz_center


def flip(voxel, invalid, flip_dim=0):
    voxel = np.flip(voxel, axis=flip_dim).copy()
    invalid = np.flip(invalid, axis=flip_dim).copy()
    return voxel, invalid


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed


cur_file = path + fs[1]
voxel_label = np.fromfile(cur_file, dtype=np.uint16).reshape(
    (-1, 1))  # voxel labels
with open(yaml_path, 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)

remapdict = semkittiyaml['learning_map']
maxkey = max(remapdict.keys())
remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
remap_lut[list(remapdict.keys())] = list(remapdict.values())
remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
remap_lut[0] = 0  # only 'empty' stays 'empty'.
learning_map = remap_lut

voxel_label = learning_map[voxel_label]
invalid = unpack(np.fromfile(cur_file.replace(
    'label', 'invalid'), dtype=np.uint8)).astype(np.float32)

voxel_label = voxel_label.reshape((256, 256, 32))
invalid = invalid.reshape((256, 256, 32))
voxel_label[invalid == 1] = 255


p = torch.randint(0, 6, (1,)).item()
if p == 6:
    if p == 0:
        voxel_label, invalid = flip(voxel_label, invalid, flip_dim=0)
    elif p == 1:
        voxel_label, invalid = flip(voxel_label, invalid, flip_dim=1)
    elif p == 2:
        voxel_label, invalid = flip(voxel_label, invalid, flip_dim=0)
        voxel_label, invalid = flip(voxel_label, invalid, flip_dim=1)


def point2voxel(preds, coords):
    grid_size = (1, 256, 256, 32)
    output = torch.zeros((preds.shape[0], grid_size[1], grid_size[2],
                         grid_size[3]), device=preds.device).type(torch.LongTensor)
    for i in range(preds.shape[0]):
        output[i, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = preds[i]
    return output


query, xyz_label, xyz_center = get_query(voxel_label)
import pdb; pdb.set_trace()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_center[xyz_label != 0])
pcd = color_point_cloud_by_labels(pcd, xyz_label[xyz_label != 0], semkittiyaml)

voxels = point2voxel(xyz_label.type(torch.LongTensor).unsqueeze(
    0), xyz_center.type(torch.LongTensor).unsqueeze(0))

# Convert the grid to a point cloud
voxel_indices = np.where(voxels[0] != 0)
labels = voxels[voxels != 0]
points = np.column_stack(voxel_indices)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points/10)
pcd = color_point_cloud_by_labels(pcd, labels, semkittiyaml)
# Create a voxel grid from the point cloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd, voxel_size=0.04)

# Visualize the voxel grid
o3d.visualization.draw_geometries([voxel_grid])
