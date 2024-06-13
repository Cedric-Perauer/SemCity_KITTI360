import open3d as o3d
import os
from tqdm import tqdm
from collections import Counter
import numpy as np
import math
from kitti360scripts.helpers.ply import parse_header


def load_kitti_poses(file_path):
    """
    Load KITTI poses from a text file.

    Args:
    - file_path (str): Path to the KITTI pose file.

    Returns:
    - poses (list of np.ndarray): List of 4x4 transformation matrices.
    """
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            line = [float(i) for i in line.split(' ')[1:]]
            pose = np.array(line).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    return poses


file_path = '/media/cedric/Datasets2/KITTI_360/data_poses/2013_05_28_drive_0000_sync/poses.txt'
poses = load_kitti_poses(file_path)

pcd_dir = '/media/cedric/Datasets2/KITTI_360/data_3d/train/2013_05_28_drive_0000_sync/static/'
pcds = os.listdir(pcd_dir)
pcds = [i for i in pcds if i.endswith('.ply')]


# Function to extract the starting number from the filename

def extract_start_number(filename):
    start_number = filename.split('_')[0]
    return int(start_number)


def are_points_inside_obb(points, obb):
    """
    Vectorized check if multiple points are inside the OBB by transforming the points into the OBB's local coordinate system.

    Args:
    - points: Nx3 numpy array of points.
    - obb: An OBB object with attributes 'center', 'R', and 'extent'.

    Returns:
    - A boolean array indicating whether each point is inside the OBB.
    """
    # Translate the points to the OBB's local origin
    local_points = points - obb.center

    # Initialize a boolean array to keep track of points inside the OBB
    inside = np.ones(local_points.shape[0], dtype=bool)

    # Project the translated points onto the OBB's local axes and check extents
    for i in range(3):
        axis = np.array(obb.R[:, i])
        extent = obb.extent[i] / 2.0

        # Calculate the projection of each point onto the current axis
        projection = np.dot(local_points, axis)

        # Update 'inside' to False for points outside the OBB's extent on this axis
        inside &= np.abs(projection) <= extent

    return inside


pcds = sorted(pcds, key=extract_start_number)

vis_list = []

invalid_color = [0.5019607843137255, 0.5019607843137255, 0.5019607843137255]

voxel_size = 0.1  # Adjust the voxel size as needed
aggregated_pcd = None
vis_list = []
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}

for i, pcd_f in tqdm(enumerate(pcds[:1])):
    pcd = o3d.io.read_point_cloud(pcd_dir + pcd_f)
    with open(pcd_dir + pcd_f, 'rb') as plyfile:
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        ext = valid_formats[fmt]
        num_points, properties = parse_header(plyfile, ext)

        data = np.fromfile(plyfile, dtype=properties, count=num_points)

    semIDs, instanceIDs = [], []
    for j in range(data.shape[0]):
        semIDs.append(data[j][5])
        instanceIDs.append(data[j][6])

    colors = np.asarray(pcd.colors)

    color_condition = np.all(colors != invalid_color, axis=1)
    color_indices = np.where(color_condition)[0]
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points)[color_indices])
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(pcd.colors)[color_indices])
    if i == 0:
        aggregated_pcd = pcd
    else:
        aggregated_pcd += pcd

vis_list.append(aggregated_pcd)
dist_travelled = 0
last_extracted = None
dist_threshold = 10

cur_idx = 20
dim = 38

for pose in poses[20:300]:
    cur_pose = pose[:2, -1]  # discard the z axis
    if last_extracted is not None:
        dist_travelled = np.linalg.norm(cur_pose-last_extracted)

    if last_extracted is None or dist_travelled > dist_threshold:
        last_extracted = cur_pose
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.1, 0.1, 0.7])  # Paint the sphere blue
        translation_vector = pose[:3, -1]
        sphere.translate(translation_vector)
        vis_list.append(sphere)
        '''
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[translation_vector[0] - 19,
                       translation_vector[1] - 19, translation_vector[2] - 19],
            max_bound=[translation_vector[0] + 19,
                       translation_vector[1] + 19, translation_vector[2] + 19]
        )
        points_within_bounding_box = aggregated_pcd.crop(bounding_box)
        o3d.visualization.draw_geometries([points_within_bounding_box])
        '''
        pos_pcd = poses[cur_idx+1][:3, -1]
        pos_last = pose[:3, -1]
        direction_vector = pos_pcd - pos_last

        direction_vector_normalized = direction_vector / \
            np.linalg.norm(direction_vector)

        y_axis = direction_vector_normalized
        z_axis = np.array([0, 0, 1]) if np.abs(
            y_axis[1]) != 1 else np.array([1, 0, 0])
        # Ensure z_axis is orthogonal to y_axis
        x_axis = np.cross(y_axis, z_axis)
        x_axis_normalized = x_axis / np.linalg.norm(x_axis)
        # Recompute z_axis to ensure orthogonality
        z_axis = np.cross(x_axis_normalized, y_axis)
        z_axis_normalized = z_axis / np.linalg.norm(z_axis)

        # Construct the rotation matrix
        rotation_matrix = np.vstack(
            [x_axis_normalized, y_axis, z_axis_normalized]).T

        # Calculate the center of the OBB (midpoint between start and end poses)
        center = pos_pcd

        # Define the extents of the OBB (length, width, height)
        extents = np.array([40, 40, 40])  # Adjust these values as needed

        # Create an Oriented Bounding Box (OBB)
        obb2 = o3d.geometry.OrientedBoundingBox(
            center, rotation_matrix, extents)

        points = np.asarray(aggregated_pcd.points)
        boolean_arr = are_points_inside_obb(points, obb2)
        ids = np.where(boolean_arr == 1)[0]

        pcd = aggregated_pcd.select_by_index(ids)

        spherec = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        spherec.compute_vertex_normals()
        spherec.paint_uniform_color([0.1, 0.1, 0.7])  # Paint the sphere blue
        translation_vector = pose[:3, -1]
        spherec.translate(translation_vector)

        spheren = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        spheren.compute_vertex_normals()
        spheren.paint_uniform_color([0.1, 0.1, 0.7])  # Paint the sphere blue
        translation_vector = poses[cur_idx+1][:3, -1]
        spheren.translate(translation_vector)

        o3d.visualization.draw_geometries([pcd, obb2, spherec, spheren])

    cur_idx += 1
