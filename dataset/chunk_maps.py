import open3d as o3d
import os
from tqdm import tqdm
from collections import Counter
import numpy as np
import math
from kitti360scripts.helpers.ply import parse_header
import yaml
import random

vis_list = []

invalid_color = [0.5019607843137255, 0.5019607843137255, 0.5019607843137255]

voxel_size = 0.1  # Adjust the voxel size as needed
aggregated_pcd = None
vis_list = []
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


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


def extract_points(poses, pose, aggregated_pcd, cur_idx):
    pos_pcd = poses[cur_idx+1][:3, -1]
    pos_last = pose[:3, -1]
    direction_vector = pos_pcd - pos_last

    direction_vector_normalized = direction_vector / \
        np.linalg.norm(direction_vector)

    y_axis = direction_vector_normalized
    z_axis = np.array([0, 0, 1]) if np.abs(
        y_axis[1]) != 1 else np.array([1, 0, 0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis_normalized = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis_normalized, y_axis)
    z_axis_normalized = z_axis / np.linalg.norm(z_axis)

    # Construct the rotation matrix
    rotation_matrix = np.vstack(
        [x_axis_normalized, y_axis, z_axis_normalized]).T

    center = pos_pcd
    extents = np.array([40, 40, 40])  # Adjust these values as needed

    obb2 = o3d.geometry.OrientedBoundingBox(
        center, rotation_matrix, extents)

    points = np.asarray(aggregated_pcd.points)
    boolean_arr = are_points_inside_obb(points, obb2)
    ids = np.where(boolean_arr == 1)[0]
    return ids


def get_ply_data(pcd_dir, pcd_f):

    with open(pcd_dir + pcd_f, 'rb') as plyfile:
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        ext = valid_formats[fmt]
        num_points, properties = parse_header(plyfile, ext)

        data = np.fromfile(plyfile, dtype=properties, count=num_points)
        return data


base_path = '/media/cedric/Datasets2/KITTI_360/data_3d/train/'
folders = os.listdir(base_path)
folder_pths = [
    base_path + folder for folder in folders if os.path.isdir(base_path + folder)]
folder_pths.sort()
min_points = 400000

out_dir = '/media/cedric/Datasets2/KITTI_360/preprocessed/'
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)

for folder in folder_pths:
    poses = load_kitti_poses(folder.replace('data_3d/train/','data_poses/') + '/poses.txt')
    pcd_dir = folder + '/static/'
    pcds = os.listdir(pcd_dir)
    pcds = [i for i in pcds if i.endswith('.ply')]
    pcds = sorted(pcds, key=extract_start_number)
    folders = os.listdir()
    semIDs, instanceIDs = [], []
    cur_out_dir = out_dir + folder.split('/')[-1] + '/'
    if os.path.exists(cur_out_dir) is False:
                os.makedirs(cur_out_dir)
    else : 
        print("skipping completed dir",folder)
        continue
    i = 0
    for pcd_f in tqdm(pcds):
        data = get_ply_data(pcd_dir, pcd_f)
        curSems = []
        curInsts = []
        points = []
        rgb = []
        for j in range(data.shape[0]):
            curSems.append(data[j][6])
            curInsts.append(data[j][7])
            points.append([data[j][0], data[j][1], data[j][2]])
            rgb.append([data[j][3], data[j][4], data[j][5]])
        curSems = np.array(curSems)
        curInsts = np.array(curInsts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.)
        colors = np.asarray(pcd.colors)
        color_condition = np.all(colors != invalid_color, axis=1)
        color_indices = np.where(color_condition)[0]
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points)[color_indices])
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[color_indices])
        curSems = curSems[color_indices]
        semIDs += curSems.tolist()
        if i == 0:
            aggregated_pcd = pcd
        else:
            aggregated_pcd += pcd
        i += 1

    vis_list.append(aggregated_pcd)
    dist_travelled = 0
    last_extracted = None
    dist_threshold = 10
    cur_idx = 0
    semIDs = np.array(semIDs)

    for pose in tqdm(poses[cur_idx:]):
        cur_pose = pose[:2, -1]  # discard the z axis
        if last_extracted is not None:
            dist_travelled = np.linalg.norm(cur_pose-last_extracted)

        if last_extracted is None or dist_travelled > dist_threshold:
            last_extracted = cur_pose
            ids = extract_points(poses, pose, aggregated_pcd, cur_idx)
            if ids.shape[0] < min_points:
                continue
            pcd = aggregated_pcd.select_by_index(ids)
            pts = np.asarray(pcd.points)
            dim_x, dim_y = abs(
                pts[:, 0].min() - pts[:, 0].max()), abs(pts[:, 1].min() - pts[:, 1].max())
            if dim_x < 25 or dim_y < 25:
                continue
            cur_sem = semIDs[ids]
            colors = np.asarray(pcd.colors)
            points = np.asarray(pcd.points)
            # o3d.visualization.draw_geometries([pcd])
            # pcd = color_point_cloud_by_labels(pcd,cur_sem)
            # o3d.visualization.draw_geometries([pcd])
            # do some storing of data here
            
            cur_out_fn = cur_out_dir + f'{cur_idx}.npz'
            np.savez(cur_out_fn, semantics=cur_sem, xyz=points,
                    colors=colors, cur_pose=pose, next_pose=poses[cur_idx+1])

            bounding_box_size = [
                38,
                38,
                38
            ]
            # Define the size of the bounding box
            # Create the axis-aligned bounding box (AABB) around the current pose
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=pose[:3,-1] - np.array(bounding_box_size) / 2,
                max_bound=pose[:3,-1] + np.array(bounding_box_size) / 2
            )
            indices_in_aabb = aabb.get_point_indices_within_bounding_box(
                aggregated_pcd.points)

            pcd_in_aabb = aggregated_pcd.select_by_index(indices_in_aabb)
            #o3d.visualization.draw_geometries([pcd_in_aabb])

            pts = np.asarray(aggregated_pcd.points)[indices_in_aabb]
            curSems = semIDs[indices_in_aabb]
            colors = np.asarray(aggregated_pcd.colors)[indices_in_aabb]
            cur_out_fn = cur_out_dir + f'{cur_idx}_axis_aligned.npz'
            np.savez(cur_out_fn, semantics=curSems, xyz=pts,
                    colors=colors, cur_pose=pose, next_pose=poses[cur_idx+1])

        cur_idx += 1
