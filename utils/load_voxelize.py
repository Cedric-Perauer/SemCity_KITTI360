import open3d as o3d 
import os 

point_cloud = o3d.io.read_point_cloud('/media/cedric/Datasets2/data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000000002_0000000385.ply')

voxel_size = 0.15  # Adjust the voxel size as needed
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)
#voxelized_point_cloud = point_cloud.voxel_down_sample(voxel_size)

o3d.visualization.draw_geometries([voxel_grid])
