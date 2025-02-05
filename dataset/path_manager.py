import os

# manual definition
PROJECT_NAMES = 'SemCity' 
SEMKITTI_DATA_PATH = '/media/cedric/Datasets2/data_odometry_voxels_all/sequences/' # the path to the sequences folder
CARLA_DATA_PATH = '' # the path to the sequences folder

# auto definition
CARLA_YAML_PATH = os.getcwd() + '/dataset/carla.yaml'
SEMKITTI_YAML_PATH = os.getcwd() + '/dataset/semantic-kitti.yaml'

# manual definition after training
AE_PATH = os.getcwd() + '/store/9_miou=83.872.pt'  # the path to the pt file 
GEN_DIFF_PATH = os.getcwd() + '/exp/diff_kitti/ema_0.9999_050000.pt' 
SSC_DIFF_PATH = os.getcwd()  + ''
