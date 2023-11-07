import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from matplotlib import pyplot as plt
import torch
import numpy as np
# from modules.overlap_transformer import featureExtracter
from modules.overlap_transformer_haomo import featureExtracter
from tools.read_samples import read_one_need_from_seq

np.set_printoptions(threshold=sys.maxsize)
from tools.utils.utils import *
import faiss
import yaml
import tqdm
import cv2


def read_one_need_from_seq(data_root_folder, file_num):
    depth_data = np.array(cv2.imread(os.path.join(data_root_folder, file_num), cv2.IMREAD_GRAYSCALE))
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def load_keyframes(keyframes_path):
    keyframe_images_path = os.path.join(keyframes_path, 'png_files')
    keyframe_poses_path = os.path.join(keyframes_path, 'poses/poses_kf.txt')

    keyframe_images = load_files(keyframe_images_path)
    keyframe_poses = load_poses(keyframe_poses_path)

    return keyframe_images, keyframe_poses


def keyframes_ground_truth(keyframes):
    pass


def keyframe_descriptors(keyframes):
    pass

class testHandler():
    pass


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config_os1_rewrite.yml'
    config = yaml.safe_load(open(config_filename))
    test_seq = config["test_config"]["test_seqs"][0]
    test_weights_path = config["test_config"]["test_weights"]
    keyframes_path = config["test_config"]["keyframes"]
    # ============================================================================

    range_imgs, poses = load_keyframes(keyframes_path)
