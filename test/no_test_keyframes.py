import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from matplotlib import pyplot as plt
import torch
import numpy as np
import scipy as sp
# from modules.overlap_transformer import featureExtracter
from modules.overlap_transformer_haomo import featureExtracter
from tools.read_samples import read_one_need_from_seq

np.set_printoptions(threshold=sys.maxsize)
from tools.utils.utils import *
import faiss
import yaml
import tqdm
import cv2


def read_image(image_path):
    depth_data = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def load_keyframes(keyframe_path):
    keyframe_images_path = os.path.join(keyframe_path, 'png_files')
    keyframe_poses_path = os.path.join(keyframe_path, 'poses/poses_kf.txt')

    keyframe_images = load_files(keyframe_images_path)
    keyframe_poses = load_poses(keyframe_poses_path)

    return keyframe_images, keyframe_poses


def load_test_frames(test_frame_path, selection_ratio=1):
    test_frame_images_path = os.path.join(test_frame_path, 'depth')
    test_frame_poses_path = os.path.join(test_frame_path, 'poses/poses.txt')

    test_frame_images = load_files(test_frame_images_path)
    test_frame_poses = load_poses(test_frame_poses_path)

    return test_frame_images, test_frame_poses


# The ground truth based on the location of the keyframes (e.g. choose the closest top n keyframes based on the
# location).
def calc_ground_truth(poses):
    frame_loc = poses[:, :3, 3]
    frame_qua = np.zeros((len(poses), 4))

    for i in range(len(poses)):
        rotation_matrix = poses[i, :3, :3]
        rotation = sp.spatial.transform.Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()
        frame_qua[i, :] = quaternion

    return frame_loc, frame_qua


def calc_descriptors(images, amodel):
    n_keyframes = len(images)
    descriptors = np.zeros((n_keyframes, 256))

    with torch.no_grad():
        for i in tqdm.tqdm(range(n_keyframes)):
            curr_batch = read_image(images[i])
            curr_batch = torch.cat((curr_batch, curr_batch), dim=0)

            amodel.eval()
            curr_descriptor = amodel(curr_batch)
            descriptors[i, :] = curr_descriptor[0, :].cpu().detach().numpy()

    descriptors = descriptors.astype('float32')
    return descriptors


def calc_top_n(keyframe_poses, keyframe_descriptors, test_frame_poses, test_frame_descriptors):
    num_test_frame = len(test_frame_poses)

    # initial searching
    nlist = 1
    k = 7
    dim_pose = 3
    dim_descriptor = 256

    quantizer_poses = faiss.IndexFlatL2(dim_pose)
    quantizer_descriptors = faiss.IndexFlatL2(dim_descriptor)

    index_poses = faiss.IndexIVFFlat(quantizer_poses, dim_pose, nlist, faiss.METRIC_L2)
    index_descriptors = faiss.IndexIVFFlat(quantizer_descriptors, dim_pose, nlist, faiss.METRIC_L2)

    if not index_poses.is_trained:
        index_poses.train(keyframe_poses)

    if not index_descriptors.is_trained:
        index_descriptors.train(keyframe_descriptors)

    index_poses.add(keyframe_poses)
    index_descriptors.add(keyframe_descriptors)

    for curr_frame_idx in range(num_test_frame):
        curr_frame_pose = test_frame_poses[curr_frame_idx, :]
        curr_frame_descriptor = test_frame_descriptors[curr_frame_idx, :]

        D_pose, I_pose = index_poses.search(curr_frame_pose, k)
        D_descriptor, I_descriptor = index_descriptors.search(curr_frame_descriptor, k)

        # search



def testHandler(keyframe_path, test_frame_path, weights_path, test_selection=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amodel = featureExtracter(height=32, width=900, channels=1, use_transformer=True).to(device)

    with torch.no_grad():
        # load model
        print(f'Load weights from {weights_path}')
        checkpoint = torch.load(weights_path)
        amodel.load_state_dict(checkpoint['state_dict'])

        # calculate ground truth and descriptors
        keyframe_images, keyframe_poses = load_keyframes(keyframe_path)
        test_frame_images, test_frame_poses = load_test_frames(test_frame_path)

        keyframe_locs, _ = calc_ground_truth(keyframe_poses)
        test_frame_locs_full, _ = calc_ground_truth(test_frame_poses)

        keyframe_descriptors = calc_descriptors(keyframe_images, amodel)
        test_frame_descriptors_full = calc_descriptors(test_frame_images, amodel)

        # select 1 sample per test_selection samples, reduce the test size
        test_frame_locs = test_frame_locs_full[::test_selection]
        test_frame_descriptors = test_frame_descriptors_full[::test_selection]







if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config_os1_rewrite.yml'
    config = yaml.safe_load(open(config_filename))
    test_seq = config["test_config"]["test_seqs"][0]
    test_folder_path = config["test_config"]["test_folder"]
    test_weights_path = config["test_config"]["test_weights"]
    keyframe_path = config["test_config"]["keyframes"]
    # ============================================================================

    testHandler(keyframe_path, test_folder_path, test_weights_path)


