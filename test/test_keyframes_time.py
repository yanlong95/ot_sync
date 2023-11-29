import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import torch
import numpy as np
from modules.overlap_transformer_haomo import featureExtracter

np.set_printoptions(threshold=sys.maxsize)
from tools.utils.utils import *
import faiss
import yaml
import tqdm
import cv2
import open3d as o3d
import time


def read_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_data = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).to(device)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def load_keyframes(keyframe_path):
    keyframe_images_path = os.path.join(keyframe_path, 'png_files')
    keyframe_poses_path = os.path.join(keyframe_path, 'poses/poses_kf.txt')

    keyframe_images = load_files(keyframe_images_path)
    keyframe_poses = load_poses(keyframe_poses_path)

    return keyframe_images, keyframe_poses


def load_one_test_frame_pcd(test_frame_path):
    current_vertex_pcd = o3d.io.read_point_cloud(test_frame_path)
    current_vertex = np.asarray(current_vertex_pcd.points, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 3))
    return current_vertex


def load_one_test_frame_npy(test_frame_path):
    return np.load(test_frame_path)


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


def testHandler(keyframe_path, test_frames_path, weights_path, top_n=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    amodel = featureExtracter(height=32, width=900, channels=1, use_transformer=True).to(device)

    with torch.no_grad():
        # load model
        print(f'Load weights from {weights_path}')
        checkpoint = torch.load(weights_path, map_location=device)
        amodel.load_state_dict(checkpoint['state_dict'])

        # calculate ground truth and descriptors
        print('loading keyframes ...')
        keyframe_images, keyframe_poses = load_keyframes(keyframe_path)

        print('calculating descriptors for keyframe ...')
        keyframe_descriptors = calc_descriptors(keyframe_images, amodel)

        # initialize searching
        nlist = 1
        dim = 256

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

        if not index.is_trained:
            index.train(keyframe_descriptors)

        index.add(keyframe_descriptors)

        # load test frame paths
        test_frame_paths = load_files(test_frames_path)

        # test time
        t_total = 0
        t_min = 2 ** 63 - 1
        t_max = 0

        t_load_data = 0
        t_calc_descriptor = 0
        t_detach = 0
        t_search = 0
        for test_frame_path in test_frame_paths:
            t_start = time.time()
            # choose loading from pcd file or npy
            point_cloud = load_one_test_frame_npy(test_frame_path)
            # point_cloud = load_one_test_frame_pcd(test_frame_path)

            # checking dimension (x, y, z)
            assert point_cloud.shape[1] == 3

            t_load_data_start = time.time()

            depth_data = range_projection(point_cloud, fov_up=22.5, fov_down=-22.5, proj_H=32, proj_W=900, max_range=45)
            curr_batch = torch.from_numpy(depth_data).type(torch.FloatTensor).to(device)
            curr_batch = torch.unsqueeze(curr_batch, dim=0)
            curr_batch = torch.unsqueeze(curr_batch, dim=0)
            curr_batch = torch.cat((curr_batch, curr_batch), dim=0)

            t_load_data_end = time.time()

            amodel.eval()
            curr_descriptor = amodel(curr_batch)

            t_calc_descriptor_end = time.time()

            curr_descriptor = curr_descriptor[0, :].cpu().detach().numpy()

            t_detach_end = time.time()

            curr_descriptor = curr_descriptor.astype('float32')
            curr_descriptor = curr_descriptor.reshape(1, -1)

            # TODO: searching
            D, I = index.search(curr_descriptor, top_n)
            top_n_keyframes_indices = I[0]
            # top_n_keyframes_poses = keyframe_poses[top_n_keyframes_indices, :]

            t_search_end = time.time()

            t_end = time.time()
            t_total += t_end - t_start
            t_min = min(t_min, t_end - t_start)
            t_max = max(t_max, t_end - t_start)

            # TODO: check both cpu and gpu
            t_load_data += t_load_data_end - t_load_data_start
            t_calc_descriptor += t_calc_descriptor_end - t_calc_descriptor_end
            t_detach += t_detach_end - t_calc_descriptor_end
            t_search += t_search_end - t_detach_end

        print('\n\n')
        print(f'Average searching time: {t_total / len(test_frame_paths)}.')
        print(f'Maximal searching time: {t_max}.')
        print(f'Minimal searching time: {t_min}.')
        print('\n\n')

        print(f'Loading time: {t_load_data / len(test_frame_paths)}.')
        print(f'Calculating time: {t_calc_descriptor / len(test_frame_paths)}.')
        print(f'Detaching time: {t_detach / len(test_frame_paths)}.')
        print(f'Searching time: {t_search / len(test_frame_paths)}')


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config_os1_rewrite.yml'
    config = yaml.safe_load(open(config_filename))
    test_seq = config["test_config"]["test_seqs"][0]
    test_keyframe_path = config["test_config"]["test_keyframes"]
    test_weights_path = config["test_config"]["test_weights"]

    test_frames_path = config["test_config"]["test_frames"]
    test_frames_path_pcd_folder = os.path.join(test_frames_path, 'pcd_files')
    test_frames_path_npy_folder = os.path.join(test_frames_path, 'npy_files')
    # ============================================================================
    testHandler(test_keyframe_path, test_frames_path_pcd_folder, test_weights_path, top_n=5)
