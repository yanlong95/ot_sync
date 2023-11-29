import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from matplotlib import pyplot as plt
import matplotlib
import torch
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, transform
from modules.overlap_transformer_haomo import featureExtracter

np.set_printoptions(threshold=sys.maxsize)
from tools.utils.utils import *
import faiss
import yaml
import tqdm
import cv2
import open3d as o3d


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


def load_test_frames(test_frame_path):
    return load_files(test_frame_path)


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


def calc_top_n(keyframe_poses, keyframe_descriptors, keyframe_voronoi_region, test_frame_poses, test_frame_descriptors,
               top_n=5):
    num_test_frame = len(test_frame_poses)

    # initialize searching
    nlist = 1
    dim_pose = 3
    dim_descriptor = 256

    quantizer_poses = faiss.IndexFlatL2(dim_pose)
    quantizer_descriptors = faiss.IndexFlatL2(dim_descriptor)

    index_poses = faiss.IndexIVFFlat(quantizer_poses, dim_pose, nlist, faiss.METRIC_L2)
    index_descriptors = faiss.IndexIVFFlat(quantizer_descriptors, dim_descriptor, nlist, faiss.METRIC_L2)

    if not index_poses.is_trained:
        index_poses.train(keyframe_poses)

    if not index_descriptors.is_trained:
        index_descriptors.train(keyframe_descriptors)

    index_poses.add(keyframe_poses)
    index_descriptors.add(keyframe_descriptors)

    positive_pred = []
    negative_pred = []
    top_n_choices = []
    positive_pred_indices = []
    negative_pred_indices = []

    for curr_frame_idx in range(num_test_frame):
        curr_frame_pose = test_frame_poses[curr_frame_idx, :].reshape(1, -1)                        # (dim,) to (1, dim)
        curr_frame_descriptor = test_frame_descriptors[curr_frame_idx, :].reshape(1, -1)

        # searching top n poses and descriptors
        # D_pose, I_pose = index_poses.search(curr_frame_pose, top_n)
        D_descriptor, I_descriptor = index_descriptors.search(curr_frame_descriptor, top_n)

        # determine if a point inside the regions
        top_n_keyframes_indices = I_descriptor[0]
        top_n_keyframes_regions = [keyframe_voronoi_region[idx] for idx in top_n_keyframes_indices]
        top_n_choices.append(top_n_keyframes_indices)

        for idx in range(top_n):
            pos_2d = curr_frame_pose[0][:2]
            region = top_n_keyframes_regions[idx]

            if region.find_simplex(pos_2d) >= 0:  # True if the point lies inside the region
                positive_pred.append(pos_2d)
                positive_pred_indices.append(curr_frame_idx)
                break

            if idx == top_n - 1:
                negative_pred.append(pos_2d)
                negative_pred_indices.append(curr_frame_idx)

    precision = len(positive_pred) / num_test_frame
    print(f'Prediction precision: {precision}.')

    return precision, positive_pred, negative_pred, positive_pred_indices, negative_pred_indices, top_n_choices


def testHandler(keyframe_path, test_frames_path, weights_path, descriptors_path, test_selection=1, load_descriptors=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amodel = featureExtracter(height=32, width=900, channels=1, use_transformer=True).to(device)

    with torch.no_grad():
        # load model
        print(f'Load weights from {weights_path}')
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        amodel.load_state_dict(checkpoint['state_dict'])

        # calculate ground truth and descriptors
        print('loading keyframes ...')
        keyframe_images, keyframe_poses = load_keyframes(keyframe_path)

        print('calculating descriptors for keyframe ...')
        keyframe_descriptors = calc_descriptors(keyframe_images, amodel)

        # load test frames from pcd files
        test_frame_paths = load_test_frames(test_frames_path)

        # choose loading from pcd file or npy
        for test_frame_path in test_frame_paths:
            point_cloud = load_one_test_frame_npy(test_frame_path)
            # point_cloud = load_one_test_frame_pcd(test_frame_path)

            # checking dimension (x, y, z)
            assert point_cloud.shape[1] == 3
            depth_data = range_projection(point_cloud, fov_up=22.5, fov_down=-22.5, proj_H=32, proj_W=900, max_range=45)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            curr_batch = torch.from_numpy(depth_data).type(torch.FloatTensor).to(device)
            curr_batch = torch.unsqueeze(curr_batch, dim=0)
            curr_batch = torch.unsqueeze(curr_batch, dim=0)
            curr_batch = torch.cat((curr_batch, curr_batch), dim=0)

            amodel.eval()
            curr_descriptor = amodel(curr_batch)
            curr_descriptor = curr_descriptor[0, :].cpu().detach().numpy()
            curr_descriptor = curr_descriptor.astype('float32')

            # TODO: searching
            # TODO: time it
            # TODO: check both cpu and gpu

        # calculate the top n choices
        precision, positive_pred, negative_pred, positive_pred_indices, negative_pred_indices, top_n_choices = \
            calc_top_n(keyframe_locs, keyframe_descriptors, keyframe_voronoi_region, test_frame_locs,
                       test_frame_descriptors, top_n=5)


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

    test_descriptors_path = config["test_config"]["test_descriptors"]
    # ============================================================================
    testHandler(test_keyframe_path, test_frames_path, test_weights_path, test_descriptors_path, test_selection=10,
                load_descriptors=True)
