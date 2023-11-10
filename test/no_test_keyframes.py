import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, transform
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
        rotation = transform.Rotation.from_matrix(rotation_matrix)
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
    k = 5
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

        # searching top n poses and descriptors
        D_pose, I_pose = index_poses.search(curr_frame_pose, k)
        D_descriptor, I_descriptor = index_descriptors.search(curr_frame_descriptor, k)

        #



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

        # keyframe_descriptors = calc_descriptors(keyframe_images, amodel)
        # test_frame_descriptors_full = calc_descriptors(test_frame_images, amodel)
        #
        # # select 1 sample per test_selection samples, reduce the test size
        # test_frame_locs = test_frame_locs_full[::test_selection]
        # test_frame_descriptors = test_frame_descriptors_full[::test_selection]

        # test 2d plot
        # keyframe_poses_plot(test_frame_locs_full, keyframe_locs)

        # test voronoi map
        calc_voronoi_map(keyframe_locs)


def calc_voronoi_map(keyframe_poses):
    xy = keyframe_poses[:, :2]
    voronoi = Voronoi(xy, incremental=True)

    # voronoi map is unbounded, set boundary by adding corner points
    x_min = min(xy[:, 0])
    x_max = max(xy[:, 0])
    y_min = min(xy[:, 1])
    y_max = max(xy[:, 1])
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_size = (x_max - x_min) / 2
    y_size = (y_max - y_min) / 2

    lower_left = [x_center - 1.2 * x_size, y_center - 1.2 * y_size]
    lower_right = [x_center + 1.2 * x_size, y_center - 1.2 * y_size]
    upper_left = [x_center - 1.2 * x_size, y_center + 1.2 * y_size]
    upper_right = [x_center + 1.2 * x_size, y_center + 1.2 * y_size]
    boundary = np.array([lower_left, lower_right, upper_left, upper_right])
    voronoi.add_points(boundary)

    # center for each region
    voronoi_centers = voronoi.points[:-len(boundary)]

    # extract the vertices positions for each region
    vertices = voronoi.vertices                 # vertices positions
    regions = voronoi.regions                   # indices of vertices for each region
    point_region = voronoi.point_region         # voronoi region (index of polyhedron)

    voronoi_vertices_indices = [regions[point_region[i]] for i in range(len(point_region) - len(boundary))]
    voronoi_vertices_poses = []                 # the vertices for all regions
    delaunay_triangulation = []                 # create delaunay triangulation to determine if a point is inside region

    for voronoi_vertex_index in voronoi_vertices_indices:
        vertices_poses = [vertices[idx] for idx in voronoi_vertex_index if idx > -1]
        voronoi_vertices_poses.append(vertices_poses)
        delaunay_triangulation.append(Delaunay(vertices_poses))

    # double check if the point lies within the region for all the regions of the map
    for i in range(len(voronoi_vertices_poses)):
        point = voronoi_centers[i]
        vertex = np.array(voronoi_vertices_poses[i])
        tri = delaunay_triangulation[i]

        if tri.find_simplex(point) < 0:                     # True if the point lies outside the region
            print(f'point {i} lies outside the region.')

        fig = voronoi_plot_2d(voronoi)
        plt.scatter(point[0], point[1], c='b', s=50, label='center')
        plt.scatter(vertex[:, 0], vertex[:, 1], c='r', s=100, label='vertices')
        plt.legend()
        plt.show()

    # fig = voronoi_plot_2d(voronoi)
    # plt.show()

    return delaunay_triangulation, voronoi_centers


def keyframe_poses_plot(poses, keyframe_poses, dim=2):
    if dim == 2:
        x = poses[:, 0]
        y = poses[:, 1]

        x_keyframe = keyframe_poses[:, 0]
        y_keyframe = keyframe_poses[:, 1]

        fig, ax = plt.subplots()
        ax.scatter(x, y, c='gold', s=2, label='Trajectory')
        ax.scatter(x_keyframe, y_keyframe, c='blue', s=3, label='Keyframe')

        plt.title('Position')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.show()

    elif dim == 3:
        xyz = poses
    else:
        raise "Error: dimension must be 2 or 3."


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


