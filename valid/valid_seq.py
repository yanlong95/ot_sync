#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: validation with KITTI 02


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')    

import torch
import numpy as np
from tools.read_samples import read_one_need_from_seq
np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm
from tools.utils.utils import *
import faiss
import time
import yaml

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def validate_seq_faiss(amodel, seq_num):
    # load config ================================================================
    # config_filename = '../config/config.yml'
    config_filename = '../config/config_os1.yml'
    config = yaml.safe_load(open(config_filename))
    seqs_root = config["data_root"]["data_root_folder"]
    valid_scan_folder = config["data_root"]["valid_scan_folder"]
    valid_scan_poses = config["data_root"]["valid_scan_poses"]
    ground_truth_folder = config["data_root"]["gt_valid_folder"]   # download needed *************************************************
    # calib_file_folder = config["data_root"]["calib_file_folder"]
    # pose_file_folder = config["data_root"]["pose_file_folder"]
    # ============================================================================
    scan_paths = load_files(valid_scan_folder)
    poses = load_poses(valid_scan_poses)

    # calib_file = calib_file_folder + seq_num + "/calib.txt"
    # poses_file = pose_file_folder + seq_num + ".txt"
    # T_cam_velo = load_calib(calib_file)
    # T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    # T_velo_cam = np.linalg.inv(T_cam_velo)
    # poses = load_poses(poses_file)
    # pose0_inv = np.linalg.inv(poses[0])
    #
    # poses_new = []
    # for pose in poses:
    #     poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
    # poses = np.array(poses_new)

    loop_num = 0

    with torch.no_grad():
        des_list = np.zeros((len(scan_paths), 256))
        time11 = time.time()

        for i in range(len(scan_paths)):
            current_batch = read_one_need_from_seq(seqs_root, str(i).zfill(6), seq_num)
            # current_batch = torch.cat((current_batch, current_batch), dim=0)          # seems useless!!!
            amodel.eval()
            current_batch_des = amodel(current_batch)  # [1,256]
            des_list[i, :] = current_batch_des[0, :].cpu().detach().numpy()

        time22 = time.time()
        cal_time = (time22-time11)/len(scan_paths)

        des_list = des_list.astype('float32')
        used_num = 0
        print(f"calculated all descriptors (calculation time: {cal_time})")
        nlist = 1
        k = 6
        d = 256
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained
        index.train(des_list)
        assert index.is_trained
        index.add(des_list)

        for i in tqdm(range(0, len(scan_paths), 10)):
            used_num = used_num + 1
            # gtm_path = ground_truth_folder + seq_num + "/overlap_"+str(i)+".npy"
            gtm_path = ground_truth_folder + f"/overlap_{i}.npy"
            ground_truth_mapping = np.load(gtm_path)
            time1 = time.time()
            D, I = index.search(des_list[i,:].reshape(1,-1), k)  # actual search
            time2 = time.time()
            time_diff = time2 - time1
            if I[:, 0] == i:
                # print("find itself")
                min_index = I[:, 1]
                min_value = D[:, 1]
            else:
                min_index = I[:, 0]
                min_value = D[:, 0]

            if ground_truth_mapping[min_index, 2] > 0.3:
                loop_num = loop_num + 1

            if i == 3000:
                # visualize the top choice position based on the descriptors
                xy_pos = poses[:, :2, 3]
                fig, ax = plt.subplots()
                norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
                mapper = cm.ScalarMappable(norm=norm)
                mapper.set_array(np.zeros_like(ground_truth_mapping[:, 2]))
                colors = np.array([mapper.to_rgba(a) for a in np.zeros_like(ground_truth_mapping[:, 2])])

                current_pos = xy_pos[i, :]
                selection_poses = xy_pos[I[0, 1:], :]

                ax.scatter(current_pos[0], current_pos[1], c='blue', s=50)
                ax.scatter(selection_poses[0, 0], selection_poses[0, 1], c='green', s=30)
                ax.scatter(selection_poses[1:, 0], selection_poses[1:, 1], c='yellow', s=30)

                ax.scatter(xy_pos[:, 0], xy_pos[:, 1], c=colors, s=1)

                ax.axis('square')
                ax.set_xlabel('X [m]')
                ax.set_ylabel('Y [m]')
                ax.set_title('Overlap Map')
                cbar = fig.colorbar(mapper, ax=ax)
                cbar.set_label('Overlap', rotation=270, weight='bold')
                plt.show()

    print("top1 rate: ", loop_num / used_num)
    return loop_num / used_num


