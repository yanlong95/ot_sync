#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: read sampled range images of KITTI sequences as single input or batch input


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from utils.utils import *
import yaml
from tools.read_all_sets_acceleration import overlap_orientation_npz_file2string_string_nparray

"""
    read one needed $file_num range image from sequence $file_num.
    Args:
        data_root_folder: dataset root of KITTI.
        file_num: the index of the needed scan (zfill 6).
        seq_num: the sequence in which the needed scan is.
"""
def read_one_need_from_seq(data_root_folder, file_num, seq_num, zfill=6):
    depth_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/depth_map/" + file_num.zfill(zfill) + ".png",
                            cv2.IMREAD_GRAYSCALE))
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


"""
    read one batch of positive samples and negative samples with respect to $f1_index in sequence $f1_seq.
    Args:
        data_root_folder: dataset root of KITTI.
        f1_index: the index of the needed scan (zfill 6).
        f1_seq: the sequence in which the needed scan is (zfill 2).
        current_frame: the overlaps matrix for one frame (pairs, 5), each row (curr_idx, ref_idx, overlap, curr_seq, ref_seq).
        overlap_thresh: 0.3 following OverlapNet.
"""
def read_one_batch_pos_neg(data_root_folder, current_frame, overlap_thresh=0.3, zfill=6):

    batch_size = 0
    for row in range(current_frame.shape[0]):
        if current_frame[row, 2] >= 0:
            batch_size += 1

    # sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 64, 900))).type(torch.FloatTensor).cuda()
    sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 32, 900))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for row in range(current_frame.shape[0]):
        idx_curr, idx_ref, overlap, seq_curr, seq_ref = current_frame[row]

        if overlap >= 0:
            # check the overlap for the reference frame
            pos_flag = False
            if overlap >= overlap_thresh:
                pos_num += 1
                pos_flag = True
            else:
                neg_num += 1

            # seems cv2 read is slow, maybe preload the image in the format of numpy array first
            path_ref = os.path.join(data_root_folder, seq_ref, 'depth_map', f'{idx_ref.zfill(zfill)}.png')
            depth_data = cv2.imread(path_ref, cv2.IMREAD_GRAYSCALE)

            depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
            depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_data_tensor
                sample_truth[pos_idx, :] = torch.tensor(overlap).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_data_tensor
                sample_truth[batch_size-neg_idx-1, :] = torch.tensor(overlap).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1

    return sample_batch, sample_truth, pos_num, neg_num

    # for j in range(len(train_imgf1)):
    #     pos_flag = False
    #     if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
    #         if train_overlap[j] > overlap_thresh:
    #             pos_num = pos_num + 1
    #             pos_flag = True
    #         else:
    #             neg_num = neg_num + 1
    #
    #         depth_data_r = \
    #             np.array(cv2.imread(data_root_folder + train_dir2[j] + "/depth_map/" + train_imgf2[j] + ".png",
    #                         cv2.IMREAD_GRAYSCALE))
    #
    #         depth_data_tensor_r = torch.from_numpy(depth_data_r).type(torch.FloatTensor).cuda()
    #         depth_data_tensor_r = torch.unsqueeze(depth_data_tensor_r, dim=0)
    #
    #         if pos_flag:
    #             sample_batch[pos_idx,:,:,:] = depth_data_tensor_r
    #             sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
    #             pos_idx = pos_idx + 1
    #         else:
    #             sample_batch[batch_size-neg_idx-1, :, :, :] = depth_data_tensor_r
    #             sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
    #             neg_idx = neg_idx + 1
    #
    #
    # return sample_batch, sample_truth, pos_num, neg_num



if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    seqs_root = config["data_root"]["data_root_folder"]
    # ============================================================================
    train_dataset = ['/home/vectr/Documents/Dataset/train/botanical_garden/overlaps/train_set_reduced_reshape.npz',
                     '/home/vectr/Documents/Dataset/train/court_of_sciences/overlaps/train_set_reduced_reshape.npz']
    labels = overlap_orientation_npz_file2string_string_nparray(train_dataset, shuffle=True)
    data_root_folder = '/home/vectr/Documents/Dataset/train/'

    sample_batch, sample_truth, num_pos, num_neg = read_one_batch_pos_neg(data_root_folder, 1, 1, labels[138], 0.3, 6)
    print(sample_batch)
    print(sample_truth)
    print(num_pos)
    print(num_neg)

