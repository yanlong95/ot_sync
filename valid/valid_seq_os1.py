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

import tqdm
import faiss
import yaml
import torch
import numpy as np
from matplotlib import pyplot as plt
from tools.read_samples import read_one_need_from_seq
from modules.overlap_transformer_haomo import featureExtracter
from tools.utils.utils import *
np.set_printoptions(threshold=sys.maxsize)


def test_chosen_seq(amodel, data_root_folder, test_seq):
    range_images = os.listdir(os.path.join(data_root_folder, test_seq, "depth_map"))

    des_list = np.zeros((len(range_images), 256))
    des_list_inv = np.zeros((len(range_images), 256))

    """Calculate the descriptors of scans"""
    print("Calculating the descriptors of scans ...")
    for i in tqdm.tqdm(range(0, len(range_images))):
        current_batch = read_one_need_from_seq(data_root_folder, str(i).zfill(6), test_seq)
        current_batch = torch.cat((current_batch, current_batch), dim=0)
        amodel.eval()
        current_batch_des = amodel(current_batch)
        des_list[i, :] = current_batch_des[0, :].cpu().detach().numpy()
        des_list_inv[i, :] = current_batch_des[1, :].cpu().detach().numpy()

    des_list = des_list.astype('float32')

    row_list = []
    remove = 100
    for i in range(remove+1, len(range_images)-1):
        nlist = 1
        k = 50
        d = 256
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained
        index.train(des_list[:i-remove, :])
        assert index.is_trained
        index.add(des_list[:i-remove, :])
        plt.clf()
        """Faiss searching"""
        D, I = index.search(des_list[i, :].reshape(1, -1), k)
        for j in range(D.shape[1]):
            """The nearest 100 frames are not considered."""
            if (i - I[:, j]) < remove:
                continue
            else:
                one_row = np.zeros((1, 3))
                one_row[:, 0] = i
                one_row[:, 1] = I[:, j]
                one_row[:, 2] = D[:, j]
                row_list.append(one_row)
                print(str(i) + "---->" + str(I[:, j]) + "  " + str(D[:, j]))

    row_list_arr = np.array(row_list)
    """Saving for the next test"""
    test_des_path = '/home/vectr/Documents/Dataset/test'
    test_des_pre_path = os.path.join(test_des_path, 'test_results_botanical_garden/predicted_des_L2_dis')
    if not os.path.exists(test_des_pre_path):
        os.makedirs(test_des_pre_path)
    np.savez_compressed(test_des_pre_path, row_list_arr)

class testHandler():
    def __init__(self, height=32, width=900, channels=1, norm_layer=None, use_transformer=True,
                 data_root_folder=None, test_seq=None, test_weights=None):
        super(testHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.data_root_folder = data_root_folder
        self.test_seq = test_seq

        self.amodel = featureExtracter(channels=self.channels, use_transformer=self.use_transformer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)

        self.parameters = self.amodel.parameters()
        self.test_weights = test_weights
        self.overlap_thresh = 0.3
        # self.overlap_thresh = 0.2

    def eval(self):
        with torch.no_grad():
            print("Loading weights from ", self.test_weights)
            checkpoint = torch.load(self.test_weights)
            self.amodel.load_state_dict(checkpoint['state_dict'])
            test_chosen_seq(self.amodel, self.data_root_folder, self.test_seq)


def validate_seq_faiss(amodel, seq_num):
    # load config ================================================================
    config_filename = '../config/config_os1.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root"]["data_root_folder"]
    test_seq = config["valid_config"]["valid_seqs"][0]
    test_weights = config["valid_config"]["valid_weights"]
    # ============================================================================
    testHandler(height=32, width=900, channels=1, norm_layer=None, use_transformer=True,
                data_root_folder=data_root_folder, test_seq=test_seq, test_weights=test_weights)



    # loop_num = 0
    #
    # with torch.no_grad():
    #     des_list = np.zeros((len(scan_paths), 256))
    #     time11 = time.time()
    #
    #     for i in range(len(scan_paths)):
    #         current_batch = read_one_need_from_seq(seqs_root, str(i).zfill(6), seq_num)
    #         # current_batch = torch.cat((current_batch, current_batch), dim=0)          # seems useless!!!
    #         amodel.eval()
    #         current_batch_des = amodel(current_batch)  # [1,256]
    #         des_list[i, :] = current_batch_des[0, :].cpu().detach().numpy()
    #
    #     time22 = time.time()
    #     cal_time = (time22-time11)/len(scan_paths)
    #
    #     des_list = des_list.astype('float32')
    #     used_num = 0
    #     print(f"calculated all descriptors (calculation time: {cal_time})")
    #     nlist = 1
    #     k = 6
    #     d = 256
    #     quantizer = faiss.IndexFlatL2(d)
    #     index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    #     assert not index.is_trained
    #     index.train(des_list)
    #     assert index.is_trained
    #     index.add(des_list)
    #
    #     for i in tqdm(range(0, len(scan_paths), 10)):
    #         used_num = used_num + 1
    #         # gtm_path = ground_truth_folder + seq_num + "/overlap_"+str(i)+".npy"
    #         gtm_path = ground_truth_folder + f"/overlap_{i}.npy"
    #         ground_truth_mapping = np.load(gtm_path)
    #         time1 = time.time()
    #         D, I = index.search(des_list[i,:].reshape(1,-1), k)  # actual search
    #         time2 = time.time()
    #         time_diff = time2 - time1
    #         if I[:, 0] == i:
    #             # print("find itself")
    #             min_index = I[:, 1]
    #             min_value = D[:, 1]
    #         else:
    #             min_index = I[:, 0]
    #             min_value = D[:, 0]
    #
    #         if ground_truth_mapping[min_index, 2] > 0.3:
    #             loop_num = loop_num + 1
    #
    # print("top1 rate: ", loop_num / used_num)
    # return loop_num / used_num


