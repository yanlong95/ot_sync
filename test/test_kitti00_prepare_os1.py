#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: generate the prediction files for the following PR, F1max and Recall@N calculation


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

"""
    Evaluation is conducted on KITTI 00 in our work.
    Args:
        amodel: pretrained model.
        data_root_folder: dataset root of KITTI.
        test_seq: "00" in our work.
"""
def test_chosen_seq(amodel, data_root_folder, test_seq, test_des_pre_path):
    range_images = os.listdir(os.path.join(data_root_folder, test_seq, "depth_map"))

    des_list = np.zeros((len(range_images), 256))
    des_list_inv = np.zeros((len(range_images), 256))

    """Calculate the descriptors of scans"""
    # print("Calculating the descriptors of scans ...")
    for i in range(0, len(range_images)):
        current_batch = read_one_need_from_seq(data_root_folder, str(i).zfill(6), test_seq)
        current_batch_inv_double = torch.cat((current_batch, current_batch), dim=-1)
        current_batch = torch.cat((current_batch, current_batch), dim=0)
        amodel.eval()
        current_batch_des = amodel(current_batch)
        des_list[i, :] = current_batch_des[0, :].cpu().detach().numpy()
        des_list_inv[i, :] = current_batch_des[1, :].cpu().detach().numpy()

    des_list = des_list.astype('float32')

    row_list = []
    remove = 60
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
            # if (i - I[:, j]) < 100:
                continue
            else:
                one_row = np.zeros((1, 3))
                one_row[:, 0] = i
                one_row[:, 1] = I[:, j]
                one_row[:, 2] = D[:, j]
                row_list.append(one_row)
                # print(str(i) + "---->" + str(I[:, j]) + "  " + str(D[:, j]))

    row_list_arr = np.array(row_list)
    # """Saving for the next test"""
    # test_des_path = '/home/vectr/Documents/Dataset/test'
    # test_des_pre_path = os.path.join(test_des_path, 'test_results_botanical_garden/predicted_des_L2_dis_20_60')
    if not os.path.exists(test_des_pre_path):
        os.makedirs(test_des_pre_path)
    np.savez_compressed(test_des_pre_path, row_list_arr)


class testHandler():
    def __init__(self, height=32, width=900, channels=1, norm_layer=None, use_transformer=True,
                 data_root_folder=None, test_seq=None, test_weights=None, train_iter=None):
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

        self.train_iter = train_iter

    def eval(self):
        with torch.no_grad():
            print("Loading weights from ", self.test_weights)
            checkpoint = torch.load(self.test_weights)
            self.amodel.load_state_dict(checkpoint['state_dict'])
            test_des_path = '/home/vectr/Documents/Dataset/test/test_results_botanical_garden/'
            if not os.path.exists(test_des_path):
                os.makedirs(test_des_path)
            test_des_pre_path = os.path.join(test_des_path, f'predicted_des_L2_dis_20_60_train_iter_{self.train_iter}')
            test_chosen_seq(self.amodel, self.data_root_folder, self.test_seq, test_des_pre_path)
            # test_chosen_seq(self.amodel, self.data_root_folder, self.test_seq)


if __name__ == '__main__':

    # load config ================================================================
    config_filename = '../config/config_os1.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root"]["data_root_folder"]
    test_seq = config["test_config"]["test_seqs"][0]
    test_weights_path = config["test_config"]["test_weights"]
    # ============================================================================

    """
        testHandler to handle with testing process.
        Args:
            height: the height of the range image (the beam number for convenience).
            width: the width of the range image (900, alone the lines of OverlapNet).
            channels: 1 for depth only in our work.
            norm_layer: None in our work for better model.
            use_transformer: Whether to use MHSA.
            data_root_folder: root of KITTI sequences. It's better to follow our file structure.
            test_seq: "00" in the evaluation.
            test_weights: pretrained weights.
    """
    num_iter = 100
    for i in tqdm.tqdm(range(1, num_iter)):
        test_weights = os.path.join(test_weights_path, f'pretrained_overlap_transformer{i}.pth.tar')
        test_handler = testHandler(height=32, width=900, channels=1, norm_layer=None, use_transformer=True,
                                   data_root_folder=data_root_folder, test_seq=test_seq, test_weights=test_weights,
                                   train_iter=i)
        test_handler.eval()
