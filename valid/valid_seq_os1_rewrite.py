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
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tools.read_samples import read_one_need_from_seq
from modules.overlap_transformer_haomo import featureExtracter
from tools.utils.utils import *
np.set_printoptions(threshold=sys.maxsize)

import time


def read_image(path):
    depth_data = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def validation(amodel):
    # ===============================================================================
    # loading paths and parameters
    config_filename = '../config/config_os1_rewrite.yml'
    with open(config_filename) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    valid_scans_folder = config['valid_config']['valid_scan_folder']
    ground_truth_folder = config['valid_config']['gt_valid_folder']
    # sequences = config['valid_config']['valid_seqs'][0]
    # ===============================================================================

    valid_scan_paths = load_files(valid_scans_folder)
    ground_truth_paths = load_files(ground_truth_folder)

    with torch.no_grad():
        num_scans = len(valid_scan_paths)
        num_valid = len(ground_truth_paths)
        descriptors = np.zeros((num_scans, 256))

        for i in tqdm.tqdm(range(num_scans)):
            # load a scan
            current_batch = read_image(valid_scan_paths[i])
            current_batch = torch.cat((current_batch, current_batch), dim=0)            # no idea why, keep it now

            # calculate descriptor
            amodel.eval()
            current_descriptor = amodel(current_batch)
            descriptors[i, :] = current_descriptor[0, :].cpu().detach().numpy()

        descriptors = descriptors.astype('float32')

        # save the descriptors to debug faster
        # np.save('/home/vectr/Documents/Dataset/train/botanical_garden/descriptors', descriptors)
        # descriptors = np.load('/home/vectr/Documents/Dataset/train/botanical_garden/descriptors.npy')

        # searching
        nlist = 1
        k = 111
        d = 256

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        if not index.is_trained:
            index.train(descriptors)

        index.add(descriptors)

        # search the closest descriptors for each one in the validation set
        remove = 50     # remove the closest scans
        top_n = 10
        num_pos_pred = 0

        scan_ids = np.arange(0, num_scans, 10)
        for i in range(num_valid):
            top_10_descriptor = []
            top_10_ground_truth = []

            scan_id = scan_ids[i]
            ground_truth = np.load(ground_truth_paths[i])
            ground_truth = ground_truth[np.argsort(ground_truth[:, 2])[::-1]]

            D, I = index.search(descriptors[scan_id, :].reshape(1, -1), k)

            for j in range(k):
                if abs(int(ground_truth[j, 1]) - scan_id) > remove and len(top_10_ground_truth) < top_n:
                    top_10_ground_truth.append(int(ground_truth[j, 1]))
                if abs(I[0, j] - scan_id) > remove and len(top_10_descriptor) < top_n:
                    top_10_descriptor.append(I[0, j])

            for idx in top_10_descriptor:
                if idx in top_10_ground_truth:
                    num_pos_pred += 1

    precision = num_pos_pred / (top_n * num_valid)
    print(f'top {top_n} precision: {precision}.')
    return precision


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amodel = featureExtracter(height=32, width=900, channels=1, use_transformer=True).to(device)
    validation(amodel)
