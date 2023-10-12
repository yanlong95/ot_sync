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


def validation(amodel):
    # ===============================================================================
    # loading paths and parameters
    config_filename = '../config/config_os1_rewrite.yml'
    with open(config_filename) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    root = config['data_root']['data_root_folder']
    valid_scans = config['data_root']['valid_scan_folder']
    valid_poses = config['data_root']['valid_scan_poses']
    ground_truth = config['data_root']['gt_valid_folder']
    seq_num = config['valid_config']['valid_seqs']
    # ===============================================================================

    with torch.no_grad():
        num_scans = len(valid_scans)
        descriptors = np.zeros((num_scans, 256))

        for i in range(num_scans):
            # load a scan
            current_batch = read_one_need_from_seq(valid_scans, str(i).zfill(6), seq_num)
            current_batch = torch.cat((current_batch, current_batch), dim=0)                # no idea why, keep it now

            # calculate descriptor
            amodel.eval()
            current_descriptor = amodel(current_batch)
            descriptors[i, :] = current_descriptor[0, :].cpu().detach().numpy()

        descriptors = descriptors.astype('float32sync')







if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amodel = featureExtracter(channels=32, use_transformer=True).to(device)
    validation(amodel)
