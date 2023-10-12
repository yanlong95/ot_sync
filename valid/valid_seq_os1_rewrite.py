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


def validation():
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

    # ===============================================================================

    pass


if __name__ == '__main__':
    validation()
