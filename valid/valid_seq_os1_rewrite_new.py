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
from modules.overlap_transformer_haomo import featureExtracter
from tools.utils.utils import *
np.set_printoptions(threshold=sys.maxsize)


def read_image(path):
    depth_data = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def validation(amodel, overlap_threshold):
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
        k = 7
        d = 256

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        if not index.is_trained:
            index.train(descriptors)

        index.add(descriptors)

        # search the closest descriptors for each one in the validation set
        top_n = 3
        num_pos_pred = 0
        # num_neg_pred = 0

        scan_ids = np.arange(0, num_scans, 10)
        for i in range(num_valid):
            scan_id = scan_ids[i]
            ground_truth = np.load(ground_truth_paths[i])

            D, I = index.search(descriptors[scan_id, :].reshape(1, -1), k)
            # D_reverse, I_reverse = index.search(descriptors[scan_id, :].reshape(1, -1), num_valid)

            if I[:, 0] == scan_id:
                min_index = I[:, 1]
            else:
                min_index = I[:, 0]

            if ground_truth[min_index, 2] > overlap_threshold:
                num_pos_pred += 1

            # max_index = I_reverse[:, -1]
            # if ground_truth[max_index, 2] < overlap_threshold:
            #     num_neg_pred += 1

    # precision = num_pos_pred / (top_n * num_valid)
    precision = num_pos_pred / num_valid
    # precision_neg = num_neg_pred / num_valid
    print(f'top {top_n} precision: {precision}.')
    # print(f'top {top_n} precision_neg: {precision_neg}')
    return precision


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amodel = featureExtracter(height=32, width=900, channels=1, use_transformer=True).to(device)
    validation(amodel)
