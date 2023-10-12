import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import copy
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn import metrics

path = '/home/vectr/PycharmProjects/overlap_transformer/kitti_dataset/kitti_seq_root/loop_gt_seq00_0.3overlap_inactive.npz'
path = '/home/vectr/Documents/Dataset/gt_overlap/botanical_garden_test/loop_gt_seq00_0.3overlap_inactive.npz'
data = np.load(path, allow_pickle=True)['arr_0']

print(len(data))

sum=0
for i in range(len(data)):
    sum += len(data[i])
print(sum)

dataset = np.array([])
print(not dataset.any())