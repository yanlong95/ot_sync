import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import numpy as np

"""Use the tools from OverlapNet"""


def overlap_orientation_npz_file2string_string_nparray(npzfilenames, shuffle=True):
    overlaps_all = []

    for npzfilename in npzfilenames:
        h = np.load(npzfilename, allow_pickle=True)
        overlaps = h['overlaps']

        """
        pair matrix (shape = [frames, pairs, 5], frame = number of frames, pairs = number of pairs (20),
        5: [current_frame_idx, reference_frame_idx, overlap, current_frame_seq, reference_frame_seq])
        """
        # seq (frames, pairs, 2)
        seq = h['seq']
        seq_repeat = np.repeat(seq, overlaps.shape[1]).reshape(overlaps.shape)

        # current_frame_idx (frame, pairs, 1)
        current_frame_idx = np.arange(overlaps.shape[0])
        current_frame_idx_repeat = np.repeat(current_frame_idx, overlaps.shape[1]).reshape((overlaps.shape[0], -1, 1))

        # concatenate frame_idx, overlaps, seq
        overlaps_seq = np.concatenate((current_frame_idx_repeat, overlaps, seq_repeat), axis=2)

        if shuffle:
            for idx in range(overlaps_seq.shape[0]):
                shuffle_idx = np.random.permutation(overlaps.shape[1])
                overlaps_seq[idx, :, :] = overlaps_seq[idx, shuffle_idx, :]

        overlaps_all.extend(overlaps_seq)

    overlaps_all = np.array(overlaps_all)
    overlaps_all[:, :, :2] = overlaps_all[:, :, :2].astype(int).astype(str)
    return overlaps_all


if __name__ == '__main__':
    train_dataset = ['/home/vectr/Documents/Dataset/train/botanical_garden/overlaps/train_set_reduced_reshape.npz',
                             '/home/vectr/Documents/Dataset/train/court_of_sciences/overlaps/train_set_reduced_reshape.npz']

    overlaps_all = overlap_orientation_npz_file2string_string_nparray(train_dataset, shuffle=True)