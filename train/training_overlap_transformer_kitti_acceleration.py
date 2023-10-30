#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: train OverlapTransformer with KITTI sequences

import os
import sys
import time

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')    
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tools.read_all_sets_acceleration import overlap_orientation_npz_file2string_string_nparray
# from modules.overlap_transformer import featureExtracter
from modules.overlap_transformer_haomo import featureExtracter
from tools.read_samples_acceleration import read_one_batch_pos_neg
from tools.read_samples_acceleration import read_one_need_from_seq
np.set_printoptions(threshold=sys.maxsize)
import modules.loss as PNV_loss
from tools.utils.utils import *
from valid.valid_seq import validate_seq_faiss
import yaml


class trainHandler():
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer=True, lr=0.001,
                 data_root_folder=None, train_set=None, training_seqs=None):
        super(trainHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.learning_rate = lr
        self.data_root_folder = data_root_folder
        self.train_set = train_set
        self.training_seqs = training_seqs

        self.amodel = featureExtracter(channels=self.channels, use_transformer=self.use_transformer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.parameters = self.amodel.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)

        # self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

        self.traindata_npzfiles = train_set
        print(self.traindata_npzfiles)
        self.labels = overlap_orientation_npz_file2string_string_nparray(self.traindata_npzfiles)

        """change the args for resuming training process"""
        self.resume = True
        self.save_name = "../weights/pretrained_overlap_transformer44.pth.tar"

        """overlap threshold follows OverlapNet"""
        self.overlap_thresh = 0.3

    def train(self):

        epochs = 100
        """resume from the saved model"""
        if self.resume:
            resume_filename = self.save_name
            print("Resuming from ", resume_filename)
            checkpoint = torch.load(resume_filename)
            starting_epoch = checkpoint['epoch']
            self.amodel.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Training From Scratch ...")
            starting_epoch = 0

        writer1 = SummaryWriter(comment="LR_0.xxxx")

        for i in range(starting_epoch+1, epochs):
            """
            self.labels consist with overlaps and seq for each frame. The shape of the numpy array is (n, 20, 5). 
            n is the total number of frames. 20 is the corresponding reference frames (max_pos = 10, pax_neg = 10).
            each row in the reference frame is [current_frame_idx, reference_frame_idx, overlap, current_frame_seq, 
            reference_frame_seq]
            """

            self.labels = overlap_orientation_npz_file2string_string_nparray(self.traindata_npzfiles, shuffle=True)

            print("=======================================================================\n\n\n")
            print("training with seq: ", self.training_seqs)
            print("total frames: ", len(self.labels))
            print("\n\n\n=======================================================================")

            loss_each_epoch = 0
            used_num = 0

            for row in range(len(self.labels)):
                """
                    check whether the query is used to train before (continue_flag==True/False).
                    TODO: More efficient method
                    seems do not have to check used or not for this data structure
                """
                """read one query range image from KITTI sequences"""
                idx_current = self.labels[row, 0, 0]
                dir_current = self.labels[row, 0, 4]
                current_batch = read_one_need_from_seq(self.data_root_folder, idx_current, dir_current, zfill=6)

                """
                    read several reference range images from KITTI sequences
                    to consist of positive samples and negative samples
                """
                current_frame_overlaps = self.labels[row]
                sample_batch, sample_truth, pos_num, neg_num = read_one_batch_pos_neg \
                    (self.data_root_folder, current_frame_overlaps, overlap_thresh=self.overlap_thresh, zfill=6)

                """
                    the balance of positive samples and negative samples.
                    TODO: Update for better training results
                """
                use_pos_num = 6
                use_neg_num = 6
                if pos_num >= use_pos_num and neg_num >= use_neg_num:
                    sample_batch = torch.cat((sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:pos_num + use_neg_num, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:use_pos_num, :], sample_truth[pos_num:pos_num+use_neg_num, :]), dim=0)
                    pos_num = use_pos_num
                    neg_num = use_neg_num
                elif pos_num >= use_pos_num:
                    sample_batch = torch.cat((sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:use_pos_num, :], sample_truth[pos_num:, :]), dim=0)
                    pos_num = use_pos_num
                elif neg_num >= use_neg_num:
                    sample_batch = sample_batch[0:pos_num+use_neg_num,:,:,:]
                    sample_truth = sample_truth[0:pos_num+use_neg_num, :]
                    neg_num = use_neg_num

                if neg_num == 0:
                    continue

                input_batch = torch.cat((current_batch, sample_batch), dim=0)

                input_batch.requires_grad_(True)
                self.amodel.train()
                self.optimizer.zero_grad()

                global_des = self.amodel(input_batch)
                o1, o2, o3 = torch.split(
                    global_des, [1, pos_num, neg_num], dim=0)
                MARGIN_1 = 0.5
                """
                    triplet_loss: Lazy for pos
                    triplet_loss_inv: Lazy for neg
                """
                loss = PNV_loss.triplet_loss(o1, o2, o3, MARGIN_1, lazy=False)

                if loss == -1:
                    continue

                # loss = PNV_loss.triplet_loss_inv(o1, o2, o3, MARGIN_1, lazy=False, use_min=True)
                loss.backward()
                self.optimizer.step()
                print(str(used_num), loss)

                if torch.isnan(loss):
                    print("Something error ...")
                    print(pos_num)
                    print(neg_num)

                loss_each_epoch = loss_each_epoch + loss.item()
                used_num = used_num + 1

            print("epoch {} loss {}".format(i, loss_each_epoch/used_num))
            print("saving weights ...")
            self.scheduler.step()

            """save trained weights and optimizer states"""
            self.save_name = "../weights/pretrained_overlap_transformer"+str(i)+".pth.tar"

            torch.save({
                'epoch': i,
                'state_dict': self.amodel.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
                self.save_name)

            print("Model Saved As " + 'pretrained_overlap_transformer' + str(i) + '.pth.tar')

            writer1.add_scalar("loss", loss_each_epoch / used_num, global_step=i)


            """a simple validation with KITTI 02"""
            print("validating ......")
            with torch.no_grad():
                # top1_rate = validate_seq_faiss(self.amodel, "02")
                top1_rate = validate_seq_faiss(self.amodel, "botanical_garden")
                writer1.add_scalar("top1_rate", top1_rate, global_step=i)


if __name__ == '__main__':
    # load config ================================================================
    # config_filename = '../config/config.yml'
    config_filename = '../config/config_os1.yml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root"]["data_root_folder"]
    training_seqs = config["training_config"]["training_seqs"]
    # ============================================================================

    # along the lines of OverlapNet
    # traindata_npzfiles = [os.path.join(data_root_folder, seq, 'overlaps/train_set.npz') for seq in training_seqs]
    traindata_npzfiles = [os.path.join(data_root_folder, seq, 'overlaps/train_set_reduced_reshape.npz') for seq in training_seqs]

    """
        trainHandler to handle with training process.
        Args:
            height: the height of the range image (the beam number for convenience).
            width: the width of the range image (900, alone the lines of OverlapNet).
            channels: 1 for depth only in our work.
            norm_layer: None in our work for better model.
            use_transformer: Whether to use MHSA.
            lr: learning rate, which needs to fine tune while training for the best performance.
            data_root_folder: root of KITTI sequences. It's better to follow our file structure.
            train_set: traindata_npzfiles (alone the lines of OverlapNet).
            training_seqs: sequences number for training (alone the lines of OverlapNet).
    """
    # train_handler = trainHandler(height=32, width=900, channels=1, norm_layer=None, use_transformer=True, lr=0.000005,
    #                              data_root_folder=data_root_folder, train_set=traindata_npzfiles, training_seqs = training_seqs)

    train_handler = trainHandler(height=32, width=900, channels=1, norm_layer=None, use_transformer=True, lr=0.000001,
                                 data_root_folder=data_root_folder, train_set=traindata_npzfiles, training_seqs=training_seqs)

    train_handler.train()
