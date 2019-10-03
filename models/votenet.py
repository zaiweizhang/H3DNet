# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from backbone_module_pairwise import Pointnet2BackbonePairwise
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
        self.backbone_net_sem_support = Pointnet2BackbonePairwise(input_feature_dim=(num_class+1), task="sem")

        #self.conv1 = torch.nn.Conv1d(256,128,1)
        #self.conv2 = torch.nn.Conv1d(128,1,1)
        #self.conv2 = torch.nn.Conv1d(128,(num_class+1)*2,1)
        #self.conv2 = torch.nn.Conv1d(128,num_class,1)
        #self.conv2 = torch.nn.Conv1d(128,2,1)

        ### Semantic Segmentation
        self.conv_sem1 = torch.nn.Conv1d(256+3,128,1) ##Pointfeature + input
        self.conv_sem2 = torch.nn.Conv1d(128,(num_class+1),1)
        self.bn_sem1 = torch.nn.BatchNorm1d(128)
        self.dropout_sem1 = torch.nn.Dropout(0.5)

        ### Support Semantic
        self.conv_support1 = torch.nn.Conv1d(256+3,128,1) ##Pointfeature + xyz
        self.conv_support2 = torch.nn.Conv1d(128,num_class,1)
        self.bn_support1 = torch.nn.BatchNorm1d(128)
        self.dropout_support1 = torch.nn.Dropout(0.5)
        
        # Hough voting
        #self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        #self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
        #    mean_size_arr, num_proposal, sampling)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)

        ### Semantic Segmentation
        sem_feature = torch.cat((end_points['fp3_features'], end_points['xyz'].transpose(2,1)), 1)
        net_sem = F.relu(self.dropout_sem1(self.bn_sem1(self.conv_sem1(sem_feature))))
        net_sem = self.conv_sem2(net_sem)

        ### Support Segmentation
        sem_support_feature = torch.cat((end_points['xyz'], net_sem.transpose(2,1)), 2)
        end_points = self.backbone_net_sem_support(sem_support_feature, end_points)
        sem_support_feature = torch.cat((end_points['sem'+'fp3_features'], end_points['xyz'].transpose(2,1)), 1)
        net_sem_support = F.relu(self.dropout_support1(self.bn_support1(self.conv_support1(sem_support_feature))))
        net_sem_support = F.relu(self.conv_support2(net_sem_support))

        end_points["pred_support_class"] = net_sem_support
        end_points["pred_sem_class"] = net_sem
        
        """
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)
        """
        #import pdb;pdb.set_trace()
        return end_points


if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model
    model = VoteNet(10,12,10,np.random.random((10,3))).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key])

    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')
