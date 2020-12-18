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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

from backbone_module_scale import Pointnet2Backbone
from voting_module import VotingModule

from proposal_module_refine import ProposalModuleRefine
from proposal_module_surface import PrimitiveModule

from dump_helper import dump_results
from loss_helper import get_loss

class HDNet_1bb(nn.Module):
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
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps', with_angle=False, scale=1):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning: 4 bb tower
        self.backbone_net1 = Pointnet2Backbone(input_feature_dim=self.input_feature_dim, scale=scale) ### Just xyz + height
        scale = max(scale, 2)
        
        ### Existence flag prediction
        self.conv_flag_z1 = torch.nn.Conv1d(256*scale,128,1) 
        self.bn_flag_z1 = torch.nn.BatchNorm1d(128)
        self.conv_flag_z2 = torch.nn.Conv1d(128,2,1)

        self.conv_flag_xy1 = torch.nn.Conv1d(256*scale,128,1)
        self.bn_flag_xy1 = torch.nn.BatchNorm1d(128)
        self.conv_flag_xy2 = torch.nn.Conv1d(128,2,1)

        self.conv_flag_line1 = torch.nn.Conv1d(256*scale,128,1)
        self.bn_flag_line1 = torch.nn.BatchNorm1d(128)
        self.conv_flag_line2 = torch.nn.Conv1d(128,2,1) 
        
        # Hough voting and clustering
        self.vgen = VotingModule(self.vote_factor, 256*scale)
        self.vgen_z = VotingModule(self.vote_factor, 256*scale)
        self.vgen_xy = VotingModule(self.vote_factor, 256*scale)
        self.vgen_line = VotingModule(self.vote_factor, 256*scale)
    
        # Vote aggregation and detection
        self.pnet_z = PrimitiveModule(num_class, num_heading_bin, num_size_cluster,
                                     mean_size_arr, num_proposal, sampling, seed_feat_dim=256*scale, numd=2)
        self.pnet_xy = PrimitiveModule(num_class, num_heading_bin, num_size_cluster,
                                     mean_size_arr, num_proposal, sampling, seed_feat_dim=256*scale, numd=1)
        self.pnet_line = PrimitiveModule(num_class, num_heading_bin, num_size_cluster,
                                        mean_size_arr, num_proposal, sampling, seed_feat_dim=256*scale, numd=0)
        
        self.pnet_final = ProposalModuleRefine(num_class, num_heading_bin, num_size_cluster,
                                   mean_size_arr, num_proposal, sampling, seed_feat_dim=256*scale, with_angle=with_angle)
        
    def forward(self, inputs, end_points, mode=""):
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
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net1(inputs['point_clouds'], end_points)

        ### Extract feature here
        xyz = end_points['fp2_xyz']
        features1 = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features1
        
        ### Combine the feature here
        features_hd_discriptor = features1

        end_points['hd_feature'] = features_hd_discriptor
        
        net_flag_z = F.relu(self.bn_flag_z1(self.conv_flag_z1(features_hd_discriptor)))
        net_flag_z = self.conv_flag_z2(net_flag_z)
        end_points["pred_flag_z"] = net_flag_z

        net_flag_xy = F.relu(self.bn_flag_xy1(self.conv_flag_xy1(features_hd_discriptor)))
        net_flag_xy = self.conv_flag_xy2(net_flag_xy)
        end_points["pred_flag_xy"] = net_flag_xy

        net_flag_line = F.relu(self.bn_flag_line1(self.conv_flag_line1(features_hd_discriptor)))
        net_flag_line = self.conv_flag_line2(net_flag_line)
        end_points["pred_flag_line"] = net_flag_line

        proposal_xyz, proposal_features, center_offset, center_residual = self.vgen(xyz, features_hd_discriptor)
        proposal_features_norm = torch.norm(proposal_features, p=2, dim=1)
        proposal_features = proposal_features.div(proposal_features_norm.unsqueeze(1))
        end_points['vote_xyz'] = proposal_xyz
        end_points['vote_features'] = proposal_features
        
        voted_z, voted_z_feature, z_offset, z_residual = self.vgen_z(xyz, features_hd_discriptor)
        voted_z_feature_norm = torch.norm(voted_z_feature, p=2, dim=1)
        voted_z_feature = voted_z_feature.div(voted_z_feature_norm.unsqueeze(1))
        end_points['vote_z'] = voted_z
        end_points['vote_z_feature'] = voted_z_feature

        voted_xy, voted_xy_feature, xy_offset, xy_residual = self.vgen_xy(xyz, features_hd_discriptor)
        voted_xy_feature_norm = torch.norm(voted_xy_feature, p=2, dim=1)
        voted_xy_feature = voted_xy_feature.div(voted_xy_feature_norm.unsqueeze(1))
        end_points['vote_xy'] = voted_xy
        end_points['vote_xy_feature'] = voted_xy_feature

        voted_line, voted_line_feature, line_offset, line_residual = self.vgen_line(xyz, features_hd_discriptor)
        voted_line_feature_norm = torch.norm(voted_line_feature, p=2, dim=1)
        voted_line_feature = voted_line_feature.div(voted_line_feature_norm.unsqueeze(1))
        end_points['vote_line'] = voted_line
        end_points['vote_line_feature'] = voted_line_feature
        
        center_z, feature_z, end_points = self.pnet_z(voted_z, voted_z_feature, end_points, mode='_z')
        center_xy, feature_xy, end_points = self.pnet_xy(voted_xy, voted_xy_feature, end_points, mode='_xy')
        center_line, feature_line, end_points = self.pnet_line(voted_line, voted_line_feature, end_points, mode='_line')

        end_points = self.pnet_final(proposal_xyz, proposal_features, center_z, feature_z, center_xy, feature_xy, center_line, feature_line, end_points)
        return end_points

