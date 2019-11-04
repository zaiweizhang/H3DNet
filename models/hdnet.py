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

from backbone_module import Pointnet2Backbone, Pointnet2BackbonePlane
from backbone_module_pairwise import Pointnet2BackbonePairwise
from backbone_module_decoder import Pointnet2BackboneDecoder
from backbone_module_dis import Pointnet2BackboneDis

from voting_module import VotingModule
from voting_module_point import VotingPointModule
from voting_module_plane import VotingPlaneModule
from mean_shift_module import MeanShiftModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss
from resnet_autoencoder import TwoStreamNetEncoder, TwoStreamNetDecoder

class HDNet(nn.Module):
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
        #assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        #self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
        #self.backbone_net = Pointnet2BackbonePlane(input_feature_dim=self.input_feature_dim)
        self.backbone_net_point = Pointnet2Backbone(input_feature_dim=self.input_feature_dim - 4) ### Just xyz + height
        self.backbone_net_sem = Pointnet2Backbone(input_feature_dim=self.input_feature_dim - 4) ### Just xyz + height
        self.backbone_net_plane = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
        self.backbone_net_voxel = TwoStreamNetEncoder()
        #self.backbone_net_other = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
        #self.backbone_net_sem_support = Pointnet2BackbonePairwise(input_feature_dim=(num_class+1), task="sem")
        #self.backbone_net_tosem_support = Pointnet2BackboneDecoder(input_feature_dim=num_class*2, task="decoder")
        #self.backbone_net_tosem_support = Pointnet2BackboneDis(input_feature_dim=num_class*2, task="decoder")

        #self.conv1 = torch.nn.Conv1d(256,128,1)
        #self.conv2 = torch.nn.Conv1d(128,1,1)
        #self.conv2 = torch.nn.Conv1d(128,(num_class+1)*2,1)
        #self.conv2 = torch.nn.Conv1d(128,num_class,1)
        #self.conv2 = torch.nn.Conv1d(128,2,1)

        ### Semantic Segmentation
        #self.conv_sem1 = torch.nn.Conv1d(256+128+7,128,1) ##Pointfeature + input
        self.conv_sem1 = torch.nn.Conv1d(128+7,128,1) ##Pointfeature + input
        self.conv_sem2 = torch.nn.Conv1d(128,(num_class+1),1)
        self.bn_sem1 = torch.nn.BatchNorm1d(128)
        self.dropout_sem1 = torch.nn.Dropout(0.5)
        
        # Hough voting
        #self.vgen = VotingModule(self.vote_factor, 256+128)
        #self.vgen_plane = VotingPlaneModule(self.vote_factor, 256+128)
        self.vgen = VotingModule(self.vote_factor, 256+256)
        self.vgen_point = VotingPointModule(self.vote_factor, 256+256)
        self.vgen_plane = VotingPlaneModule(self.vote_factor, 256+256)
        #self.vgen = MeanShiftModule(self.vote_factor, 256)    
        self.vgen_voxel = TwoStreamNetDecoder()

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)
        
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
        batch_size = inputs['plane_label'].shape[0]

        end_points = self.backbone_net_point(inputs['point_clouds'], end_points)
        end_points = self.backbone_net_point(inputs['point_clouds'], end_points, mode='sem')
        end_points = self.backbone_net_plane(inputs['plane_label'], end_points, mode='plane')
        end_points = self.backbone_net_voxel(inputs['voxel_label'], end_points)
        
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        xyz_plane = end_points['fp2_xyz'+'plane']
        features_plane = end_points['fp2_features'+'plane']
        end_points['seed_inds'+'plane'] = end_points['fp2_inds'+'plane']
        end_points['seed_xyz'+'plane'] = xyz_plane
        end_points['seed_features'+'plane'] = features_plane

        features_vox = pc_util.voxel_to_pt_feature_batch(end_points['vox_latent_feature'], xyz)
        features_vox = features_vox.contiguous().transpose(2,1)
    
        xyz_sem = end_points['fp2_xyz'+'sem']
        features_sem = end_points['fp2_features'+'sem']
        end_points['seed_inds'+'sem'] = end_points['fp2_inds'+'sem']
        end_points['seed_xyz'+'sem'] = xyz_sem
        end_points['seed_features'+'sem'] = features_sem
        
        #features_combine_point = torch.cat((features, features_plane, features_sem.detach()), 1)
        features_combine_point = torch.cat((features, features_plane, features_vox), 1)
        features_combine_sem = features_sem
        #features_combine_sem = torch.cat((features.detach(), features_plane.detach(), features_sem), 1)
        #features_combine_plane = torch.cat((features, features_plane, features_sem.detach()), 1)
        features_combine_plane = torch.cat((features, features_plane, features_vox), 1)
        allfeat = torch.cat((xyz, torch.cat((features, features_plane), 1).contiguous().transpose(2,1)), 2)
        features_other_vox = pc_util.pt_to_voxel_feature_batch(allfeat)
        features_combine_vox = torch.cat((end_points['vox_latent_feature'], features_other_vox), 1)
        #features_combine_vox = end_points['vox_latent_feature']

        '''
        features_combine = torch.cat((features, features_plane), 1)
        '''
        proposal_xyz, proposal_features = self.vgen(xyz, features_combine_point)
        
        proposal_features_norm = torch.norm(proposal_features, p=2, dim=1)
        proposal_features = proposal_features.div(proposal_features_norm.unsqueeze(1))
        end_points['vote_xyz'] = proposal_xyz
        end_points['vote_features'] = proposal_features
        
        voted_xyz, voted_xyz_corner = self.vgen_point(xyz, features_combine_point)
        #features_norm = torch.norm(features, p=2, dim=1)
        #features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz_center'] = voted_xyz
        end_points['vote_xyz_corner'] = voted_xyz_corner

        end_points = self.vgen_voxel(features_combine_vox, end_points)
        
        seed_plane = torch.gather(inputs['plane_label'][:,:,4:], 1, end_points['seed_inds'].long().unsqueeze(-1).repeat(1,1,4))

        xyz_plane = torch.cat((xyz, seed_plane), -1)
        
        net_upper, net_lower, net_left, net_right, net_front, net_back = self.vgen_plane(xyz_plane, features_combine_plane)

        end_points['upper_rot'] = net_upper[:,:3,:].transpose(2,1).contiguous()
        end_points['upper_off'] = net_upper[:,3,:].contiguous()

        end_points['lower_rot'] = net_lower[:,:3,:].transpose(2,1).contiguous()
        end_points['lower_off'] = net_lower[:,3,:].contiguous()

        end_points['left_rot'] = net_left[:,:3,:].transpose(2,1).contiguous()
        end_points['left_off'] = net_left[:,3,:].contiguous()

        end_points['right_rot'] = net_right[:,:3,:].transpose(2,1).contiguous()
        end_points['right_off'] = net_right[:,3,:].contiguous()

        end_points['front_rot'] = net_front[:,:3,:].transpose(2,1).contiguous()
        end_points['front_off'] = net_front[:,3,:].contiguous()

        end_points['back_rot'] = net_back[:,:3,:].transpose(2,1).contiguous()
        end_points['back_off'] = net_back[:,3,:].contiguous()

        ### Semantic Segmentation
        features_for_sem = torch.cat((features_combine_sem, xyz_plane.transpose(2,1).contiguous()), 1)
        net_sem = F.relu(self.dropout_sem1(self.bn_sem1(self.conv_sem1(features_for_sem))))
        net_sem = self.conv_sem2(net_sem)
        end_points["pred_sem_class"] = net_sem

        end_points = self.pnet(proposal_xyz, proposal_features, end_points)
        """
        if end_points['use_support']:
            end_points = self.pnet(xyz_support, features_support, end_points, mode='_support')
            end_points = self.pnet(xyz_bsupport, features_bsupport, end_points, mode='_bsupport')
            #xyz = torch.cat((xyz, xyz_support, xyz_bsupport), 1)
            #features = torch.cat((features, features_support, features_bsupport), 2)
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
