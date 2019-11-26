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
import cue_to_voxfield

from backbone_module import Pointnet2Backbone, Pointnet2BackbonePlane
from backbone_module_pairwise import Pointnet2BackbonePairwise
from backbone_module_decoder import Pointnet2BackboneDecoder
from backbone_module_dis import Pointnet2BackboneDis

from voting_module import VotingModule
from voting_module_point import VotingPointModule
from voting_module_corner import VotingCornerModule
from voting_module_plane import VotingPlaneModule
from mean_shift_module import MeanShiftModule
from proposal_module_hd import ProposalModule
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
        self.backbone_net_center = Pointnet2Backbone(input_feature_dim=self.input_feature_dim - 4) ### Just xyz + height
        self.backbone_net_corner = Pointnet2Backbone(input_feature_dim=self.input_feature_dim - 4) ### Just xyz + height
        #self.backbone_net_sem = Pointnet2Backbone(input_feature_dim=self.input_feature_dim - 4) ### Just xyz + height
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

        '''
        ### Semantic Segmentation
        #self.conv_sem1 = torch.nn.Conv1d(256+128+7,128,1) ##Pointfeature + input
        #self.conv_sem1 = torch.nn.Conv1d(128+512+7,128,1) ##Pointfeature + input
        self.conv_sem1 = torch.nn.Conv1d(128+128+7,128,1) ##Pointfeature + input
        self.conv_sem2 = torch.nn.Conv1d(128,(num_class),1)
        self.bn_sem1 = torch.nn.BatchNorm1d(128)
        self.dropout_sem1 = torch.nn.Dropout(0.5)

        #self.conv_sem3 = torch.nn.Conv1d(128+512+7,128,1) ##Pointfeature + input
        self.conv_sem3 = torch.nn.Conv1d(128+128+7,128,1) ##Pointfeature + input
        self.conv_sem4 = torch.nn.Conv1d(128,(num_class),1)
        self.bn_sem2 = torch.nn.BatchNorm1d(128)
        self.dropout_sem2 = torch.nn.Dropout(0.5)

        #self.conv_sem5 = torch.nn.Conv1d(128+512+7,128,1) ##Pointfeature + input
        self.conv_sem5 = torch.nn.Conv1d(128+128+7,128,1) ##Pointfeature + input
        self.conv_sem6 = torch.nn.Conv1d(128,(num_class),1)
        self.bn_sem3 = torch.nn.BatchNorm1d(128)
        self.dropout_sem3 = torch.nn.Dropout(0.5)
        '''
        # Hough voting
        #self.vgen = VotingModule(self.vote_factor, 256+128)
        #self.vgen_plane = VotingPlaneModule(self.vote_factor, 256+128)
        self.vgen = VotingModule(self.vote_factor, 128)
        self.vgen_corner = VotingCornerModule(self.vote_factor, 128)
        #self.vgen_corner = VotingPointModule(self.vote_factor, 256)
        self.vgen_plane = VotingPlaneModule(self.vote_factor, 128)
        #self.vgen = MeanShiftModule(self.vote_factor, 256)    
        self.vgen_voxel = TwoStreamNetDecoder()

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                   mean_size_arr, num_proposal, sampling, seed_feat_dim=128)
        
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

        end_points = self.backbone_net_center(inputs['point_clouds'], end_points)
        end_points = self.backbone_net_corner(inputs['point_clouds'], end_points, mode='corner')
        #end_points = self.backbone_net_point(inputs['point_clouds'], end_points, mode='sem')
        end_points = self.backbone_net_plane(inputs['plane_label'], end_points, mode='plane')
        end_points = self.backbone_net_voxel(inputs['voxel_label'], end_points)
        
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        xyz_corner = end_points['fp2_xyz'+'corner']
        features_corner = end_points['fp2_features'+'corner']

        xyz_plane = end_points['fp2_xyz'+'plane']
        features_plane = end_points['fp2_features'+'plane']
        end_points['seed_inds'+'plane'] = end_points['fp2_inds'+'plane']
        end_points['seed_xyz'+'plane'] = xyz_plane
        end_points['seed_features'+'plane'] = features_plane

        newxyz = torch.matmul(xyz, end_points['aug_rot'].float())
        newxyz = torch.stack(((newxyz[:,:,0])*(-1*end_points['aug_yz'].unsqueeze(-1).float()), (newxyz[:,:,1])*(-1*end_points['aug_xz'].unsqueeze(-1).float()), newxyz[:,:,2]), 2)

        if inputs['sunrgbd']:
            features_vox = pc_util.voxel_to_pt_feature_batch_sunrgbd(end_points['vox_latent_feature'], newxyz)
        else:
            features_vox = pc_util.voxel_to_pt_feature_batch(end_points['vox_latent_feature'], newxyz)
        features_vox = features_vox.contiguous().transpose(2,1)
    
        #xyz_sem = end_points['fp2_xyz'+'sem']
        #features_sem = end_points['fp2_features'+'sem']
        #end_points['seed_inds'+'sem'] = end_points['fp2_inds'+'sem']
        #end_points['seed_xyz'+'sem'] = xyz_sem
        #end_points['seed_features'+'sem'] = features_sem

        #features_combine_center = torch.cat((features, features_corner, features_plane), 1)
        #features_combine_corner = torch.cat((features, features_corner, features_plane), 1)
        #features_combine_plane = torch.cat((features, features_corner, features_plane), 1)
        features_combine_center = features
        features_combine_corner = features_corner
        features_combine_plane = features_plane
        
        #features_combine_point = torch.cat((features, features_plane.detach(), features_vox.detach()), 1)
        #features_combine_point = torch.cat((features, features_plane, features_vox), 1)
        #features_combine_point = torch.cat((features, features_plane), 1)
        #features_combine_sem = torch.cat((features.detach(), features_plane.detach(), features_sem), 1)
        #features_combine_plane = torch.cat((features, features_plane, features_sem.detach()), 1)
        #features_combine_plane = torch.cat((features.detach(), features_plane, features_vox.detach()), 1)
        #features_combine_plane = torch.cat((features, features_plane, features_vox), 1)
        #features_combine_plane = torch.cat((features, features_plane), 1)
        #end_points["feature_map"] = features_combine_point
        allfeat = torch.cat((newxyz, torch.cat((features, features_plane), 1).contiguous().transpose(2,1)), 2)
        if inputs['sunrgbd']:
            features_other_vox = pc_util.pt_to_voxel_feature_batch_sunrgbd(allfeat)
        else:
            features_other_vox = pc_util.pt_to_voxel_feature_batch(allfeat)
        #features_combine_vox = torch.cat((end_points['vox_latent_feature'], features_other_vox.detach()), 1)
        features_combine_vox = torch.cat((end_points['vox_latent_feature'], features_other_vox.detach()), 1)
        #features_combine_vox = end_points['vox_latent_feature']

        '''
        features_combine = torch.cat((features, features_plane), 1)
        '''
        proposal_xyz, proposal_features, proposal_cueness = self.vgen(xyz, features_combine_center)
        
        proposal_features_norm = torch.norm(proposal_features, p=2, dim=1)
        proposal_features = proposal_features.div(proposal_features_norm.unsqueeze(1))
        end_points['vote_xyz'] = proposal_xyz
        end_points['vote_features'] = proposal_features
        end_points['cueness_scores'+'center'] = proposal_cueness

        voted_xyz_corner1, voted_xyz_corner1_feature, voted_xyz_corner2, voted_xyz_corner2_feature, corner_cueness = self.vgen_corner(xyz_corner, features_combine_corner)
        
        #proposal_features_norm = torch.norm(proposal_features, p=2, dim=1)
        #proposal_features = proposal_features.div(proposal_features_norm.unsqueeze(1))
        #end_points['vote_xyz'] = proposal_xyz
        #end_points['vote_features'] = proposal_features
        
        #voted_xyz, voted_xyz_corner, center_feature, corner_feature = self.vgen_point(xyz, features_combine_point)
        #voted_xyz_corner, voted_xyz_corner_feature = self.vgen_point(xyz, features_combine_point)
        #features_norm = torch.norm(features, p=2, dim=1)
        #features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz_corner1'] = voted_xyz_corner1
        end_points['vote_xyz_corner1_feature'] = voted_xyz_corner1_feature
        end_points['vote_xyz_corner2'] = voted_xyz_corner2
        end_points['vote_xyz_corner2_feature'] = voted_xyz_corner2_feature
        end_points['vote_xyz_corner_center'] = (voted_xyz_corner1 + voted_xyz_corner2) / 2.0
        end_points['cueness_scores'+'corner'] = corner_cueness
        
        voted_xyz_corner = torch.cat((voted_xyz_corner1, voted_xyz_corner2), 1)        
        voted_xyz_corner_feature = torch.cat((voted_xyz_corner1_feature, voted_xyz_corner2_feature), 2)
        
        end_points = self.vgen_voxel(features_combine_vox, end_points, inputs)
        
        seed_plane = torch.gather(inputs['plane_label'][:,:,4:], 1, end_points['seed_inds'].long().unsqueeze(-1).repeat(1,1,4))

        xyz_plane = torch.cat((xyz, seed_plane), -1)
        
        #net_upper, net_lower, net_left, net_right, net_front, net_back, plane_feature = self.vgen_plane(xyz_plane, features_combine_plane)
        plane_xyz, plane_features, end_points = self.vgen_plane(xyz_plane, features_combine_plane, end_points)

        end_points = self.pnet(proposal_xyz, proposal_features, voted_xyz_corner, voted_xyz_corner_feature, plane_xyz, plane_features, end_points)
        ### Semantic Segmentation
        #features_combine_sem_point = torch.cat((features_sem, center_feature), 1)
        #features_combine_sem_corner = torch.cat((features_sem, corner_feature), 1)#corner_feature
        #features_combine_sem_plane = torch.cat((features_sem, plane_feature), 1)#plane_feature
        '''
        features_for_sem = torch.cat((features_combine_sem_point, xyz_plane.transpose(2,1).contiguous()), 1)
        net_sem = F.relu(self.dropout_sem1(self.bn_sem1(self.conv_sem1(features_for_sem))))
        net_sem = self.conv_sem2(net_sem)
        end_points["pred_sem_class1"] = net_sem
        features_for_sem = torch.cat((features_combine_sem_corner, xyz_plane.transpose(2,1).contiguous()), 1)
        net_sem = F.relu(self.dropout_sem2(self.bn_sem2(self.conv_sem3(features_for_sem))))
        net_sem = self.conv_sem4(net_sem)
        end_points["pred_sem_class2"] = net_sem
        features_for_sem = torch.cat((features_combine_sem_plane, xyz_plane.transpose(2,1).contiguous()), 1)
        net_sem = F.relu(self.dropout_sem3(self.bn_sem3(self.conv_sem5(features_for_sem))))
        net_sem = self.conv_sem6(net_sem)
        end_points["pred_sem_class3"] = net_sem

        end_points["pred_sem_class"] = torch.stack((torch.argmax(end_points["pred_sem_class1"], 1), torch.argmax(end_points["pred_sem_class2"], 1), torch.argmax(end_points["pred_sem_class3"], 1)), 1)
        end_points["pred_sem_class_top3"] = torch.stack((torch.topk(end_points["pred_sem_class1"], 3, dim=1)[1], torch.topk(end_points["pred_sem_class2"], 3, dim=1)[1], torch.topk(end_points["pred_sem_class3"], 3, dim=1)[1]), 1)
        '''
        """
        if end_points['use_support']:
            end_points = self.pnet(xyz_support, features_support, end_points, mode='_support')
            end_points = self.pnet(xyz_bsupport, features_bsupport, end_points, mode='_bsupport')
            #xyz = torch.cat((xyz, xyz_support, xyz_bsupport), 1)
            #features = torch.cat((features, features_support, features_bsupport), 2)
        """
        #import pdb;pdb.set_trace()
        return end_points

class OBJNet(nn.Module):
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
        self.weight = torch.nn.Linear(10,1) ###2 + 2 + 6

    def forward(self, inputs, end_points, batch_data_label, mode=""):
        ### 3D Field
        center_cue = torch.squeeze(end_points["vote_xyz_center"])
        #center_cue_field = cue_to_voxfield.get_3d_field(center_cue)
        #center_cue_field = cue_to_voxfield.trilinear_interpolation_window(center_cue, window_size=5, vs_x=0.1, vs_y=0.1, vs_z=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84, zmin=-0.2, zmax=2.68)
        center_cue_field = cue_to_voxfield.trilinear_interpolation(center_cue, vs_x=0.1, vs_y=0.1, vs_z=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84, zmin=-0.2, zmax=2.68)
        corner_cue = torch.squeeze(end_points["vote_xyz_corner"])
        #corner_cue_field = cue_to_voxfield.trilinear_interpolation_window(corner_cue, window_size=5, vs_x=0.1, vs_y=0.1, vs_z=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84, zmin=-0.2, zmax=2.68)
        corner_cue_field = cue_to_voxfield.trilinear_interpolation(corner_cue, vs_x=0.1, vs_y=0.1, vs_z=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84, zmin=-0.2, zmax=2.68)
        vox_center_cue = end_points["vox_pred1"]
        vox_center_cue_field = torch.squeeze(vox_center_cue)
        #vox_center_cue_field = torch.squeeze(torch.nn.functional.interpolate(vox_center_cue, scale_factor=(0.6,0.6,0.6), mode='trilinear'))
        vox_corner_cue = end_points["vox_pred2"]
        #vox_corner_cue_field = torch.squeeze(torch.nn.functional.interpolate(vox_corner_cue, scale_factor=(0.6,0.6,0.6), mode='trilinear'))
        vox_corner_cue_field = torch.squeeze(vox_corner_cue)

        ### 1D Field
        plane_z0_cue = torch.squeeze(end_points['z_off0'])
        #plane_z0_cue_field = cue_to_voxfield.linear_interpolation_window(plane_z0_cue, window_size=3, vs_x=0.1,xmin=-3.84, xmax=3.84)
        plane_z0_cue_field = cue_to_voxfield.linear_interpolation(plane_z0_cue, vs_x=0.1,xmin=-3.84, xmax=3.84)
        plane_z1_cue = end_points['z_off1']
        #plane_z1_cue_field = cue_to_voxfield.linear_interpolation_window(plane_z1_cue, window_size=3, vs_x=0.1,xmin=-3.84, xmax=3.84)
        plane_z1_cue_field = cue_to_voxfield.linear_interpolation(plane_z1_cue, vs_x=0.1,xmin=-3.84, xmax=3.84)

        ### 2D Field
        plane_x0_cue = torch.stack([torch.squeeze(torch.argmax(end_points['x_angle'], 1)).float() + torch.squeeze(end_points['x_res']), torch.squeeze(end_points['x_off0'])], 1)
        #plane_x0_cue_field = cue_to_voxfield.bilinear_interpolation_window(plane_x0_cue, window_size=5, vs_x=12, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84)
        plane_x0_cue_field = cue_to_voxfield.bilinear_interpolation(plane_x0_cue, vs_x=1, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84)
        plane_x1_cue = torch.stack([torch.squeeze(torch.argmax(end_points['x_angle'], 1)).float() + torch.squeeze(end_points['x_res']), torch.squeeze(end_points['x_off1'])], 1)
        #plane_x1_cue_field = cue_to_voxfield.bilinear_interpolation_window(plane_x1_cue, window_size=5, vs_x=12, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84)
        plane_x1_cue_field = cue_to_voxfield.bilinear_interpolation(plane_x1_cue, vs_x=1, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84)
        plane_y0_cue = torch.stack([torch.squeeze(torch.argmax(end_points['y_angle'], 1)).float() + torch.squeeze(end_points['y_res']), torch.squeeze(end_points['y_off0'])], 1)
        #plane_y0_cue_field = cue_to_voxfield.bilinear_interpolation_window(plane_y0_cue, window_size=5, vs_x=12, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84)
        plane_y0_cue_field = cue_to_voxfield.bilinear_interpolation(plane_y0_cue, vs_x=1, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84)
        plane_y1_cue = torch.stack([torch.squeeze(torch.argmax(end_points['y_angle'], 1)).float() + torch.squeeze(end_points['y_res']), torch.squeeze(end_points['y_off1'])], 1)
        #plane_y1_cue_field = cue_to_voxfield.bilinear_interpolation_window(plane_y1_cue, window_size=5, vs_x=12, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84)
        plane_y1_cue_field = cue_to_voxfield.bilinear_interpolation(plane_y1_cue, vs_x=1, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84)

        ###
        gt_bbox = torch.squeeze(inputs['gt_bboxes'])
        center, corner, planex0, planex1, planey0, planey1, planez0, planez1 = cue_to_voxfield.get_oriented_cues_batch_torch(gt_bbox, end_points)
        corner = corner.contiguous().view(-1,3)
        pert_bbox = torch.squeeze(inputs['pert_bboxes'])
        pert_center, pert_corner, pert_planex0, pert_planex1, pert_planey0, pert_planey1, pert_planez0, pert_planez1 = cue_to_voxfield.get_oriented_cues_batch_torch(pert_bbox, end_points)
        pert_corner = pert_corner.contiguous().view(-1,3)
        '''
        print (np.unique(batch_data_label['plane_votes_z0'].cpu().numpy()))
        print (np.unique(planez0[:].cpu().numpy()))
        print (np.unique(batch_data_label['plane_votes_z1'].cpu().numpy()))
        print (np.unique(planez1[:].cpu().numpy()))
        print (np.unique(batch_data_label['plane_votes_x'].cpu().numpy()))
        print (np.unique(planex0[:,0].cpu().numpy()))
        print (np.unique(batch_data_label['plane_votes_x0'].cpu().numpy()))
        print (np.unique(planex0[:,1].cpu().numpy()))
        print (np.unique(batch_data_label['plane_votes_x1'].cpu().numpy()))
        print (np.unique(planex1[:,1].cpu().numpy()))
        print (np.unique(batch_data_label['plane_votes_y'].cpu().numpy()))
        print (np.unique(planey0[:,0].cpu().numpy()))
        print (np.unique(batch_data_label['plane_votes_y0'].cpu().numpy()))
        print (np.unique(planey0[:,1].cpu().numpy()))
        print (np.unique(batch_data_label['plane_votes_y1'].cpu().numpy()))
        print (np.unique(planey1[:,1].cpu().numpy()))
        '''
        pt_center = cue_to_voxfield.get_center_or_corner_potential_function(center, center_cue_field.detach())
        #pt_center = torch.sum(pt_center*batch_data_label['box_label_mask'])
        pt_center = torch.sum(pt_center[:batch_data_label['num_instance'][0]])
        pt_corner = cue_to_voxfield.get_center_or_corner_potential_function(corner, corner_cue_field.detach())
        #corner_expand = torch.stack([batch_data_label['box_label_mask']]*8)
        pt_corner = torch.sum(pt_corner)
        vox_center = cue_to_voxfield.get_center_or_corner_potential_function(center, vox_center_cue_field.detach())
        vox_center = torch.sum(vox_center[:batch_data_label['num_instance'][0]])
        vox_corner = cue_to_voxfield.get_center_or_corner_potential_function(corner, vox_corner_cue_field.detach())
        vox_corner = torch.sum(vox_corner)
        pt_planex0 = cue_to_voxfield.get_xy_plane_potential_function(planex0, plane_x0_cue_field.detach())
        pt_planex0 = torch.sum(pt_planex0[:batch_data_label['num_instance'][0]])
        pt_planex1 = cue_to_voxfield.get_xy_plane_potential_function(planex1, plane_x1_cue_field.detach())
        pt_planex1 = torch.sum(pt_planex1[:batch_data_label['num_instance'][0]])
        pt_planey0 = cue_to_voxfield.get_xy_plane_potential_function(planey0, plane_y0_cue_field.detach())
        pt_planey0 = torch.sum(pt_planey0[:batch_data_label['num_instance'][0]])
        pt_planey1 = cue_to_voxfield.get_xy_plane_potential_function(planey1, plane_y1_cue_field.detach())
        pt_planey1 = torch.sum(pt_planey1[:batch_data_label['num_instance'][0]])
        pt_planez0 = cue_to_voxfield.get_z_plane_potential_function(torch.squeeze(planez0), plane_z0_cue_field.detach())
        pt_planez0 = torch.sum(pt_planez0[:batch_data_label['num_instance'][0]])
        pt_planez1 = cue_to_voxfield.get_z_plane_potential_function(torch.squeeze(planez1), plane_z1_cue_field.detach())
        pt_planez1 = torch.sum(pt_planez1[:batch_data_label['num_instance'][0]])

        pert_pt_center = cue_to_voxfield.get_center_or_corner_potential_function(center, center_cue_field.detach())
        #pert_pt_center = torch.sum(pert_pt_center*batch_data_label['box_label_mask'])
        pert_pt_center = torch.sum(pert_pt_center[:batch_data_label['num_instance'][0]])
        pert_pt_corner = cue_to_voxfield.get_center_or_corner_potential_function(corner, corner_cue_field.detach())
        pert_pt_corner = torch.sum(pert_pt_corner)
        pert_vox_center = cue_to_voxfield.get_center_or_corner_potential_function(center, vox_center_cue_field.detach())
        pert_vox_center = torch.sum(pert_vox_center[:batch_data_label['num_instance'][0]])
        pert_vox_corner = cue_to_voxfield.get_center_or_corner_potential_function(corner, vox_corner_cue_field.detach())
        pert_vox_corner = torch.sum(pert_vox_corner)
        pert_pt_planex0 = cue_to_voxfield.get_xy_plane_potential_function(planex0, plane_x0_cue_field.detach())
        pert_pt_planex0 = torch.sum(pert_pt_planex0[:batch_data_label['num_instance'][0]])
        pert_pt_planex1 = cue_to_voxfield.get_xy_plane_potential_function(planex1, plane_x1_cue_field.detach())
        pert_pt_planex1 = torch.sum(pert_pt_planex1[:batch_data_label['num_instance'][0]])
        pert_pt_planey0 = cue_to_voxfield.get_xy_plane_potential_function(planey0, plane_y0_cue_field.detach())
        pert_pt_planey0 = torch.sum(pert_pt_planey0[:batch_data_label['num_instance'][0]])
        pert_pt_planey1 = cue_to_voxfield.get_xy_plane_potential_function(planey1, plane_y1_cue_field.detach())
        pert_pt_planey1 = torch.sum(pert_pt_planey1[:batch_data_label['num_instance'][0]])
        pert_pt_planez0 = cue_to_voxfield.get_z_plane_potential_function(torch.squeeze(planez0), plane_z0_cue_field.detach())
        pert_pt_planez0 = torch.sum(pert_pt_planez0[:batch_data_label['num_instance'][0]])
        pert_pt_planez1 = cue_to_voxfield.get_z_plane_potential_function(torch.squeeze(planez1), plane_z1_cue_field.detach())
        pert_pt_planez1 = torch.sum(pert_pt_planez1[:batch_data_label['num_instance'][0]])

        potential_vec = torch.stack((pt_center, pt_corner, vox_center, vox_corner, pt_planex0, pt_planex1, pt_planey0, pt_planey1, pt_planez0, pt_planez1))
        #potential_vec = torch.stack((pt_center))
        #potential_vec = pt_center.unsqueeze(0)
        pert_potential_vec = torch.stack((pert_pt_center, pert_pt_corner, pert_vox_center, pert_vox_corner, pert_pt_planex0, pert_pt_planex1, pert_pt_planey0, pert_pt_planey1, pert_pt_planez0, pert_pt_planez1))
        #pert_potential_vec = torch.stack((pert_pt_center))
        #pert_potential_vec = pert_pt_center.unsqueeze(0)
        
        end_points['gt_potential'] = self.weight(potential_vec)
        end_points['pert_potential'] = self.weight(pert_potential_vec)
        return end_points
        #center_vox = pc_util.trilinear_interpolation(voted_xyz)
        #import pdb;pdb.set_trace()

class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(min=0.0,max=1.01)
            module.weight.data=w
        
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
