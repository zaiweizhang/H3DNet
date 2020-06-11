# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from model_util_sunrgbd import SunrgbdDatasetConfig
from pointnet2_modules import PointnetSAModuleVotes
from pointnet2_modules import PointnetSAModuleMatch
import pointnet2_utils
from nn_distance import nn_distance
from box_util import get_surface_line_points_batch_pytorch


UPPER_THRESH = 100.0
SURFACE_THRESH = 0.5
MATCH_THRESH = 0.5
LINE_THRESH = 0.5

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3

FAR_MATCH_THRESHOLD = 0.6
NEAR_MATCH_THRESHOLD = 0.3

FAR_COMB_THRESHOLD = 0.4
NEAR_COMB_THRESHOLD = 0.3

MASK_SURFACE_THRESHOLD = 0.3
LABEL_SURFACE_THRESHOLD = 0.3
MASK_LINE_THRESHOLD = 0.3
LABEL_LINE_THRESHOLD = 0.3

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr, mode=''):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    if mode == 'opt':
        start = 2
        objectness_scores = net_transposed[:,:,0:2]
        end_points['objectness_scores'+mode] = objectness_scores
    else:
        start = 2
        objectness_scores = net_transposed[:,:,0:2]
        end_points['objectness_scores'+mode] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'+mode] # (batch_size, num_proposal, 3)
    end_points['centerres'+mode] = net_transposed[:,:,start:start+3]
    center = base_xyz + net_transposed[:,:,start:start+3] # (batch_size, num_proposal, 3)
    end_points['center'+mode] = center

    heading_scores = net_transposed[:,:,start+3:start+3+num_heading_bin]
    heading_residuals_normalized = net_transposed[:,:,start+3+num_heading_bin:start+3+num_heading_bin*2]
    end_points['heading_scores'+mode] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'+mode] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'+mode] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    if mode == 'opt':
        size_scores = net_transposed[:,:,start+3+num_heading_bin*2:start+3+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net_transposed[:,:,start+3+num_heading_bin*2+num_size_cluster:start+3+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
        end_points['size_scores'+mode] = size_scores
        end_points['size_residuals_normalized'+mode] = size_residuals_normalized
        end_points['size_residuals'+mode] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    else:
        size_scores = net_transposed[:,:,start+3+num_heading_bin*2:start+3+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net_transposed[:,:,start+3+num_heading_bin*2+num_size_cluster:start+3+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
        end_points['size_scores'+mode] = size_scores
        end_points['size_residuals_normalized'+mode] = size_residuals_normalized
        end_points['size_residuals'+mode] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    if mode == 'opt':
        sem_cls_scores = net_transposed[:,:,start+3+num_heading_bin*2:start+3+num_heading_bin*2+num_size_cluster] # Bxnum_proposalx10
        #sem_cls_scores = net_transposed[:,:,start+3+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
        end_points['sem_cls_scores'+mode] = sem_cls_scores
    else:
        sem_cls_scores = net_transposed[:,:,start+3+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
        end_points['sem_cls_scores'+mode] = sem_cls_scores
    if mode == 'center':
        return end_points['center'+mode], end_points['size_residuals'+mode], end_points['size_scores'+mode], end_points
    else:
        return end_points

class ProposalModuleRefine(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256, with_angle=False):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.num_proposal_comb = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.with_angle = with_angle
        self.vote_aggregation_corner = []
        self.vote_aggregation_plane = []

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        ### surface center matching
        self.match_surface_center = PointnetSAModuleMatch( 
                npoint=self.num_proposal*6,
                radius=0.5,
                nsample=32,
                mlp=[128+6, 128, 64, 32],
                use_xyz=True,
                normalize_xyz=True
            )

        ### line center matching
        self.match_line_center = PointnetSAModuleMatch( 
                npoint=self.num_proposal*12,
                radius=0.5,
                nsample=32,
                mlp=[128+12, 128, 64, 32],
                use_xyz=True,
                normalize_xyz=True
            )
        
        # Initial object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        # Compute the matching scores
        self.conv_match1 = torch.nn.Conv1d(32,32,1)
        self.conv_match2 = torch.nn.Conv1d(32,2,1)
        self.bn_match1 = torch.nn.BatchNorm1d(32)

        # Compute the semantic matching scores
        self.conv_match_sem1 = torch.nn.Conv1d(32,32,1)
        self.conv_match_sem2 = torch.nn.Conv1d(32,2,1)
        self.bn_match_sem1 = torch.nn.BatchNorm1d(32)

        # Surface feature aggregation
        self.conv_surface1 = torch.nn.Conv1d(32,32,1)
        self.conv_surface2 = torch.nn.Conv1d(32,32,1)
        self.bn_surface1 = torch.nn.BatchNorm1d(32)
        self.bn_surface2 = torch.nn.BatchNorm1d(32)

        # Line feature aggregation
        self.conv_line1 = torch.nn.Conv1d(32,32,1)
        self.conv_line2 = torch.nn.Conv1d(32,32,1)
        self.bn_line1 = torch.nn.BatchNorm1d(32)
        self.bn_line2 = torch.nn.BatchNorm1d(32)

        ### Final object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv_refine1 = torch.nn.Conv1d(192*3,128,1)
        self.conv_refine2 = torch.nn.Conv1d(128,128,1)
        self.conv_refine3 = torch.nn.Conv1d(128,128,1)
        self.conv_refine4 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)

        self.bn_refine1 = torch.nn.BatchNorm1d(128)
        self.bn_refine2 = torch.nn.BatchNorm1d(128)
        self.bn_refine3 = torch.nn.BatchNorm1d(128)
        
        self.softmax_normal = torch.nn.Softmax(dim=1)
        
    def forward(self, xyz, features, center_z, z_feature, center_xy, xy_feature, center_line, line_feature, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            original_features = features
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds            
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyzcenter'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
        batch_size = xyz.shape[0]
        object_proposal = xyz.shape[1]
        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net)))
        original_feature = features.contiguous()
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        final_feature = net
        center_vote, size_vote, sizescore_vote, end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, mode='center')
    
        ### Create surface center here
        ### Extract surface points and features here
        ind_normal_z = self.softmax_normal(end_points["pred_flag_z"])
        end_points["pred_z_ind"] = (ind_normal_z[:,1,:] > SURFACE_THRESH).detach().float()
        z_sel = (ind_normal_z[:,1,:] <= SURFACE_THRESH).detach().float()
        offset = torch.ones_like(center_z) * UPPER_THRESH
        z_center = center_z + offset*z_sel.unsqueeze(-1)
        z_sem = end_points["sem_cls_scores_z"]

        ind_normal_xy = self.softmax_normal(end_points["pred_flag_xy"])
        end_points["pred_xy_ind"] = (ind_normal_xy[:,1,:] > SURFACE_THRESH).detach().float()
        xy_sel = (ind_normal_xy[:,1,:] <= SURFACE_THRESH).detach().float()
        offset = torch.ones_like(center_xy) * UPPER_THRESH
        xy_center = center_xy + offset*xy_sel.unsqueeze(-1)
        xy_sem = end_points["sem_cls_scores_xy"]
        
        surface_center_pred = torch.cat((z_center, xy_center), dim=1)
        end_points['surface_center_pred'] = surface_center_pred
        end_points['surface_sem_pred'] = torch.cat((z_sem, xy_sem), dim=1)
        surface_center_feature_pred = torch.cat((z_feature, xy_feature), dim=2)
        surface_center_feature_pred = torch.cat((torch.zeros((batch_size, 6, surface_center_feature_pred.shape[2])).cuda(), surface_center_feature_pred), dim=1)

        ### Extract line points and features here
        ind_normal_line = self.softmax_normal(end_points["pred_flag_line"])
        end_points["pred_line_ind"] = (ind_normal_line[:,1,:] > LINE_THRESH).detach().float()
        line_sel = (ind_normal_line[:,1,:] <= SURFACE_THRESH).detach().float()
        offset = torch.ones_like(center_line) * UPPER_THRESH
        line_center = center_line + offset*line_sel.unsqueeze(-1)
        end_points['line_center_pred'] = line_center
        end_points['line_sem_pred'] = end_points["sem_cls_scores_line"]

        end_points['aggregated_vote_xyzopt'] = xyz
        
        ### Extract the object center here
        obj_center = center_vote.contiguous()
        size_residual = size_vote.contiguous()
        pred_size_class = torch.argmax(sizescore_vote.contiguous(), -1)
        pred_size_residual = torch.gather(size_vote.contiguous(), 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3))
        mean_size_class_batched = torch.ones_like(size_residual) * torch.from_numpy(self.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        pred_size_avg = torch.gather(mean_size_class_batched, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3))
        obj_size = (pred_size_avg.squeeze(2) + pred_size_residual.squeeze(2)).detach()

        pred_heading_class = torch.argmax(end_points['heading_scores'+'center'].detach(), -1) # B,num_proposal
        pred_heading_residual = torch.gather(end_points['heading_residuals'+'center'].detach(), 2,
                                                 pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_residual.squeeze_(2)

        # if self.num_class == 18:
        if not self.with_angle:
            pred_heading = torch.zeros_like(pred_heading_class)
        else:
            config = SunrgbdDatasetConfig()
            pred_heading = pred_heading_class.float()*(2*np.pi/float(config.num_heading_bin)) + pred_heading_residual 
            
        obj_surface_center, obj_line_center = get_surface_line_points_batch_pytorch(obj_size, pred_heading, obj_center)
        obj_surface_feature = original_feature.repeat(1,1,6)
        end_points['surface_center_object'] = obj_surface_center
        # Add an indicator for different surfaces
        obj_upper_indicator = torch.zeros((batch_size, object_proposal, 6)).cuda()
        obj_upper_indicator[:,:,0] = 1
        obj_lower_indicator = torch.zeros((batch_size, object_proposal, 6)).cuda()
        obj_lower_indicator[:,:,1] = 1
        obj_front_indicator = torch.zeros((batch_size, object_proposal, 6)).cuda()
        obj_front_indicator[:,:,2] = 1
        obj_back_indicator = torch.zeros((batch_size, object_proposal, 6)).cuda()
        obj_back_indicator[:,:,3] = 1
        obj_left_indicator = torch.zeros((batch_size, object_proposal, 6)).cuda()
        obj_left_indicator[:,:,4] = 1
        obj_right_indicator = torch.zeros((batch_size, object_proposal, 6)).cuda()
        obj_right_indicator[:,:,5] = 1
        obj_surface_indicator = torch.cat((obj_upper_indicator, obj_lower_indicator, obj_front_indicator, obj_back_indicator, obj_left_indicator, obj_right_indicator), dim=1).transpose(2,1).contiguous()
        obj_surface_feature = torch.cat((obj_surface_indicator, obj_surface_feature), dim=1)
        
        obj_line_feature = original_feature.repeat(1,1,12)
        end_points['line_center_object'] = obj_line_center
        # Add an indicator for different lines
        obj_line_indicator0 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator0[:,0,:] = 1
        obj_line_indicator1 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator1[:,1,:] = 1
        obj_line_indicator2 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator2[:,2,:] = 1
        obj_line_indicator3 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator3[:,3,:] = 1
        
        obj_line_indicator4 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator4[:,4,:] = 1
        obj_line_indicator5 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator5[:,5,:] = 1
        obj_line_indicator6 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator6[:,6,:] = 1
        obj_line_indicator7 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator7[:,7,:] = 1

        obj_line_indicator8 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator8[:,8,:] = 1
        obj_line_indicator9 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator9[:,9,:] = 1
        obj_line_indicator10 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator10[:,10,:] = 1
        obj_line_indicator11 = torch.zeros((batch_size, 12, object_proposal)).cuda()
        obj_line_indicator11[:,11,:] = 1

        obj_line_indicator = torch.cat((obj_line_indicator0, obj_line_indicator1, obj_line_indicator2, obj_line_indicator3, obj_line_indicator4, obj_line_indicator5, obj_line_indicator6, obj_line_indicator7, obj_line_indicator8, obj_line_indicator9, obj_line_indicator10, obj_line_indicator11), dim=2)
        obj_line_feature = torch.cat((obj_line_indicator, obj_line_feature), dim=1)
        
        surface_xyz, surface_features, _ = self.match_surface_center(torch.cat((obj_surface_center, surface_center_pred), dim=1), torch.cat((obj_surface_feature, surface_center_feature_pred), dim=2))
        line_feature = torch.cat((torch.zeros((batch_size, 12, line_feature.shape[2])).cuda(), line_feature), dim=1)
        line_xyz, line_features, _ = self.match_line_center(torch.cat((obj_line_center, line_center), dim=1), torch.cat((obj_line_feature, line_feature), dim=2))

        combine_features = torch.cat((surface_features.contiguous(), line_features.contiguous()), dim=2)

        match_features = F.relu(self.bn_match1(self.conv_match1(combine_features)))
        match_score = self.conv_match2(match_features)
        end_points["match_scores"] = match_score.transpose(2,1).contiguous()

        match_features_sem = F.relu(self.bn_match_sem1(self.conv_match_sem1(combine_features)))
        match_score_sem = self.conv_match_sem2(match_features_sem)
        end_points["match_scores_sem"] = match_score_sem.transpose(2,1).contiguous()

        surface_features = F.relu(self.bn_surface1(self.conv_surface1(surface_features)))
        surface_features = F.relu(self.bn_surface2(self.conv_surface2(surface_features)))

        line_features = F.relu(self.bn_line1(self.conv_line1(line_features)))
        line_features = F.relu(self.bn_line2(self.conv_line2(line_features)))
        
        surface_features = surface_features.view(batch_size, -1, 6, object_proposal).contiguous()
        line_features = line_features.view(batch_size, -1, 12, object_proposal).contiguous()

        # Combine all surface and line features
        surface_pool_feature = surface_features.view(batch_size, -1, object_proposal).contiguous()
        line_pool_feature = line_features.view(batch_size, -1, object_proposal).contiguous()
        
        combine_feature = torch.cat((surface_pool_feature, line_pool_feature), dim=1)

        net = F.relu(self.bn_refine1(self.conv_refine1(combine_feature)))
        net += original_feature
        net = F.relu(self.bn_refine2(self.conv_refine2(net)))
        net = F.relu(self.bn_refine3(self.conv_refine3(net)))
        net = self.conv_refine4(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, mode='opt')
        return end_points

