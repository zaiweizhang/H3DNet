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
from pointnet2_modules import PointnetSAModuleVotes
from pointnet2_modules import PointnetSAModuleMatch
from pointnet2_modules import PointnetSAModulePairwise
import pointnet2_utils

UPPER_THRESH = 10.0
SURFACE_THRESH = 0.5
LINE_THRESH = 0.5

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr, mode=''):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    if mode == 'opt':
        start = 0
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

    if False:#mode == 'corner':
        import pdb;pdb.set_trace()
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
        
    sem_cls_scores = net_transposed[:,:,start+3+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points['sem_cls_scores'+mode] = sem_cls_scores
    return end_points


class ProposalModuleRefine(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.num_proposal_comb = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
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
                radius=0.3,
                nsample=32,
                mlp=[128, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        ### line center matching
        self.match_line_center = PointnetSAModuleMatch( 
                npoint=self.num_proposal*12,
                radius=0.3,
                nsample=32,
                mlp=[128, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.conv_match1 = torch.nn.Conv1d(128,64,1)
        self.conv_match2 = torch.nn.Conv1d(64,2,1)
        self.bn_match1 = torch.nn.BatchNorm1d(64)

        self.conv_surface1 = torch.nn.Conv1d(128,128,1)
        self.conv_surface2 = torch.nn.Conv1d(128,128,1)
        self.bn_surface1 = torch.nn.BatchNorm1d(128)
        self.bn_surface2 = torch.nn.BatchNorm1d(128)

        self.conv_line1 = torch.nn.Conv1d(128,128,1)
        self.conv_line2 = torch.nn.Conv1d(128,128,1)
        self.bn_line1 = torch.nn.BatchNorm1d(128)
        self.bn_line2 = torch.nn.BatchNorm1d(128)
        
        self.conv_refine1 = torch.nn.Conv1d(256,128,1)
        self.conv_refine2 = torch.nn.Conv1d(128,128,1)
        #self.conv_refine3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.conv_refine3 = torch.nn.Conv1d(128,3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn_refine1 = torch.nn.BatchNorm1d(128)
        self.bn_refine2 = torch.nn.BatchNorm1d(128)
        
        self.softmax_normal = torch.nn.Softmax(dim=1)
        
    def forward(self, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            xyz = end_points['vote_xyz']
            features = end_points['vote_features']
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
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, mode='center')

        ### Create surface center here
        ### Extract surface points and features here
        ind_normal_z = self.softmax_normal(end_points["pred_sem_class_z"])
        z_sel = (ind_normal_z[:,1,:] <= SURFACE_THRESH).detach().float()
        z_center = end_points["center_z"]
        offset = torch.ones_like(z_center) * UPPER_THRESH
        z_center += offset*z_sel.unsqueeze(-1)
        z_feature = end_points['aggregated_feature_z']

        ind_normal_xy = self.softmax_normal(end_points["pred_sem_class_xy"])
        xy_sel = (ind_normal_xy[:,1,:] <= SURFACE_THRESH).detach().float()
        xy_center = end_points["center_xy"]
        offset = torch.ones_like(xy_center) * UPPER_THRESH
        xy_center += offset*xy_sel.unsqueeze(-1)
        xy_feature = end_points['aggregated_feature_xy']

        surface_center_pred = torch.cat((z_center, xy_center), dim=1)
        end_points['surface_center_pred'] = surface_center_pred
        surface_center_feature_pred = torch.cat((z_feature, xy_feature), dim=2)

        ### Extract line points and features here
        ind_normal_line = self.softmax_normal(end_points["pred_sem_class_line"])
        line_sel = (ind_normal_line[:,1,:] <= SURFACE_THRESH).detach().float()
        line_center = end_points["center_line"]
        offset = torch.ones_like(line_center) * UPPER_THRESH
        line_center += offset*line_sel.unsqueeze(-1)
        line_feature = end_points['aggregated_feature_line']
        end_points['line_center_pred'] = line_center
        
        ### Extract the object center here
        obj_center = end_points['center'+'center']
        end_points['aggregated_vote_xyzopt'] = obj_center
        size_residual = end_points['size_residuals'+'center']
        pred_size_class = torch.argmax(end_points['size_scores'+'center'], -1)
        pred_size_residual = torch.gather(end_points['size_residuals'+'center'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3))
        mean_size_class_batched = torch.ones_like(size_residual) * torch.from_numpy(self.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        pred_size_avg = torch.gather(mean_size_class_batched, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3))
        obj_size = (pred_size_avg.squeeze(2) + pred_size_residual.squeeze(2)).detach()
        ### Get the object surface center here
        offset = torch.zeros_like(obj_size)
        offset[:,:,2] = obj_size[:,:,2] / 2.0
        obj_upper_surface_center = obj_center + offset
        obj_lower_surface_center = obj_center - offset
        offset = torch.zeros_like(obj_size)
        offset[:,:,1] = obj_size[:,:,1] / 2.0
        obj_front_surface_center = obj_center + offset
        obj_back_surface_center = obj_center - offset
        offset = torch.zeros_like(obj_size)
        offset[:,:,0] = obj_size[:,:,0] / 2.0
        obj_left_surface_center = obj_center + offset
        obj_right_surface_center = obj_center - offset
        obj_surface_center = torch.cat((obj_upper_surface_center, obj_lower_surface_center, obj_front_surface_center, obj_back_surface_center, obj_left_surface_center, obj_right_surface_center), dim=1)
        obj_surface_feature = features.repeat(1,1,6)
        end_points['surface_center_object'] = obj_surface_center

        ## Get the object line center here
        offset_x = torch.zeros_like(obj_size)
        offset_y = torch.zeros_like(obj_size)
        offset_z = torch.zeros_like(obj_size)
        offset_x[:,:,0] = obj_size[:,:,0] / 2.0
        offset_y[:,:,1] = obj_size[:,:,1] / 2.0
        offset_z[:,:,2] = obj_size[:,:,2] / 2.0
        obj_line_center_0 = obj_center + offset_z + offset_x
        obj_line_center_1 = obj_center + offset_z - offset_x
        obj_line_center_2 = obj_center + offset_z + offset_y
        obj_line_center_3 = obj_center + offset_z - offset_y
        
        obj_line_center_4 = obj_center - offset_z + offset_x
        obj_line_center_5 = obj_center - offset_z - offset_x
        obj_line_center_6 = obj_center - offset_z + offset_y
        obj_line_center_7 = obj_center - offset_z - offset_y

        obj_line_center_8 = obj_center + offset_x + offset_y
        obj_line_center_9 = obj_center + offset_x - offset_y
        obj_line_center_10 = obj_center - offset_x + offset_y
        obj_line_center_11 = obj_center - offset_x - offset_y

        obj_line_center = torch.cat((obj_line_center_0, obj_line_center_1, obj_line_center_2, obj_line_center_3, obj_line_center_4, obj_line_center_5, obj_line_center_6, obj_line_center_7, obj_line_center_8, obj_line_center_9, obj_line_center_10, obj_line_center_11), dim=1)
        obj_line_feature = features.repeat(1,1,12)
        end_points['line_center_object'] = obj_line_center
        
        surface_xyz, surface_features, _ = self.match_surface_center(torch.cat((obj_surface_center, surface_center_pred), dim=1), torch.cat((obj_surface_feature, surface_center_feature_pred), dim=2))

        line_xyz, line_features, _ = self.match_line_center(torch.cat((obj_line_center, line_center), dim=1), torch.cat((obj_line_feature, line_feature), dim=2))

        combine_features = torch.cat((surface_features, line_features), dim=2)
        
        match_features = F.relu(self.bn_match1(self.conv_match1(combine_features)))
        match_score = self.conv_match2(match_features)
        end_points["match_scores"] = match_score.transpose(2,1).contiguous()
        match_score = match_score.view(batch_size, -1, object_proposal, 12+6).contiguous()
        _, inds_obj = torch.max(match_score[:,1,:,:], -1)
        end_points['objectness_scores'+'opt'] = torch.gather(match_score, -1, inds_obj.unsqueeze(-1).repeat(1,1,2).transpose(2,1).unsqueeze(-1)).squeeze(-1).transpose(2,1).contiguous()
        
        surface_features = F.relu(self.bn_surface1(self.conv_surface1(surface_features)))
        surface_features = F.relu(self.bn_surface2(self.conv_surface2(surface_features))).view(batch_size, -1, 6, object_proposal).contiguous()

        line_features = F.relu(self.bn_line1(self.conv_line1(line_features)))
        line_features = F.relu(self.bn_line2(self.conv_line2(line_features))).view(batch_size, -1, 12, object_proposal).contiguous()
        
        surface_pool_feature = F.max_pool2d(surface_features, (6,1), stride=1).squeeze(2)
        line_pool_feature = F.max_pool2d(line_features, (12,1), stride=1).squeeze(2)

        combine_feature = torch.cat((surface_pool_feature, line_pool_feature), dim=1)

        net = F.relu(self.bn_refine1(self.conv_refine1(combine_feature))) 
        net = F.relu(self.bn_refine2(self.conv_refine2(net))) 
        net = self.conv_refine3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, mode='opt')
        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
