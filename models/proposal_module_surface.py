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
import pointnet2_utils

def decode_scores(net, end_points, num_class, mode=''):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    #objectness_scores = net_transposed[:,:,0:2]
    #end_points['objectness_scores'+mode] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'+mode] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,0:3] # (batch_size, num_proposal, 3)
    end_points['center'+mode] = center

    #heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    #heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    #end_points['heading_scores'+mode] = heading_scores # Bxnum_proposalxnum_heading_bin
    #end_points['heading_residuals_normalized'+mode] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    #end_points['heading_residuals'+mode] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    if mode == '_z':
        end_points['size_residuals'+mode] = net_transposed[:,:,3:5]
        sem_cls_scores = net_transposed[:,:,5:] # Bxnum_proposalx10
        end_points['sem_cls_scores'+mode] = sem_cls_scores
    elif mode == '_xy':
        end_points['size_residuals'+mode] = net_transposed[:,:,3:4]
        sem_cls_scores = net_transposed[:,:,4:] # Bxnum_proposalx10
        end_points['sem_cls_scores'+mode] = sem_cls_scores
    else:
        sem_cls_scores = net_transposed[:,:,3:] # Bxnum_proposalx10
        end_points['sem_cls_scores'+mode] = sem_cls_scores
    return center, end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256, numd=1):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True,
                same_idx=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,3+numd+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points, mode=''):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        """
        if mode != '':
            xyz, features, fps_inds = self.vote_aggregation(xyz, features, inds=end_points['aggregated_vote_inds'].detach().int())
            ##sample_inds = fps_inds
            #xyz_transpose = xyz.transpose(2,1).contiguous()
            #xyz = pointnet2_utils.gather_operation(xyz_transpose, end_points['aggregated_vote_inds'].detach().int())
            #xyz = xyz.transpose(2,1).contiguous()
            #features = pointnet2_utils.gather_operation(features, end_points['aggregated_vote_inds'].detach().int())
            end_points['aggregated_vote_xyz'+mode] = xyz
            end_points['aggregated_vote_inds'+mode] = end_points['aggregated_vote_inds'] # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
                        
            # --------- PROPOSAL GENERATION ---------
            net = F.relu(self.bn1(self.conv1(features))) 
            net = F.relu(self.bn2(self.conv2(net))) 
            net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
            
            end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, mode=mode)
            return end_points
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
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
        end_points['aggregated_vote_xyz'+mode] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'+mode] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
        end_points['aggregated_feature'+mode] = features
        
        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        last_net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(last_net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        newcenter, end_points = decode_scores(net, end_points, self.num_class, mode=mode)
        return newcenter.contiguous(), features.contiguous(), end_points

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
