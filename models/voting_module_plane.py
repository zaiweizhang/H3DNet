# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Voting module: generate votes from XYZ and features of seed points.

Date: July, 2019
Author: Charles R. Qi and Or Litany
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class VotingPlaneModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim+3+4, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+1+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

        self.conv4 = torch.nn.Conv1d(self.in_dim+3+4, self.in_dim, 1)
        self.conv5 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv6 = torch.nn.Conv1d(self.in_dim, (1+self.out_dim) * self.vote_factor, 1)
        self.bn3 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn4 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz_plane, seed_features, end_points):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz_plane.shape[0]
        num_seed = seed_xyz_plane.shape[1]
        num_vote = num_seed*self.vote_factor

        seed_xyz = seed_xyz_plane[:,:,:3]
        seed_plane = seed_xyz_plane[:,:,3:]

        new_seed_features = torch.cat((seed_xyz_plane.transpose(2,1).contiguous(), seed_features), 1)
        
        net = F.relu(self.bn1(self.conv1(new_seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+1+self.out_dim)
        offset = net[:,:,:,:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

        #net_plane = F.relu(self.bn3(self.conv4(new_seed_features))) 
        #net_plane = F.relu(self.bn4(self.conv5(net_plane))) 
        #net_plane = self.conv6(net_plane) # (batch_size, (3+out_dim)*vote_factor, num_seed)

        #net_plane = net_plane.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 1+self.out_dim)
        offset_plane = net[:,:,:,3]
        vote_plane = seed_plane[:,:,-1].unsqueeze(2) + offset_plane
        vote_plane = vote_plane.contiguous().view(batch_size, num_vote)

        #new_plane = torch.cat((seed_plane[:,:,:3], vote_plane), -1)
        end_points["plane_rem"] = torch.sum(seed_plane[:,:,:3] * vote_xyz, -1) + vote_plane
        end_points["plane_off"] = vote_plane.contiguous().view(batch_size, num_vote, 1)
        
        residual_features = net[:,:,:,4:] # (batch_size, num_seed, vote_factor, out_dim)
        #residual_features2 = net_plane[:,:,:,1:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features#1 + residual_features2
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features, net[:,:,:,0:2].view(batch_size, num_vote, 2).contiguous()
 
if __name__=='__main__':
    net = VotingModule(2, 256).cuda()
    xyz, features = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print('xyz', xyz.shape)
    print('features', features.shape)

