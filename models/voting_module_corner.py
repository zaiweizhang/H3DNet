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

class VotingCornerModule(nn.Module):
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
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

        self.conv4 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv5 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv6 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)

        self.conv7 = torch.nn.Conv1d(self.in_dim*2, 2 * self.vote_factor, 1)
        
        self.bn3 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn4 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed*self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net_c1 = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net_c1) # (batch_size, (3+out_dim)*vote_factor, num_seed)
                
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        
        residual_features = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()

        net = F.relu(self.bn3(self.conv4(seed_features))) 
        net_c2 = F.relu(self.bn4(self.conv5(net))) 
        net = self.conv6(net_c2) # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net_cueness = self.conv7(torch.cat((net_c1, net_c2), 1))
        
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset = net[:,:,:,0:3]
        vote_xyz_other = seed_xyz.unsqueeze(2) + offset
        vote_xyz_other = vote_xyz_other.contiguous().view(batch_size, num_vote, 3)
        
        residual_features_other = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features_other = seed_features.transpose(2,1).unsqueeze(2) + residual_features_other
        vote_features_other = vote_features_other.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features_other = vote_features_other.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features, vote_xyz_other, vote_features_other, net_cueness.transpose(2,1).contiguous()
 
if __name__=='__main__':
    net = VotingModule(2, 256).cuda()
    xyz, features = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print('xyz', xyz.shape)
    print('features', features.shape)

