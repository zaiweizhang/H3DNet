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

class VotingModule(nn.Module):
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

        self.conv4 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv5 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv6 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)

        self.conv_corner1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv_corner2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv_corner3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        
        self.conv_support1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv_support2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv_bsupport1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv_bsupport2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
                
        self.conv3_support = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        self.conv3_bsupport = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        #self.conv3_support_center = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        #self.conv3_support_offset = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        #self.conv3_bsupport_center = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        #self.conv3_bsupport_offset = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

        self.bn3 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn4 = torch.nn.BatchNorm1d(self.in_dim)

        self.bn_corner1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn_corner2 = torch.nn.BatchNorm1d(self.in_dim)
        
        self.bn_support1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn_support2 = torch.nn.BatchNorm1d(self.in_dim)
        
        self.bn_bsupport1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn_bsupport2 = torch.nn.BatchNorm1d(self.in_dim)
                
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
        net_feature = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net_feature) # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net_bcenter = F.relu(self.bn3(self.conv4(seed_features))) 
        net_bcenter = F.relu(self.bn2(self.conv5(net_bcenter))) 
        net_bcenter = self.conv6(net_bcenter) # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net_corner = F.relu(self.bn_corner1(self.conv_corner1(seed_features)))
        net_corner = F.relu(self.bn_corner2(self.conv_corner2(net_corner))) 
        net_corner = self.conv_corner3(net_corner)
        
        net_support = F.relu(self.bn_support1(self.conv_support1(seed_features)))
        net_support = F.relu(self.bn_support2(self.conv_support2(net_support))) 
        net_support = self.conv3_support(net_support) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        #net_support_center = F.relu(self.bn_support1(self.conv_support1(seed_features)))
        #net_support_center = F.relu(self.bn_support2(self.conv_support2(net_support_center)))
        #net_support_center = self.conv3_support_center(net_support_center) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        #net_support_offset = F.relu(self.bn_support3(self.conv_support3(seed_features)))
        #net_support_offset = F.relu(self.bn_support4(self.conv_support4(net_support_offset))) 
        #net_support_offset = self.conv3_support_offset(net_support_offset) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        #net_support = net_support_center + net_support_offset
        
        net_bsupport = F.relu(self.bn_bsupport1(self.conv_bsupport1(seed_features)))
        net_bsupport = F.relu(self.bn_bsupport2(self.conv_bsupport2(net_bsupport))) 
        net_bsupport = self.conv3_bsupport(net_bsupport) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        #net_bsupport_center = F.relu(self.bn_bsupport1(self.conv_bsupport1(seed_features)))
        #net_bsupport_center = F.relu(self.bn_bsupport2(self.conv_bsupport2(net_bsupport_center)))
        #net_bsupport_center = self.conv3_bsupport_center(net_bsupport_center) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        #net_bsupport_offset = F.relu(self.bn_bsupport3(self.conv_bsupport3(seed_features)))
        #net_bsupport_offset = F.relu(self.bn_bsupport4(self.conv_bsupport4(net_bsupport_offset))) 
        #net_bsupport_offset = self.conv3_bsupport_offset(net_bsupport_offset) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        #net_bsupport = net_bsupport_center + net_bsupport_offset
        
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

        net_bcenter = net_bcenter.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset = net_bcenter[:,:,:,0:3]
        vote_xyz_bcenter = seed_xyz.unsqueeze(2) + offset
        vote_xyz_bcenter = vote_xyz_bcenter.contiguous().view(batch_size, num_vote, 3)

        net_corner = net_corner.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset = net_corner[:,:,:,0:3]
        vote_xyz_corner = seed_xyz.unsqueeze(2) + offset
        vote_xyz_corner = vote_xyz_corner.contiguous().view(batch_size, num_vote, 3)

        net_support = net_support.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset_support = net_support[:,:,:,0:3]
        vote_xyz_support = seed_xyz.unsqueeze(2) + offset_support
        vote_xyz_support = vote_xyz_support.contiguous().view(batch_size, num_vote, 3)
        #net_support_center = net_support_center.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        #offset_support_center = net_support_center[:,:,:,0:3]
        #vote_xyz_support_center = seed_xyz.unsqueeze(2) + offset_support_center
        #vote_xyz_support_center = vote_xyz_support_center.contiguous().view(batch_size, num_vote, 3)
        #net_support_offset = net_support_offset.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        #offset_support_offset = net_support_offset[:,:,:,0:3]
        #vote_xyz_support_offset = seed_xyz.unsqueeze(2) + offset_support_offset
        #vote_xyz_support_offset = vote_xyz_support_offset.contiguous().view(batch_size, num_vote, 3)

        net_bsupport = net_bsupport.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset_bsupport = net_bsupport[:,:,:,0:3]
        vote_xyz_bsupport = seed_xyz.unsqueeze(2) + offset_bsupport
        vote_xyz_bsupport = vote_xyz_bsupport.contiguous().view(batch_size, num_vote, 3)
        """
        net_bsupport_center = net_bsupport_center.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset_bsupport_center = net_bsupport_center[:,:,:,0:3]
        vote_xyz_bsupport_center = seed_xyz.unsqueeze(2) + offset_bsupport_center
        vote_xyz_bsupport_center = vote_xyz_bsupport_center.contiguous().view(batch_size, num_vote, 3)
        net_bsupport_offset = net_bsupport_offset.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset_bsupport_offset = net_bsupport_offset[:,:,:,0:3]
        vote_xyz_bsupport_offset = seed_xyz.unsqueeze(2) + offset_bsupport_offset
        vote_xyz_bsupport_offset = vote_xyz_bsupport_offset.contiguous().view(batch_size, num_vote, 3)
        """
        residual_features = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()

        residual_features_support = net_support[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features_support = seed_features.transpose(2,1).unsqueeze(2) + residual_features_support
        vote_features_support = vote_features_support.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features_support = vote_features_support.transpose(2,1).contiguous()

        residual_features_bsupport = net_bsupport[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features_bsupport = seed_features.transpose(2,1).unsqueeze(2) + residual_features_bsupport
        vote_features_bsupport = vote_features_bsupport.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features_bsupport = vote_features_bsupport.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features, vote_xyz_bcenter, vote_xyz_corner, vote_xyz_support, vote_features_support, vote_xyz_bsupport, vote_features_bsupport
 
if __name__=='__main__':
    net = VotingModule(2, 256).cuda()
    xyz, features = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print('xyz', xyz.shape)
    print('features', features.shape)
