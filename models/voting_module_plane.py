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
        self.vote_factor = 1#vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = 3+2#xyz rotation + d1 + d2

        self.conv_upper1 = torch.nn.Conv1d(self.in_dim + 3 + 4, self.in_dim, 1)
        self.conv_upper2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv_upper3 = torch.nn.Conv1d(self.in_dim, (self.out_dim) * self.vote_factor, 1)

        #self.conv_lower1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.conv_lower2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.conv_lower3 = torch.nn.Conv1d(self.in_dim, (self.out_dim) * self.vote_factor, 1)

        self.conv_front1 = torch.nn.Conv1d(self.in_dim+3+4, self.in_dim, 1)
        self.conv_front2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv_front3 = torch.nn.Conv1d(self.in_dim, (self.out_dim) * self.vote_factor, 1)

        #self.conv_back1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.conv_back2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.conv_back3 = torch.nn.Conv1d(self.in_dim, (self.out_dim) * self.vote_factor, 1)

        self.conv_left1 = torch.nn.Conv1d(self.in_dim+3+4, self.in_dim, 1)
        self.conv_left2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv_left3 = torch.nn.Conv1d(self.in_dim, (self.out_dim) * self.vote_factor, 1)

        #self.conv_right1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.conv_right2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        #self.conv_right3 = torch.nn.Conv1d(self.in_dim, (self.out_dim) * self.vote_factor, 1)
        
        self.bn_upper1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn_upper2 = torch.nn.BatchNorm1d(self.in_dim)

        #self.bn_lower1 = torch.nn.BatchNorm1d(self.in_dim)
        #self.bn_lower2 = torch.nn.BatchNorm1d(self.in_dim)

        self.bn_front1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn_front2 = torch.nn.BatchNorm1d(self.in_dim)

        #self.bn_back1 = torch.nn.BatchNorm1d(self.in_dim)
        #self.bn_back2 = torch.nn.BatchNorm1d(self.in_dim)

        self.bn_left1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn_left2 = torch.nn.BatchNorm1d(self.in_dim)

        #self.bn_right1 = torch.nn.BatchNorm1d(self.in_dim)
        #self.bn_right2 = torch.nn.BatchNorm1d(self.in_dim)
                
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
        seed_features = torch.cat((seed_xyz.transpose(2,1).contiguous(), seed_features), 1)
        
        net_upper = F.relu(self.bn_upper1(self.conv_upper1(seed_features)))
        #net_upper = torch.cat((seed_xyz.transpose(2,1).contiguous(), net_upper), 1)
        net_upper = F.relu(self.bn_upper2(self.conv_upper2(net_upper)))
        #net_upper = self.conv_upper3(net_upper) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        #net_upper = torch.cat((seed_xyz.transpose(2,1).contiguous(), net_upper), 1)
        net_upper_lower = self.conv_upper3(net_upper) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        net_upper = net_upper_lower[:,:4,:]
        net_lower = torch.cat((net_upper_lower[:,:3,:], net_upper_lower[:,4,:].unsqueeze(1)), 1)

        '''
        net_lower = F.relu(self.bn_lower1(self.conv_lower1(seed_features))) 
        net_lower = F.relu(self.bn_lower2(self.conv_lower2(net_lower))) 
        net_lower = self.conv_lower3(net_lower) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        '''
        
        net_front = F.relu(self.bn_front1(self.conv_front1(seed_features)))
        #net_front = torch.cat((seed_xyz.transpose(2,1).contiguous(), net_front), 1)
        net_front = F.relu(self.bn_front2(self.conv_front2(net_front)))
        #net_front = torch.cat((seed_xyz.transpose(2,1).contiguous(), net_front), 1)
        #net_front = self.conv_front3(net_front) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        net_front_back = self.conv_front3(net_front) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        net_front = net_front_back[:,:4,:]
        net_back = torch.cat((net_front_back[:,:3,:], net_front_back[:,4,:].unsqueeze(1)), 1)

        '''
        net_back = F.relu(self.bn_back1(self.conv_back1(seed_features))) 
        net_back = F.relu(self.bn_back2(self.conv_back2(net_back))) 
        net_back = self.conv_back3(net_back) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        '''

        net_left = F.relu(self.bn_left1(self.conv_left1(seed_features)))
        #net_left = torch.cat((seed_xyz.transpose(2,1).contiguous(), net_left), 1)
        net_left = F.relu(self.bn_left2(self.conv_left2(net_left)))
        #net_left = torch.cat((seed_xyz.transpose(2,1).contiguous(), net_left), 1)
        #net_left = self.conv_left3(net_left) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        net_left_right = self.conv_left3(net_left) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        net_left = net_left_right[:,:4,:]
        net_right = torch.cat((net_left_right[:,:3,:], net_left_right[:,4,:].unsqueeze(1)), 1)

        '''
        net_right = F.relu(self.bn_right1(self.conv_right1(seed_features))) 
        net_right = F.relu(self.bn_right2(self.conv_right2(net_right))) 
        net_right = self.conv_right3(net_right) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        '''
        
        return net_upper, net_lower, net_left, net_right, net_front, net_back
 
if __name__=='__main__':
    net = VotingModule(2, 256).cuda()
    xyz, features = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print('xyz', xyz.shape)
    print('features', features.shape)
