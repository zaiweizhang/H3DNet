# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

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

SIZE_THRESHOLD = 0.4
SIZE_FAR_THRESHOLD = 0.6

GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness
SEM_CLS_WEIGHTS = [0.4,0.6] # put larger weights on positive objectness
OBJECTNESS_CLS_WEIGHTS_REFINE = [0.3,0.7] # put larger weights on positive objectness

EPOCH_THRESH = 400

def compute_potential_loss(model, end_points):
    gt_potential = end_points['gt_potential']
    pert_potential = end_points['pert_potential']

    param_sum = torch.sum(model.weight.weight[0])
    
    crit = nn.MSELoss()
    pert_loss = 10e6*(param_sum - 1.0)*(param_sum - 1.0)-crit(gt_potential, pert_potential)# + 10000*(param_sum - 1.0)*(param_sum - 1.0)
    print (model.weight.weight[0])
    #import pdb;pdb.set_trace()
    
    return pert_loss


def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_surface_center_loss(end_points, mode='_z'):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'+mode].shape[0]
    num_seed = end_points['seed_xyz'+mode].shape[1] # B,num_seed,3
    #if mode == '_z':
    #    vote_xyz = end_points['vote_z'] # B,num_seed*vote_factor,3
    #else:
    vote_xyz = end_points['vote'+mode] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'+mode].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    if mode == '_line':
        seed_gt_votes_mask = torch.gather(end_points['point_line_mask'], 1, seed_inds)
    else:
        seed_gt_votes_mask = torch.gather(end_points['point_boundary_mask'+mode], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3)
    if mode == '_line':
        seed_gt_votes = torch.gather(end_points['point_line_offset'], 1, seed_inds_expand)
    else:
        seed_gt_votes = torch.gather(end_points['point_boundary_offset'+mode], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'+mode]#.repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, 1, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_vote_center_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz_center'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_voxel_l2_loss(pred, target, w95, w8, w5):
    l2_loss = nn.MSELoss()(pred, target)
    
    mask = (target>0.9).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w95*nn.MSELoss(reduction='mean')(p1, t1)
    
    mask = (target>0.8).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w8*nn.MSELoss(reduction='mean')(p1, t1)
    mask = (target>0.5).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w5*nn.MSELoss(reduction='mean')(p1, t1)
    mask = (target>0.3).float()
    loss = l2_loss
    return loss

def compute_voxel_loss(pred, gt, w10=1.0, w8=0.0):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    pos_inds8 = (gt>0.8).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    loss = 0

    pos_loss = w10*torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    pos_loss8 = w8*torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds8

    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    pos_loss8 = pos_loss8.sum()
    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss + pos_loss8) / num_pos
    return loss


def compute_corner_plane_loss(end_points):
    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz_corner'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    seed_gt_votes_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    
    #upper_loss = torch.sum(end_points['upper_rot'] * vote_xyz_corner, -1) + end_points['upper_off']
    #lower_loss = torch.sum(end_points['lower_rot'] * vote_xyz_corner, -1) + end_points['lower_off']
    #left_loss = torch.sum(end_points['left_rot'] * vote_xyz_corner, -1) + end_points['left_off']
    #right_loss = torch.sum(end_points['right_rot'] * vote_xyz_corner, -1) + end_points['right_off']
    #front_loss = torch.sum(end_points['front_rot'] * vote_xyz_corner, -1) + end_points['front_off']
    #back_loss = torch.sum(end_points['back_rot'] * vote_xyz_corner, -1) + end_points['back_off']
    upper_loss = vote_xyz[:,:,2] + end_points['z_off0']
    lower_loss = vote_xyz[:,:,2] + end_points['z_off1']
    left_loss = vote_xyz[:,:,0] + end_points['x_off0']
    right_loss = vote_xyz[:,:,0] + end_points['x_off1']
    front_loss = vote_xyz[:,:,1] + end_points['y_off0']
    back_loss = vote_xyz[:,:,1] + end_points['y_off1']
    
    reg_loss = torch.abs(torch.stack((upper_loss, lower_loss, left_loss, right_loss, front_loss, back_loss), -1))
    reg_dist,_ = torch.min(reg_loss, dim=-1)

    reg_loss = torch.sum(reg_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return reg_loss

def compute_corner_center_loss(end_points):

    vote_center = end_points['vote_xyz']
    vote_corner_center = end_points['vote_xyz_corner_center']

    reg_dist = torch.abs(vote_center - vote_corner_center)

    #reg_loss = torch.sum(reg_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    reg_loss = torch.mean(reg_dist)
    return reg_loss

def compute_center_plane_loss(end_points):
    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    seed_gt_votes_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    
    #upper_loss = torch.sum(end_points['upper_rot'] * vote_xyz, -1) + end_points['z_off0']
    #lower_loss = torch.sum(end_points['lower_rot'] * vote_xyz, -1) + end_points['lower_off']
    #left_loss = torch.sum(end_points['left_rot'] * vote_xyz, -1) + end_points['left_off']
    #right_loss = torch.sum(end_points['right_rot'] * vote_xyz, -1) + end_points['right_off']
    #front_loss = torch.sum(end_points['front_rot'] * vote_xyz, -1) + end_points['front_off']
    #back_loss = torch.sum(end_points['back_rot'] * vote_xyz, -1) + end_points['back_off']
    upper_loss = vote_xyz[:,:,2] + end_points['z_off0']
    lower_loss = vote_xyz[:,:,2] + end_points['z_off1']
    left_loss = vote_xyz[:,:,0] + end_points['x_off0']
    right_loss = vote_xyz[:,:,0] + end_points['x_off1']
    front_loss = vote_xyz[:,:,1] + end_points['y_off0']
    back_loss = vote_xyz[:,:,1] + end_points['y_off1']

    #reg_dist = torch.abs(upper_loss + lower_loss + left_loss + right_loss + front_loss + back_loss)
    reg_loss1 = torch.abs(upper_loss + lower_loss)
    reg_loss2 = torch.abs(left_loss + right_loss)
    reg_loss3 = torch.abs(front_loss + back_loss)

    reg_dist,_ = torch.min(torch.stack((reg_loss1, reg_loss2, reg_loss3), -1), -1)
    
    reg_loss = torch.sum(reg_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return reg_loss

def compute_plane_loss(end_points):
    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'+'plane'].shape[0]
    num_seed = end_points['seed_xyz'+'plane'].shape[1] # B,num_seed,3
    seed_inds = end_points['seed_inds'+'plane'].long() # B,num_seed in [0,num_points-1]
    
    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    #seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_inds_expand_off = seed_inds.view(batch_size,num_seed,1).repeat(1,1,GT_VOTE_FACTOR)

    offset = end_points['plane_off']
    ### Loss for off0 regression
    #seed_off_reshape = end_points['plane_votes_offset'].contiguous().view(batch_size*num_seed, -1, 1)
    seed_off_reshape = torch.gather(end_points['plane_votes_offset'], 1, seed_inds_expand_off)
    seed_off_reshape =  seed_off_reshape.view(batch_size*num_seed, -1, 1)
    plane_off_reshape = offset.view(batch_size*num_seed, -1, 1)
    
    dist1, _, dist2, _ = nn_distance(plane_off_reshape, seed_off_reshape, l1=True)
    off_dist, _ = torch.min(dist2, dim=1)
    off_dist = off_dist.view(batch_size, num_seed)
    off_loss = torch.sum(off_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    reg_dist = torch.abs(end_points['plane_rem'])
    reg_loss = torch.sum(reg_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    
    return off_loss+reg_loss*5, off_loss, reg_loss*5
    #return angle_loss + sign_loss + res_loss + off0_loss + off1_loss, angle_loss, res_loss, sign_loss, off0_loss, off1_loss
    
def compute_objcue_vote_loss(end_points, mode=''):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)

    if mode == 'normal':
        vote_xyz_corner = end_points['normal_xyz'] # B,num_seed*vote_factor,3
        seed_gt_votes_corner = torch.gather(end_points['point_layout_normal'], 1, seed_inds_expand)
    else:
        vote_xyz_corner = end_points['vote_xyz_'+mode] # B,num_seed*vote_factor,3
        seed_gt_votes_corner = torch.gather(end_points['vote_label_'+mode], 1, seed_inds_expand)
    seed_gt_votes_corner += end_points['seed_xyz'].repeat(1,1,3)

    if mode == 'plane':
        seed_gt_votes_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    elif mode == 'normal':
        seed_gt_votes_mask = torch.gather(end_points['point_layout_mask'], 1, seed_inds)
    else:
        seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    
    # Compute the min of min of distance
    vote_xyz_reshape_corner = vote_xyz_corner.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape_corner = seed_gt_votes_corner.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1_corner, _, dist2_corner, _ = nn_distance(vote_xyz_reshape_corner, seed_gt_votes_reshape_corner, l1=True)
    votes_dist_corner, _ = torch.min(dist2_corner, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist_corner = votes_dist_corner.view(batch_size, num_seed)
    vote_loss_corner = torch.sum(votes_dist_corner*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    return vote_loss_corner

def compute_cueness_loss_nndistance(end_points, mode=''):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)

    if mode=='center':
        vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    else:
        vote_xyz = end_points['vote_xyz_corner_center'] # B,num_seed*vote_factor,3

    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1
    
    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()

    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()

    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1    
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    #objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['cueness_scores'+mode]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    return objectness_loss, objectness_label, objectness_mask#, object_assignment

def compute_cueness_loss(end_points, mode=''):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)

    if mode=='center':
        vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    else:
        vote_xyz = end_points['vote_xyz_corner_center'] # B,num_seed*vote_factor,3

    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    gt_votes = end_points['seed_xyz'] + seed_gt_votes[:,:,:3]

    B = gt_votes.shape[0]
    K = gt_votes.shape[1]
    
    diff = vote_xyz - gt_votes
    dist = torch.sum(diff**2, dim=-1)
    #dist1, ind1, dist2, _ = nn_distance(vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Set assignment
    #object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1
    
    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise

    euclidean_dist = torch.sqrt(dist+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()

    objectness_label[euclidean_dist<NEAR_THRESHOLD] = 1    
    objectness_mask[euclidean_dist<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist>FAR_THRESHOLD] = 1
    objectness_mask = objectness_mask.float() * seed_gt_votes_mask.float()
    
    # Compute objectness loss
    objectness_scores = end_points['cueness_scores'+mode]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    return objectness_loss, objectness_label, objectness_mask#, object_assignment

def compute_objectness_loss(end_points, mode=''):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K2 = gt_center.shape[1]
    
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'+mode] ### Vote xyz is the same for all
    K = aggregated_vote_xyz.shape[1]

    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    
    ### Get the corresponding object center
    if mode == 'opt':
        obj_center = torch.gather(end_points['center_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3))
        gt_size = torch.gather(end_points['size_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K) from (B,K2)
        gt_sem = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
        
        offset_x = torch.zeros_like(gt_size)
        offset_y = torch.zeros_like(gt_size)
        offset_z = torch.zeros_like(gt_size)
        offset_x[:,:,0] = gt_size[:,:,0] / 2.0
        offset_y[:,:,1] = gt_size[:,:,1] / 2.0
        offset_z[:,:,2] = gt_size[:,:,2] / 2.0        
        ### Extract the ground truth surface label here
        obj_surface_center_0 = obj_center + offset_z
        obj_surface_center_1 = obj_center - offset_z
        obj_surface_center_2 = obj_center + offset_y
        obj_surface_center_3 = obj_center - offset_y
        obj_surface_center_4 = obj_center + offset_x
        obj_surface_center_5 = obj_center - offset_x
        
        ### Extract the ground truth line label here
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

        obj_surface_center = torch.cat((obj_surface_center_0, obj_surface_center_1, obj_surface_center_2, obj_surface_center_3, obj_surface_center_4, obj_surface_center_5), dim=1)
        obj_line_center = torch.cat((obj_line_center_0, obj_line_center_1, obj_line_center_2, obj_line_center_3, obj_line_center_4, obj_line_center_5, obj_line_center_6, obj_line_center_7, obj_line_center_8, obj_line_center_9, obj_line_center_10, obj_line_center_11), dim=1)

        pred_surface_center = end_points['surface_center_pred']
        pred_line_center = end_points['line_center_pred']

        pred_obj_surface_center = end_points["surface_center_object"]
        pred_obj_line_center = end_points["line_center_object"]

        surface_sem = torch.argmax(end_points['surface_sem_pred'], dim=2).float()
        line_sem = torch.argmax(end_points['line_sem_pred'], dim=2).float()
        
        #surface_sem_gt = torch.cat((end_points['surface_sem_gt_z'][:,:,-1], end_points['surface_sem_gt_xy'][:,:,-1]), dim=1)
        #line_sem_gt = end_points['surface_sem_gt_line'][:,:,-1]
        
        dist_surface, surface_ind, _, _ = nn_distance(obj_surface_center, pred_surface_center)
        dist_line, line_ind, _, _ = nn_distance(obj_line_center, pred_line_center)

        surface_sel = torch.gather(pred_surface_center, 1, surface_ind.unsqueeze(-1).repeat(1,1,3))
        line_sel = torch.gather(pred_line_center, 1, line_ind.unsqueeze(-1).repeat(1,1,3))
        surface_sel_sem = torch.gather(surface_sem, 1, surface_ind)
        line_sel_sem = torch.gather(line_sem, 1, line_ind)
        #surface_sel_sem_gt = torch.gather(surface_sem_gt, 1, surface_ind)
        #line_sel_sem_gt = torch.gather(line_sem_gt, 1, line_ind)
        surface_sel_sem_gt = gt_sem.repeat(1,6).float()
        line_sel_sem_gt = gt_sem.repeat(1,12).float()

        end_points["surface_sel"] = surface_sel
        end_points["line_sel"] = line_sel
        end_points["surface_sel_sem"] = surface_sel_sem
        end_points["line_sel_sem"] = line_sel_sem
        
        euclidean_dist_surface = torch.sqrt(dist_surface+1e-6)
        euclidean_dist_line = torch.sqrt(dist_line+1e-6)
        objectness_label_surface = torch.zeros((B,K*6), dtype=torch.long).cuda()
        objectness_mask_surface = torch.zeros((B,K*6)).cuda()
        objectness_label_line = torch.zeros((B,K*12), dtype=torch.long).cuda()
        objectness_mask_line = torch.zeros((B,K*12)).cuda()
        objectness_label_surface_sem = torch.zeros((B,K*6), dtype=torch.long).cuda()
        objectness_label_line_sem = torch.zeros((B,K*12), dtype=torch.long).cuda()

        #euclidean_dist_obj_surface = torch.sqrt(torch.sum((obj_surface_center - pred_obj_surface_center)**2, dim=-1)+1e-6)
        #euclidean_dist_obj_line = torch.sqrt(torch.sum((obj_line_center - pred_obj_line_center)**2, dim=-1)+1e-6)
        euclidean_dist_obj_surface = torch.sqrt(torch.sum((pred_obj_surface_center - surface_sel)**2, dim=-1)+1e-6)
        euclidean_dist_obj_line = torch.sqrt(torch.sum((pred_obj_line_center - line_sel)**2, dim=-1)+1e-6)

    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1    
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1
    if mode == 'opt':
        objectness_label_surface_obj = objectness_label.repeat(1,6)
        objectness_mask_surface_obj = objectness_mask.repeat(1,6)
        objectness_label_line_obj = objectness_label.repeat(1,12)
        objectness_mask_line_obj = objectness_mask.repeat(1,12)
        
        #objectness_mask_surface[(euclidean_dist_surface<MASK_SURFACE_THRESHOLD)] = 1
        objectness_mask_surface = objectness_mask_surface_obj
        #objectness_mask_line[(euclidean_dist_line<MASK_LINE_THRESHOLD)] = 1
        objectness_mask_line = objectness_mask_line_obj
        objectness_label_surface[(euclidean_dist_obj_surface<LABEL_SURFACE_THRESHOLD)*(euclidean_dist_surface<MASK_SURFACE_THRESHOLD)] = 1
        objectness_label_surface_sem[(euclidean_dist_obj_surface<LABEL_SURFACE_THRESHOLD)*(euclidean_dist_surface<MASK_SURFACE_THRESHOLD)*(surface_sel_sem==surface_sel_sem_gt)] = 1
        #objectness_label_surface[(euclidean_dist_obj_surface<LABEL_SURFACE_THRESHOLD)] = 1
        objectness_label_surface *= objectness_label_surface_obj
        objectness_label_surface_sem *= objectness_label_surface_obj
        #objectness_label_surface = objectness_label_surface_obj
        objectness_label_line[(euclidean_dist_obj_line<LABEL_LINE_THRESHOLD)*(euclidean_dist_line<MASK_LINE_THRESHOLD)] = 1
        objectness_label_line_sem[(euclidean_dist_obj_line<LABEL_LINE_THRESHOLD)*(euclidean_dist_line<MASK_LINE_THRESHOLD)*(line_sel_sem==line_sel_sem_gt)] = 1
        #objectness_label_line[(euclidean_dist_obj_line<LABEL_LINE_THRESHOLD)] = 1    
        objectness_label_line *= objectness_label_line_obj
        objectness_label_line_sem *= objectness_label_line_obj
        #objectness_label_line = objectness_label_line_obj
        
    # Compute objectness loss
    if mode == 'opt':
        objectness_scores = end_points["match_scores"]#end_points['objectness_scores'+'refine']
        temp_objectness_label = torch.cat((objectness_label_surface, objectness_label_line), 1)
        temp_objectness_label_sem = torch.cat((objectness_label_surface_sem, objectness_label_line_sem), 1)
        #end_points['match_gt'] = temp_objectness_label
        temp_objectness_mask = torch.cat((objectness_mask_surface, objectness_mask_line), 1)
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS_REFINE).cuda(), reduction='none')
        objectness_loss = criterion(objectness_scores.transpose(2,1), temp_objectness_label)
        objectness_loss = torch.sum(objectness_loss * temp_objectness_mask)/(torch.sum(temp_objectness_mask)+1e-6)

        #objectness_scores = end_points['objectness_scores'+mode]#end_points['objectness_scores'+'refine']
        #criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
        #objectness_loss_refine = criterion(objectness_scores.transpose(2,1), objectness_label)
        #objectness_loss_refine = torch.sum(objectness_loss_refine * objectness_mask)/(torch.sum(objectness_mask)+1e-6)
        
        ### Temprory test
        #match_score = end_points["match_gt"]
        #match_score = match_score.view(B, -1, 12+6, 256).transpose(3,2).contiguous()
        #_, inds_obj = torch.max(match_score[:,0,:,:], -1)
        #_, inds_obj = torch.topk(match_score[:,1,:,:], k=3, dim=-1)
        #nd_points['objectness_scores'+'opt'] = torch.mean(torch.gather(match_score, -1, inds_obj.unsqueeze(1).repeat(1,2,1,1)), dim=-1).transpose(2,1).contiguous()
        #end_points['objectness_scores'+'opt'] = torch.gather(match_score, -1, inds_obj.unsqueeze(-1).transpose(2,1).unsqueeze(-1)).squeeze(-1).transpose(2,1).contiguous().squeeze(-1).float()
        return objectness_loss, objectness_label, objectness_mask, temp_objectness_label, temp_objectness_label_sem, temp_objectness_mask, object_assignment
    else:
        objectness_scores = end_points['objectness_scores'+mode]
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
        objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
        objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

        return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config, mode=''):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment'+mode]
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center'+mode]
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'+mode].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'+mode].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized'+mode]*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'+mode].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized'+mode]*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'+mode].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def compute_matching_box_loss(end_points, config, mode=''):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment'+mode]
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center'+mode]
    obj_center = end_points['center'+mode]
    #pred_center = end_points['center'+'center']
    #obj_center = end_points['center'+'center']
    #size_residual = end_points['size_residuals'+'center']
    #size_residual_normalized = end_points['size_residuals_normalized'+'center']
    size_residual = end_points['size_residuals'+mode]
    size_residual_normalized = end_points['size_residuals_normalized'+mode]
    pred_size_class = torch.argmax(end_points['size_scores'+'center'].contiguous(), -1).detach()
    pred_size_residual = torch.gather(size_residual, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3))
    mean_size_class_batched = torch.ones_like(size_residual) * torch.from_numpy(config.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_avg = torch.gather(mean_size_class_batched, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).detach()
    obj_size = pred_size_avg.squeeze(2) + pred_size_residual.squeeze(2)
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
    pred_obj_surface_center = torch.cat((obj_upper_surface_center, obj_lower_surface_center, obj_front_surface_center, obj_back_surface_center, obj_left_surface_center, obj_right_surface_center), dim=1)

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
    pred_obj_line_center = torch.cat((obj_line_center_0, obj_line_center_1, obj_line_center_2, obj_line_center_3, obj_line_center_4, obj_line_center_5, obj_line_center_6, obj_line_center_7, obj_line_center_8, obj_line_center_9, obj_line_center_10, obj_line_center_11), dim=1)

    source_point = torch.cat((pred_obj_surface_center, pred_obj_line_center), 1)
    
    surface_target = end_points["surface_sel"]
    line_target = end_points["line_sel"]

    target_point = torch.cat((surface_target, line_target), 1)

    objectness_match_label = end_points['objectness_match_label'+'opt'].float()
    objectness_match_label_sem = end_points['objectness_match_label_sem'+'opt'].float()

    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'+mode].float()
    centroid_reg_loss1 = torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    dist_match = torch.sqrt(torch.sum((source_point - target_point)**2, dim=-1)+1e-6)
    centroid_reg_loss3 = torch.sum(dist_match*objectness_match_label)/(torch.sum(objectness_match_label)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2 + centroid_reg_loss3

    ### Compute the original size loss
    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    #criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    #size_class_loss = criterion_size_class(end_points['size_scores'+mode].transpose(2,1), size_class_label) # (B,K)
    #size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(size_residual_normalized*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'+mode].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    pred_surface_semantic = end_points['surface_sel_sem'].detach()
    pred_line_semantic = end_points['line_sel_sem'].detach()
    pred_semantic = torch.cat((pred_surface_semantic, pred_line_semantic), 1)

    objectness_match_label_sem = end_points['objectness_match_label_sem'+'opt'].float()
    obj_sem_pred = end_points['sem_cls_scores'+mode].transpose(2,1).repeat(1,1,18)
    sem_cls_loss_reg = criterion_sem_cls(obj_sem_pred, pred_semantic.long()) # (B,K)
    sem_cls_loss_reg = torch.sum(sem_cls_loss_reg * objectness_match_label_sem)/(torch.sum(objectness_match_label_sem)+1e-6)
    
    return center_loss+size_residual_normalized_loss+0.1*sem_cls_loss+0.1*sem_cls_loss_reg, centroid_reg_loss1 + centroid_reg_loss2, centroid_reg_loss3, size_residual_normalized_loss, sem_cls_loss, sem_cls_loss_reg
    # return center_loss+size_residual_normalized_loss, centroid_reg_loss1 + centroid_reg_loss2, centroid_reg_loss3, size_residual_normalized_loss, sem_cls_loss, sem_cls_loss_reg
    #return center_loss, centroid_reg_loss1 + centroid_reg_loss2, centroid_reg_loss3, size_residual_normalized_loss, sem_cls_loss, sem_cls_loss_reg

def compute_boxsem_loss(end_points, config, mode=''):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'+mode].shape[0]
    num_seed = end_points['seed_xyz'+mode].shape[1] # B,num_seed,3
    vote_xyz = end_points['center'+mode] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'+mode].long() # B,num_seed in [0,num_points-1]

    num_proposal = end_points['aggregated_vote_xyz'+mode].shape[1] # B,num_seed,3
    
    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    if mode == '_line':
        seed_gt_votes_mask = torch.gather(end_points['point_line_mask'], 1, seed_inds)
    else:
        seed_gt_votes_mask = torch.gather(end_points['point_boundary_mask'+mode], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3)
    if mode == '_z':
        seed_inds_expand_sem = seed_inds.view(batch_size,num_seed,1).repeat(1,1,6)
    elif mode == '_xy':
        seed_inds_expand_sem = seed_inds.view(batch_size,num_seed,1).repeat(1,1,5)
    else:
        seed_inds_expand_sem = seed_inds.view(batch_size,num_seed,1).repeat(1,1,4)
    if mode == '_line':
        seed_gt_votes = torch.gather(end_points['point_line_offset'], 1, seed_inds_expand)
        seed_gt_sem = torch.gather(end_points['point_line_sem'], 1, seed_inds_expand_sem)
    else:
        seed_gt_votes = torch.gather(end_points['point_boundary_offset'+mode], 1, seed_inds_expand)
        seed_gt_sem = torch.gather(end_points['point_boundary_sem'+mode], 1, seed_inds_expand_sem)
    seed_gt_votes += end_points['seed_xyz'+mode]#.repeat(1,1,3)

    '''
    seed_inds_expand = end_points['aggregated_vote_inds'+mode].long().view(batch_size,num_proposal,1).repeat(1,1,3*GT_VOTE_FACTOR)
    if mode == '_z':
        seed_inds_expand_sem = end_points['aggregated_vote_inds'+mode].long().view(batch_size,num_proposal,1).repeat(1,1,6*GT_VOTE_FACTOR)
    else:
        seed_inds_expand_sem = end_points['aggregated_vote_inds'+mode].long().view(batch_size,num_proposal,1).repeat(1,1,5*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(seed_gt_votes, 1, seed_inds_expand)
    seed_gt_sem = torch.gather(seed_gt_sem, 1, seed_inds_expand_sem)
    seed_gt_votes_mask = torch.gather(seed_gt_votes_mask, 1, end_points['aggregated_vote_inds'+mode].long())
    '''
    end_points['surface_center_gt'+mode] = seed_gt_votes
    end_points['surface_sem_gt'+mode] = seed_gt_sem
    end_points['surface_mask_gt'+mode] = seed_gt_votes_mask
    
    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_proposal, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_proposal, 1, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_proposal)
    center_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    # Compute the min of min of distance
    if mode != '_line':
        size_xyz = end_points['size_residuals'+mode].contiguous() # B,num_seed*vote_factor,3
        if mode == '_z':
            size_xyz_reshape = size_xyz.view(batch_size*num_proposal, -1, 2).contiguous() # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
            seed_gt_votes_reshape = seed_gt_sem[:,:,3:5].view(batch_size*num_proposal, 1, 2).contiguous() # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
        else:
            size_xyz_reshape = size_xyz.view(batch_size*num_proposal, -1, 1).contiguous()# from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
            seed_gt_votes_reshape = seed_gt_sem[:,:,3:4].view(batch_size*num_proposal, 1, 1).contiguous() # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
        # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
        dist1, _, dist2, _ = nn_distance(size_xyz_reshape, seed_gt_votes_reshape, l1=True)
        size_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
        size_dist = size_dist.view(batch_size, num_proposal)
        size_loss = torch.sum(size_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)    
    else:
        size_loss = torch.tensor(0)
        
    # 3.4 Semantic cls loss
    sem_cls_label = seed_gt_sem[:,:,-1].long()#torch.gather(end_points['sem_cls_label'+mode], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'+mode].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    return center_loss, size_loss, sem_cls_loss

def compute_support_loss(end_points, config):
    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment'+mode]
    batch_size = object_assignment.shape[0]
    
    # Compute center loss
    pred_center = end_points['center'+mode]
    gt_center = end_points['center_label'+mode][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    
    box_label_mask = end_points['box_label_mask'+mode]
    objectness_label = end_points['objectness_label'+mode].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Get predicted size
    size_class_label = torch.gather(end_points['size_class_label'+mode], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'+mode].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'+mode], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized'+mode]*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

def compute_sem_cls_loss(end_points, mode):
    # 3.4 Semantic cls loss
    #sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'+mode].shape[0]
    num_seed = end_points['seed_xyz'+mode].shape[1] # B,num_seed,3
    seed_inds = end_points['seed_inds'+mode].long() # B,num_seed in [0,num_points-1]

    #seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,GT_VOTE_FACTOR)
    seed_inds_expand = seed_inds.view(batch_size,num_seed)#.repeat(1,1,GT_VOTE_FACTOR)

    if mode == '_line':
        seed_gt_votes_mask = torch.gather(end_points['point_line_mask'], 1, seed_inds).float()
    else:
        seed_gt_votes_mask = torch.gather(end_points['point_boundary_mask'+mode], 1, seed_inds).float()
        
    end_points['sem_mask'] = seed_gt_votes_mask

    if mode == '_line':
        sem_cls_label = torch.gather(end_points['point_line_mask'], 1, seed_inds)
    else:
        sem_cls_label = torch.gather(end_points['point_boundary_mask'+mode], 1, seed_inds)

    end_points['sub_point_sem_cls_label'+mode] = sem_cls_label
    num_class = end_points['pred_sem_class'+mode].shape[1]

    pred1 = end_points['pred_sem_class'+mode]#.transpose(2,1).contiguous().view(-1, num_class)
    
    criterion = nn.CrossEntropyLoss(torch.Tensor(SEM_CLS_WEIGHTS).cuda(), reduction='none')
    sem_loss = criterion(pred1, sem_cls_label.long())
    sem_loss = torch.mean(sem_loss.float())#/(torch.sum(seed_gt_votes_mask)+1e-6)
    #sem_loss = torch.sum(sem_loss.float() * seed_gt_votes_mask)/(torch.sum(seed_gt_votes_mask)+1e-6)

    return sem_loss
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')

    #target1 = sem_cls_label[...,0].view(-1)
    target1 = sem_cls_label[...,0].view(-1)
    #target2 = sem_cls_label[...,1].view(-1)
    #target3 = sem_cls_label[...,2].view(-1)
    
    

    #sem_cls_loss_final1, _ = torch.min(torch.stack((sem_cls_loss1, sem_cls_loss2, sem_cls_loss3), 1), dim=1)
    #import pdb;pdb.set_trace()
    #pred2 = end_points['pred_sem_class2'].transpose(2,1).contiguous().view(-1, num_class)
    #pred3 = end_points['pred_sem_class3'].transpose(2,1).contiguous().view(-1, num_class)

    sem_cls_loss1 = criterion_sem_cls(pred1, target1)
    sem_cls_loss2 = criterion_sem_cls(pred1, target2)
    sem_cls_loss3 = criterion_sem_cls(pred1, target3)

    sem_cls_loss_final1, _ = torch.min(torch.stack((sem_cls_loss1, sem_cls_loss2, sem_cls_loss3), 1), dim=1)

    '''
    sem_cls_loss1 = torch.sum(sem_cls_loss_final1*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6) + 0.2*torch.sum(sem_cls_loss_final1*(1.0 - seed_gt_votes_mask.view(-1)).float())/(torch.sum(1.0 - seed_gt_votes_mask.view(-1).float())+1e-6)
    sem_cls_loss2 = torch.sum(sem_cls_loss_final2*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6) + 0.2*torch.sum(sem_cls_loss_final2*(1.0 - seed_gt_votes_mask.view(-1)).float())/(torch.sum(1.0 - seed_gt_votes_mask.view(-1).float())+1e-6)
    sem_cls_loss3 = torch.sum(sem_cls_loss_final3*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6) + 0.2*torch.sum(sem_cls_loss_final3*(1.0 - seed_gt_votes_mask.view(-1)).float())/(torch.sum(1.0 - seed_gt_votes_mask.view(-1).float())+1e-6)
    '''
    sem_cls_loss1 = torch.sum(sem_cls_loss_final1*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6)
    #sem_cls_loss1 = torch.mean(sem_cls_loss_final1)
    #sem_cls_loss2 = torch.sum(sem_cls_loss_final2*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6)
    #sem_cls_loss3 = torch.sum(sem_cls_loss_final3*seed_gt_votes_mask_plane.view(-1).float())/(torch.sum(seed_gt_votes_mask_plane.view(-1).float())+1e-6)
    return sem_cls_loss1#+sem_cls_loss2+sem_cls_loss3
    

def compute_matching_loss(end_points):
    """ Compute matching loss for the centers.
    Note:
        TO BE IMPROVED. Currently waiting for hdnet to generate:
            end_points['matching_scores_surface']: [B, 256*6, 2]
            end_points['matching_scores_line']: [B, 256*12, 2]
    Args:
        end_points: dict (read-only)

    Returns:
        matching_loss: scalar Tensor
    """ 
    # surface matching
    surface_center_object = end_points['surface_center_object']
    surface_center_pred = end_points['surface_center_pred']
    B = surface_center_pred.shape[0]
    Ksurface= surface_center_object.shape[1] # 256*6
    K2surface = surface_center_pred.shape[1] # 4096 
    dist1, ind1, dist2, _ = nn_distance(surface_center_object, surface_center_pred) # dist1: BxK, dist2: BxK2

    euclidean_dist1_surface = torch.sqrt(dist1+1e-6)
    objectness_label_surface = torch.zeros((B,Ksurface), dtype=torch.long).cuda()
    objectness_mask_surface = torch.zeros((B,Ksurface)).cuda()
    objectness_label_surface[euclidean_dist1_surface<NEAR_MATCH_THRESHOLD] = 1
    objectness_mask_surface[euclidean_dist1_surface<NEAR_MATCH_THRESHOLD] = 1
    objectness_mask_surface[euclidean_dist1_surface>FAR_MATCH_THRESHOLD] = 1

    matching_scores_surface = end_points['matching_scores_surface'] # [B, 256*6, 2]
    criterion_surface = nn.CrossEntropyLoss(reduction='none')
    objectness_loss_surface = criterion_surface(matching_scores_surface.transpose(2,1), objectness_label_surface)
    objectness_loss_surface = torch.sum(objectness_loss_surface * objectness_mask_surface)/(torch.sum(objectness_mask_surface)+1e-6)

    # line matching
    line_center_object = end_points['line_center_object']
    line_center_pred = end_points['line_center_pred']
    B = line_center_pred.shape[0]
    Kline= line_center_object.shape[1] # 256*12
    K2line = line_center_pred.shape[1] # 4096 
    dist1, ind1, dist2, _ = nn_distance(line_center_object, line_center_pred) # dist1: BxK, dist2: BxK2

    euclidean_dist1_line = torch.sqrt(dist1+1e-6)
    objectness_label_line = torch.zeros((B,Kline), dtype=torch.long).cuda()
    objectness_mask_line = torch.zeros((B,Kline)).cuda()
    objectness_label_line[euclidean_dist1_line<NEAR_MATCH_THRESHOLD] = 1
    objectness_mask_line[euclidean_dist1_line<NEAR_MATCH_THRESHOLD] = 1
    objectness_mask_line[euclidean_dist1_line>FAR_MATCH_THRESHOLD] = 1

    matching_scores_line = end_points['matching_scores_line'] # [B, 256*12, 2]
    criterion_line = nn.CrossEntropyLoss(reduction='none')
    objectness_loss_line = criterion_line(matching_scores_line.transpose(2,1), objectness_label_line)
    objectness_loss_line = torch.sum(objectness_loss_line * objectness_mask_line)/(torch.sum(objectness_mask_line)+1e-6)

    matching_loss = objectness_loss_surface + objectness_loss_line

    return matching_loss




def get_loss(inputs, end_points, config, is_votenet_training, is_refine_training, net=None):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """
    if inputs['sunrgbd']:
        wcenter = 1
        wcorner = 1
        wfocal = 0
        wl2 = 3
        w9_fcen = 100
        w8_fcen = 100
        w9_fcor = 100
        w8_fcor = 100
        
        w9_cen = 50
        w8_cen = 350
        w5_cen = 0
        w9_cor = 100
        w8_cor = 350
        w5_cor = 0
    else:
        wcenter = 1
        wcorner = 1
        wfocal = 0#2
        wl2 = 1
        w9_fcen = 30
        w8_fcen = 30
        w9_fcor = 30
        w8_fcor = 30
        
        w9_cen = 20
        w8_cen = 250
        w5_cen = 0
        w9_cor = 60
        w8_cor = 250
        w5_cor = 0

    sem_loss_z = compute_sem_cls_loss(end_points, mode='_z')*30 # torch.tensor(0)#compute_sem_cls_loss(end_points)*10
    end_points['sem_loss_z'] = sem_loss_z

    sem_loss_xy = compute_sem_cls_loss(end_points, mode='_xy')*30 # torch.tensor(0)#compute_sem_cls_loss(end_points)*10
    end_points['sem_loss_xy'] = sem_loss_xy

    sem_loss_line = compute_sem_cls_loss(end_points, mode='_line')*30 # torch.tensor(0)#compute_sem_cls_loss(end_points)*10
    end_points['sem_loss_line'] = sem_loss_line

    vote_loss_z = compute_surface_center_loss(end_points, mode='_z')*10
    end_points['vote_loss_z'] = vote_loss_z

    vote_loss_xy = compute_surface_center_loss(end_points, mode='_xy')*10
    end_points['vote_loss_xy'] = vote_loss_xy

    vote_loss_line = compute_surface_center_loss(end_points, mode='_line')*10
    end_points['vote_loss_line'] = vote_loss_line

    center_lossz, size_lossz, sem_lossz = compute_boxsem_loss(end_points, config, mode='_z')
    end_points['center_lossz'] = center_lossz
    end_points['size_lossz'] = size_lossz
    end_points['sem_lossz'] = sem_lossz
    end_points['surface_lossz'] = center_lossz*0.5 + size_lossz*0.5 + sem_lossz

    center_lossxy, size_lossxy, sem_lossxy = compute_boxsem_loss(end_points, config, mode='_xy')
    end_points['center_lossxy'] = center_lossxy
    end_points['size_lossxy'] = size_lossxy
    end_points['sem_lossxy'] = sem_lossxy
    end_points['surface_lossxy'] = center_lossxy*0.5 + size_lossxy*0.5 + sem_lossxy

    center_lossline, size_lossline, sem_lossline = compute_boxsem_loss(end_points, config, mode='_line')
    end_points['center_lossline'] = center_lossline
    end_points['size_lossline'] = size_lossline
    end_points['sem_lossline'] = sem_lossline
    end_points['surface_lossline'] = center_lossline*0.5 + size_lossline*0.5 + sem_lossline
    
    end_points['objcue_loss'] = sem_loss_z + sem_loss_xy + sem_loss_line + vote_loss_z + vote_loss_xy + vote_loss_line + end_points['surface_lossz'] + end_points['surface_lossxy'] + end_points['surface_lossline']*2
    #return end_points['loss'], end_points

    ### Init Proposal loss
    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
                                                                                compute_objectness_loss(end_points, mode='center')
    end_points['objectness_loss'+'center'] = objectness_loss
    end_points['objectness_label'+'center'] = objectness_label
    end_points['objectness_mask'+'center'] = objectness_mask
    end_points['object_assignment'+'center'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
                              torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
                              torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    objectness_loss_opt, objectness_label_opt, objectness_mask_opt, objectness_label_match, objectness_label_match_sem, objectness_mask_match, object_assignment_opt = \
                                                                                compute_objectness_loss(end_points, mode='opt')
    end_points['objectness_loss'+'opt'] = objectness_loss_opt
    end_points['objectness_label'+'opt'] = objectness_label_opt
    end_points['objectness_mask'+'opt'] = objectness_mask_opt
    
    end_points['objectness_match_label'+'opt'] = objectness_label_match
    end_points['objectness_match_label_sem'+'opt'] = objectness_label_match_sem
    end_points['objectness_match_mask'+'opt'] = objectness_mask_match
    
    end_points['object_assignment'+'opt'] = object_assignment_opt
    total_num_proposal_opt = objectness_label_match.shape[0]*objectness_label_match.shape[1]
    end_points['cover_ratio_opt'] = \
                              torch.sum(objectness_mask_match.float().cuda())/float(total_num_proposal_opt)
    end_points['pos_ratio_opt'] = \
                              torch.sum(objectness_label_match.float().cuda())/float(total_num_proposal_opt)#torch.sum(objectness_mask_match.float().cuda())
    end_points['pos_obj_ratio_opt'] = \
                              torch.sum((torch.max(objectness_label_match.float().view(objectness_label.shape[0], 18, objectness_label.shape[1]), dim=1)[0]).cuda())/torch.sum(objectness_label.float().cuda())#torch.sum(objectness_mask_match.float().cuda())
    
    end_points['neg_ratio_opt'] = \
                              torch.sum(objectness_mask_match.float())/float(total_num_proposal_opt) - end_points['pos_ratio_opt']
    end_points['sem_ratio_opt'] = \
                              torch.sum(objectness_label_match_sem.float().cuda())/torch.sum(objectness_label_match.float().cuda())

    assert(np.array_equal(objectness_label.detach().cpu().numpy(), objectness_label_opt.detach().cpu().numpy()))
    assert(np.array_equal(objectness_mask.detach().cpu().numpy(), objectness_mask_opt.detach().cpu().numpy()))
    assert(np.array_equal(object_assignment.detach().cpu().numpy(), object_assignment_opt.detach().cpu().numpy()))
    
    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config, mode='center')
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Box loss and sem cls loss finetune
    """
    center_loss_opt, heading_cls_loss_opt, heading_reg_loss_opt, size_cls_loss_opt, size_reg_loss_opt, sem_cls_loss_opt = \
        compute_box_and_sem_cls_loss(end_points, config, mode='opt')
    end_points['center_loss_opt'] = center_loss_opt
    end_points['heading_cls_loss_opt'] = heading_cls_loss_opt
    end_points['heading_reg_loss_opt'] = heading_reg_loss_opt
    end_points['size_cls_loss_opt'] = size_cls_loss_opt
    end_points['size_reg_loss_opt'] = size_reg_loss_opt
    end_points['sem_cls_loss_opt'] = sem_cls_loss_opt
    box_loss_opt = center_loss_opt + 0.1*heading_cls_loss_opt + heading_reg_loss_opt + 0.1*size_cls_loss_opt + size_reg_loss_opt
    end_points['box_loss_opt'] = box_loss_opt
    """
    center_loss_opt, original_center, new_center, original_size, sem_cls_loss_opt, sem_cls_reg_loss = compute_matching_box_loss(end_points, config, mode='opt')
    end_points['center_loss_opt'] = center_loss_opt
    end_points['center_loss_original'] = original_center
    end_points['size_reg_loss_original'] = original_size
    end_points['center_loss_new'] = new_center
    end_points['sem_cls_loss_opt'] = sem_cls_loss_opt
    end_points['sem_cls_reg_loss'] = sem_cls_reg_loss

    # Final loss function
    proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.5*objectness_loss_opt + center_loss_opt
    """
    if is_votenet_training and (not is_refine_training):
        proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.5*objectness_loss_opt + center_loss_opt
    elif (not is_votenet_training) and is_refine_training:
        proposalloss = 0.5*objectness_loss_opt + box_loss_opt + 0.1*sem_cls_loss_opt
    elif is_votenet_training and is_refine_training:
        proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.5*objectness_loss_opt + box_loss_opt + 0.1*sem_cls_loss_opt
    else:
        exit(1)
    """
    '''
    if inputs['epoch'] < EPOCH_THRESH:
        #proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0*box_loss_opt + 0*0.1*sem_cls_loss_opt
        #proposalloss = vote_loss + center_cueloss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.5*objectness_losscorner + box_losscorner + 0.1*sem_cls_losscorner + 0.5*objectness_lossplane + box_lossplane + 0.1*sem_cls_lossplane
        #proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.5*objectness_losscorner + box_losscorner + 0.1*sem_cls_losscorner + 0.5*objectness_lossplane + box_lossplane + 0.1*sem_cls_lossplane
        proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.5*objectness_losscorner + box_losscorner + 0.1*sem_cls_losscorner + 0.5*objectness_lossplane + box_lossplane + 0.1*sem_cls_lossplane + 0.5*objectness_losscomb + box_losscomb + 0.1*sem_cls_losscomb
    #elif inputs['epoch'] < EPOCH_THRESH+80:
    #    proposalloss = 0*vote_loss + 0*0.5*objectness_loss + 0*box_loss + 0*0.1*sem_cls_loss + box_loss_opt + 0.1*sem_cls_loss_opt# + objectness_reg_loss
    else:
        #proposalloss = vote_loss + center_cueloss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + box_loss_opt + 0.1*sem_cls_loss_opt
        proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + box_loss_opt + 0.1*sem_cls_loss_opt
    '''
    proposalloss *= 10
    loss = proposalloss + end_points['objcue_loss']
    end_points['init_proposal_loss'] = proposalloss
    end_points['loss'] = loss ### Add the initial proposal loss term
        
    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'+'center'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc
    """
    obj_pred_val = torch.argmax(end_points['objectness_scores'+'opt'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label_opt.long()).float()*objectness_mask_opt)/(torch.sum(objectness_mask_opt)+1e-6)
    end_points['obj_acc_opt'] = obj_acc
    """
    obj_pred_val = torch.argmax(end_points['match_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label_match.long()).float()*objectness_mask_match)/(torch.sum(objectness_mask_match)+1e-6)
    end_points['obj_acc_match'] = obj_acc
    return loss, end_points
