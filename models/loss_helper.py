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
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

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
    vote_xyz_corner = end_points['vote_xyz_corner'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    seed_gt_votes_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    
    upper_loss = torch.sum(end_points['upper_rot'] * vote_xyz_corner, -1) + end_points['upper_off']
    lower_loss = torch.sum(end_points['lower_rot'] * vote_xyz_corner, -1) + end_points['lower_off']
    left_loss = torch.sum(end_points['left_rot'] * vote_xyz_corner, -1) + end_points['left_off']
    right_loss = torch.sum(end_points['right_rot'] * vote_xyz_corner, -1) + end_points['right_off']
    front_loss = torch.sum(end_points['front_rot'] * vote_xyz_corner, -1) + end_points['front_off']
    back_loss = torch.sum(end_points['back_rot'] * vote_xyz_corner, -1) + end_points['back_off']

    reg_loss = torch.abs(torch.stack((upper_loss, lower_loss, left_loss, right_loss, front_loss, back_loss), -1))
    reg_dist,_ = torch.min(reg_loss, dim=-1)

    reg_loss = torch.sum(reg_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return reg_loss

def compute_center_plane_loss(end_points):
    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz_center'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    seed_gt_votes_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    
    upper_loss = torch.sum(end_points['upper_rot'] * vote_xyz, -1) + end_points['upper_off']
    lower_loss = torch.sum(end_points['lower_rot'] * vote_xyz, -1) + end_points['lower_off']
    left_loss = torch.sum(end_points['left_rot'] * vote_xyz, -1) + end_points['left_off']
    right_loss = torch.sum(end_points['right_rot'] * vote_xyz, -1) + end_points['right_off']
    front_loss = torch.sum(end_points['front_rot'] * vote_xyz, -1) + end_points['front_off']
    back_loss = torch.sum(end_points['back_rot'] * vote_xyz, -1) + end_points['back_off']

    #reg_dist = torch.abs(upper_loss + lower_loss + left_loss + right_loss + front_loss + back_loss)
    reg_loss1 = torch.abs(upper_loss + lower_loss)
    reg_loss2 = torch.abs(left_loss + right_loss)
    reg_loss3 = torch.abs(front_loss + back_loss)

    reg_dist,_ = torch.min(torch.stack((reg_loss1, reg_loss2, reg_loss3), -1), -1)
    
    reg_loss = torch.sum(reg_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return reg_loss

def compute_plane_loss_old(end_points, mode='upper'):
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
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_inds_expand_off = seed_inds.view(batch_size,num_seed,1).repeat(1,1,GT_VOTE_FACTOR)

    vote_upper_rot = end_points[mode+'_rot'] # B,num_seed*vote_factor,3
    vote_upper_rot_reshape = vote_upper_rot.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    vote_upper_offset = end_points[mode+'_off'] # B,num_seed*vote_factor,3
    vote_upper_offset_reshape = vote_upper_offset.view(batch_size*num_seed, -1, 1) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3

    seed_rot_votes = torch.gather(end_points['plane_votes_rot_'+mode], 1, seed_inds_expand)
    seed_off_votes = torch.gather(end_points['plane_votes_off_'+mode], 1, seed_inds_expand_off)   
    seed_gt_rot_reshape = seed_rot_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    seed_gt_off_reshape = seed_off_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 1) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    dist1, _, dist2, _ = nn_distance(vote_upper_rot_reshape, seed_gt_rot_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_rot_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    dist1, _, dist2, _ = nn_distance(vote_upper_offset_reshape, seed_gt_off_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_off_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    
    #return vote_rot_loss * 0.5 + vote_off_loss, vote_rot_loss*0.5, vote_off_loss
    return vote_rot_loss * 0.5 + vote_off_loss, vote_rot_loss*0.5, vote_off_loss
    #plane_upper_loss, plane_lower_loss, plane_left_loss, plane_right_loss, plane_front_loss, plane_back_loss, plane_support_loss, plane_bsupport_loss = 

def compute_plane_loss(end_points, mode='x'):
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
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_inds_expand_off = seed_inds.view(batch_size,num_seed,1).repeat(1,1,GT_VOTE_FACTOR)

    rot_meta = torch.gather(end_points['plane_votes_'+mode], 1, seed_inds_expand)
    off0_meta = torch.gather(end_points['plane_votes_'+mode+'0'], 1, seed_inds_expand_off)
    off1_meta = torch.gather(end_points['plane_votes_'+mode+'1'], 1, seed_inds_expand_off)
    
    ### Loss for the angle class
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')

    end_points[mode+'_gt'] = rot_meta[:,:,[0,3,6]].long()
    
    angle_loss1 = criterion_sem_cls(end_points[mode+'_angle'], rot_meta[:,:,0].long())
    angle_loss2 = criterion_sem_cls(end_points[mode+'_angle'], rot_meta[:,:,3].long())
    angle_loss3 = criterion_sem_cls(end_points[mode+'_angle'], rot_meta[:,:,6].long())

    angle_loss, _ = torch.min(torch.stack((angle_loss1, angle_loss2, angle_loss3), 2), dim=2)
    angle_loss = torch.sum(angle_loss*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    
    ### Loss for the sign class
    plane_sign_reshape = end_points[mode+'_sign'].view(batch_size*num_seed, -1, 1)
    seed_sign = rot_meta[:,:,[2,5,8]]
    seed_sign_reshape = seed_sign.view(batch_size*num_seed, GT_VOTE_FACTOR, 1)
    dist1, _, dist2, _ = nn_distance(plane_sign_reshape, seed_sign_reshape, l1=True)
    sign_dist, _ = torch.min(dist2, dim=1)
    sign_dist = sign_dist.view(batch_size, num_seed)
    sign_loss = torch.sum(sign_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    ### Loss for res regression    
    plane_res_reshape = end_points[mode+'_res'].view(batch_size*num_seed, -1, 1)
    seed_res = rot_meta[:,:,[1,4,7]]
    seed_res_reshape = seed_res.view(batch_size*num_seed, GT_VOTE_FACTOR, 1)
    dist1, _, dist2, _ = nn_distance(plane_res_reshape, seed_res_reshape, l1=True)
    res_dist, _ = torch.min(dist2, dim=1)
    res_dist = res_dist.view(batch_size, num_seed)
    res_loss = torch.sum(res_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    ### Loss for off0 regression
    plane_off0_reshape = end_points[mode+'_off0'].contiguous().view(batch_size*num_seed, -1, 1)
    seed_off0_reshape = off0_meta.view(batch_size*num_seed, GT_VOTE_FACTOR, 1)
    dist1, _, dist2, _ = nn_distance(plane_off0_reshape, seed_off0_reshape, l1=True)
    off0_dist, _ = torch.min(dist2, dim=1)
    off0_dist = off0_dist.view(batch_size, num_seed)
    off0_loss = torch.sum(off0_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    ### Loss for off1 regression
    plane_off1_reshape = end_points[mode+'_off1'].contiguous().view(batch_size*num_seed, -1, 1)
    seed_off1_reshape = off1_meta.view(batch_size*num_seed, GT_VOTE_FACTOR, 1)
    dist1, _, dist2, _ = nn_distance(plane_off1_reshape, seed_off1_reshape, l1=True)
    off1_dist, _ = torch.min(dist2, dim=1)
    off1_dist = off1_dist.view(batch_size, num_seed)
    off1_loss = torch.sum(off1_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    
    return angle_loss + sign_loss + res_loss + off0_loss + off1_loss, angle_loss, res_loss, sign_loss, off0_loss, off1_loss
    
def compute_objcue_vote_loss(end_points):
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
    
    vote_xyz_corner = end_points['vote_xyz_corner'] # B,num_seed*vote_factor,3
    seed_gt_votes_corner = torch.gather(end_points['vote_label_corner'], 1, seed_inds_expand)
    seed_gt_votes_corner += end_points['seed_xyz'].repeat(1,1,3)

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
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'+mode]
    gt_center = end_points['center_label'+mode][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores'+mode]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

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
    gt_center = end_points['center_label'+mode][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask'+mode]
    objectness_label = end_points['objectness_label'+mode].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'+mode], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'+mode].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'+mode], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized'+mode]*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
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

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'+mode], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'+mode].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

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

def compute_sem_cls_loss(end_points):
    # 3.4 Semantic cls loss
    #sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'+'sem'].shape[0]
    num_seed = end_points['seed_xyz'+'sem'].shape[1] # B,num_seed,3
    seed_inds = end_points['seed_inds'+'sem'].long() # B,num_seed in [0,num_points-1]

    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,GT_VOTE_FACTOR)
    
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_gt_votes_mask_plane = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    
    end_points['sem_mask'] = seed_gt_votes_mask
    end_points['sem_mask_plane'] = seed_gt_votes_mask_plane
    
    sem_cls_label = torch.gather(end_points['point_sem_cls_label'], 1, seed_inds_expand)
    end_points['sub_point_sem_cls_label'] = sem_cls_label
    num_class = end_points['pred_sem_class1'].shape[1]
    
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')

    target1 = sem_cls_label[...,0].view(-1)
    target2 = sem_cls_label[...,1].view(-1)
    target3 = sem_cls_label[...,2].view(-1)
    
    pred1 = end_points['pred_sem_class1'].transpose(2,1).contiguous().view(-1, num_class)
    pred2 = end_points['pred_sem_class2'].transpose(2,1).contiguous().view(-1, num_class)
    pred3 = end_points['pred_sem_class3'].transpose(2,1).contiguous().view(-1, num_class)

    sem_cls_loss1 = criterion_sem_cls(pred1, target1)
    sem_cls_loss2 = criterion_sem_cls(pred1, target2)
    sem_cls_loss3 = criterion_sem_cls(pred1, target3)

    sem_cls_loss_final1, _ = torch.min(torch.stack((sem_cls_loss1, sem_cls_loss2, sem_cls_loss3), 1), dim=1)

    sem_cls_loss1 = criterion_sem_cls(pred2, target1)
    sem_cls_loss2 = criterion_sem_cls(pred2, target2)
    sem_cls_loss3 = criterion_sem_cls(pred2, target3)

    sem_cls_loss_final2, _ = torch.min(torch.stack((sem_cls_loss1, sem_cls_loss2, sem_cls_loss3), 1), dim=1)

    sem_cls_loss1 = criterion_sem_cls(pred3, target1)
    sem_cls_loss2 = criterion_sem_cls(pred3, target2)
    sem_cls_loss3 = criterion_sem_cls(pred3, target3)

    sem_cls_loss_final3, _ = torch.min(torch.stack((sem_cls_loss1, sem_cls_loss2, sem_cls_loss3), 1), dim=1)

    '''
    sem_cls_loss1 = torch.sum(sem_cls_loss_final1*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6) + 0.2*torch.sum(sem_cls_loss_final1*(1.0 - seed_gt_votes_mask.view(-1)).float())/(torch.sum(1.0 - seed_gt_votes_mask.view(-1).float())+1e-6)
    sem_cls_loss2 = torch.sum(sem_cls_loss_final2*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6) + 0.2*torch.sum(sem_cls_loss_final2*(1.0 - seed_gt_votes_mask.view(-1)).float())/(torch.sum(1.0 - seed_gt_votes_mask.view(-1).float())+1e-6)
    sem_cls_loss3 = torch.sum(sem_cls_loss_final3*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6) + 0.2*torch.sum(sem_cls_loss_final3*(1.0 - seed_gt_votes_mask.view(-1)).float())/(torch.sum(1.0 - seed_gt_votes_mask.view(-1).float())+1e-6)
    '''
    sem_cls_loss1 = torch.sum(sem_cls_loss_final1*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6)
    sem_cls_loss2 = torch.sum(sem_cls_loss_final2*seed_gt_votes_mask.view(-1).float())/(torch.sum(seed_gt_votes_mask.view(-1).float())+1e-6)
    sem_cls_loss3 = torch.sum(sem_cls_loss_final3*seed_gt_votes_mask_plane.view(-1).float())/(torch.sum(seed_gt_votes_mask_plane.view(-1).float())+1e-6)
    return sem_cls_loss1+sem_cls_loss2+sem_cls_loss3
    
def get_loss(end_points, config):
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
    wcenter = 3
    wcorner = 2
    wfocal = 2
    wl2 = 1
    w9_fcen = 25
    w8_fcen = 25
    w9_fcor = 25
    w8_fcor = 25

    w9_cen = 20
    w8_cen = 250
    w5_cen = 0
    w9_cor = 60
    w8_cor = 250
    w5_cor = 0
    
    # Compute support vote loss
    if end_points['use_objcue']:
        vote_loss_center = compute_vote_center_loss(end_points)*10
        end_points['vote_loss_center'] = vote_loss_center
        vote_loss_corner = compute_objcue_vote_loss(end_points)*10
        end_points['vote_loss_corner'] = vote_loss_corner
        sem_loss = compute_sem_cls_loss(end_points)*10 # torch.tensor(0)#compute_sem_cls_loss(end_points)*10
        end_points['sem_loss'] = sem_loss
        ## get the voxel loss here
        voxel_center_loss_focal = compute_voxel_loss(end_points['vox_pred1'], end_points['vox_center'], w9_fcen, w8_fcen)
        voxel_center_loss_l2 = compute_voxel_l2_loss(end_points['vox_pred1'], end_points['vox_center'], w9_cen, w8_cen, w5_cen)
        end_points['voxel_loss_center'] = wfocal*voxel_center_loss_focal + wl2*voxel_center_loss_l2

        voxel_corner_loss_focal = compute_voxel_loss(end_points['vox_pred2'], end_points['vox_corner'], w9_fcor, w8_fcor)
        voxel_corner_loss_l2 = compute_voxel_l2_loss(end_points['vox_pred2'], end_points['vox_corner'], w9_cor, w8_cor, w5_cor)
        end_points['voxel_loss_corner'] = wfocal*voxel_corner_loss + wl2*voxel_corner_loss_l2

        end_points['voxel_loss'] = voxel_center_loss*wcenter + voxel_corner_loss*wcorner
        #end_points['vote_loss_support_center'] = support_center
        #end_points['vote_loss_bsupport_center'] = bsupport_center
        #end_points['vote_loss_support_offset'] = support_offset
        #end_points['vote_loss_bsupport_offset'] = bsupport_offset
    else:
        end_points['vote_loss_corner'] = torch.tensor(0)
        #end_points['vote_loss_support_center'] = torch.tensor(0)
        #end_points['vote_loss_bsupport_center'] = torch.tensor(0)
        #end_points['vote_loss_support_offset'] = torch.tensor(0)
        #end_points['vote_loss_bsupport_offset'] = torch.tensor(0)

    if end_points['use_plane']:
        plane_z_loss, z_angle, z_res, z_sign, z_off0, z_off1 = compute_plane_loss(end_points, mode='z')
        plane_x_loss, x_angle, x_res, x_sign, x_off0, x_off1 = compute_plane_loss(end_points, mode='x')
        plane_y_loss, y_angle, y_res, y_sign, y_off0, y_off1 = compute_plane_loss(end_points, mode='y')

        end_points['plane_x_loss'] = plane_x_loss
        end_points['plane_x_loss_angle'] = x_angle
        end_points['plane_x_loss_res'] = x_res
        end_points['plane_x_loss_sign'] = x_sign
        end_points['plane_x_loss_off0'] = x_off0
        end_points['plane_x_loss_off1'] = x_off1

        end_points['plane_y_loss'] = plane_y_loss
        end_points['plane_y_loss_angle'] = y_angle
        end_points['plane_y_loss_res'] = y_res
        end_points['plane_y_loss_sign'] = y_sign
        end_points['plane_y_loss_off0'] = y_off0
        end_points['plane_y_loss_off1'] = y_off1

        end_points['plane_z_loss'] = plane_z_loss
        end_points['plane_z_loss_angle'] = z_angle
        end_points['plane_z_loss_res'] = z_res
        end_points['plane_z_loss_sign'] = z_sign
        end_points['plane_z_loss_off0'] = z_off0
        end_points['plane_z_loss_off1'] = z_off1
    else:
        end_points['plane_upper_loss'] = torch.tensor(0)
        end_points['plane_lower_loss'] = torch.tensor(0)
        end_points['plane_left_loss'] = torch.tensor(0)
        end_points['plane_right_loss'] = torch.tensor(0)
        end_points['plane_front_loss'] = torch.tensor(0)
        end_points['plane_back_loss'] = torch.tensor(0)

    if end_points['use_plane']:
        #loss_plane = plane_upper_loss + plane_lower_loss + plane_left_loss + plane_right_loss + plane_front_loss + plane_back_loss
        loss_plane = plane_x_loss + plane_y_loss + plane_z_loss
        loss_plane *= 10
        end_points['loss_plane'] = loss_plane
        
        loss = loss_plane + vote_loss_center + vote_loss_corner + sem_loss + 50*end_points['voxel_loss']# + loss_plane_corner + loss_plane_center
        end_points['loss'] = loss
        return loss, end_points

    ### Init Proposal loss
    # Vote loss
    vote_loss = compute_vote_loss(end_points)*10
    end_points['vote_loss'] = vote_loss
        
    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
                                                                                compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
                              torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
                              torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss
    
    # Final loss function
    proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    proposalloss *= 10
    loss += proposalloss
    end_points['init_proposal_loss'] = proposalloss
    end_points['loss'] = loss ### Add the initial proposal loss term
        
    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points
