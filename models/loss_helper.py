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

SIZE_THRESHOLD = 0.4
SIZE_FAR_THRESHOLD = 0.6

GT_VOTE_FACTOR = 1#3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness
SEM_CLS_WEIGHTS = [0.4,0.6] # put larger weights on positive objectness
OBJECTNESS_CLS_WEIGHTS_REFINE = [0.2,0.8] # put larger weights on positive objectness

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
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    if mode == '_line':
        seed_gt_votes = torch.gather(end_points['point_line_offset'], 1, seed_inds_expand)
    else:
        seed_gt_votes = torch.gather(end_points['point_boundary_offset'+mode], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'+mode]#.repeat(1,1,3)

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

def compute_plane_loss_old(end_points, mode='x'):
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

    return off0_loss + off1_loss, angle_loss, res_loss, sign_loss, off0_loss, off1_loss
    #return angle_loss + sign_loss + res_loss + off0_loss + off1_loss, angle_loss, res_loss, sign_loss, off0_loss, off1_loss

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
    if 'refine' in mode:
        feature = end_points['aggregated_vote_mrf']
        F = feature.shape[1] // 2
        K = feature.shape[2]
        N = feature.shape[3]

        ### For location constraint
        aggregated_vote_xyz1 = feature[:,2:2+3,:,:].view(B,3,K*N).transpose(2,1).contiguous()
        aggregated_vote_xyz2 = feature[:,F+2:F+2+3,:,:].view(B,3,K*N).transpose(2,1).contiguous()
        aggregated_vote_xyz = torch.cat((aggregated_vote_xyz1, aggregated_vote_xyz2), 1)

        #aggregated_vote_xyz = feature[:,2:2+3,:,0].transpose(2,1).contiguous()
        #aggregated_vote_sem = end_points['aggregated_vote_sem'][:,:,:,0].transpose(2,1).contiguous()
        #aggregated_vote_sem = torch.argmax(aggregated_vote_sem, -1)

        K *= N
    else:
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
    if 'refine' in mode:
        objectness_label_sub = torch.zeros((B,K//N), dtype=torch.long).cuda()
        objectness_mask_sub = torch.zeros((B,K//N)).cuda()
    '''
    if False:#mode == 'corner':
        size_gt = end_points['gt_bbox'][:, :, 3:6]
        size_assign = torch.gather(size_gt, 1, (object_assignment[:, :, None].repeat(1, 1, 3)))
        
        size_init_pps = end_points['aggregated_vote_xyz'+mode+'size']
        
        size_dist = torch.norm((size_init_pps / size_assign - 1), dim=2)

        objectness_label[(euclidean_dist1<NEAR_THRESHOLD) & \
                         (size_dist < SIZE_THRESHOLD)] = 1
        objectness_mask[(euclidean_dist1<NEAR_THRESHOLD) & \
                         (size_dist < SIZE_THRESHOLD)] = 1
        objectness_mask[(euclidean_dist1>FAR_THRESHOLD) & \
                         (size_dist > SIZE_FAR_THRESHOLD)] = 1
    else:
        objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1    
        objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1
    '''
    #if mode == 'refine':
    if 'refine' in mode:
        ###Pairwise on Geometry
        objectness_label[(euclidean_dist1[:,:K]<NEAR_COMB_THRESHOLD)*(euclidean_dist1[:,K:]<NEAR_COMB_THRESHOLD)] = 1
        
        objectness_mask[(euclidean_dist1[:,:K]<NEAR_COMB_THRESHOLD)*(euclidean_dist1[:,K:]<NEAR_COMB_THRESHOLD)] = 1
        objectness_mask[(euclidean_dist1[:,:K]>FAR_COMB_THRESHOLD)*(euclidean_dist1[:,K:]>FAR_COMB_THRESHOLD)] = 1

        ### Remove the self idx
        '''
        idx = end_points['temp'+'idx']

        check_tensor = torch.tensor([np.stack([np.arange(0,K//N)]*N, -1)]*B).cuda()
        check_diag = (1.0 - (idx.float() == check_tensor.float()).float())

        objectness_mask = (objectness_mask.view(B,K//N,N).contiguous() * check_diag).view(B, K).contiguous()
        '''
        
        end_points['gt_mrf'] = objectness_label.view(B,-1,N)
        
        '''
        #sem_gt = end_points['sem_cls_label']
        #sem_assign = torch.gather(sem_gt, 1, object_assignment)
        if mode == 'refine1':
            objectness_mask_sub[euclidean_dist1[:,:K]<NEAR_COMB_THRESHOLD] = 1
            objectness_mask_sub[euclidean_dist1[:,:K]<=FAR_COMB_THRESHOLD] = 1
        else:
            objectness_label_sub[(euclidean_dist1<NEAR_COMB_THRESHOLD) &
                                 (sem_assign == aggregated_vote_sem)] = 1
            objectness_mask_sub[euclidean_dist1[:,:K]<NEAR_COMB_THRESHOLD] = 1
        #objectness_mask_sub[euclidean_dist1[:,:K]<NEAR_COMB_THRESHOLD] = 1
        #objectness_mask_sub[euclidean_dist1[:,:K]>FAR_COMB_THRESHOLD] = 1

        objectness_label_mat1 = objectness_label_sub.unsqueeze(-1).repeat(1,1,N)
        objectness_label_mat2 = objectness_label_sub.unsqueeze(-2).repeat(1,N,1)
        objectness_label_mat = objectness_label_mat1 * objectness_label_mat2
        end_points['gt_mrf'] = objectness_label_mat
        
        objectness_label1 = objectness_label_sub.unsqueeze(-1).repeat(1,1,N).view(B, K).contiguous()
        objectness_label2 = objectness_label_sub.unsqueeze(-2).repeat(1,N,1).view(B, K).contiguous()
        objectness_label = objectness_label1 * objectness_label2

        objectness_mask1 = objectness_mask_sub.unsqueeze(-1).repeat(1,1,N).view(B, K).contiguous()
        objectness_mask2 = objectness_mask_sub.unsqueeze(-2).repeat(1,N,1).view(B, K).contiguous()
        objectness_mask = objectness_mask1 * objectness_mask2
        '''
        '''
        objectness_label2[euclidean_dist1[:,K:]<NEAR_COMB_THRESHOLD] = 1
        
        objectness_label1 = torch.zeros((B,K), dtype=torch.long).cuda()
        objectness_label2 = torch.zeros((B,K), dtype=torch.long).cuda()
        objectness_mask1 = torch.zeros((B,K)).cuda()
        objectness_mask2 = torch.zeros((B,K)).cuda()
   
        objectness_label = objectness_label1 * objectness_label2
        objectness_mask = objectness_mask1 * objectness_mask2
        '''
        
        #objectness_mask_orig = torch.eye(K//N).cuda()
        #objectness_mask_orig = (1 - objectness_mask_orig.unsqueeze(0).repeat(B,1,1)).view(B, K).contiguous()

        #objectness_mask = objectness_mask * objectness_mask_orig
        #objectness_mask[euclidean_dist1<NEAR_COMB_THRESHOLD] = 1
        #objectness_mask[euclidean_dist1>FAR_COMB_THRESHOLD] = 1
    else:
        objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1    
        objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1
        
    # Compute objectness loss
    if 'refine' in mode:
        objectness_scores = end_points['objectness_scores'+'refine']
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS_REFINE).cuda(), reduction='none')
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
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    if mode == '_z':
        seed_inds_expand_sem = seed_inds.view(batch_size,num_seed,1).repeat(1,1,6*GT_VOTE_FACTOR)
    elif mode == '_xy':
        seed_inds_expand_sem = seed_inds.view(batch_size,num_seed,1).repeat(1,1,5*GT_VOTE_FACTOR)
    else:
        seed_inds_expand_sem = seed_inds.view(batch_size,num_seed,1).repeat(1,1,4*GT_VOTE_FACTOR)
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
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_proposal, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
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
            seed_gt_votes_reshape = seed_gt_sem[:,:,3:5].view(batch_size*num_proposal, GT_VOTE_FACTOR, 2).contiguous() # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
        else:
            size_xyz_reshape = size_xyz.view(batch_size*num_proposal, -1, 1).contiguous()# from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
            seed_gt_votes_reshape = seed_gt_sem[:,:,3:4].view(batch_size*num_proposal, GT_VOTE_FACTOR, 1).contiguous() # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
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
    objectness_loss_surface = criterion(matching_scores_surface.transpose(2,1), objectness_label_surfacel)
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
    objectness_loss_line = criterion(matching_scores_line.transpose(2,1), objectness_label_line)
    objectness_loss_line = torch.sum(objectness_loss_line * objectness_mask_line)/(torch.sum(objectness_mask_line)+1e-6)

    matching_loss = objectness_loss_surface + objectness_loss_line

    return matching_loss




def get_loss(inputs, end_points, config, net=None):
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
    end_points['sem_loss_line'] = sem_loss_line*2

    vote_loss_z = compute_surface_center_loss(end_points, mode='_z')*10
    end_points['vote_loss_z'] = vote_loss_z

    vote_loss_xy = compute_surface_center_loss(end_points, mode='_xy')*10
    end_points['vote_loss_xy'] = vote_loss_xy

    vote_loss_line = compute_surface_center_loss(end_points, mode='_line')*10
    end_points['vote_loss_line'] = vote_loss_line*2

    center_lossz, size_lossz, sem_lossz = compute_boxsem_loss(end_points, config, mode='_z')
    end_points['center_lossz'] = center_lossz
    end_points['size_lossz'] = size_lossz
    end_points['sem_lossz'] = sem_lossz
    end_points['surface_lossz'] = center_lossz*3 + size_lossz*3 + sem_lossz

    center_lossxy, size_lossxy, sem_lossxy = compute_boxsem_loss(end_points, config, mode='_xy')
    end_points['center_lossxy'] = center_lossxy
    end_points['size_lossxy'] = size_lossxy
    end_points['sem_lossxy'] = sem_lossxy
    end_points['surface_lossxy'] = center_lossxy*3 + size_lossxy*3 + sem_lossxy

    center_lossline, size_lossline, sem_lossline = compute_boxsem_loss(end_points, config, mode='_line')
    end_points['center_lossline'] = center_lossline
    end_points['size_lossline'] = size_lossline
    end_points['sem_lossline'] = sem_lossline
    end_points['surface_lossline'] = center_lossline*3 + size_lossline*3 + sem_lossline
    
    end_points['loss'] = sem_loss_z + sem_loss_xy + sem_loss_line + vote_loss_z + vote_loss_xy + vote_loss_line + end_points['surface_lossz'] + end_points['surface_lossxy'] + end_points['surface_lossline']*2
    return end_points['loss'], end_points
    ### New loss here
    '''
    sem_loss = compute_sem_cls_loss(end_points)*10 # torch.tensor(0)#compute_sem_cls_loss(end_points)*10
    end_points['sem_loss'] = sem_loss
    normal_loss = compute_objcue_vote_loss(end_points, mode='normal')*10
    end_points['normal_loss'] = normal_loss
    loss = sem_loss + normal_loss
    end_points['loss'] = loss
    return loss, end_points
    '''
    
    # Compute support vote loss
    if True:#end_points['use_objcue']:
        vote_loss_corner1 = compute_objcue_vote_loss(end_points, mode='corner1')*10
        vote_loss_corner2 = compute_objcue_vote_loss(end_points, mode='corner2')*10
        end_points['vote_loss_corner1'] = vote_loss_corner1
        end_points['vote_loss_corner2'] = vote_loss_corner2
        corner_cueloss, corner_cuelabel, corner_cuemask = compute_cueness_loss(end_points, mode='corner')
        corner_cueloss *= 10
        end_points['vote_loss_corner_cue'] = corner_cueloss
        end_points['corner_cue'] = corner_cuelabel
        vote_loss_plane = compute_objcue_vote_loss(end_points, mode='plane')*10
        end_points['vote_loss_plane'] = vote_loss_plane
        sem_loss = compute_sem_cls_loss(end_points)*30 # torch.tensor(0)#compute_sem_cls_loss(end_points)*10
        end_points['sem_loss'] = sem_loss
        ## get the voxel loss here
        
        if inputs['sunrgbd']:
            voxel_center_loss_focal = 0
        else:
            voxel_center_loss_focal = compute_voxel_loss(end_points['vox_pred1'], end_points['vox_center'], w9_fcen, w8_fcen)
        voxel_center_loss_l2 = compute_voxel_l2_loss(end_points['vox_pred1'], end_points['vox_center'], w9_cen, w8_cen, w5_cen)
        #end_points['voxel_loss_center'] = wfocal*voxel_center_loss_focal + wl2*voxel_center_loss_l2
        end_points['voxel_loss_center'] = wl2*voxel_center_loss_l2
        '''
        if inputs['sunrgbd']:
            voxel_corner_loss_focal = 0
        else:
            voxel_corner_loss_focal = compute_voxel_loss(end_points['vox_pred2'], end_points['vox_corner'], w9_fcor, w8_fcor)
        voxel_corner_loss_l2 = compute_voxel_l2_loss(end_points['vox_pred2'], end_points['vox_corner'], w9_cor, w8_cor, w5_cor)
        end_points['voxel_loss_corner'] = wfocal*voxel_corner_loss_focal + wl2*voxel_corner_loss_l2

        end_points['voxel_loss'] =  end_points['voxel_loss_center']*wcenter + end_points['voxel_loss_corner']*wcorner
        '''
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

    if True:#end_points['use_plane']:
        plane_loss, plane_offset_loss, plane_reg_loss = compute_plane_loss(end_points)
        #plane_z_loss, z_angle, z_res, z_sign, z_off0, z_off1 = compute_plane_loss(end_points, mode='z')
        #plane_x_loss, x_angle, x_res, x_sign, x_off0, x_off1 = compute_plane_loss(end_points, mode='x')
        #plane_y_loss, y_angle, y_res, y_sign, y_off0, y_off1 = compute_plane_loss(end_points, mode='y')

        end_points['plane_offset_loss'] = plane_offset_loss
        end_points['plane_reg_loss'] = plane_reg_loss
        #end_points['plane_x_loss'] = plane_x_loss
        #end_points['plane_x_loss_angle'] = x_angle
        #end_points['plane_x_loss_res'] = x_res
        #end_points['plane_x_loss_sign'] = x_sign
        #end_points['plane_x_loss_off0'] = x_off0
        #end_points['plane_x_loss_off1'] = x_off1

        #end_points['plane_y_loss'] = plane_y_loss
        #end_points['plane_y_loss_angle'] = y_angle
        #end_points['plane_y_loss_res'] = y_res
        #end_points['plane_y_loss_sign'] = y_sign
        #end_points['plane_y_loss_off0'] = y_off0
        #end_points['plane_y_loss_off1'] = y_off1

        #end_points['plane_z_loss'] = plane_z_loss
        #end_points['plane_z_loss_angle'] = z_angle
        #end_points['plane_z_loss_res'] = z_res
        #end_points['plane_z_loss_sign'] = z_sign
        #end_points['plane_z_loss_off0'] = z_off0
        #end_points['plane_z_loss_off1'] = z_off1
    else:
        end_points['plane_upper_loss'] = torch.tensor(0)
        end_points['plane_lower_loss'] = torch.tensor(0)
        end_points['plane_left_loss'] = torch.tensor(0)
        end_points['plane_right_loss'] = torch.tensor(0)
        end_points['plane_front_loss'] = torch.tensor(0)
        end_points['plane_back_loss'] = torch.tensor(0)

    if inputs['opt_proposal']:
        loss = compute_potential_loss(net, end_points)
        end_points['proposal_loss'] = loss
        return loss, end_points
    
    if True:#end_points['use_plane']:
        #loss_plane = plane_upper_loss + plane_lower_loss + plane_left_loss + plane_right_loss + plane_front_loss + plane_back_loss
        loss_plane = plane_loss# + plane_y_loss + plane_z_loss
        loss_plane *= 10
        end_points['loss_plane'] = loss_plane

        #loss_plane_corner = compute_corner_plane_loss(end_points)
        #end_points['loss_plane_corner'] = loss_plane_corner*50
        #loss_plane_center = compute_center_plane_loss(end_points)
        #end_points['loss_plane_center'] = loss_plane_center*50

        #corner_reg = compute_corner_center_loss(end_points)*50
        #end_points["corner_reg_loss"] = corner_reg
        loss = loss_plane + (vote_loss_corner1 + vote_loss_corner2)/2.0 + end_points['voxel_loss_center'] + sem_loss# + corner_cueloss + 0*corner_reg + 0*end_points['voxel_loss'] #+ 5*loss_plane_corner + 5*loss_plane_center
        end_points['loss'] = loss
        #return loss, end_points

    ### Init Proposal loss
    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    center_cueloss, center_cuelabel, center_cuemask = compute_cueness_loss(end_points, mode='center')
    #center_cueloss *= 10
    end_points['vote_loss_center_cue'] = center_cueloss
    end_points['center_cue'] = corner_cuelabel

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

    objectness_losscorner, objectness_labelcorner, objectness_maskcorner, object_assignmentcorner = \
                                                                                compute_objectness_loss(end_points, mode='corner')
    end_points['objectness_loss'+'corner'] = objectness_losscorner
    end_points['objectness_label'+'corner'] = objectness_labelcorner
    end_points['objectness_mask'+'corner'] = objectness_maskcorner
    end_points['object_assignment'+'corner'] = object_assignmentcorner
    total_num_proposalcorner = objectness_labelcorner.shape[0]*objectness_labelcorner.shape[1]
    end_points['pos_ratiocorner'] = \
                              torch.sum(objectness_labelcorner.float().cuda())/float(total_num_proposalcorner)
    end_points['neg_ratiocorner'] = \
                              torch.sum(objectness_maskcorner.float())/float(total_num_proposalcorner) - end_points['pos_ratiocorner']

    objectness_lossplane, objectness_labelplane, objectness_maskplane, object_assignmentplane = \
                                                                                compute_objectness_loss(end_points, mode='plane')
    end_points['objectness_loss'+'plane'] = objectness_lossplane
    end_points['objectness_label'+'plane'] = objectness_labelplane
    end_points['objectness_mask'+'plane'] = objectness_maskplane
    end_points['object_assignment'+'plane'] = object_assignmentplane
    total_num_proposalplane = objectness_labelplane.shape[0]*objectness_labelplane.shape[1]
    end_points['pos_ratioplane'] = \
                              torch.sum(objectness_labelplane.float().cuda())/float(total_num_proposalplane)
    end_points['neg_ratioplane'] = \
                              torch.sum(objectness_maskplane.float())/float(total_num_proposalplane) - end_points['pos_ratioplane']

    objectness_losscomb, objectness_labelcomb, objectness_maskcomb, object_assignmentcomb = \
                                                                                compute_objectness_loss(end_points, mode='comb')
    end_points['objectness_loss'+'comb'] = objectness_losscomb
    end_points['objectness_label'+'comb'] = objectness_labelcomb
    end_points['objectness_mask'+'comb'] = objectness_maskcomb
    end_points['object_assignment'+'comb'] = object_assignmentcomb
    total_num_proposalcomb = objectness_labelcomb.shape[0]*objectness_labelcomb.shape[1]
    end_points['pos_ratiocomb'] = \
                              torch.sum(objectness_labelcomb.float().cuda())/float(total_num_proposalcomb)
    end_points['neg_ratiocomb'] = \
                              torch.sum(objectness_maskcomb.float())/float(total_num_proposalcomb) - end_points['pos_ratiocomb']
    '''
    objectness_lossrefine1, objectness_labelrefine1, objectness_maskrefine1, object_assignmentrefine1 = \
                                                                                compute_objectness_loss(end_points, mode='refine1')
    end_points['objectness_loss'+'refine1'] = objectness_lossrefine1
    end_points['objectness_label'+'refine1'] = objectness_labelrefine1
    end_points['objectness_mask'+'refine1'] = objectness_maskrefine1
    end_points['object_assignment'+'refine1'] = object_assignmentrefine1
    total_num_proposalrefine1 = objectness_labelrefine1.shape[0]*objectness_labelrefine1.shape[1]
    end_points['pos_ratiorefine1'] = \
                              torch.sum(objectness_labelrefine1.float().cuda())/float(total_num_proposalrefine1)
    end_points['neg_ratiorefine1'] = \
                              torch.sum(objectness_maskrefine1.float())/float(total_num_proposalrefine1) - end_points['pos_ratiorefine1']
    
    objectness_lossrefine2, objectness_labelrefine2, objectness_maskrefine2, object_assignmentrefine2 = \
                                                                                compute_objectness_loss(end_points, mode='refine2')
    end_points['objectness_loss'+'refine2'] = objectness_lossrefine2
    end_points['objectness_label'+'refine2'] = objectness_labelrefine2
    end_points['objectness_mask'+'refine2'] = objectness_maskrefine2
    end_points['object_assignment'+'refine2'] = object_assignmentrefine2
    total_num_proposalrefine2 = objectness_labelrefine2.shape[0]*objectness_labelrefine2.shape[1]
    end_points['pos_ratiorefine2'] = \
                              torch.sum(objectness_labelrefine2.float().cuda())/float(total_num_proposalrefine2)
    end_points['neg_ratiorefine2'] = \
                              torch.sum(objectness_maskrefine2.float())/float(total_num_proposalrefine2) - end_points['pos_ratiorefine2']
    '''
    # Obj loss finetune
    '''
    objectness_loss_opt, objectness_label_opt, objectness_mask_opt, object_assignment_opt = \
                                                                                compute_objectness_loss(end_points, mode='opt')
    end_points['objectness_loss'+'opt'] = objectness_loss_opt
    end_points['objectness_label'+'opt'] = objectness_label_opt
    end_points['objectness_mask'+'opt'] = objectness_mask_opt
    end_points['object_assignment'+'opt'] = object_assignment_opt
    total_num_proposal_opt = objectness_label_opt.shape[0]*objectness_label_opt.shape[1]
    end_points['pos_ratio_opt'] = \
                              torch.sum(objectness_label_opt.float().cuda())/float(total_num_proposal_opt)
    end_points['neg_ratio_opt'] = \
                              torch.sum(objectness_mask_opt.float())/float(total_num_proposal_opt) - end_points['pos_ratio_opt']
    #objectness_reg_loss = compute_objectness_reg_loss(end_points)
    '''
    
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

    center_losscorner, heading_cls_losscorner, heading_reg_losscorner, size_cls_losscorner, size_reg_losscorner, sem_cls_losscorner = \
        compute_box_and_sem_cls_loss(end_points, config, mode='corner')
    end_points['center_losscorner'] = center_losscorner
    end_points['heading_cls_losscorner'] = heading_cls_losscorner
    end_points['heading_reg_losscorner'] = heading_reg_losscorner
    end_points['size_cls_losscorner'] = size_cls_losscorner
    end_points['size_reg_losscorner'] = size_reg_losscorner
    end_points['sem_cls_losscorner'] = sem_cls_losscorner
    box_losscorner = center_losscorner + 0.1*heading_cls_losscorner + heading_reg_losscorner + 0.1*size_cls_losscorner + size_reg_losscorner
    end_points['box_losscorner'] = box_losscorner

    center_lossplane, heading_cls_lossplane, heading_reg_lossplane, size_cls_lossplane, size_reg_lossplane, sem_cls_lossplane = \
        compute_box_and_sem_cls_loss(end_points, config, mode='plane')
    end_points['center_lossplane'] = center_lossplane
    end_points['heading_cls_lossplane'] = heading_cls_lossplane
    end_points['heading_reg_lossplane'] = heading_reg_lossplane
    end_points['size_cls_lossplane'] = size_cls_lossplane
    end_points['size_reg_lossplane'] = size_reg_lossplane
    end_points['sem_cls_lossplane'] = sem_cls_lossplane
    box_lossplane = center_lossplane + 0.1*heading_cls_lossplane + heading_reg_lossplane + 0.1*size_cls_lossplane + size_reg_lossplane
    end_points['box_lossplane'] = box_lossplane

    center_losscomb, heading_cls_losscomb, heading_reg_losscomb, size_cls_losscomb, size_reg_losscomb, sem_cls_losscomb = \
        compute_box_and_sem_cls_loss(end_points, config, mode='comb')
    end_points['center_losscomb'] = center_losscomb
    end_points['heading_cls_losscomb'] = heading_cls_losscomb
    end_points['heading_reg_losscomb'] = heading_reg_losscomb
    end_points['size_cls_losscomb'] = size_cls_losscomb
    end_points['size_reg_losscomb'] = size_reg_losscomb
    end_points['sem_cls_losscomb'] = sem_cls_losscomb
    box_losscomb = center_losscomb + 0.1*heading_cls_losscomb + heading_reg_losscomb + 0.1*size_cls_losscomb + size_reg_losscomb
    end_points['box_losscomb'] = box_losscomb

    '''
    center_lossrefine, heading_cls_lossrefine, heading_reg_lossrefine, size_cls_lossrefine, size_reg_lossrefine, sem_cls_lossrefine = \
        compute_box_and_sem_cls_loss(end_points, config, mode='refine')
    end_points['center_lossrefine'] = center_lossrefine
    end_points['heading_cls_lossrefine'] = heading_cls_lossrefine
    end_points['heading_reg_lossrefine'] = heading_reg_lossrefine
    end_points['size_cls_lossrefine'] = size_cls_lossrefine
    end_points['size_reg_lossrefine'] = size_reg_lossrefine
    end_points['sem_cls_lossrefine'] = sem_cls_lossrefine
    box_lossrefine = center_lossrefine + 0.1*heading_cls_lossrefine + heading_reg_lossrefine + 0.1*size_cls_lossrefine + size_reg_lossrefine
    end_points['box_lossrefine'] = box_lossrefine
    '''
    # Box loss and sem cls loss finetune
    '''
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
    '''
    # Final loss function
    proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.5*objectness_losscorner + box_losscorner + 0.1*sem_cls_losscorner + 0.5*objectness_lossplane + box_lossplane + 0.1*sem_cls_lossplane + 0.5*objectness_losscomb + box_losscomb + 0.1*sem_cls_losscomb# + 0.5*2*objectness_lossrefine2# + 0.5*2*objectness_lossrefine2 # + box_lossrefine + 0.1*sem_cls_lossrefine
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
    loss += proposalloss
    end_points['init_proposal_loss'] = proposalloss
    end_points['loss'] = loss ### Add the initial proposal loss term
        
    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'+'center'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc_center'] = obj_acc

    obj_pred_val = torch.argmax(end_points['objectness_scores'+'corner'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_labelcorner.long()).float()*objectness_maskcorner)/(torch.sum(objectness_maskcorner)+1e-6)
    end_points['obj_acc_corner'] = obj_acc

    obj_pred_val = torch.argmax(end_points['objectness_scores'+'plane'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_labelplane.long()).float()*objectness_maskplane)/(torch.sum(objectness_maskplane)+1e-6)
    end_points['obj_acc_plane'] = obj_acc

    obj_pred_val = torch.argmax(end_points['objectness_scores'+'comb'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_labelcomb.long()).float()*objectness_maskcomb)/(torch.sum(objectness_maskcomb)+1e-6)
    end_points['obj_acc_comb'] = obj_acc

    '''
    obj_pred_val = torch.argmax(end_points['objectness_scores'+'refine'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_labelrefine2.long()).float()*objectness_maskrefine2)/(torch.sum(objectness_maskrefine2)+1e-6)
    end_points['obj_acc_refine2'] = obj_acc
    '''
    return loss, end_points
