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

def compute_support_vote_loss(end_points):
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

    #vote_xyz_support = end_points['vote_xyz_support'] # B,num_seed*vote_factor,3
    #vote_xyz_bsupport = end_points['vote_xyz_bsupport'] # B,num_seed*vote_factor,3
    vote_xyz_support_center = end_points['vote_xyz_support_center'] # B,num_seed*vote_factor,3
    vote_xyz_support_offset = end_points['vote_xyz_support_offset'] # B,num_seed*vote_factor,3
    vote_xyz_bsupport_center = end_points['vote_xyz_bsupport_center'] # B,num_seed*vote_factor,3
    vote_xyz_bsupport_offset = end_points['vote_xyz_bsupport_offset'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    
    seed_gt_votes_mask_support = torch.gather(end_points['vote_label_mask_support'], 1, seed_inds)
    #seed_gt_votes_support = torch.gather(end_points['vote_label_support'], 1, seed_inds_expand)
    seed_gt_votes_support_center = torch.gather(end_points['vote_label_support_middle'], 1, seed_inds_expand)
    seed_gt_votes_support_offset = torch.gather(end_points['vote_label_support_offset'], 1, seed_inds_expand)
    seed_gt_votes_support_center += end_points['seed_xyz'].repeat(1,1,3)
    seed_gt_votes_support_offset += end_points['seed_xyz'].repeat(1,1,3)

    seed_gt_votes_mask_bsupport = torch.gather(end_points['vote_label_mask_bsupport'], 1, seed_inds)
    #seed_gt_votes_bsupport = torch.gather(end_points['vote_label_bsupport'], 1, seed_inds_expand)
    seed_gt_votes_bsupport_center = torch.gather(end_points['vote_label_bsupport_middle'], 1, seed_inds_expand)
    seed_gt_votes_bsupport_offset = torch.gather(end_points['vote_label_bsupport_offset'], 1, seed_inds_expand)
    seed_gt_votes_bsupport_center += end_points['seed_xyz'].repeat(1,1,3)
    seed_gt_votes_bsupport_offset += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    """
    vote_xyz_reshape_support = vote_xyz_support.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape_support = seed_gt_votes_support.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1_support, _, dist2_support, _ = nn_distance(vote_xyz_reshape_support, seed_gt_votes_reshape_support, l1=True)
    votes_dist_support, _ = torch.min(dist2_support, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist_support = votes_dist_support.view(batch_size, num_seed)
    vote_loss_support = torch.sum(votes_dist_support*seed_gt_votes_mask_support.float())/(torch.sum(seed_gt_votes_mask_support.float())+1e-6)
    """
    vote_xyz_reshape_support_center = vote_xyz_support_center.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape_support_center = seed_gt_votes_support_center.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1_support_center, _, dist2_support_center, _ = nn_distance(vote_xyz_reshape_support_center, seed_gt_votes_reshape_support_center, l1=True)
    votes_dist_support_center, _ = torch.min(dist2_support_center, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist_support_center = votes_dist_support_center.view(batch_size, num_seed)
    vote_loss_support_center = torch.sum(votes_dist_support_center*seed_gt_votes_mask_support.float())/(torch.sum(seed_gt_votes_mask_support.float())+1e-6)
    vote_xyz_reshape_support_offset = vote_xyz_support_offset.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape_support_offset = seed_gt_votes_support_offset.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1_support_offset, _, dist2_support_offset, _ = nn_distance(vote_xyz_reshape_support_offset, seed_gt_votes_reshape_support_offset, l1=True)
    votes_dist_support_offset, _ = torch.min(dist2_support_offset, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist_support_offset = votes_dist_support_offset.view(batch_size, num_seed)
    vote_loss_support_offset = torch.sum(votes_dist_support_offset*seed_gt_votes_mask_support.float())/(torch.sum(seed_gt_votes_mask_support.float())+1e-6)
    vote_loss_support = vote_loss_support_center + vote_loss_support_offset
    
    # Compute the min of min of distance
    """
    vote_xyz_reshape_bsupport = vote_xyz_bsupport.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape_bsupport = seed_gt_votes_bsupport.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1_bsupport, _, dist2_bsupport, _ = nn_distance(vote_xyz_reshape_bsupport, seed_gt_votes_reshape_bsupport, l1=True)
    votes_dist_bsupport, _ = torch.min(dist2_bsupport, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist_bsupport = votes_dist_bsupport.view(batch_size, num_seed)
    vote_loss_bsupport = torch.sum(votes_dist_bsupport*seed_gt_votes_mask_bsupport.float())/(torch.sum(seed_gt_votes_mask_bsupport.float())+1e-6)
    """
    vote_xyz_reshape_bsupport_center = vote_xyz_bsupport_center.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape_bsupport_center = seed_gt_votes_bsupport_center.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1_bsupport_center, _, dist2_bsupport_center, _ = nn_distance(vote_xyz_reshape_bsupport_center, seed_gt_votes_reshape_bsupport_center, l1=True)
    votes_dist_bsupport_center, _ = torch.min(dist2_bsupport_center, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist_bsupport_center = votes_dist_bsupport_center.view(batch_size, num_seed)
    vote_loss_bsupport_center = torch.sum(votes_dist_bsupport_center*seed_gt_votes_mask_bsupport.float())/(torch.sum(seed_gt_votes_mask_bsupport.float())+1e-6)
    vote_xyz_reshape_bsupport_offset = vote_xyz_bsupport_offset.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape_bsupport_offset = seed_gt_votes_bsupport_offset.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1_bsupport_offset, _, dist2_bsupport_offset, _ = nn_distance(vote_xyz_reshape_bsupport_offset, seed_gt_votes_reshape_bsupport_offset, l1=True)
    votes_dist_bsupport_offset, _ = torch.min(dist2_bsupport_offset, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist_bsupport_offset = votes_dist_bsupport_offset.view(batch_size, num_seed)
    vote_loss_bsupport_offset = torch.sum(votes_dist_bsupport_offset*seed_gt_votes_mask_bsupport.float())/(torch.sum(seed_gt_votes_mask_bsupport.float())+1e-6)
    vote_loss_bsupport = vote_loss_bsupport_center + vote_loss_bsupport_offset
    return vote_loss_support, vote_loss_bsupport

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
    if end_points['use_support']:
        use_support = 1
    else:
        use_support = 0
    
    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss
    
    # Compute support vote loss
    if end_points['use_support']:
        vote_loss_support, vote_loss_bsupport = compute_support_vote_loss(end_points)
        end_points['vote_loss_support'] = vote_loss_support
        end_points['vote_loss_bsupport'] = vote_loss_bsupport
    else:
        support_vote_loss, bsupport_vote_loss = 0,0
        end_points['vote_loss_support'] = torch.tensor(0)
        end_points['vote_loss_bsupport'] = torch.tensor(0)
        
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
    if end_points['use_support']:
        objectness_loss_support, objectness_label_support, objectness_mask_support, object_assignment_support = \
                                                                                compute_objectness_loss(end_points, mode='_support')
        end_points['objectness_loss_support'] = objectness_loss_support
        end_points['objectness_label_support'] = objectness_label_support
        end_points['objectness_mask_support'] = objectness_mask_support
        end_points['object_assignment_support'] = object_assignment_support
        total_num_proposal_support = objectness_label_support.shape[0]*objectness_label_support.shape[1]
        end_points['pos_ratio_support'] = \
                                  torch.sum(objectness_label_support.float().cuda())/float(total_num_proposal_support)
        end_points['neg_ratio_support'] = \
                                  torch.sum(objectness_mask_support.float())/float(total_num_proposal_support) - end_points['pos_ratio_support']
        objectness_loss_bsupport, objectness_label_bsupport, objectness_mask_bsupport, object_assignment_bsupport = \
                                                                                compute_objectness_loss(end_points, mode='_bsupport')
        end_points['objectness_loss_bsupport'] = objectness_loss_bsupport
        end_points['objectness_label_bsupport'] = objectness_label_bsupport
        end_points['objectness_mask_bsupport'] = objectness_mask_bsupport
        end_points['object_assignment_bsupport'] = object_assignment_bsupport
        total_num_proposal_bsupport = objectness_label_bsupport.shape[0]*objectness_label_bsupport.shape[1]
        end_points['pos_ratio_bsupport'] = \
                                  torch.sum(objectness_label_bsupport.float().cuda())/float(total_num_proposal_bsupport)
        end_points['neg_ratio_bsupport'] = \
                                  torch.sum(objectness_mask_bsupport.float())/float(total_num_proposal_bsupport) - end_points['pos_ratio_bsupport']

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
    if end_points['use_support']:
        center_loss_support, heading_cls_loss_support, heading_reg_loss_support, size_cls_loss_support, size_reg_loss_support, sem_cls_loss_support = \
        compute_box_and_sem_cls_loss(end_points, config, mode='_support')
        end_points['center_loss_support'] = center_loss_support
        end_points['heading_cls_loss_support'] = heading_cls_loss_support
        end_points['heading_reg_loss_support'] = heading_reg_loss_support
        end_points['size_cls_loss_support'] = size_cls_loss_support
        end_points['size_reg_loss_support'] = size_reg_loss_support
        end_points['sem_cls_loss_support'] = sem_cls_loss_support
        box_loss_support = center_loss_support + 0.1*heading_cls_loss_support + heading_reg_loss_support + 0.1*size_cls_loss_support + size_reg_loss_support
        end_points['box_loss_support'] = box_loss_support
        center_loss_bsupport, heading_cls_loss_bsupport, heading_reg_loss_bsupport, size_cls_loss_bsupport, size_reg_loss_bsupport, sem_cls_loss_bsupport = \
        compute_box_and_sem_cls_loss(end_points, config, mode='_bsupport')
        end_points['center_loss_bsupport'] = center_loss_bsupport
        end_points['heading_cls_loss_bsupport'] = heading_cls_loss_bsupport
        end_points['heading_reg_loss_bsupport'] = heading_reg_loss_bsupport
        end_points['size_cls_loss_bsupport'] = size_cls_loss_bsupport
        end_points['size_reg_loss_bsupport'] = size_reg_loss_bsupport
        end_points['sem_cls_loss_bsupport'] = sem_cls_loss_bsupport
        box_loss_bsupport = center_loss_bsupport + 0.1*heading_cls_loss_bsupport + heading_reg_loss_bsupport + 0.1*size_cls_loss_bsupport + size_reg_loss_bsupport
        end_points['box_loss_bsupport'] = box_loss_bsupport
    
    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss
    if end_points['use_support']:
        loss_support = vote_loss_support + 0.5*objectness_loss_support + box_loss_support + 0.1*sem_cls_loss_support
        loss_support *= 10
        end_points['loss_support'] = loss_support
        loss_bsupport = vote_loss_bsupport + 0.5*objectness_loss_bsupport + box_loss_bsupport + 0.1*sem_cls_loss_bsupport
        loss_bsupport *= 10
        end_points['loss_bsupport'] = loss_bsupport
        
    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc
    if end_points['use_support']:
        obj_pred_val_support = torch.argmax(end_points['objectness_scores_support'], 2) # B,K
        obj_acc_support = torch.sum((obj_pred_val_support==objectness_label_support.long()).float()*objectness_mask_support)/(torch.sum(objectness_mask_support)+1e-6)
        end_points['obj_acc_support'] = obj_acc_support
        obj_pred_val_bsupport = torch.argmax(end_points['objectness_scores_bsupport'], 2) # B,K
        obj_acc_bsupport = torch.sum((obj_pred_val_bsupport==objectness_label_bsupport.long()).float()*objectness_mask_bsupport)/(torch.sum(objectness_mask_bsupport)+1e-6)
        end_points['obj_acc_bsupport'] = obj_acc_bsupport

    if end_points['use_support']:
        loss = loss + loss_support + loss_bsupport
    return loss, end_points
