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
from box_util import get_surface_line_points_batch_pytorch

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3

MASK_SURFACE_THRESHOLD = 0.3
LABEL_SURFACE_THRESHOLD = 0.3
MASK_LINE_THRESHOLD = 0.3
LABEL_LINE_THRESHOLD = 0.3

GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness
SEM_CLS_WEIGHTS = [0.4,0.6] # put larger weights on positive objectness
OBJECTNESS_CLS_WEIGHTS_REFINE = [0.3,0.7] # put larger weights on positive objectness

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

def compute_primitive_center_loss(end_points, mode='_z'):
    """ Compute primitive center loss: similar to votenet clustering
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    #if mode == '_z':
    #    vote_xyz = end_points['vote_z'] # B,num_seed*vote_factor,3
    #else:
    vote_xyz = end_points['vote'+mode] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

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
    seed_gt_votes += end_points['seed_xyz']#.repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, 1, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_proposal_loss(end_points, mode=''):
    """ Compute objectness loss for the initial proposal 
    and also find the initial proposal with detected primitives
    """ 
    # Associate proposal and GT objects by point-to-point distances
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K2 = gt_center.shape[1]
    
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'+'center'] ### Vote xyz is the same for all
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
    
    ### Get the corresponding proposal with detected object proposal
    if mode == 'opt':
        obj_center = torch.gather(end_points['center_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3))
        gt_size = torch.gather(end_points['size_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K) from (B,K2)
        gt_heading = torch.gather(end_points['heading_label'], 1, object_assignment) # select (B,K) from (B,K2)
        gt_sem = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
        end_points['selected_sem'] = gt_sem
        
        ### gt for primitive matching
        obj_surface_center, obj_line_center = get_surface_line_points_batch_pytorch(gt_size, gt_heading, obj_center)

        pred_surface_center = end_points['surface_center_pred']
        pred_line_center = end_points['line_center_pred']

        pred_obj_surface_center = end_points["surface_center_object"]
        pred_obj_line_center = end_points["line_center_object"]

        surface_sem = torch.argmax(end_points['surface_sem_pred'], dim=2).float()
        line_sem = torch.argmax(end_points['line_sem_pred'], dim=2).float()
        
        dist_surface, surface_ind, _, _ = nn_distance(obj_surface_center, pred_surface_center)
        dist_line, line_ind, _, _ = nn_distance(obj_line_center, pred_line_center)

        surface_sel = torch.gather(pred_surface_center, 1, surface_ind.unsqueeze(-1).repeat(1,1,3))
        line_sel = torch.gather(pred_line_center, 1, line_ind.unsqueeze(-1).repeat(1,1,3))
        surface_sel_sem = torch.gather(surface_sem, 1, surface_ind)
        line_sel_sem = torch.gather(line_sem, 1, line_ind)

        surface_sel_sem_gt = gt_sem.unsqueeze(-1).repeat(1,1,6).view(B,-1).float()
        line_sel_sem_gt = gt_sem.unsqueeze(-1).repeat(1,1,12).view(B,-1).float()

        end_points["surface_sel"] = obj_surface_center
        end_points["line_sel"] = obj_line_center
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
        
        euclidean_dist_obj_surface = torch.sqrt(torch.sum((pred_obj_surface_center - surface_sel)**2, dim=-1)+1e-6)
        euclidean_dist_obj_line = torch.sqrt(torch.sum((pred_obj_line_center - line_sel)**2, dim=-1)+1e-6)

    ### Objectness score just with centers
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1 
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    if mode == 'opt':
        objectness_label_surface_obj = objectness_label.repeat(1,6)
        objectness_mask_surface_obj = objectness_mask.repeat(1,6)
        objectness_label_line_obj = objectness_label.repeat(1,12)
        objectness_mask_line_obj = objectness_mask.repeat(1,12)
        
        objectness_mask_surface = objectness_mask_surface_obj
        objectness_mask_line = objectness_mask_line_obj
        objectness_label_surface[(euclidean_dist_obj_surface<LABEL_SURFACE_THRESHOLD)*(euclidean_dist_surface<MASK_SURFACE_THRESHOLD)] = 1
        objectness_label_surface_sem[(euclidean_dist_obj_surface<LABEL_SURFACE_THRESHOLD)*(euclidean_dist_surface<MASK_SURFACE_THRESHOLD)*(surface_sel_sem==surface_sel_sem_gt)] = 1

        objectness_label_line[(euclidean_dist_obj_line<LABEL_LINE_THRESHOLD)*(euclidean_dist_line<MASK_LINE_THRESHOLD)] = 1
        objectness_label_line_sem[(euclidean_dist_obj_line<LABEL_LINE_THRESHOLD)*(euclidean_dist_line<MASK_LINE_THRESHOLD)*(line_sel_sem==line_sel_sem_gt)] = 1

    if mode == 'opt':
        objectness_scores = end_points["match_scores"]#match scores for each geometric primitive
        temp_objectness_label = torch.cat((objectness_label_surface, objectness_label_line), 1)
        temp_objectness_label_sem = torch.cat((objectness_label_surface_sem, objectness_label_line_sem), 1)
        temp_objectness_mask = torch.cat((objectness_mask_surface, objectness_mask_line), 1)
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS_REFINE).cuda(), reduction='none')
        objectness_loss = criterion(objectness_scores.transpose(2,1), temp_objectness_label)
        objectness_loss = torch.sum(objectness_loss * temp_objectness_mask)/(torch.sum(temp_objectness_mask)+1e-6)

        objectness_scores_sem = end_points["match_scores_sem"]#match scores for the semantics of primitives
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS_REFINE).cuda(), reduction='none')
        objectness_loss_sem = criterion(objectness_scores_sem.transpose(2,1), temp_objectness_label_sem)
        objectness_loss_sem = torch.sum(objectness_loss_sem * temp_objectness_mask)/(torch.sum(temp_objectness_mask)+1e-6)

        end_points['objectness_match_label_cue'] = torch.cat((objectness_label_surface, objectness_label_line), 1)
        objectness_label_surface *= objectness_label_surface_obj
        objectness_label_line *= objectness_label_line_obj
        end_points['objectness_match_label_plusscore'] = torch.cat((objectness_label_surface, objectness_label_line), 1)

        #  Optional semantic matching
        # temp_objectness_label_sem = torch.cat((objectness_label_surface_sem, objectness_label_line_sem), 1)
        # objectness_match_mask = (torch.sum(temp_objectness_label_sem.view(B, 18, K), dim=1) >= 1).float()
        # end_points['objectness_match_label_plusscore_sem'] = objectness_match_mask
        
        objectness_label_surface_sem *= objectness_label_surface_obj                                                                         
        objectness_label_line_sem *= objectness_label_line_obj                                                                               
        end_points['objectness_match_label_plusscore_sem'] = torch.cat((objectness_label_surface_sem, objectness_label_line_sem), 1) 


        objectness_scores = end_points['objectness_scores'+mode]
        objectness_match_mask = (torch.sum(temp_objectness_label.view(B, 18, K), dim=1) >= 1).float()
        
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
        objectness_loss_refine = criterion(objectness_scores.transpose(2,1), objectness_label)
        objectness_loss_refine1 = torch.sum(objectness_loss_refine * objectness_match_mask)/(torch.sum(objectness_match_mask)+1e-6)
        objectness_loss_refine2 = torch.sum(objectness_loss_refine * objectness_mask)/(torch.sum(objectness_mask)+1e-6)
        
        return objectness_loss+objectness_loss_sem+(objectness_loss_refine1+objectness_loss_refine2)/2.0, objectness_label, objectness_mask, temp_objectness_label, temp_objectness_label_sem, temp_objectness_mask, object_assignment, objectness_loss, objectness_loss_sem
        
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

    if mode != 'refine':
        object_assignment = end_points['object_assignment'+'center']
    else:
        object_assignment = end_points['object_assignment'+mode]
        #object_assignment = end_points['object_assignment'+'center']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center'+mode]
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    if mode != 'refine':
        objectness_label = end_points['objectness_label'+'center'].float()
    else:
        objectness_label = end_points['objectness_label'+mode].float()
        #objectness_label = end_points['objectness_label'+'center'].float()
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

def compute_matching_potential_loss(end_points, config, mode=''):
    """ Compute potential function loss with computed object proposals
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

    ### Get the heading here

    pred_heading_class = torch.argmax(end_points['heading_scores'+'center'].detach(), -1) # B,num_proposal
    if config.dataset == 'scannet':
        pred_heading = torch.zeros_like(pred_heading_class).float()
    elif config.dataset == 'sunrgbd':
        ''' here
        pred_heading_opt = end_points['heading_residuals'+mode]*(np.pi/float(config.num_heading_bin)) # (8, 256, 12)
        '''
        pred_heading_residual = torch.gather(end_points['heading_residuals'+mode].detach(), 2,
                                             pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_residual.squeeze_(2)
        pred_heading = pred_heading_class.float()*(2*np.pi/float(config.num_heading_bin)) + pred_heading_residual
    else:
        AssertionError('Dataset Config Error!')
    

    size_residual = end_points['size_residuals'+mode]
    size_residual_normalized = end_points['size_residuals_normalized'+mode]
    pred_size_class = torch.argmax(end_points['size_scores'+'center'].contiguous(), -1).detach()
    pred_size_residual = torch.gather(size_residual, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3))
    mean_size_class_batched = torch.ones_like(size_residual) * torch.from_numpy(config.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_avg = torch.gather(mean_size_class_batched, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).detach()
    
    obj_size = pred_size_avg.squeeze(2) + pred_size_residual.squeeze(2)# + size_residual_opt
    
    ### Get the object surface center here
    pred_obj_surface_center, pred_obj_line_center = get_surface_line_points_batch_pytorch(obj_size, pred_heading, obj_center)
    
    source_point = torch.cat((pred_obj_surface_center, pred_obj_line_center), 1)

    surface_target = end_points["surface_sel"]
    line_target = end_points["line_sel"]
    target_point = torch.cat((surface_target, line_target), 1)

    objectness_match_label = end_points['objectness_match_label_plusscore'].float()
    objectness_match_label_sem = end_points['objectness_match_label_plusscore_sem'].float()
        
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'+mode].float()
    centroid_reg_loss1 = torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    dist_match = torch.sqrt(torch.sum((source_point - target_point)**2, dim=-1)+1e-6)
    centroid_reg_loss3 = torch.sum(dist_match*objectness_match_label)/(torch.sum(objectness_match_label)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2 + centroid_reg_loss3
    
    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'+mode].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
    #heading_class_loss = torch.tensor(0)
    
    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    if False:#mode == 'opt':
        heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized'+'center']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
        heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)
    else:
        heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized'+mode]*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
        heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)
    
    ### Compute the original size loss
    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    
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
    # Optional for semantic optimization
    '''
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'+mode].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss1 = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
    sem_cls_loss2 = torch.sum(sem_cls_loss * objectness_match_label_sem)/(torch.sum(objectness_match_label_sem)+1e-6)
    '''

    if config.dataset == 'scannet':
        return centroid_reg_loss1 + centroid_reg_loss2 + centroid_reg_loss3 + size_residual_normalized_loss
    elif config.dataset == 'sunrgbd':
        return centroid_reg_loss1 + centroid_reg_loss2 + centroid_reg_loss3 + size_residual_normalized_loss + heading_residual_normalized_loss
    else:
        AssertionError('Config Error!')


def compute_primitivesem_loss(end_points, config, mode=''):
    """ Compute final geometric primitive center and semantic
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['center'+mode] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

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
    seed_gt_votes += end_points['seed_xyz']

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
    ### Need to remove this soon
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
    end_points['supp_sem'+mode] = sem_cls_label
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'+mode].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    return center_loss, size_loss, sem_cls_loss

def compute_flag_loss(end_points, mode):
    # Compute existence flag for face and edge centers
    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    seed_inds_expand = seed_inds.view(batch_size,num_seed)

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
    num_class = end_points['pred_flag'+mode].shape[1]

    pred_flag = end_points['pred_flag'+mode]
    
    criterion = nn.CrossEntropyLoss(torch.Tensor(SEM_CLS_WEIGHTS).cuda(), reduction='none')
    sem_loss = criterion(pred_flag, sem_cls_label.long())
    sem_loss = torch.mean(sem_loss.float())

    return sem_loss

def get_loss(inputs, end_points, config):
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
    ### Geometric Primitive Prediction

    ### Existence flag pred
    ### Separate 
    flag_loss_z = compute_flag_loss(end_points, mode='_z')*30
    end_points['flag_loss_z'] = flag_loss_z

    flag_loss_xy = compute_flag_loss(end_points, mode='_xy')*30
    end_points['flag_loss_xy'] = flag_loss_xy
    
    flag_loss_line = compute_flag_loss(end_points, mode='_line')*30
    end_points['flag_loss_line'] = flag_loss_line

    vote_loss_z = compute_primitive_center_loss(end_points, mode='_z')*10
    end_points['vote_loss_z'] = vote_loss_z

    vote_loss_xy = compute_primitive_center_loss(end_points, mode='_xy')*10
    end_points['vote_loss_xy'] = vote_loss_xy

    vote_loss_line = compute_primitive_center_loss(end_points, mode='_line')*10
    end_points['vote_loss_line'] = vote_loss_line

    center_lossz, size_lossz, sem_lossz = compute_primitivesem_loss(end_points, config, mode='_z')
    end_points['center_lossz'] = center_lossz
    end_points['size_lossz'] = size_lossz
    end_points['sem_lossz'] = sem_lossz
    end_points['surface_lossz'] = center_lossz*0.5 + size_lossz*0.5 + sem_lossz
        
    center_lossxy, size_lossxy, sem_lossxy = compute_primitivesem_loss(end_points, config, mode='_xy')
    end_points['center_lossxy'] = center_lossxy
    end_points['size_lossxy'] = size_lossxy
    end_points['sem_lossxy'] = sem_lossxy
    end_points['surface_lossxy'] = center_lossxy*0.5 + size_lossxy*0.5 + sem_lossxy

    center_lossline, size_lossline, sem_lossline = compute_primitivesem_loss(end_points, config, mode='_line')
    end_points['center_lossline'] = center_lossline
    end_points['size_lossline'] = size_lossline
    end_points['sem_lossline'] = sem_lossline
    end_points['surface_lossline'] = center_lossline*0.5 + size_lossline*0.5 + sem_lossline
    
    end_points['objcue_loss'] = flag_loss_z + flag_loss_xy + flag_loss_line + vote_loss_z + vote_loss_xy + vote_loss_line + end_points['surface_lossz'] + end_points['surface_lossxy'] + end_points['surface_lossline']*2
    
        
    ### Init Proposal loss
    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
                                                                            compute_proposal_loss(end_points, mode='center')
    end_points['objectness_loss'+'center'] = objectness_loss
    end_points['objectness_label'+'center'] = objectness_label
    end_points['objectness_mask'+'center'] = objectness_mask
    end_points['object_assignment'+'center'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
                              torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
                              torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    objectness_loss_opt, objectness_label_opt, objectness_mask_opt, objectness_label_match, objectness_label_match_sem, objectness_mask_match, object_assignment_opt, objectness_loss_cue, objectness_loss_sem = \
                                                                                                                                                                                                                 compute_proposal_loss(end_points, mode='opt')
    end_points['objectness_loss'+'opt'] = objectness_loss_opt
    end_points['objectness_loss'+'_cue'] = objectness_loss_cue
    end_points['objectness_loss'+'_sem'] = objectness_loss_sem
    
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
                                      torch.sum(((torch.max(objectness_label_match.float().view(objectness_label.shape[0], 18, objectness_label.shape[1]), dim=1)[0])*objectness_label.float()).cuda())/torch.sum(objectness_label.float().cuda())

    end_points['neg_ratio_opt'] = \
                                  torch.sum(objectness_mask_match.float())/float(total_num_proposal_opt) - end_points['pos_ratio_opt']
    end_points['sem_ratio_opt'] = \
                                  torch.sum(objectness_label_match_sem.float().cuda())/torch.sum(objectness_label_match.float().cuda())
    assert(np.array_equal(objectness_label.detach().cpu().numpy(), objectness_label_opt.detach().cpu().numpy()))
    assert(np.array_equal(objectness_mask.detach().cpu().numpy(), objectness_mask_opt.detach().cpu().numpy()))
    assert(np.array_equal(object_assignment.detach().cpu().numpy(), object_assignment_opt.detach().cpu().numpy()))
    
    # Box loss and sem cls loss for initial proposal
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
                                                                                                  compute_box_and_sem_cls_loss(end_points, config, mode='center')
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss + 0.1*sem_cls_loss
    end_points['box_loss'] = box_loss

    potential_loss = compute_matching_potential_loss(end_points, config, mode='opt')
    end_points['potential_loss'] = potential_loss

    proposalloss = vote_loss + 0.5*objectness_loss + box_loss + 0.5*objectness_loss_opt + potential_loss
    # proposalloss = vote_loss + 0.5*objectness_loss + box_loss +  0.1*sem_cls_loss + 0.5*objectness_loss_opt + potential_loss
    proposalloss *= 10
    loss = proposalloss + end_points['objcue_loss']
    end_points['init_proposal_loss'] = proposalloss
    end_points['loss'] = loss ### Add the initial proposal loss term    
        
    # --------------------------------------------
    # Some other statistics
    
    obj_pred_val = torch.argmax(end_points['objectness_scores'+'center'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc
    obj_pred_val = torch.argmax(end_points['match_scores'], 2) # B,K
    #obj_acc = torch.sum((obj_pred_val==objectness_label_match.long()).float()*objectness_mask_match)/(torch.sum(objectness_mask_match)+1e-6)
    obj_acc = torch.sum((obj_pred_val==objectness_label_match.long())).float() / float(obj_pred_val.shape[0]*obj_pred_val.shape[1])
    end_points['obj_acc_match'] = obj_acc

    obj_pred_val = torch.argmax(end_points['match_scores_sem'], 2) # B,K
    #obj_acc = torch.sum((obj_pred_val==objectness_label_match.long()).float()*objectness_mask_match)/(torch.sum(objectness_mask_match)+1e-6)
    obj_acc = torch.sum((obj_pred_val==objectness_label_match_sem.long())).float() / float(obj_pred_val.shape[0]*obj_pred_val.shape[1])
    end_points['obj_acc_match_sem'] = obj_acc

    return loss, end_points
