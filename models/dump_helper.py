# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import scipy.io as sio
import scipy

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def params2bbox(center, xsize, ysize, zsize, angle):
    ''' from bbox_center, angle and size to bbox
    @Args:
        center: (3)
        x/y/zsize: scalar
        angle: -pi ~ pi
    @Returns:
        bbox: 8 x 3, order:
         [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
          [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    '''
    vx = np.array([np.cos(angle), np.sin(angle), 0])
    vy = np.array([-np.sin(angle), np.cos(angle), 0])
    vx = vx * np.abs(xsize) / 2
    vy = vy * np.abs(ysize) / 2
    vz = np.array([0, 0, np.abs(zsize) / 2])
    bbox = np.array([\
        center - vx - vy - vz, center - vx - vy + vz,
        center - vx + vy - vz, center - vx + vy + vz,
        center + vx - vy - vz, center + vx - vy + vz,
        center + vx + vy - vz, center + vx + vy + vz])
    return bbox

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def dump_results(end_points, dump_dir, config, dataset, opt_ang, mode='opt'):
    '''
        similar to dump results
        scan_names: all scan names
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))
        
    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]
    
    # NETWORK OUTPUTS
    seed_xyz_z = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    seed_xyz_xy = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    seed_xyz_line = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    
    gt_center = end_points['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    gt_num = end_points['num_instance'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    scan_idxes = end_points['scan_idx'].detach().cpu().numpy()

    pred_center = end_points['vote_xyz'].detach().cpu().numpy()

    aggregated_vote_xyz = end_points['aggregated_vote_xyz'+mode].detach().cpu().numpy()
    objectness_scores = end_points['objectness_scores'+mode].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'+mode].detach().cpu().numpy() # (B,K,3)

    pred_heading_class = torch.argmax(end_points['heading_scores'+'center'], -1) # B,num_proposal
    if opt_ang:
        pred_heading_residual = torch.gather(end_points['heading_residuals'+'opt'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    else:
        pred_heading_residual = torch.gather(end_points['heading_residuals'+'center'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1

    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    
    pred_size_class = torch.argmax(end_points['size_scores'+'center'], -1) # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'+mode], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3
    pred_sem_cls = torch.argmax(end_points['sem_cls_scores'+'center'], -1) # B, num_proposal
    pred_sem_cls = pred_sem_cls.detach().cpu().numpy()

    pred_mask = end_points['pred_mask'] # B,num_proposal

    # LABELS
    gt_center = end_points['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'].cpu().numpy() # B,K2
    gt_heading_class = end_points['heading_class_label'].cpu().numpy() # B,K2
    gt_heading_residual = end_points['heading_residual_label'].cpu().numpy() # B,K2
    gt_size_class = end_points['size_class_label'].cpu().numpy() # B,K2
    gt_size_residual = end_points['size_residual_label'].cpu().numpy() # B,K2,3
    objectness_label = end_points['objectness_label'+mode].detach().cpu().numpy() # (B,K,)
    objectness_mask = end_points['objectness_mask'+mode].detach().cpu().numpy() # (B,K,)
    sem_cls_label = end_points['sem_cls_label'].detach().cpu().numpy()

    ### Boundary points
    boundary_gt_z = end_points['sub_point_sem_cls_label'+'_z'].detach().cpu().numpy()
    boundary_pred_z = end_points['pred_flag'+'_z'].detach().cpu().numpy()
    boundary_gt_xy = end_points['sub_point_sem_cls_label'+'_xy'].detach().cpu().numpy()
    boundary_pred_xy = end_points['pred_flag'+'_xy'].detach().cpu().numpy()
    boundary_gt_line = end_points['sub_point_sem_cls_label'+'_line'].detach().cpu().numpy()
    boundary_pred_line = end_points['pred_flag'+'_line'].detach().cpu().numpy()

    gt_center_z = end_points['surface_center_gt_z'].detach().cpu().numpy()
    gt_sem_z = end_points['surface_sem_gt_z'].detach().cpu().numpy()
    gt_mask_z = end_points['surface_mask_gt_z'].detach().cpu().numpy()

    gt_center_xy = end_points['surface_center_gt_xy'].detach().cpu().numpy()
    gt_sem_xy = end_points['surface_sem_gt_xy'].detach().cpu().numpy()
    gt_mask_xy = end_points['surface_mask_gt_xy'].detach().cpu().numpy()

    gt_center_line = end_points['surface_center_gt_line'].detach().cpu().numpy()
    gt_sem_line = end_points['surface_sem_gt_line'].detach().cpu().numpy()
    gt_mask_line = end_points['surface_mask_gt_line'].detach().cpu().numpy()
    
    pred_center_z = end_points['center_z'].detach().cpu().numpy()
    pred_center_xy = end_points['center_xy'].detach().cpu().numpy()
    pred_center_line = end_points['center_line'].detach().cpu().numpy()
    
    pred_size_z = end_points['size_residuals_z'].detach().cpu().numpy()
    pred_size_xy = end_points['size_residuals_xy'].detach().cpu().numpy()

    pred_sem_z = end_points['sem_cls_scores_z'].detach().cpu().numpy()
    pred_sem_xy = end_points['sem_cls_scores_xy'].detach().cpu().numpy()
    pred_sem_line = end_points['sem_cls_scores_line'].detach().cpu().numpy()

    num_proposal = pred_center.shape[1]
    for i in range(batch_size):
        idx = scan_idxes[i]
        scan = dataset.scan_names[idx]
        print('-' * 30)
        print(scan)
        print('-' * 30)
    
        box_pred_list = []
        box_gt_list = []
        obb_pred_list = []
        obb_gt_list = []

        for j in range(num_proposal):
            obb = config.param2obb2(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                            pred_size_class[i,j], pred_size_residual[i,j])
            obb_pred_list.append(np.hstack([obb, pred_sem_cls[i, j] + 1])) # ATTENTION: need to + 1
            box = params2bbox(obb[:3], obb[3], obb[4], obb[5], obb[6])
            box_pred_list.append(box)
        obb_pred_mat = np.array(obb_pred_list)
        
        for j in range(gt_center.shape[1]):
            if gt_mask[i, j] == 0: continue
            obb = config.param2obb2(gt_center[i,j,0:3], gt_heading_class[i,j], gt_heading_residual[i,j],
                            gt_size_class[i,j], gt_size_residual[i,j])
            obb_gt_list.append(np.hstack([obb, sem_cls_label[i, j] + 1])) # ATTENTION: need to + 1
            box = params2bbox(obb[:3], obb[3], obb[4], obb[5], obb[6])
            box_gt_list.append(box)
        obb_gt_mat = np.array(obb_gt_list)

        scipy.io.savemat(dump_dir + mode + scan + '_gt.mat', {'gt': obb_gt_mat})
        scipy.io.savemat(dump_dir + mode + scan + '_boundary_z.mat', {'gt': boundary_gt_z[i,...], 'pred': boundary_pred_z[i,...], 'origpc': point_clouds[i,...], 'seedpc': seed_xyz_z[i,...], 'gt_center': gt_center_z[i,...], 'gt_sem': gt_sem_z[i,...], 'gt_mask': gt_mask_z[i,...], 'pred_center': pred_center_z[i,...], 'pred_sem': pred_sem_z[i,...], 'pred_size': pred_size_z[i,...]})
        scipy.io.savemat(dump_dir + mode + scan + '_boundary_xy.mat', {'gt': boundary_gt_xy[i,...], 'pred': boundary_pred_xy[i,...], 'origpc': point_clouds[i,...], 'seedpc': seed_xyz_xy[i,...], 'gt_center': gt_center_xy[i,...], 'gt_sem': gt_sem_xy[i,...], 'gt_mask': gt_mask_xy[i,...], 'pred_center': pred_center_xy[i,...], 'pred_sem': pred_sem_xy[i,...], 'pred_size': pred_size_xy[i,...]})
        scipy.io.savemat(dump_dir + mode + scan + '_boundary_line.mat', {'gt': boundary_gt_line[i,...], 'pred': boundary_pred_line[i,...], 'origpc': point_clouds[i,...], 'seedpc': seed_xyz_line[i,...], 'gt_center': gt_center_line[i,...], 'gt_sem': gt_sem_line[i,...], 'gt_mask': gt_mask_line[i,...], 'pred_center': pred_center_line[i,...], 'pred_sem': pred_sem_line[i,...]})

        # uncomment to visualize
        # Dump predicted bounding boxes
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)
        select_idx = np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1)
        box_pred_nms_list = []
        obb_pred_nms_list = []
        for i, val in enumerate(select_idx.tolist()):
            if val:
                box_pred_nms_list.append(box_pred_list[i])
                obb_pred_nms_list.append(obb_pred_list[i])

        votenet_pred_nms_arr = np.array(obb_pred_nms_list)
        np.save(dump_dir + mode + scan + '_nms.npy', votenet_pred_nms_arr)
