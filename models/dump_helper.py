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
from sklearn.cluster import DBSCAN

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def dump_results(end_points, dump_dir, config, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
        vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # OTHERS
    pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        if 'vote_xyz' in end_points:
            pc_util.write_ply(end_points['vote_xyz'][i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(idx_beg+i)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                pred_size_class[i,j], pred_size_residual[i,j])
                obbs.append(obb)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                pc_util.write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH,:], os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:], os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    gt_center = end_points['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'].cpu().numpy() # B,K2
    gt_heading_class = end_points['heading_class_label'].cpu().numpy() # B,K2
    gt_heading_residual = end_points['heading_residual_label'].cpu().numpy() # B,K2
    gt_size_class = end_points['size_class_label'].cpu().numpy() # B,K2
    gt_size_residual = end_points['size_residual_label'].cpu().numpy() # B,K2,3
    objectness_label = end_points['objectness_label'].detach().cpu().numpy() # (B,K,)
    objectness_mask = end_points['objectness_mask'].detach().cpu().numpy() # (B,K,)

    for i in range(batch_size):
        if np.sum(objectness_label[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_label[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_mask[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_mask[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(gt_center[i,:,0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply'%(idx_beg+i)))
        pc_util.write_ply_color(pred_center[i,:,0:3], objectness_label[i,:], os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.obj'%(idx_beg+i)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i,j] == 0: continue
            obb = config.param2obb(gt_center[i,j,0:3], gt_heading_class[i,j], gt_heading_residual[i,j],
                            gt_size_class[i,j], gt_size_residual[i,j])
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_gt_bbox.ply'%(idx_beg+i)))

    # OPTIONALL, also dump prediction and gt details
    if 'vote_xyz_bcenter' in end_points:
        for ii in range(batch_size):
            pc_util.write_ply(end_points['vote_xyz_bcenter'][ii,:,:], os.path.join(dump_dir, '%06d_vgen_bcenter_pc.ply'%(idx_beg+ii)))
    if 'vote_xyz_corner' in end_points:
        for ii in range(batch_size):
            pc_util.write_ply(end_points['vote_xyz_corner'][ii,:,:], os.path.join(dump_dir, '%06d_vgen_corner_pc.ply'%(idx_beg+ii)))
    if 'vote_xyz_support' in end_points:
        for ii in range(batch_size):
            pc_util.write_ply(end_points['vote_xyz_support'][ii,:,:], os.path.join(dump_dir, '%06d_vgen_support_pc.ply'%(idx_beg+ii)))
    if 'vote_xyz_bsupport' in end_points:
        for ii in range(batch_size):
            pc_util.write_ply(end_points['vote_xyz_bsupport'][ii,:,:], os.path.join(dump_dir, '%06d_vgen_bsupport_pc.ply'%(idx_beg+ii)))
    
            
    if 'batch_pred_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_pred_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' '+str(t[2]))
                fout.write('\n')
            fout.close()
    if 'batch_gt_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_gt_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()

def dump_planes(end_points, dump_dir, config, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    #point_clouds = end_points['point_clouds'].cpu().numpy()
    #batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    #vote_xyz = end_points['vote_xyz'+mode] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_po

    #seed_gt_votes_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    seed_gt_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds).detach().cpu().numpy()
    new_ind = torch.stack([seed_inds]*3, -1)

    ### Gt planes
    mode = 'upper'
    seed_gt_upper_rot = torch.gather(end_points['plane_votes_rot_'+mode][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_upper_off = torch.gather(end_points['plane_votes_off_'+mode][:,:,0], 1, seed_inds).detach().cpu().numpy()

    mode = 'lower'
    seed_gt_lower_rot = torch.gather(end_points['plane_votes_rot_'+mode][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_lower_off = torch.gather(end_points['plane_votes_off_'+mode][:,:,0], 1, seed_inds).detach().cpu().numpy()

    mode = 'left'
    seed_gt_left_rot = torch.gather(end_points['plane_votes_rot_'+mode][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_left_off = torch.gather(end_points['plane_votes_off_'+mode][:,:,0], 1, seed_inds).detach().cpu().numpy()

    mode = 'right'
    seed_gt_right_rot = torch.gather(end_points['plane_votes_rot_'+mode][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_right_off = torch.gather(end_points['plane_votes_off_'+mode][:,:,0], 1, seed_inds).detach().cpu().numpy()

    mode = 'front'
    seed_gt_front_rot = torch.gather(end_points['plane_votes_rot_'+mode][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_front_off = torch.gather(end_points['plane_votes_off_'+mode][:,:,0], 1, seed_inds).detach().cpu().numpy()

    mode = 'back'
    seed_gt_back_rot = torch.gather(end_points['plane_votes_rot_'+mode][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_back_off = torch.gather(end_points['plane_votes_off_'+mode][:,:,0], 1, seed_inds).detach().cpu().numpy()

    sio.savemat(os.path.join(dump_dir,'sample_output_plane.mat'), {'seed_xyz':seed_xyz, 'mask':seed_gt_mask, 'gt_rot_upper': seed_gt_upper_rot, 'gt_off_upper': seed_gt_upper_off, 'gt_rot_lower': seed_gt_lower_rot, 'gt_off_lower': seed_gt_lower_off, 'gt_rot_right': seed_gt_right_rot, 'gt_off_right': seed_gt_right_off, 'gt_rot_left': seed_gt_left_rot, 'gt_off_left': seed_gt_left_off, 'gt_rot_front': seed_gt_front_rot, 'gt_off_front': seed_gt_front_off, 'gt_rot_back': seed_gt_back_rot, 'gt_off_back': seed_gt_back_off, 'corner': (end_points['vote_label']+end_points['vote_label_corner']).detach().cpu().numpy(), 'rot_upper': end_points['upper_rot'].detach().cpu().numpy(), 'off_upper': end_points['upper_off'].detach().cpu().numpy(), 'rot_lower': end_points['lower_rot'].detach().cpu().numpy(), 'off_lower': end_points['lower_off'].detach().cpu().numpy(), 'rot_front': end_points['front_rot'].detach().cpu().numpy(), 'off_front': end_points['front_off'].detach().cpu().numpy(), 'rot_back': end_points['back_rot'].detach().cpu().numpy(), 'off_back': end_points['back_off'].detach().cpu().numpy(), 'rot_left': end_points['left_rot'].detach().cpu().numpy(), 'off_left': end_points['left_off'].detach().cpu().numpy(), 'rot_right': end_points['right_rot'].detach().cpu().numpy(), 'off_right': end_points['right_off'].detach().cpu().numpy()})

    ### Center and Corner
    seed_gt_center = torch.gather(end_points['vote_label'][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_center += seed_xyz
    seed_pred_center = end_points['vote_xyz'][:,:,:3].detach().cpu().numpy()

    seed_gt_corner = torch.gather(end_points['vote_label_corner'][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_corner += seed_xyz
    seed_pred_corner = end_points['vote_xyz_corner'][:,:,:3].detach().cpu().numpy()

    
    sio.savemat(os.path.join(dump_dir,'sample_output_point.mat'), {'seed_xyz':seed_xyz, 'mask':seed_gt_mask, 'gt_center': seed_gt_center, 'gt_corner': seed_gt_corner, 'pred_center': seed_pred_center, 'pred_corner': seed_pred_corner})

def get_plane_xy(plane_equ):
    orgplane = np.zeros([plane_equ.shape[0], 4])
    for i in range(len(plane_equ)):
        organg = np.clip(plane_equ[i,0]+np.clip(plane_equ[i,1], a_min=0, a_max=1), a_min=0, a_max=12)*(np.pi/12) - np.pi/2
        slope = np.tan(organg)
        if plane_equ[i,2] >= 0:
            norm = np.linalg.norm([-slope, 1])
        else:
            norm = np.linalg.norm([-slope, 1]) * (-1.0)
        orgplane[i,:] = np.array([-slope/norm, 1/norm, 0, plane_equ[i,-1]])
        #return np.sum(np.sum(para_points*orgplane[:3], 1) + orgplane[3]) / 4.0 < LOWER_THRESH
    return orgplane#np.array()

def get_plane_z(plane_equ):
    orgplane = np.zeros([plane_equ.shape[0], 4])
    for i in range(len(plane_equ)):
        orgplane[i,:] = np.array([0, 0, 1.0, plane_equ[i,-1]])
    return orgplane
    
def dump_objcue(input_points, end_points, dump_dir, config, dataset, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = input_points['point_clouds'].cpu().numpy()
    #batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    #vote_xyz = end_points['vote_xyz'+mode] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_po

    #seed_gt_votes_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds)
    seed_gt_mask = torch.gather(end_points['plane_label_mask'], 1, seed_inds).detach().cpu().numpy()
    new_ind = torch.stack([seed_inds]*3, -1)

    ### Center and Corner
    seed_gt_center = torch.gather(end_points['vote_label'][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_center += seed_xyz
    seed_pred_center = end_points['vote_xyz'][:,:,:3].detach().cpu().numpy()

    seed_gt_corner = torch.gather(end_points['vote_label_corner'][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gt_corner += seed_xyz
    seed_pred_corner = end_points['vote_xyz_corner'][:,:,:3].detach().cpu().numpy()

    seed_gt_sem = torch.gather(end_points['point_sem_cls_label'][...,0], 1, seed_inds).detach().cpu().numpy()
    seed_pred_sem = end_points['pred_sem_class'].detach().cpu().numpy()
    seed_pred_sem3 = end_points['pred_sem_class_top3'].detach().cpu().numpy()

    feature_map = end_points['feature_map'].detach().cpu().numpy()
    #sio.savemat(os.path.join(dump_dir,'point_cue.mat'), {'full_pc': point_clouds[:,:,:3], 'seed_xyz':seed_xyz, 'mask':seed_gt_mask, 'gt_center': seed_gt_center, 'gt_corner': seed_gt_corner, 'pred_center': seed_pred_center, 'pred_corner': seed_pred_corner})
    ### Visulization here
    new_ind = new_ind.detach().cpu().numpy()
    for i in range(len(seed_gt_center)):
        inds = np.where(seed_gt_mask[i,...] == 1)[0]
        pc_util.pc2obj(point_clouds[i,:,:3], os.path.join(dump_dir,'pc_%d.obj' % i))
        pc_util.pc2obj(seed_xyz[i, inds, ...], os.path.join(dump_dir,'subpc_%d.obj' % i))
        pc_util.pc2obj(seed_gt_center[i,inds,...], os.path.join(dump_dir,'subpc_gt_center_%d.obj' % i))
        pc_util.pc2obj(seed_pred_center[i,inds,...], os.path.join(dump_dir,'subpc_pred_center_%d.obj' % i))
        pc_util.write_ply_label(seed_xyz[i, inds, ...], seed_gt_sem[i,inds], os.path.join(dump_dir,'subpc_gt_sem_%d.ply' % i), 38)
        #pc_util.write_ply_label(seed_xyz[i, inds, ...], seed_pred_sem[i,inds], os.path.join(dump_dir,'subpc_pred_sem_%d.ply' % i), 38)
    
        #pred_center = seed_pred_center[i,...]
        #db = DBSCAN(eps=0.3, min_samples=10).fit(pred_center)
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        #labels = db.labels_

        #cluster_centers = []
        #for label in labels:
        #    if label >= 0:
        #        cluster_centers.append(np.mean(pred_center[np.where(labels==label)[0]], 0))
        #if len(cluster_centers) == 0:
        #pc_util.pc2obj(np.array(cluster_centers), os.path.join(dump_dir, 'subpc_cluster_center_%d.obj' % i))

        #pc_util.pc2obj(seed_gt_corner[i,inds,...], 'subpc_gt_corner_%d.obj' % i)
        #pc_util.pc2obj(seed_pred_corner[i,inds,...], 'subpc_pred_corner_%d.obj' % i)
        
        sio.savemat(os.path.join(dump_dir,dataset.scan_names[end_points['scan_idx'][i]]+'_point_objcue.mat'), {'full_pc': point_clouds[i,:,:3], 'sub_pc':seed_xyz[i,...], 'feature_map':feature_map[i,...], 'subpc_mask':seed_gt_mask[i,...], 'gt_center': seed_gt_center[i,...], 'gt_corner': seed_gt_corner[i,...], 'pred_center': seed_pred_center[i,...], 'pred_corner': seed_pred_corner[i,...], 'gt_sem': seed_gt_sem[i,...], 'pred_sem': seed_pred_sem[i,...],  'pred_sem_top3': seed_pred_sem3[i,...]})

    ### Gt planes
    new_ind = torch.tensor(new_ind).cuda()

    seed_gtx_x = torch.gather(end_points['plane_votes_x'][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gtx_left = torch.gather(end_points['plane_votes_x0'][:,:,0], 1, seed_inds).detach().cpu().numpy()
    seed_gtx_right = torch.gather(end_points['plane_votes_x1'][:,:,0], 1, seed_inds).detach().cpu().numpy()

    seed_gty_y = torch.gather(end_points['plane_votes_y'][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gty_front = torch.gather(end_points['plane_votes_y0'][:,:,0], 1, seed_inds).detach().cpu().numpy()
    seed_gty_back = torch.gather(end_points['plane_votes_y1'][:,:,0], 1, seed_inds).detach().cpu().numpy()

    seed_gtz_z = torch.gather(end_points['plane_votes_z'][:,:,:3], 1, new_ind).detach().cpu().numpy()
    seed_gtz_lower = torch.gather(end_points['plane_votes_z0'][:,:,0], 1, seed_inds).detach().cpu().numpy()
    seed_gtz_upper = torch.gather(end_points['plane_votes_z1'][:,:,0], 1, seed_inds).detach().cpu().numpy()
    
    zangle = np.argmax(end_points['z_angle'].detach().cpu().numpy(), 1) # (batch_size, (3+out_dim)*vote_factor, num_seed)
    zres = end_points['z_res'].detach().cpu().numpy().squeeze(1)
    zsign = end_points['z_sign'].detach().cpu().numpy().squeeze(1)
    zoff0 = end_points['z_d0'].detach().cpu().numpy()
    zoff1 = end_points['z_d1'].detach().cpu().numpy()
    plane_lower = np.stack([zangle, np.clip(zres, a_min=0, a_max=1), zsign, zoff0], -1)
    plane_upper = np.stack([zangle, np.clip(zres, a_min=0, a_max=1), zsign, zoff1], -1)

    xangle = np.argmax(end_points['x_angle'].detach().cpu().numpy(), 1) # (batch_size, (3+out_dim)*vote_factor, num_seed)
    xres = end_points['x_res'].detach().cpu().numpy().squeeze(1)
    xsign = end_points['x_sign'].detach().cpu().numpy().squeeze(1)
    xoff0 = end_points['x_d0'].detach().cpu().numpy()
    xoff1 = end_points['x_d1'].detach().cpu().numpy()
    plane_left = np.stack([xangle, np.clip(xres, a_min=0, a_max=1), xsign, xoff0], -1)
    plane_right = np.stack([xangle, np.clip(xres, a_min=0, a_max=1), xsign, xoff1], -1)

    yangle = np.argmax(end_points['y_angle'].detach().cpu().numpy(), 1) # (batch_size, (3+out_dim)*vote_factor, num_seed)
    yres = end_points['y_res'].detach().cpu().numpy().squeeze(1)
    ysign = end_points['y_sign'].detach().cpu().numpy().squeeze(1)
    yoff0 = end_points['y_d0'].detach().cpu().numpy()
    yoff1 = end_points['y_d1'].detach().cpu().numpy()
    plane_front = np.stack([yangle, np.clip(yres, a_min=0, a_max=1), ysign, yoff0], -1)
    plane_back = np.stack([yangle, np.clip(yres, a_min=0, a_max=1), ysign, yoff1], -1)
    
    for i in range(len(plane_lower)):
        #sio.savemat(os.path.join(dump_dir,end_points['scan_name'][i]+'_plane_objcue.mat'), {'full_pc':seed_xyz[i,...], 'subpc_mask':seed_gt_mask[i,...], 'gt_upper': seed_gt_upper_rot[i,...], 'gt_off_upper': seed_gt_upper_off[i,...], 'gt_rot_lower': seed_gt_lower_rot[i,...], 'gt_off_lower': seed_gt_lower_off[i,...], 'gt_rot_right': seed_gt_right_rot[i,...], 'gt_off_right': seed_gt_right_off[i,...], 'gt_rot_left': seed_gt_left_rot[i,...], 'gt_off_left': seed_gt_left_off[i,...], 'gt_rot_front': seed_gt_front_rot[i,...], 'gt_off_front': seed_gt_front_off[i,...], 'gt_rot_back': seed_gt_back_rot[i,...], 'gt_off_back': seed_gt_back_off[i,...], 'rot_upper': end_points['upper_rot'][i,...].detach().cpu().numpy(), 'off_upper': end_points['upper_off'][i,...].detach().cpu().numpy(), 'rot_lower': end_points['lower_rot'][i,...].detach().cpu().numpy(), 'off_lower': end_points['lower_off'][i,...].detach().cpu().numpy(), 'rot_front': end_points['front_rot'][i,...].detach().cpu().numpy(), 'off_front': end_points['front_off'][i,...].detach().cpu().numpy(), 'rot_back': end_points['back_rot'][i,...].detach().cpu().numpy(), 'off_back': end_points['back_off'][i,...].detach().cpu().numpy(), 'rot_left': end_points['left_rot'][i,...].detach().cpu().numpy(), 'off_left': end_points['left_off'][i,...].detach().cpu().numpy(), 'rot_right': end_points['right_rot'][i,...].detach().cpu().numpy(), 'off_right': end_points['right_off'][i,...].detach().cpu().numpy()})
        seed_gt_lower = get_plane_z(np.concatenate([seed_gtz_z[i,...], np.expand_dims(seed_gtz_lower[i,...], -1)], -1))
        seed_gt_upper = get_plane_z(np.concatenate([seed_gtz_z[i,...], np.expand_dims(seed_gtz_upper[i,...], -1)], -1))
        seed_gt_left = get_plane_xy(np.concatenate([seed_gtx_x[i,...], np.expand_dims(seed_gtx_left[i,...], -1)], -1))
        seed_gt_right = get_plane_xy(np.concatenate([seed_gtx_x[i,...], np.expand_dims(seed_gtx_right[i,...], -1)], -1))
        seed_gt_front = get_plane_xy(np.concatenate([seed_gty_y[i,...], np.expand_dims(seed_gty_front[i,...], -1)], -1))
        seed_gt_back = get_plane_xy(np.concatenate([seed_gty_y[i,...], np.expand_dims(seed_gty_back[i,...], -1)], -1))

        pred_lower = get_plane_z(plane_lower[i,...])
        pred_upper = get_plane_z(plane_upper[i,...])
        pred_left = get_plane_xy(plane_left[i,...])
        pred_right = get_plane_xy(plane_right[i,...])
        pred_front = get_plane_xy(plane_front[i,...])
        pred_back = get_plane_xy(plane_back[i,...])
        
        sio.savemat(os.path.join(dump_dir,dataset.scan_names[end_points['scan_idx'][i]]+'_plane_objcue.mat'), {'full_pc':seed_xyz[i,...], 'subpc_mask':seed_gt_mask[i,...], 'gt_upper': seed_gt_upper, 'gt_lower': seed_gt_lower, 'gt_right': seed_gt_right, 'gt_left': seed_gt_left, 'gt_front': seed_gt_front, 'gt_back': seed_gt_back, 'pred_upper': pred_upper, 'pred_lower': pred_lower, 'pred_front': pred_front, 'pred_back': pred_back, 'pred_left': pred_left, 'pred_right': pred_right})

        
    pred_center_vox = end_points['vox_pred1']
    pred_corner_vox = end_points['vox_pred2']
    for i in range(pred_center_vox.shape[0]):
        name = end_points['scan_name'][i]
        center = pc_util.volume_to_point_cloud(pred_center_vox.detach().cpu().numpy()[i,0], thres=0.75)
        corner = pc_util.volume_to_point_cloud(pred_corner_vox.detach().cpu().numpy()[i,0], thres=0.9)
        pt_center = pc_util.volume_pt_to_pt(center, 0.0625, xmin=-4.0, xmax=4.0,ymin=0.0, ymax=8.0, zmin=-2.0, zmax=2.0)
        pt_corner = pc_util.volume_pt_to_pt(corner, 0.0625, xmin=-4.0, xmax=4.0,ymin=0.0, ymax=8.0, zmin=-2.0, zmax=2.0)
        print(pt_center.shape, pt_corner.shape)
        center_label = np.zeros((pt_center.shape[0], 3))
        corner_label = np.zeros((pt_corner.shape[0], 3))
        sio.savemat(os.path.join(dump_dir, name+'_center_0.0625_vox.mat'), {'center_vox': pt_center, 'center_vox_label':center_label})
        sio.savemat(os.path.join(dump_dir, name+'_corner_0.0625_vox.mat'), {'corner_vox': pt_corner, 'corner_vox_label':corner_label})

    """
    # Voxel cues
    sem_path =  '/tmp2/bosun/data/scannet/scannet_train_detection_data_vox/'
    pred_center_vox = end_points['vox_pred1']
    pred_corner_vox = end_points['vox_pred2']
    for i in range(pred_center_vox.shape[0]):
        name = end_points['scan_name'][i]
        center = pc_util.volume_to_point_cloud(pred_center_vox.detach().cpu().numpy()[i,0], thres=0.4)
        corner = pc_util.volume_to_point_cloud(pred_corner_vox.detach().cpu().numpy()[i,0], thres=0.95)
        if center.shape[0]==0:
            continue
        pt_center = pc_util.get_pred_pts(center, vsize=0.06, eps=2)
        pt_corner = volume_pt_to_pt(corner, 0.06, xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68)
#         pt_corner = pc_util.get_pred_pts(corner, vsize=0.06, eps=2)
        crop_ids = []
        for j in range(pt_center.shape[0]):
            if pt_center[j,0]<100:
              crop_ids.append(j)
        pt_center = pt_center[crop_ids]
        print(pt_center.shape, pt_corner.shape)
        if pt_center.shape[0]==0 or  pt_corner.shape[0]==0:
            print('***0 point***', name)
            continue
        center_label = np.zeros((pt_center.shape[0], 3))
        corner_label = np.zeros((pt_corner.shape[0], 3))
        for j in range(3):
            sem = np.load(os.path.join(sem_path, name+'_sem_pt_top%d_from3.npy'%(j)))
            if sem.shape[0]==0:
                print('***0 sem point***', name)
                continue
            center_label[:,j] = pc_util.point_add_sem_label(pt_center, sem, k=10)
            corner_label[:,j] = pc_util.point_add_sem_label(pt_corner, sem, k=50)
       
        sio.savemat(os.path.join(dump_dir, name+'_center_0.06_vox.mat'), {'center_vox': pt_center, 'center_label':center_label})
        sio.savemat(os.path.join(dump_dir, name+'_corner_0.06_vox.mat'), {'corner_vox': pt_corner, 'corner_label':corner_label})
    """

    
    
