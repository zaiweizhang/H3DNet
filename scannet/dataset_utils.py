#!/usr/bin/env python
# coding=utf-8

import numpy as np
import scipy.io as sio
import os

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


def bboxParams2bbox(bbox_param):
    ''' from bbox_center, angle and size to bbox
    @Args:
        bbox_param: (7,), (center, xsize, ysize, zsize, angle):
    @Returns:
        bbox: 8 x 3, order:
         [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
          [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    '''
    center = bbox_param[:3]
    xsize = bbox_param[3]
    ysize = bbox_param[4]
    zsize = bbox_param[5]
    angle = bbox_param[6]
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


def load_pps_and_gt(data_dir, scan_name):
    ''' load bbox_initial_proposals & bbox_gt
    @Returns:
        bbox_init_pps: (256, 8)
        bbox_gt: (num_gt_bbox, 8)
    '''
    bbox_init_pps = sio.loadmat(os.path.join(data_dir, 'initial_proposal_pred_gt', scan_name + '_pred.mat'))
    bbox_init_pps = bbox_init_pps['pred'].astype(np.float32)
    bbox_gt = sio.loadmat(os.path.join(data_dir, 'initial_proposal_pred_gt', scan_name + '_gt.mat'))
    bbox_gt = bbox_gt['gt2'].reshape(-1, 8).astype(np.float32)
    return bbox_init_pps, bbox_gt


def judge_points_in_bbox(points, bbox):
    '''
    @Args:
        points: Nx3
        bbox: 8x3, order matters
             [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
              [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    '''
    center = (bbox[0] + bbox[7]) / 2
    xaxis = bbox[4] - bbox[0]; xlength = np.linalg.norm(xaxis); xaxis = xaxis / xlength
    yaxis = bbox[2] - bbox[0]; ylength = np.linalg.norm(yaxis); yaxis = yaxis / ylength
    zaxis = bbox[1] - bbox[0]; zlength = np.linalg.norm(zaxis); zaxis = zaxis / zlength

    displacement = (points - center).T
    px = np.abs(np.dot(xaxis, displacement)) * 2 <= xlength
    py = np.abs(np.dot(yaxis, displacement)) * 2 <= ylength
    pz = np.abs(np.dot(zaxis, displacement)) * 2 <= zlength
    # inside_idx = np.where(px * py * pz)
    # return inside_idx

    return (px * py * pz).astype(np.float32)



def load_data(data_dir, scan_name):
    ''' load cues and bbox_initial_proposals & bbox_gt
    @Returns:
        cues: all with sem as the last column
        cues['center_vox']: [n_pn, 6] 
        cues['center_pn'] : [1024, 6] 
        cues['corner_vox']: [n_vx, 6] 
        cues['corner_pn'] : [1024, 6] 
        cues['pred_upper']: [1024, 7] 
        cues['pred_lower']: [1024, 7] 
        cues['pred_right']: [1024, 7] 
        cues['pred_left'] : [1024, 7] 
        cues['pred_front']: [1024, 7] 
        cues['pred_back'] : [1024, 7] 
    '''
    cues = {}
    center_vox_label = sio.loadmat(os.path.join(data_dir, 'voxel', scan_name + '_center_0.06_vox_0.6.mat'))
    center_vox = np.hstack([center_vox_label['center_vox'], center_vox_label['center_label']])
    cues['center_vox'] = center_vox.astype(np.float32)

    corner_vox_label = sio.loadmat(os.path.join(data_dir, 'voxel', scan_name + '_corner_0.06_vox_0.7.mat'))
    corner_vox = np.hstack([corner_vox_label['corner_vox'], corner_vox_label['corner_label']])
    cues['corner_vox'] = corner_vox.astype(np.float32)

    pred_point = sio.loadmat(os.path.join(data_dir, 'point', scan_name + '_point_objcue.mat'))
    center_pn = np.hstack([pred_point['pred_center'], pred_point['pred_sem'].reshape(-1, 3)])
    cues['center_pn'] = center_pn.astype(np.float32)
    corner_pn = np.hstack([pred_point['pred_corner'], pred_point['pred_sem'].reshape(-1, 3)])
    cues['corner_pn'] = corner_pn.astype(np.float32)
    
    pred_plane = sio.loadmat(os.path.join(data_dir, 'plane', scan_name + '_plane_objcue.mat'))
    cues['pred_upper'] = np.hstack([pred_plane['pred_upper'], pred_point['pred_sem'].reshape(-1, 3)]).astype(np.float32)
    cues['pred_lower'] = np.hstack([pred_plane['pred_lower'], pred_point['pred_sem'].reshape(-1, 3)]).astype(np.float32)
    cues['pred_right'] = np.hstack([pred_plane['pred_right'], pred_point['pred_sem'].reshape(-1, 3)]).astype(np.float32)
    cues['pred_left']  = np.hstack([pred_plane['pred_left'] , pred_point['pred_sem'].reshape(-1, 3)]).astype(np.float32)
    cues['pred_front'] = np.hstack([pred_plane['pred_front'], pred_point['pred_sem'].reshape(-1, 3)]).astype(np.float32)
    cues['pred_back']  = np.hstack([pred_plane['pred_back'] , pred_point['pred_sem'].reshape(-1, 3)]).astype(np.float32)

    bbox_init_pps = sio.loadmat(os.path.join(data_dir, 'initial_proposal_pred_gt', scan_name + '_pred.mat'))
    bbox_init_pps = bbox_init_pps['pred'].astype(np.float32)
    bbox_gt = sio.loadmat(os.path.join(data_dir, 'initial_proposal_pred_gt', scan_name + '_gt.mat'))
    bbox_gt = bbox_gt['gt2'].reshape(-1, 8).astype(np.float32)

    return cues, bbox_init_pps, bbox_gt


def dists_bbox2cues(cues, cur_bbox):
    ''' cues not contain sem
    '''
    VOX_CUE_NOT_EXIST_DIST = 4
    center, corners = extract_center_corners(cur_bbox)
    
    dists = {}
    # center 
    if cues['center_vox'].shape[0]!=0:
        dif = cues['center_vox'] - center # (K, 3)
        dists['center_vox'] = np.min(np.sum(dif**2, 1))
    else:
        dists['center_vox'] = VOX_CUE_NOT_EXIST_DIST

    dif = cues['center_pn'] - center # (K, 3)
    dists['center_pn'] = np.min(np.sum(dif**2, 1))

    # corner
    if cues['corner_vox'].shape[0]!=0:
        for i in range(8):
            dif = cues['corner_vox'] - corners[i]
            dists['corner_vox' + str(i)] = np.min(np.sum(dif**2, 1))
    else:
        for i in range(8):
            dists['corner_vox' + str(i)] = VOX_CUE_NOT_EXIST_DIST

    for i in range(8):
        dif = cues['corner_pn'] - corners[i]
        dists['corner_pn' + str(i)] = np.min(np.sum(dif**2, 1))
        
    # plane 
    sqrDis = face_2_plane_sqrDis(corners[[0,1,2,3]], cues['pred_upper'])
    dists['face_upper'] = np.min(sqrDis)
    sqrDis = face_2_plane_sqrDis(corners[[4,5,6,7]], cues['pred_lower'])
    dists['face_lower'] = np.min(sqrDis)
    sqrDis = face_2_plane_sqrDis(corners[[0,2,4,6]], cues['pred_right'])
    dists['face_right'] = np.min(sqrDis)
    sqrDis = face_2_plane_sqrDis(corners[[1,3,5,7]], cues['pred_left'])
    dists['face_left'] = np.min(sqrDis)
    sqrDis = face_2_plane_sqrDis(corners[[2,3,6,7]], cues['pred_front'])
    dists['face_front'] = np.min(sqrDis)
    sqrDis = face_2_plane_sqrDis(corners[[0,1,4,5]], cues['pred_back'])
    dists['face_back'] = np.min(sqrDis)
    return dists


   
def face_2_plane_sqrDis(corners, planes):
    '''
    @Args:
        corners: (4, 3)
        planes: (M, 4)
    @Returns:
        dis: (M,)
    '''
    dis = np.zeros((4, planes.shape[0]))
    for i in range(4):
        dis[i] = (corners[i,0]*planes[:,0]+corners[i,1]*planes[:,1]+corners[i,2]*planes[:,2]+planes[:,3])
    dis = dis**2
    dis = np.sum(dis, 0)/4
    return dis


def get_oriented_corners(bbx):
    center = bbx[0:3]
    xsize = bbx[3]
    ysize = bbx[4]
    zsize = bbx[5]
    angle = bbx[6]
    vx = np.array([np.cos(angle), np.sin(angle), 0])
    vy = np.array([-np.sin(angle), np.cos(angle), 0])
    vx = vx * np.abs(xsize) / 2
    vy = vy * np.abs(ysize) / 2
    vz = np.array([0, 0, np.abs(zsize) / 2])
    corners = np.zeros((8,3))
    corners[0] = center + vx + vy + vz
    corners[1] = center - vx + vy + vz
    corners[2] = center + vx - vy + vz
    corners[3] = center - vx - vy + vz
    corners[4] = center + vx + vy - vz
    corners[5] = center - vx + vy - vz
    corners[6] = center + vx - vy - vz
    corners[7] = center - vx - vy - vz
    return corners


def extract_center_corners(bbx): 
    '''
    @Args:
        bbx: (8,)
    @Returns:
        centers: [1, 3]
        corners: [8, 3]
    '''
    centers = bbx[0:3][None, ...]
    corners= get_oriented_corners(bbx)
    return centers, corners
 

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------
def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

