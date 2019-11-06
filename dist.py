#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy.io as sio
import os

POINT_THRESHOLD = 0.2

def judge_point(gt_point, gt_sem, pred_point_all, pred_sem_all, use_sem=False, sem_num=1):
    ''' 
    point: (3,)
    gt_point_all: (m, 3)
    '''
    dists = np.linalg.norm(pred_point_all - gt_point, axis=1)
    dist = np.min(dists)
    idx = np.argmin(dists)
    if use_sem:
        if sem_num == 1:
            return np.int((dist < POINT_THRESHOLD) and (pred_sem_all[idx] == gt_sem))
        elif sem_num == 3:
            return np.int((dist < POINT_THRESHOLD) and ((pred_sem_all[idx, 0] == gt_sem) or (pred_sem_all[idx, 1] == gt_sem) or (pred_sem_all[idx, 2] == gt_sem)))
        else:
            raise AssertionError('sem_num == 1 or 3')
    else:
        return np.int((dist < POINT_THRESHOLD))

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


if __name__ == "__main__":
    gt_dir = './initial_proposal/'
    pred_dir = './votenet_nms_result/'
    vox_dir = './new_voxel_cues_18cls/'

    val_list = []
    with open('./scannetv2_val.txt', 'r') as f:
        for line in f.readlines():
            val_list.append(line[:-1])

    total_num_list = []
    pred_exist_num_list = []
    vox_exist_num_list = []
    pred_exist_list = []
    vox_exist_list = []
    for scan_name in val_list:
        # if scan_name != 'scene0025_00':
        #     continue

        gt_obb_all = sio.loadmat(os.path.join(gt_dir, scan_name + '_gt.mat'))
        gt_obb_all = gt_obb_all['gt']
        pred_obb_all = np.load(os.path.join(pred_dir, scan_name + '_nms.npy'))
        vox = sio.loadmat(os.path.join(vox_dir, scan_name + '_center_0.06_vox.mat'))
        corner = sio.loadmat(os.path.join(vox_dir, scan_name + '_corner_0.06_vox.mat'))
        center_vox = vox['center_vox']
        center_label = vox['center_label']
        corner_vox = corner['corner_vox']
        corner_label = corner['corner_label']


        # ---------- center -----------
        s_pred = 0
        s_vox = 0
        pred_list = []
        vox_list = []
        for gt_obb in gt_obb_all:
            is_pred_exist = judge_point(gt_obb[:3], gt_obb[-1], pred_obb_all[:, :3], pred_obb_all[:, -1], use_sem=True)
            is_vox_exist = judge_point(gt_obb[:3], gt_obb[-1], center_vox, center_label, use_sem=True, sem_num=3)
            pred_list.append(is_pred_exist)
            vox_list.append(is_vox_exist)
            s_pred += is_pred_exist
            s_vox += is_vox_exist

        pred_exist_list.append(np.array(pred_list))
        vox_exist_list.append(np.array(vox_list))

        total_num_list.append(gt_obb_all.shape[0])
        pred_exist_num_list.append(s_pred)
        vox_exist_num_list.append(s_vox)


        # ---------- corner -----------
        corner_num_exist_list = []
        for gt_obb in gt_obb_all:
            s = 0
            bbox = params2bbox(gt_obb[:3], gt_obb[3], gt_obb[4], gt_obb[5], gt_obb[6])
            for cnr in bbox:
                # s += judge_point(cnr, gt_obb[-1], corner_vox, corner_label, use_sem=True, sem_num=3)
                s += judge_point(cnr, gt_obb[-1], corner_vox, corner_label, use_sem=False)

            corner_num_exist_list.append(s)

        gt_obb_all_exist = np.hstack([gt_obb_all, np.array(corner_num_exist_list).reshape(-1, 1)])
        '''
        here
        '''




    total_num = np.array(total_num_list)                                              
    pred_num = np.array(pred_exist_num_list)                                          
    vox_num = np.array(vox_exist_num_list)
    print('-' * 30)
    print(np.sum(pred_num) / np.sum(total_num))
    print('-' * 30)
    print(np.sum(vox_num) / np.sum(total_num))
    print('-' * 30)

