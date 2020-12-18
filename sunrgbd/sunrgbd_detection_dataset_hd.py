# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are length of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import sunrgbd_utils
from sunrgbd_utils import extract_pc_in_box3d
from model_util_sunrgbd import SunrgbdDatasetConfig

DC = SunrgbdDatasetConfig() # dataset specific config
MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # sunrgbd color is in 0~1

DIST_THRESH = 0.1#0.2
VAR_THRESH = 5e-3
CENTER_THRESH = 0.1
LOWER_THRESH = 1e-6
NUM_POINT = 50
NUM_POINT_LINE = 10
LINE_THRESH = 0.1#0.2
MIND_THRESH = 0.1

NUM_POINT_SEM_THRESHOLD = 1

def check_upright(para_points):
    return (para_points[0][-1] == para_points[1][-1]) and (para_points[1][-1] == para_points[2][-1]) and (para_points[2][-1] == para_points[3][-1])

def check_z(plane_equ, para_points):
    return np.sum(para_points[:,2] + plane_equ[-1]) / 4.0 < LOWER_THRESH

def clockwise2counter(angle):
    ''' 
    @Args:
        angle: clockwise from x axis, from 0 to 2*pi, 
    @Returns:
        theta: counter clockwise, -pi / 2 ~ pi / 2, +x~+y: (0, pi/2), +x~-y: (0, -pi/2)
    '''
    return -((angle + np.pi / 2) % np.pi) + np.pi / 2;

def point2line_dist(points, a, b): 
    '''
    @Args:
        points: (N, 3)
        a / b: (3,)
    @Returns:
        distance: (N,)
    '''
    x = b - a 
    t = np.dot(points - a, x) / np.dot(x, x) 
    c = a + t[:, None] * np.tile(x, (t.shape[0], 1)) 
    return np.linalg.norm(points - c, axis=1) 

def get_linesel(points, corners, direction):
    ''' corners:
    [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
     [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    '''
    if direction == 'lower':
        sel1 = point2line_dist(points, corners[0], corners[2]) < LINE_THRESH
        sel2 = point2line_dist(points, corners[4], corners[6]) < LINE_THRESH
        sel3 = point2line_dist(points, corners[0], corners[4]) < LINE_THRESH
        sel4 = point2line_dist(points, corners[2], corners[6]) < LINE_THRESH
        return sel1, sel2, sel3, sel4
    elif direction == 'upper':
        sel1 = point2line_dist(points, corners[1], corners[3]) < LINE_THRESH
        sel2 = point2line_dist(points, corners[5], corners[7]) < LINE_THRESH
        sel3 = point2line_dist(points, corners[1], corners[5]) < LINE_THRESH
        sel4 = point2line_dist(points, corners[3], corners[7]) < LINE_THRESH
        return sel1, sel2, sel3, sel4
    elif direction == 'left':
        sel1 = point2line_dist(points, corners[0], corners[1]) < LINE_THRESH
        sel2 = point2line_dist(points, corners[2], corners[3]) < LINE_THRESH
        return sel1, sel2
    elif direction == 'right':
        sel1 = point2line_dist(points, corners[4], corners[5]) < LINE_THRESH
        sel2 = point2line_dist(points, corners[6], corners[7]) < LINE_THRESH
        return sel1, sel2
    else:
        AssertionError('direction = lower / upper / left')


def get_linesel2(points, ymin, ymax, zmin, zmax, axis=0):
    #sel3 = sweep(points, axis, ymax, 2, zmin, zmax)
    #sel4 = sweep(points, axis, ymax, 2, zmin, zmax)
    sel3 = np.abs(points[:,axis] - ymin) < LINE_THRESH
    sel4 = np.abs(points[:,axis] - ymax) < LINE_THRESH
    return sel3, sel4


''' ATTENTION: SUNRGBD, size_label is only half the actual size
'''
def params2bbox(center, size, angle):
    ''' from bbox_center, angle and size to bbox
    @Args:
        center: (3,)
        size: (3,)
        angle: -pi ~ pi, +x~+y: (0, pi/2), +x~-y: (0, -pi/2)
    @Returns:
        bbox: 8 x 3, order:
         [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
          [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    '''
    xsize = size[0] 
    ysize = size[1]
    zsize = size[2]
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



class SunrgbdDetectionVotesDataset(Dataset):
    def __init__(self, data_path=None, split_set='train', num_points=20000,
        use_color=False, use_height=False, use_v1=False,
        augment=False, scan_idx_list=None):

        assert(num_points<=50000)
        self.use_v1 = use_v1 
        if use_v1:
            self.data_path = os.path.join(data_path, 'sunrgbd_pc_bbox_votes_50k_v1_' + split_set)
            # self.data_path = os.path.join('/scratch/cluster/yanght/Dataset/sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_' + split_set)
        else:
            AssertionError("v2 data is not prepared")

        self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(self.data_path)])))

        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        scan_name = self.scan_names[idx]
        point_color_sem = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc'] # Nx6
        bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy') # K,8
        point_votes = np.load(os.path.join(self.data_path, scan_name)+'_votes.npz')['point_votes'] # Nx10

        semantics37 = point_color_sem[:, 6]
        semantics10 = np.array([DC.class37_2_class10[k] for k in semantics37])
        semantics10_multi = [DC.class37_2_class10_multi[k] for k in semantics37]
        if not self.use_color:
            point_cloud = point_color_sem[:, 0:3]
        else:
            point_cloud = point_color_sem[:,0:6]
            point_cloud[:,3:6] = (point_color_sem[:,3:6]-MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                bboxes[:,0] = -1 * bboxes[:,0]
                bboxes[:,6] = np.pi - bboxes[:,6]
                point_votes[:,[1,4,7]] = -1 * point_votes[:,[1,4,7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))
            point_votes_end[:,4:7] = np.dot(point_cloud[:,0:3] + point_votes[:,4:7], np.transpose(rot_mat))
            point_votes_end[:,7:10] = np.dot(point_cloud[:,0:3] + point_votes[:,7:10], np.transpose(rot_mat))

            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            bboxes[:,0:3] = np.dot(bboxes[:,0:3], np.transpose(rot_mat))
            bboxes[:,6] -= rot_angle
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]
            point_votes[:,4:7] = point_votes_end[:,4:7] - point_cloud[:,0:3]
            point_votes[:,7:10] = point_votes_end[:,7:10] - point_cloud[:,0:3]

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:,3:6] + MEAN_COLOR_RGB
                rgb_color *= (1+0.4*np.random.random(3)-0.2) # brightness change for each channel
                rgb_color += (0.1*np.random.random(3)-0.05) # color shift for each channel
                rgb_color += np.expand_dims((0.05*np.random.random(point_cloud.shape[0])-0.025), -1) # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0])>0.3,-1)
                point_cloud[:,3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random()*0.3+0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio,3),0)
            point_cloud[:,0:3] *= scale_ratio
            bboxes[:,0:3] *= scale_ratio
            bboxes[:,3:6] *= scale_ratio
            point_votes[:,1:4] *= scale_ratio
            point_votes[:,4:7] *= scale_ratio
            point_votes[:,7:10] *= scale_ratio
            if self.use_height:
                point_cloud[:,-1] *= scale_ratio[0,0]

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0],:] = bboxes

        # new items
        box3d_angles = np.zeros((MAX_NUM_OBJ,))

        point_boundary_mask_z = np.zeros(self.num_points)
        point_boundary_mask_xy = np.zeros(self.num_points)
        point_boundary_offset_z = np.zeros([self.num_points, 3])
        point_boundary_offset_xy = np.zeros([self.num_points, 3])
        point_boundary_sem_z = np.zeros([self.num_points, 3+2+1])
        point_boundary_sem_xy = np.zeros([self.num_points, 3+1+1])
        point_line_mask = np.zeros(self.num_points)
        point_line_offset = np.zeros([self.num_points, 3])
        point_line_sem = np.zeros([self.num_points, 3+1])

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
            box3d_size = bbox[3:6]*2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size
            box3d_angles[i] = bbox[6]

        target_bboxes_mask = label_mask 
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:,0])
            ymin = np.min(corners_3d[:,1])
            zmin = np.min(corners_3d[:,2])
            xmax = np.max(corners_3d[:,0])
            ymax = np.max(corners_3d[:,1])
            zmax = np.max(corners_3d[:,2])
            target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i,:] = target_bbox

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        semantics37 = semantics37[choices]
        semantics10 = semantics10[choices]
        semantics10_multi = [semantics10_multi[i] for i in choices]
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        # box angle is -pi to pi
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners = params2bbox(bbox[:3], 2 * bbox[3:6], clockwise2counter(bbox[6])) 
            # corners_votenet = sunrgbd_utils.my_compute_box_3d(bbox[:3], bbox[3:6], bbox[6])

            try:
                x_all_cls, ind_all_cls = extract_pc_in_box3d(point_cloud, corners)
            except:
                continue
            ind_all_cls = np.where(ind_all_cls)[0] # T/F to index
            # find point with same semantic as bbox, note semantics is 37 cls in sunrgbd

            # ind = ind_all_cls[np.where(semantics10[ind_all_cls] == bbox[7])[0]]
            ind = []
            for j in ind_all_cls:
                if bbox[7] in semantics10_multi[j]:
                    ind.append(j)
            ind = np.array(ind)

            if ind.shape[0] < NUM_POINT_SEM_THRESHOLD:
                pass
            else:
                x = point_cloud[ind, :3]

                ###Get bb planes and boundary points
                plane_lower_temp = np.array([0,0,1,-corners[6,-1]])
                para_points = np.array([corners[1], corners[3], corners[5], corners[7]])
                newd = np.sum(para_points * plane_lower_temp[:3], 1)
                if check_upright(para_points) and plane_lower_temp[0]+plane_lower_temp[1] < LOWER_THRESH:
                    plane_lower = np.array([0,0,1,plane_lower_temp[-1]]) 
                    plane_upper = np.array([0,0,1,-np.mean(newd)])
                else:
                    import pdb;pdb.set_trace()
                    print ("error with upright")
                if check_z(plane_upper, para_points) == False:
                    import pdb;pdb.set_trace()
                ### Get the boundary points here
                #alldist = np.abs(np.sum(point_cloud[:,:3]*plane_lower[:3], 1) + plane_lower[-1])
                alldist = np.abs(np.sum(x*plane_lower[:3], 1) + plane_lower[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,0] >= xmin) & (point_cloud[:,0] <= xmax) & (point_cloud[:,1] >= ymin) & (point_cloud[:,1] <= ymax)

                ## Get lower four lines
                line_sel1, line_sel2, line_sel3, line_sel4 = get_linesel(x[sel], corners, 'lower')
                if np.sum(line_sel1) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = (corners[0] + corners[2]) / 2.0
                    point_line_offset[ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(line_sel2) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = (corners[4] + corners[6]) / 2.0
                    point_line_offset[ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(line_sel3) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel3]] = 1.0
                    linecenter = (corners[0] + corners[4]) / 2.0
                    point_line_offset[ind[sel][line_sel3]] = linecenter - x[sel][line_sel3]
                    point_line_sem[ind[sel][line_sel3]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(line_sel4) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel4]] = 1.0
                    linecenter = (corners[2] + corners[6]) / 2.0
                    point_line_offset[ind[sel][line_sel4]] = linecenter - x[sel][line_sel4]
                    point_line_sem[ind[sel][line_sel4]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])

                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    # center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, np.mean(x[sel][:,2])])
                    center = (corners[0] + corners[6]) / 2.0
                    center[2] = np.mean(x[sel][:,2])
                    sel_global = ind[sel]
                    point_boundary_mask_z[sel_global] = 1.0
                    point_boundary_sem_z[sel_global] = np.array([center[0], center[1], center[2], np.linalg.norm(corners[4] - corners[0]), np.linalg.norm(corners[2] - corners[0]), bbox[7]])
                    point_boundary_offset_z[sel_global] = center - x[sel]
                    
                '''
                ### Check for middle z surfaces
                [count, val] = np.histogram(alldist, bins=20)
                mind_middle = val[np.argmax(count)]
                sel_pre = np.copy(sel)
                sel = np.abs(alldist - mind_middle) < DIST_THRESH
                if np.abs(np.mean(x[sel_pre][:,2]) - np.mean(x[sel][:,2])) > MIND_THRESH:
                    ### Do not use line for middle surfaces
                    if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                        center = (corners[0] + corners[6]) / 2.0
                        center[2] = np.mean(x[sel][:,2])
                        # center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, np.mean(x[sel][:,2])])
                        sel_global = ind[sel]
                        point_boundary_mask_z[sel_global] = 1.0
                        point_boundary_sem_z[sel_global] = np.array([center[0], center[1], center[2], np.linalg.norm(corners[4] - corners[0]), np.linalg.norm(corners[2] - corners[0]), bbox[7]])
                        point_boundary_offset_z[sel_global] = center - x[sel]
                '''
                    
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_upper[:3], 1) + plane_upper[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,0] >= xmin) & (point_cloud[:,0] <= xmax) & (point_cloud[:,1] >= ymin) & (point_cloud[:,1] <= ymax)

                ## Get upper four lines
                line_sel1, line_sel2, line_sel3, line_sel4 = get_linesel(x[sel], corners, 'upper')
                if np.sum(line_sel1) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = (corners[1] + corners[3]) / 2.0
                    point_line_offset[ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(line_sel2) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = (corners[5] + corners[7]) / 2.0
                    point_line_offset[ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(line_sel3) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel3]] = 1.0
                    linecenter = (corners[1] + corners[5]) / 2.0
                    point_line_offset[ind[sel][line_sel3]] = linecenter - x[sel][line_sel3]
                    point_line_sem[ind[sel][line_sel3]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(line_sel4) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel4]] = 1.0
                    linecenter = (corners[3] + corners[7]) / 2.0
                    point_line_offset[ind[sel][line_sel4]] = linecenter - x[sel][line_sel4]
                    point_line_sem[ind[sel][line_sel4]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    # center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, np.mean(x[sel][:,2])])
                    center = (corners[1] + corners[7]) / 2.0
                    center[2] = np.mean(x[sel][:,2])
                    sel_global = ind[sel]
                    point_boundary_mask_z[sel_global] = 1.0
                    point_boundary_sem_z[sel_global] = np.array([center[0], center[1], center[2], np.linalg.norm(corners[5] - corners[1]), np.linalg.norm(corners[3] - corners[1]), bbox[7]])
                    point_boundary_offset_z[sel_global] = center - x[sel]
                    
                v1 = corners[3] - corners[2]
                v2 = corners[2] - corners[0]
                cp = np.cross(v1, v2)
                d = -np.dot(cp,corners[0])
                a,b,c = cp
                plane_left_temp = np.array([a, b, c, d])
                para_points = np.array([corners[4], corners[5], corners[6], corners[7]])
                ### Normalize xy here
                plane_left_temp /= np.linalg.norm(plane_left_temp[:3])
                newd = np.sum(para_points * plane_left_temp[:3], 1)
                if plane_left_temp[2] < LOWER_THRESH:
                    plane_left = plane_left_temp#np.array([cls,res,tempsign,plane_left_temp[-1]]) 
                    plane_right = np.array([plane_left_temp[0], plane_left_temp[1], plane_left_temp[2], -np.mean(newd)])
                else:
                    import pdb;pdb.set_trace()
                    print ("error with upright")
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_left[:3], 1) + plane_left[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,2] >= zmin) & (point_cloud[:,2] <= zmax) & (point_cloud[:,1] >= ymin) & (point_cloud[:,1] <= ymax)
                ## Get upper four lines
                line_sel1, line_sel2 = get_linesel(x[sel], corners, 'left')
                if np.sum(line_sel1) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = (corners[0] + corners[1]) / 2.0
                    point_line_offset[ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(line_sel2) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = (corners[2] + corners[3]) / 2.0
                    point_line_offset[ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    # center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (corners[0, 2] + corners[1, 2])/2.0])
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    # point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], corners[1, 2] - corners[0, 2], bbox[7]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]

                '''
                [count, val] = np.histogram(alldist, bins=20)
                mind_middle = val[np.argmax(count)]
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,2] >= zmin) & (point_cloud[:,2] <= zmax) & (point_cloud[:,1] >= ymin) & (point_cloud[:,1] <= ymax)
                ## Get upper four lines
                sel_pre = np.copy(sel)
                sel = np.abs(alldist - mind_middle) < DIST_THRESH
                if np.abs(np.mean(x[sel_pre][:,0]) - np.mean(x[sel][:,0])) > MIND_THRESH:
                    ### Do not use line for middle surfaces
                    if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                        # center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                        center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (corners[0, 2] + corners[1, 2])/2.0])
                        sel_global = ind[sel]
                        point_boundary_mask_xy[sel_global] = 1.0
                        point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], corners[1, 2] - corners[0, 2], bbox[7]])
                        point_boundary_offset_xy[sel_global] = center - x[sel]
                '''

                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_right[:3], 1) + plane_right[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,2] >= zmin) & (point_cloud[:,2] <= zmax) & (point_cloud[:,1] >= ymin) & (point_cloud[:,1] <= ymax)
                line_sel1, line_sel2 = get_linesel(x[sel], corners, 'right')
                if np.sum(line_sel1) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = (corners[4] + corners[5]) / 2.0
                    point_line_offset[ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(line_sel2) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = (corners[6] + corners[7]) / 2.0
                    point_line_offset[ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = np.array([linecenter[0], linecenter[1], linecenter[2], bbox[7]])
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    # center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (corners[4, 2] + corners[5, 2])/2.0])
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], corners[5, 2] - corners[4, 2], bbox[7]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]

                #plane_front_temp = leastsq(residuals, [0,1,0,0], args=(None, np.array([corners[0], corners[1], corners[4], corners[5]]).T))[0]
                v1 = corners[0] - corners[4]
                v2 = corners[4] - corners[5]
                cp = np.cross(v1, v2)
                d = -np.dot(cp,corners[5])
                a,b,c = cp
                plane_front_temp = np.array([a, b, c, d])
                para_points = np.array([corners[2], corners[3], corners[6], corners[7]])
                plane_front_temp /= np.linalg.norm(plane_front_temp[:3])
                newd = np.sum(para_points * plane_front_temp[:3], 1)
                if plane_front_temp[2] < LOWER_THRESH:
                    plane_front = plane_front_temp#np.array([cls,res,tempsign,plane_front_temp[-1]]) 
                    plane_back = np.array([plane_front_temp[0], plane_front_temp[1], plane_front_temp[2], -np.mean(newd)])
                else:
                    import pdb;pdb.set_trace()
                    print ("error with upright")
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_front[:3], 1) + plane_front[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,0] >= xmin) & (point_cloud[:,0] <= xmax) & (point_cloud[:,2] >= zmin) & (point_cloud[:,2] <= zmax)
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    # center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (corners[0, 2] + corners[1, 2])/2.0])
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], corners[1, 2] - corners[0, 2], bbox[7]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]

                '''
                [count, val] = np.histogram(alldist, bins=20)
                mind_middle = val[np.argmax(count)]
                sel_pre = np.copy(sel)
                sel = np.abs(alldist - mind_middle) < DIST_THRESH
                if np.abs(np.mean(x[sel_pre][:,1]) - np.mean(x[sel][:,1])) > MIND_THRESH:
                    ### Do not use line for middle surfaces
                    if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                        # center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                        center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (corners[0, 2] + corners[1, 2])/2.0])
                        sel_global = ind[sel]
                        point_boundary_mask_xy[sel_global] = 1.0
                        point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], corners[1, 2] - corners[0, 2], bbox[7]])
                        point_boundary_offset_xy[sel_global] = center - x[sel]
                ''' 
                    
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_back[:3], 1) + plane_back[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,0] >= xmin) & (point_cloud[:,0] <= xmax) & (point_cloud[:,2] >= zmin) & (point_cloud[:,2] <= zmax)
                    # center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (corners[2, 2] + corners[3, 2])/2.0])
                    #point_boundary_offset_xy[sel] = center - x[sel]
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], corners[3, 2] - corners[2, 2], bbox[7]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]


        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes

        # new items
        ret_dict['size_label'] = box3d_sizes.astype(np.float32)
        ret_dict['heading_label'] = box3d_angles.astype(np.float32)
        if self.use_height:
            ret_dict['floor_height'] = floor_height

        ret_dict['point_boundary_mask_z'] = point_boundary_mask_z.astype(np.float32)
        ret_dict['point_boundary_mask_xy'] = point_boundary_mask_xy.astype(np.float32)
        ret_dict['point_boundary_offset_z'] = point_boundary_offset_z.astype(np.float32)
        ret_dict['point_boundary_offset_xy'] = point_boundary_offset_xy.astype(np.float32)
        ret_dict['point_boundary_sem_z'] = point_boundary_sem_z.astype(np.float32)
        ret_dict['point_boundary_sem_xy'] = point_boundary_sem_xy.astype(np.float32)

        ret_dict['point_line_mask'] = point_line_mask.astype(np.float32)
        ret_dict['point_line_offset'] = point_line_offset.astype(np.float32)
        ret_dict['point_line_sem'] = point_line_sem.astype(np.float32)

        return ret_dict

def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds,3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds,6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')

def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_ply(label[mask==1,:], 'gt_centroids.ply')

def get_sem_cls_statistics():
    """ Compute number of objects for each semantic class """
    d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i%10==0: print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)

if __name__=='__main__':
    d = SunrgbdDetectionVotesDataset(use_height=True, use_color=False, use_v1=True, augment=True)
    for i in range(1000):
        print('-' * 50)
        print(i)
        sample = d[i]
    # sample = d[200]
    import ipdb; ipdb.set_trace()
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        sample['heading_class_label'], sample['heading_residual_label'],
        sample['size_class_label'], sample['size_residual_label'])
