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
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import sunrgbd_utils
from model_util_sunrgbd import SunrgbdDatasetConfig
import scipy.io as sio
from scipy.cluster.vq import vq, kmeans, whiten

DC = SunrgbdDatasetConfig() # dataset specific config
MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # sunrgbd color is in 0~1

class SunrgbdDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, use_v1=False,
                 augment=False, scan_idx_list=None, vsize=0.06, use_tsdf=0, use_18cls=1,center_dev=2.0, corner_dev=1.0):

        assert(num_points<=50000)
        self.use_v1 = use_v1
        ROOT_DIR = '/scratch/cluster/zaiwei92/dataset/'
        if use_v1:
            self.data_path = os.path.join(ROOT_DIR,
                'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_%s'%(split_set))
            self.data_plane_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_all_plane')
        else:
            self.data_path = os.path.join(ROOT_DIR,
                'sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_%s'%(split_set))
            
        #self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height

        ### Vox parameters
        self.vsize = vsize
        self.center_dev = center_dev
        self.corner_dev = corner_dev
        self.use_tsdf = use_tsdf
        self.use_18cls = use_18cls
        
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
        point_cloud = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc'] # Nx6
        bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy') # K,8
        cues = np.load(os.path.join(self.data_path, scan_name)+'_votes_cue.npz')
        planes = sio.loadmat(os.path.join(self.data_plane_path, scan_name)+'_planar.mat')
        assert(np.array_equal(planes['xyz'], point_cloud[:,:3]))
        point_votes = cues['point_votes'] # Nx10
        point_corner_votes = cues['point_votes_corner'] # Nx9
        point_sem = cues['point_sem']
        plane_mask = cues['plane_mask']
        plane_upper = cues['plane_upper']
        plane_lower = cues['plane_lower']
        plane_left = cues['plane_left']
        plane_right = cues['plane_right']
        plane_front = cues['plane_front']
        plane_back = cues['plane_back']
        plane_label = planes['params']
        
        if True:#not self.use_color: ### Do not use color 
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
        # ------------------------------- DATA AUGMENTATION ------------------------------
        point_yz = -1
        point_xz = -1
        point_rot = np.eye(3).astype(np.float32)
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_yz = 1
                point_cloud[:,0] = -1 * point_cloud[:,0]
                plane_label[:,0] = -1 * plane_label[:,0]
                
                plane_front[:,0] = -1 * plane_front[:,0]
                plane_back[:,0] = -1 * plane_back[:,0]
                plane_upper[:,0] = -1 * plane_upper[:,0]
                plane_lower[:,0] = -1 * plane_lower[:,0]
                plane_left[:,0] = -1 * plane_left[:,0]
                plane_right[:,0] = -1 * plane_right[:,0]
                
                bboxes[:,0] = -1 * bboxes[:,0]
                bboxes[:,6] = np.pi - bboxes[:,6]
                point_votes[:,[1,4,7]] = -1 * point_votes[:,[1,4,7]]
                point_corner_votes[:,[0,3,6]] = -1 * point_corner_votes[:,[0,3,6]]
              
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle).astype(np.float32)

            point_rot = rot_mat
            
            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))
            point_votes_end[:,4:7] = np.dot(point_cloud[:,0:3] + point_votes[:,4:7], np.transpose(rot_mat))
            point_votes_end[:,7:10] = np.dot(point_cloud[:,0:3] + point_votes[:,7:10], np.transpose(rot_mat))

            point_corner_votes_end = np.zeros_like(point_corner_votes)
            point_corner_votes_end[:,0:3] = np.dot(point_cloud[:,0:3] + point_corner_votes[:,0:3], np.transpose(rot_mat))
            point_corner_votes_end[:,3:6] = np.dot(point_cloud[:,0:3] + point_corner_votes[:,3:6], np.transpose(rot_mat))
            point_corner_votes_end[:,6:9] = np.dot(point_cloud[:,0:3] + point_corner_votes[:,6:9], np.transpose(rot_mat))
            
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            plane_label[:,0:3] = np.transpose(np.dot(rot_mat, np.transpose(plane_label[:,0:3])))

            plane_left[:,0:3] = np.transpose(np.dot(rot_mat, np.transpose(plane_left[:,0:3])))
            plane_right[:,0:3] = np.transpose(np.dot(rot_mat, np.transpose(plane_right[:,0:3])))
            plane_upper[:,0:3] = np.transpose(np.dot(rot_mat, np.transpose(plane_upper[:,0:3])))
            plane_lower[:,0:3] = np.transpose(np.dot(rot_mat, np.transpose(plane_lower[:,0:3])))
            plane_front[:,0:3] = np.transpose(np.dot(rot_mat, np.transpose(plane_front[:,0:3])))
            plane_back[:,0:3] = np.transpose(np.dot(rot_mat, np.transpose(plane_back[:,0:3])))
            
            bboxes[:,0:3] = np.dot(bboxes[:,0:3], np.transpose(rot_mat))
            bboxes[:,6] -= rot_angle
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]
            point_votes[:,4:7] = point_votes_end[:,4:7] - point_cloud[:,0:3]
            point_votes[:,7:10] = point_votes_end[:,7:10] - point_cloud[:,0:3]
            point_corner_votes[:,0:3] = point_corner_votes_end[:,0:3] - point_cloud[:,0:3]
            point_corner_votes[:,3:6] = point_corner_votes_end[:,3:6] - point_cloud[:,0:3]
            point_corner_votes[:,6:9] = point_corner_votes_end[:,6:9] - point_cloud[:,0:3]
            
            # Augment RGB color
            if False:#self.use_color:
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
            plane_label[:,0:3] /= scale_ratio

            plane_front[:,0:3] /= scale_ratio
            plane_back[:,0:3] /= scale_ratio
            plane_left[:,0:3] /= scale_ratio
            plane_right[:,0:3] /= scale_ratio
            plane_upper[:,0:3] /= scale_ratio
            plane_lower[:,0:3] /= scale_ratio
            
            bboxes[:,0:3] *= scale_ratio
            bboxes[:,3:6] *= scale_ratio
            point_votes[:,1:4] *= scale_ratio
            point_votes[:,4:7] *= scale_ratio
            point_votes[:,7:10] *= scale_ratio
            point_corner_votes[:,0:3] *= scale_ratio
            point_corner_votes[:,3:6] *= scale_ratio
            point_corner_votes[:,6:9] *= scale_ratio
            if self.use_height:
                point_cloud[:,-1] *= scale_ratio[0,0]

        # load voxel data
        vox = pc_util.point_cloud_to_voxel_scene(point_cloud[:,0:3])
        bbx_for_vox = np.unique(bboxes, axis=0)
        bbx_for_vox_processed = pc_util.process_bbx(bbx_for_vox)
        vox_center = pc_util.center_to_volume_gaussion(bbx_for_vox_processed, dev=self.center_dev)
        corner_vox = pc_util.get_corner(bbx_for_vox_processed) # without angle 
        # corner_vox = pc_util.get_oriented_corners(bbx_for_vox) # with angle
        vox_corner = pc_util.point_to_volume_gaussion(corner_vox, dev=self.corner_dev)
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
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]
        point_votes_corner = point_corner_votes[choices,:]
        point_sem = point_sem[choices,:]
        plane_mask = plane_mask[choices]
        plane_upper = plane_upper[choices,:]
        plane_lower = plane_lower[choices,:]
        plane_left = plane_left[choices,:]#cues['plane_votes_left']
        plane_right = plane_right[choices,:]#cues['plane_votes_right']
        plane_front = plane_front[choices,:]#cues['plane_votes_front']
        plane_back = plane_back[choices,:]#cues['plane_votes_back']
        plane_label = plane_label[choices,:]

        plane_votes_rot_front = np.concatenate([plane_front[:,0:3], plane_front[:,4:7], plane_front[:,8:11]], 1)
        plane_votes_off_front = plane_front[:,[3,7,11]]

        plane_votes_rot_back = np.concatenate([plane_back[:,0:3], plane_back[:,4:7], plane_back[:,8:11]], 1)
        plane_votes_off_back = plane_back[:,[3,7,11]]

        plane_votes_rot_lower = np.concatenate([plane_lower[:,0:3], plane_lower[:,4:7], plane_lower[:,8:11]], 1)
        plane_votes_off_lower = plane_lower[:,[3,7,11]]

        plane_votes_rot_upper = np.concatenate([plane_upper[:,0:3], plane_upper[:,4:7], plane_upper[:,8:11]], 1)
        plane_votes_off_upper = plane_upper[:,[3,7,11]]

        plane_votes_rot_left = np.concatenate([plane_left[:,0:3], plane_left[:,4:7], plane_left[:,8:11]], 1)
        plane_votes_off_left = plane_left[:,[3,7,11]]

        plane_votes_rot_right = np.concatenate([plane_right[:,0:3], plane_right[:,4:7], plane_right[:,8:11]], 1)
        plane_votes_off_right = plane_right[:,[3,7,11]]
        
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

        ret_dict['point_sem_cls_label'] = point_sem.astype(np.int64)
        ret_dict['vote_label_corner'] = point_votes_corner.astype(np.float32)

        ret_dict['plane_label'] = np.concatenate([point_cloud, plane_label], 1).astype(np.float32)
        ret_dict['plane_label_mask'] = plane_mask.astype(np.float32)
        ret_dict['plane_votes_rot_front'] = plane_votes_rot_front.astype(np.float32)
        ret_dict['plane_votes_off_front'] = plane_votes_off_front.astype(np.float32)
        
        ret_dict['plane_votes_rot_back'] = plane_votes_rot_back.astype(np.float32)
        ret_dict['plane_votes_off_back'] = plane_votes_off_back.astype(np.float32)
        
        ret_dict['plane_votes_rot_left'] = plane_votes_rot_left.astype(np.float32)
        ret_dict['plane_votes_off_left'] = plane_votes_off_left.astype(np.float32)
        
        ret_dict['plane_votes_rot_right'] = plane_votes_rot_right.astype(np.float32)
        ret_dict['plane_votes_off_right'] = plane_votes_off_right.astype(np.float32)
        
        ret_dict['plane_votes_rot_lower'] = plane_votes_rot_lower.astype(np.float32)
        ret_dict['plane_votes_off_lower'] = plane_votes_off_lower.astype(np.float32)
        
        ret_dict['plane_votes_rot_upper'] = plane_votes_rot_upper.astype(np.float32)
        ret_dict['plane_votes_off_upper'] = plane_votes_off_upper.astype(np.float32)

        ret_dict['scan_name'] = scan_name

        ret_dict['voxel'] =np.expand_dims(vox.astype(np.float32), 0)
#         ret_dict['sem_voxel'] =np.array(sem_vox, np.float32)
        ret_dict['vox_center'] = np.expand_dims(np.array(vox_center, np.float32), 0)
        ret_dict['vox_corner'] = np.expand_dims(np.array(vox_corner, np.float32), 0)

        ret_dict['aug_yz'] = point_yz
        ret_dict['aug_xz'] = point_xz
        ret_dict['aug_rot'] = point_rot
        
        return ret_dict

def viz_votes(pc, point_votes, point_votes_mask, name='test'):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds,3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds,6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, name+'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, name+'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, name+'pc_obj_voted3.ply')

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

def viz_plane(point_planes, point_planes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_planes_mask==1)
    pc_plane = point_planes[inds,:]
    cmap = pc_util.write_ply_color_multi(pc_plane[:,:3], pc_plane[:,4:], 'pc_obj_planes{}.ply'.format(name))
    return cmap

def viz_plane_perside(point_planes, point_planes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_planes_mask==1)
    pc_plane = point_planes[inds,:]

    #whitened = whiten(pc_plane[:,3:7])
    '''
    planes = {}

    count = 0
    for plane in pc_plane[:,3:7]:
        check = 1
        for k in planes:
            if np.array_equal(planes[k], plane):
                check *= 0
                break
        if check == 1:
            planes[count] = plane
            count += 1
    '''
    planes = kmeans(pc_plane[:,3:7], 40)[0] ### Get 40 planes
    #temp_planes = []
    #for k in planes:
    #    temp_planes.append(planes[k])
    #planes = np.stack(temp_planes)
    #import pdb;pdb.set_trace()
    
    final_scene = pc_plane[:,:3]
    final_labels = np.zeros(pc_plane.shape[0])
    cur_scene = pc_plane[:,:3]
    count = 0
    for j in range(len(planes)):
        cur_plane = planes[j,:]#np.stack([planes[j,:]]*cur_scene.shape[0])
        if np.sum(cur_plane) == 0:
            continue
        ### Sample 1000 points
        choice = np.random.choice(cur_scene.shape[0], 500, replace=False)
        ### get z
        xy = cur_scene[choice,:2]
        z = -(np.sum(planes[j,:2]*xy, 1) + planes[j,3]) / planes[j,2]
        new_xyz = np.concatenate([xy,np.expand_dims(z, -1)], 1)
        #pc_util.write_ply_label(np.concatenate([final_scene, new_xyz], 0), np.concatenate([final_labels, np.ones(500)*(count+1)], 0), 'just_plane_%d.ply' % (j), count+2)
        #import pdb;pdb.set_trace()
        final_scene = np.concatenate([final_scene, new_xyz], 0)
        final_labels = np.concatenate([final_labels, np.ones(500)*(count+1)], 0)
        count += 1
    #pc_util.write_ply_label(cur_scene, np.squeeze(labels), '%d_plane_visual.ply' % i, len(planes))
    #import pdb;pdb.set_trace()
    pc_util.write_ply_label(final_scene, final_labels, 'pc_obj_planes_oneside{}.ply'.format(name), count+1)
    #cmap = pc_util.write_ply_color_multi(pc_plane[:,:3], pc_plane[:,3:], 'pc_obj_planes{}.ply'.format(name))

def viz_plane_leftperside(point_planes, point_planes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_planes_mask==1)
    pc_plane = point_planes[inds,:]

    planes = kmeans(pc_plane[:,3:7], 40)[0] ### Get 40 planes
    #temp_planes = []
    #for k in planes:
    #    temp_planes.append(planes[k])
    #planes = np.stack(temp_planes)
    #import pdb;pdb.set_trace()
    
    final_scene = pc_plane[:,:3]
    final_labels = np.zeros(pc_plane.shape[0])
    cur_scene = pc_plane[:,:3]
    count = 0
    for j in range(len(planes)):
        cur_plane = planes[j,:]#np.stack([planes[j,:]]*cur_scene.shape[0])
        if np.sum(cur_plane) == 0:
            continue
        ### Sample 1000 points
        choice = np.random.choice(cur_scene.shape[0], 500, replace=False)
        ### get z
        xz = np.stack([cur_scene[choice,0], cur_scene[choice,2]], 1)
        y = -(np.sum(np.array([planes[j,0], planes[j,2]])*xz, 1) + planes[j,3]) / planes[j,1]
        new_xyz = np.stack([xz[:,0],y,xz[:,1]], 1)
        #pc_util.write_ply_label(np.concatenate([final_scene, new_xyz], 0), np.concatenate([final_labels, np.ones(500)*(count+1)], 0), 'just_plane_%d.ply' % (j), count+2)
        #import pdb;pdb.set_trace()
        final_scene = np.concatenate([final_scene, new_xyz], 0)
        final_labels = np.concatenate([final_labels, np.ones(500)*(count+1)], 0)
        count += 1
    #pc_util.write_ply_label(cur_scene, np.squeeze(labels), '%d_plane_visual.ply' % i, len(planes))
    #import pdb;pdb.set_trace()
    pc_util.write_ply_label(final_scene, final_labels, 'pc_obj_planes_oneside{}.ply'.format(name), count+1)
    #cmap = pc_util.write_ply_color_multi(pc_plane[:,:3], pc_plane[:,3:], 'pc_obj_planes{}.ply'.format(name))

    
if __name__=='__main__':
    d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sample = d[200]
    example = sample
    i_example = 200
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        sample['heading_class_label'], sample['heading_residual_label'],
        sample['size_class_label'], sample['size_residual_label'])
    pc_util.write_ply_label(example['point_clouds'][:,:3], example['point_sem_cls_label'][:,0], 'pc_sem_{}.ply'.format(str(i_example)),  18)
    viz_votes(example['point_clouds'], example['vote_label_corner'],
              example['vote_label_mask'],name=str(i_example)+'corner')
    viz_plane(example['plane_label'],
              example['plane_label_mask'],name=str(i_example)+'plane')
    viz_plane_perside(np.concatenate([example['plane_label'][:,:3], np.concatenate([example['plane_votes_rot_upper'][:,:3], np.expand_dims(example['plane_votes_off_upper'][:,0], -1)], 1)], 1), example['plane_label_mask'],name=str(i_example)+'plane_oneside')
    viz_plane_leftperside(np.concatenate([example['plane_label'][:,:3], np.concatenate([example['plane_votes_rot_front'][:,:3], np.expand_dims(example['plane_votes_off_front'][:,0], -1)], 1)], 1), example['plane_label_mask'],name=str(i_example)+'plane_oneside_front')
