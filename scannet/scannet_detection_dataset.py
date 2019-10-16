# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from model_util_scannet import rotate_aligned_boxes

from model_util_scannet import ScannetDatasetConfig

from scipy.optimize import linear_sum_assignment
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 100#64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

def pdist2(x1, x2):
    """ Computes the squared Euclidean distance between all pairs """
    C = -2*np.matmul(x1,x2.T)
    nx = np.sum(np.square(x1),1,keepdims=True)
    ny = np.sum(np.square(x2),1,keepdims=True)
    costMatrix = (C + ny.T) + nx
    return costMatrix

def find_idx(targetbb, selected_centers, selected_centers_support, selected_centers_bsupport):
    center_matrix = np.stack(selected_centers)
    assert(center_matrix.shape[0] == targetbb.shape[0])
    #costMatrix = np.zeros((selected_centers.shape[0], selected_centers.shape[0]))
    costMatrix = pdist2(center_matrix, targetbb)
    row_ind, col_ind = linear_sum_assignment(costMatrix)
    idx2bb = {i:row_ind[i] for i in range(center_matrix.shape[0])}
    support_idx = []
    bsupport_idx = []
    for center in selected_centers_support:
        check = 0
        for idx in range(len(selected_centers)):
            if np.array_equal(selected_centers[idx], center):
                check = 1
                break
        if check == 0:
            print("error with data")
        if idx not in support_idx:
            support_idx.append(int(idx2bb[idx]))
    for center in selected_centers_bsupport:
        check = 0
        for idx in range(len(selected_centers)):
            if np.array_equal(selected_centers[idx], center):
                check = 1
                break
        if check == 0:
            print("error with data")
        if idx not in bsupport_idx:
            bsupport_idx.append(int(idx2bb[idx]))
    return support_idx, bsupport_idx

class ScannetDetectionDataset(Dataset):
       
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, augment=False):

        self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data')
        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))
        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                'scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('illegal split name')
            return
        
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.augment = augment
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        
        scan_name = self.scan_names[idx]        
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name)+'_vert.npy')
        instance_labels = np.load(os.path.join(self.data_path, scan_name)+'_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name)+'_sem_label.npy')
        support_labels = np.load(os.path.join(self.data_path, scan_name)+'_support_label.npy')
        support_instance_labels = np.load(os.path.join(self.data_path, scan_name)+'_support_instance_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy')
        
        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0
            pcl_color = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
            
        # ------------------------------- LABELS ------------------------------        
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        target_bboxes_support = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask_support = np.zeros((MAX_NUM_OBJ))    
        angle_classes_support = np.zeros((MAX_NUM_OBJ,))
        angle_residuals_support = np.zeros((MAX_NUM_OBJ,))
        size_classes_support = np.zeros((MAX_NUM_OBJ,))
        size_residuals_support = np.zeros((MAX_NUM_OBJ, 3))

        target_bboxes_bsupport = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask_bsupport = np.zeros((MAX_NUM_OBJ))    
        angle_classes_bsupport = np.zeros((MAX_NUM_OBJ,))
        angle_residuals_bsupport = np.zeros((MAX_NUM_OBJ,))
        size_classes_bsupport = np.zeros((MAX_NUM_OBJ,))
        size_residuals_bsupport = np.zeros((MAX_NUM_OBJ, 3))

        before_sample = np.unique(instance_labels)
        while True:
            orig_point_cloud = np.copy(point_cloud)
            temp_point_cloud, choices = pc_util.random_sampling(orig_point_cloud,
                                                           self.num_points, return_choices=True)
            after_sample = np.unique(instance_labels[choices])
            if np.array_equal(before_sample, after_sample):
                point_cloud = temp_point_cloud
                break
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
                        
        pcl_color = pcl_color[choices]
        
        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

        # ------------------------------- SUPPORT RELATION ------------------------------
        support_labels = support_labels[choices]
        support_instance_labels = support_instance_labels[choices]
        
        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_votes = np.zeros([self.num_points, 3])
        point_votes_bbcenter = np.zeros([self.num_points, 3])
        point_votes_corner = np.zeros([self.num_points, 3])
        point_votes_support = np.zeros([self.num_points, 3])
        point_votes_bsupport = np.zeros([self.num_points, 3])
        point_votes_support_middle = np.zeros([self.num_points, 3])
        point_votes_bsupport_middle = np.zeros([self.num_points, 3])
        point_votes_support_offset = np.zeros([self.num_points, 3])
        point_votes_bsupport_offset = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        point_votes_support_mask = np.zeros(self.num_points)
        point_votes_bsupport_mask = np.zeros(self.num_points)
        num_instance = len(instance_bboxes)
        #assert(num_instance == len(np.unique(instance_labels)) - 1)
        """
        if 0 in np.unique(instance_labels):
            if ((len(np.unique(instance_labels)) - 1 - 1)*2 == support_instance_labels.shape[1]) == False:
                import pdb;pdb.set_trace()
        else:
            assert((len(np.unique(instance_labels)) - 1)*2 == support_instance_labels.shape[1])
        """
        selected_instances = []
        selected_centers = []
        selected_centers_support = []
        selected_centers_bsupport = []
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                
                lower = x.min(0)
                upper = x.max(0)
                center = 0.5*(lower + upper)
                bbcenters = [np.array([0.5*(lower[0] + upper[0]), 0.5*(lower[1] + upper[1]), lower[2]]), np.array([0.5*(lower[0] + upper[0]), 0.5*(lower[1] + upper[1]), upper[2]]), np.array([lower[0], 0.5*(lower[1] + upper[1]), 0.5*(lower[2] + upper[2])]), np.array([upper[0], 0.5*(lower[1] + upper[1]), 0.5*(lower[2] + upper[2])]), np.array([0.5*(lower[0] + upper[0]), lower[1], 0.5*(lower[2] + upper[2])]), np.array([0.5*(lower[0] + upper[0]), upper[1], 0.5*(lower[2] + upper[2])])]
                corners = [np.array([lower[0], lower[1], lower[2]]), np.array([upper[0], lower[1], lower[2]]), np.array([lower[0], upper[1], lower[2]]), np.array([upper[0], upper[1], lower[2]]), np.array([lower[0], lower[1], upper[2]]), np.array([upper[0], lower[1], upper[2]]), np.array([lower[0], upper[1], upper[2]]), np.array([upper[0], upper[1], upper[2]])]
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
                
                xtemp = np.stack([x]*len(bbcenters))
                dist = np.sum(np.square(xtemp - np.expand_dims(bbcenters, 1)), axis=2)
                sel = np.argmin(dist, 0)
                xtemp = np.stack([x]*len(corners))
                dist = np.sum(np.square(xtemp - np.expand_dims(corners, 1)), axis=2)
                sel_corner = np.argmin(dist, 0)
                for i in range(len(ind)):
                    point_votes_bbcenter[ind[i], :] = bbcenters[sel[i]] - x[i,:]
                    point_votes_corner[ind[i], :] = corners[sel[i]] - x[i,:]
                selected_instances.append(i_instance)
                selected_centers.append(center)
                
                # find votes for supported object
                j_instance = np.unique(support_instance_labels[ind][:, :(num_instance-1)])
                if 0 in j_instance:
                    j_instance = j_instance[1:]
                if len(j_instance) > 0:
                    centers = []
                    bbcenters = []
                    #offset = []
                    for j_inst in j_instance:
                        if j_inst == 0:
                            continue
                        if j_inst not in np.unique(instance_labels):
                            continue
                        jind = np.where(instance_labels == j_inst)[0]
                        if semantic_labels[jind[0]] in DC.nyu40ids:
                            j = point_cloud[jind,:3]
                            jcenter = 0.5*(j.min(0) + j.max(0))
                            centers.append(jcenter)
                            #offset.append(jcenter - x)
                            lower = j.min(0)
                            upper = j.max(0)
                            bbcenters.append(np.array([lower[0], lower[1], lower[2]]))
                            bbcenters.append(np.array([upper[0], lower[1], lower[2]]))
                            bbcenters.append(np.array([lower[0], upper[1], lower[2]]))
                            bbcenters.append(np.array([upper[0], upper[1], lower[2]]))
                            
                            selected_centers_support.append(jcenter)
                    if len(bbcenters) > 0:
                        #offset = np.stack(offset)
                        #xtemp = np.stack([x]*len(centers))
                        xtemp = np.stack([x]*len(bbcenters))
                        dist = np.sum(np.square(xtemp - np.expand_dims(bbcenters, 1)), axis=2)
                        sel = np.argmin(dist, 0)

                        #other_sel = np.argmin(dist, 1)
                        #other_centers = x[other_sel]
                        #other_dist = np.sum(np.square(xtemp - np.expand_dims(other_centers, 1)), axis=2)
                        #sel_near = np.argmin(other_dist, 0)
                        #othercenter = np.stack([other_centers]*len(ind))
                        
                        #tempcenter = np.stack([np.array(bbcenters)]*len(ind))
                        tempcenter = np.array(bbcenters)
                        for i in range(len(ind)):
                            point_votes_support[ind[i], :] = tempcenter[sel[i], :] - x[i,:]
                            #point_votes_support_middle[ind[i], :] = othercenter[i, sel_near[i], :] - x[i,:]
                            #point_votes_support_offset[ind[i], :] = tempcenter[i, sel[i], :] - othercenter[i, sel_near[i], :]
                        point_votes_support_mask[ind] = 1.0
            
                # find votes for bsupported object
                j_instance = np.unique(support_instance_labels[ind][:, (num_instance-1):])
                if 0 in j_instance:
                    j_instance = j_instance[1:]
                if len(j_instance) > 0:
                    centers = []
                    bbcenters = []
                    #offset = []
                    for j_inst in j_instance:
                        if j_inst == 0:
                            continue
                        if j_inst not in np.unique(instance_labels):
                            continue
                        jind = np.where(instance_labels == j_inst)[0]
                        if semantic_labels[jind[0]] in DC.nyu40ids:
                            j = point_cloud[jind,:3]
                            jcenter = 0.5*(j.min(0) + j.max(0))
                            centers.append(jcenter)
                            #offset.append(jcenter - x)
                            lower = x.min(0)
                            upper = x.max(0)
                            bbcenters.append(np.array([lower[0], lower[1], lower[2]]))
                            bbcenters.append(np.array([upper[0], lower[1], lower[2]]))
                            bbcenters.append(np.array([lower[0], upper[1], lower[2]]))
                            bbcenters.append(np.array([upper[0], upper[1], lower[2]]))
                            selected_centers_bsupport.append(jcenter)
                    if len(bbcenters) > 0:
                        #offset = np.stack(offset)
                        xtemp = np.stack([x]*len(bbcenters))
                        dist = np.sum(np.square(xtemp - np.expand_dims(bbcenters, 1)), axis=2)
                        sel = np.argmin(dist, 0)

                        #other_sel = np.argmin(dist, 1)
                        #other_centers = x[other_sel]
                        #other_dist = np.sum(np.square(xtemp - np.expand_dims(other_centers, 1)), axis=2)
                        #sel_near = np.argmin(other_dist, 0)
                        #othercenter = np.stack([other_centers]*len(ind))

                        #tempcenter = np.stack([np.array(centers)]*len(ind))
                        tempcenter = np.array(bbcenters)
                        for i in range(len(ind)):
                            point_votes_bsupport[ind[i], :] = tempcenter[sel[i], :] - x[i,:]
                            #point_votes_bsupport_middle[ind[i], :] = othercenter[i, sel_near[i], :] - x[i,:]
                            #point_votes_bsupport_offset[ind[i], :] = tempcenter[i, sel[i], :] - othercenter[i, sel_near[i], :]
                        point_votes_bsupport_mask[ind] = 1.0

        support_idx, bsupport_idx = find_idx(target_bboxes[0:instance_bboxes.shape[0],0:3], selected_centers, selected_centers_support, selected_centers_bsupport)
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical
        point_votes_bbcenter = np.tile(point_votes_bbcenter, (1, 3)) # make 3 votes identical
        point_votes_corner = np.tile(point_votes_corner, (1, 3)) # make 3 votes identical
        point_votes_support = np.tile(point_votes_support, (1, 3)) # make 3 votes identical
        #point_votes_support_middle = np.tile(point_votes_support_middle, (1, 3)) # make 3 votes identical
        #point_votes_support_offset = np.tile(point_votes_support_offset, (1, 3)) # make 3 votes identical
        point_votes_bsupport = np.tile(point_votes_bsupport, (1, 3)) # make 3 votes identical
        #point_votes_bsupport_middle = np.tile(point_votes_bsupport_middle, (1, 3)) # make 3 votes identical
        #point_votes_bsupport_offset = np.tile(point_votes_bsupport_offset, (1, 3)) # make 3 votes identical 

        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:,-1]]   
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]

        ret_dict = {}
                
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [DC.nyu40id2class[x] for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]                
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)

        ### Label for support
        target_bboxes_support[0:len(support_idx),...] = target_bboxes[support_idx,...]
        ret_dict['center_label_support'] = target_bboxes_support.astype(np.float32)[:,0:3]
        angle_classes_support[0:len(support_idx),...] = angle_classes[support_idx,...]
        ret_dict['heading_class_label_support'] = angle_classes_support.astype(np.int64)
        angle_residuals_support[0:len(support_idx),...] = angle_residuals[support_idx,...]
        ret_dict['heading_residual_label_support'] = angle_residuals_support.astype(np.float32)
        size_classes_support[0:len(support_idx),...] = size_classes[support_idx, ...]
        ret_dict['size_class_label_support'] = size_classes_support.astype(np.int64)
        size_residuals_support[0:len(support_idx),...] = size_residuals[support_idx, ...]
        ret_dict['size_residual_label_support'] = size_residuals_support.astype(np.float32)
        
        target_bboxes_semcls_support = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls_support[0:len(support_idx)] = target_bboxes_semcls[support_idx]
        ret_dict['sem_cls_label_support'] = target_bboxes_semcls_support.astype(np.int64)
        target_bboxes_mask_support[0:len(support_idx),...] = target_bboxes_mask[support_idx, ...]
        ret_dict['box_label_mask_support'] = target_bboxes_mask_support.astype(np.float32)

        ### Label for bsupport
        target_bboxes_bsupport[0:len(bsupport_idx),...] = target_bboxes[bsupport_idx,...]
        ret_dict['center_label_bsupport'] = target_bboxes_bsupport.astype(np.float32)[:,0:3]
        angle_classes_bsupport[0:len(bsupport_idx),...] = angle_classes[bsupport_idx,...]
        ret_dict['heading_class_label_bsupport'] = angle_classes_bsupport.astype(np.int64)
        angle_residuals_bsupport[0:len(bsupport_idx),...] = angle_residuals[bsupport_idx,...]
        ret_dict['heading_residual_label_bsupport'] = angle_residuals_bsupport.astype(np.float32)
        size_classes_bsupport[0:len(bsupport_idx),...] = size_classes[bsupport_idx, ...]
        ret_dict['size_class_label_bsupport'] = size_classes_bsupport.astype(np.int64)
        size_residuals_bsupport[0:len(bsupport_idx),...] = size_residuals[bsupport_idx, ...]
        ret_dict['size_residual_label_bsupport'] = size_residuals_bsupport.astype(np.float32)
        
        target_bboxes_semcls_bsupport = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls_bsupport[0:len(bsupport_idx)] = target_bboxes_semcls[bsupport_idx]
        ret_dict['sem_cls_label_bsupport'] = target_bboxes_semcls_bsupport.astype(np.int64)
        target_bboxes_mask_bsupport[0:len(bsupport_idx),...] = target_bboxes_mask[bsupport_idx, ...]
        ret_dict['box_label_mask_bsupport'] = target_bboxes_mask_bsupport.astype(np.float32)

        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_bcenter'] = point_votes_bbcenter.astype(np.float32)
        ret_dict['vote_label_corner'] = point_votes_corner.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['vote_label_support'] = point_votes_support.astype(np.float32)
        #ret_dict['vote_label_support_middle'] = point_votes_support_middle.astype(np.float32)
        #ret_dict['vote_label_support_offset'] = point_votes_support_offset.astype(np.float32)
        ret_dict['vote_label_mask_support'] = point_votes_support_mask.astype(np.int64)
        ret_dict['vote_label_bsupport'] = point_votes_bsupport.astype(np.float32)
        #ret_dict['vote_label_bsupport_middle'] = point_votes_bsupport_middle.astype(np.float32)
        #ret_dict['vote_label_bsupport_offset'] = point_votes_bsupport_offset.astype(np.float32)
        ret_dict['vote_label_mask_bsupport'] = point_votes_bsupport_mask.astype(np.int64)
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['pcl_color'] = pcl_color
        ret_dict['num_instance'] = num_instance
        
        return ret_dict
        
############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]    
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))
    
def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals, name=''):
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
        heading_angle = 0 # hard code to 0
        box_size = DC.mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)        
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask==1,:], 'gt_centroids{}.ply'.format(name))

    
if __name__=='__main__': 
    dset = ScannetDetectionDataset(use_height=True, num_points=40000)
    for i_example in range(1000):
        example = dset.__getitem__(i_example)
        pc_util.write_ply(example['point_clouds'], 'pc_{}.ply'.format(i_example))    
        viz_votes(example['point_clouds'], example['vote_label'],
                  example['vote_label_mask'],name=i_example)
        viz_votes(example['point_clouds'], example['vote_label_bcenter'],
                  example['vote_label_mask'],name=str(i_example)+'bcenter')
        viz_votes(example['point_clouds'], example['vote_label_corner'],
                  example['vote_label_mask'],name=str(i_example)+'corner')
        viz_votes(example['point_clouds'], example['vote_label_support'],
                  example['vote_label_mask_support'],name=str(i_example)+'support')
        #viz_votes(example['point_clouds'], example['vote_label_support_middle'],example['vote_label_mask_support'],name=str(i_example)+'support_middle')
        viz_votes(example['point_clouds'], example['vote_label_bsupport'],
                  example['vote_label_mask_bsupport'],name=str(i_example)+'bsupport')
        #viz_votes(example['point_clouds'], example['vote_label_bsupport_middle'],example['vote_label_mask_bsupport'],name=str(i_example)+'bsupport_middle')
        #viz_votes(example['point_clouds'], example['vote_label_support_offset'],example['vote_label_mask_support'],name=str(i_example)+'support_offset')
        #viz_votes(example['point_clouds'], example['vote_label_bsupport_offset'],example['vote_label_mask_bsupport'],name=str(i_example)+'bsupport_offset')
        import pdb;pdb.set_trace()
        """
        viz_obb(pc=example['point_clouds'], label=example['center_label'],
            mask=example['box_label_mask'],
            angle_classes=None, angle_residuals=None,
            size_classes=example['size_class_label'], size_residuals=example['size_residual_label'],
            name=i_example)
        viz_obb(pc=example['point_clouds'], label=example['center_label_support'],
            mask=example['box_label_mask_support'],
            angle_classes=None, angle_residuals=None,
            size_classes=example['size_class_label_support'], size_residuals=example['size_residual_label_support'],
            name=str(i_example)+'support')
        viz_obb(pc=example['point_clouds'], label=example['center_label_bsupport'],
            mask=example['box_label_mask_bsupport'],
            angle_classes=None, angle_residuals=None,
            size_classes=example['size_class_label_bsupport'], size_residuals=example['size_residual_label_bsupport'],
            name=str(i_example)+'bsupport')
        import pdb;pdb.set_trace()
        """
        
