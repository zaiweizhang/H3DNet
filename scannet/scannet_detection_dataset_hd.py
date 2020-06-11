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

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DIST_THRESH = 0.2
VAR_THRESH = 1e-2
CENTER_THRESH = 0.1
LOWER_THRESH = 1e-6
NUM_POINT = 100
NUM_POINT_LINE = 10
LINE_THRESH = 0.2
MIND_THRESH = 0.1

def check_upright(para_points):
    return (para_points[0][-1] == para_points[1][-1]) and (para_points[1][-1] == para_points[2][-1]) and (para_points[2][-1] == para_points[3][-1])

def check_z(plane_equ, para_points):
    return np.sum(para_points[:,2] + plane_equ[-1]) / 4.0 < LOWER_THRESH

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
    return bbox, (center - vx - vy - vz)[0], (center - vx - vy - vz)[1], (center - vx - vy - vz)[2], (center + vx + vy + vz)[0], (center + vx + vy + vz)[1], (center + vx + vy + vz)[2],

def get_linesel(points, xmin, xmax, ymin, ymax):
    sel1 = np.abs(points[:,0] - xmin) < LINE_THRESH
    sel2 = np.abs(points[:,0] - xmax) < LINE_THRESH
    sel3 = np.abs(points[:,1] - ymin) < LINE_THRESH
    sel4 = np.abs(points[:,1] - ymax) < LINE_THRESH
    return sel1, sel2, sel3, sel4
    
def get_linesel2(points, ymin, ymax, zmin, zmax, axis=0):
    sel3 = np.abs(points[:,axis] - ymin) < LINE_THRESH
    sel4 = np.abs(points[:,axis] - ymax) < LINE_THRESH
    return sel3, sel4

class ScannetDetectionDataset(Dataset):
       
    def __init__(self, data_path=None, split_set='train', num_points=20000, center_dev=2.0, corner_dev=1.0,
                 use_color=False, use_height=False, augment=False, use_angle=False, vsize=0.06, use_tsdf=0, use_18cls=1):

        # self.data_path = os.path.join('/scratch/cluster/yanght/Dataset/', 'scannet_train_detection_data')
        self.data_path = data_path
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
        self.use_angle = use_angle
        self.augment = augment

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
        meta_vertices = np.load(os.path.join(self.data_path, scan_name)+'_all_noangle_40cls.npy') ### Need to change the name here
        
        instance_labels = meta_vertices[:,-2]
        semantic_labels = meta_vertices[:,-1]
        
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
        angle_label = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        ### For statistics
        surface_cue = np.zeros((MAX_NUM_OBJ))
        line_cue = np.zeros((MAX_NUM_OBJ,))
        
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
        meta_vertices = meta_vertices[choices]
        
        pcl_color = pcl_color[choices]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                # target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                meta_vertices[:, 0] = -1 * meta_vertices[:, 0]                
                meta_vertices[:, 6] = -1 * meta_vertices[:, 6]
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                # target_bboxes[:,1] = -1 * target_bboxes[:,1]
                meta_vertices[:, 1] = -1 * meta_vertices[:, 1]
                meta_vertices[:, 6] = -1 * meta_vertices[:, 6]
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle).astype(np.float32)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            meta_vertices[:, :6] = rotate_aligned_boxes(meta_vertices[:, :6], rot_mat)
            meta_vertices[:, 6] += rot_angle
        
        # ------------------------------- Plane and point ------------------------------
        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)

        point_boundary_mask_z = np.zeros(self.num_points)
        point_boundary_mask_xy = np.zeros(self.num_points)
        point_boundary_offset_z = np.zeros([self.num_points, 3])
        point_boundary_offset_xy = np.zeros([self.num_points, 3])
        point_boundary_sem_z = np.zeros([self.num_points, 3+2+1])
        point_boundary_sem_xy = np.zeros([self.num_points, 3+1+1])

        point_line_mask = np.zeros(self.num_points)
        point_line_offset = np.zeros([self.num_points, 3])
        point_line_sem = np.zeros([self.num_points, 3+1])

        point_sem_label = np.zeros(self.num_points)
        
        selected_instances = []
        selected_centers = []
        selected_centers_support = []
        selected_centers_bsupport = []
        obj_meta = []

        counter = -1
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]

            if semantic_labels[ind[0]] in DC.nyu40ids:
                counter += 1
                idx_instance = counter
                x = point_cloud[ind,:3]
                ### Meta information here
                meta = meta_vertices[ind[0]]
                obj_meta.append(meta)
                
                ### Get the centroid here
                center = meta[:3]

                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
                point_sem_label[ind] = DC.nyu40id2class_sem[meta[-1]]
                                
                ### Corners
                corners, xmin, ymin, zmin, xmax, ymax, zmax = params2bbox(center, meta[3], meta[4], meta[5], meta[6])
                
                ## Get lower four lines
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
                alldist = np.abs(np.sum(x*plane_lower[:3], 1) + plane_lower[-1])
                mind = np.min(alldist)
                sel = np.abs(alldist - mind) < DIST_THRESH
                
                ## Get lower four lines
                line_sel1, line_sel2, line_sel3, line_sel4 = get_linesel(x[sel], xmin, xmax, ymin, ymax)
                if np.sum(line_sel1) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = np.mean(x[sel][line_sel1], axis=0)
                    linecenter[1] = (ymin+ymax)/2.0
                    point_line_offset[ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(line_sel2) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = np.mean(x[sel][line_sel2], axis=0)
                    linecenter[1] = (ymin+ymax)/2.0
                    point_line_offset[ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(line_sel3) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel3]] = 1.0
                    linecenter = np.mean(x[sel][line_sel3], axis=0)
                    linecenter[0] = (xmin+xmax)/2.0
                    point_line_offset[ind[sel][line_sel3]] = linecenter - x[sel][line_sel3]
                    point_line_sem[ind[sel][line_sel3]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(line_sel4) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel4]] = 1.0
                    linecenter = np.mean(x[sel][line_sel4], axis=0)
                    linecenter[0] = (xmin+xmax)/2.0
                    point_line_offset[ind[sel][line_sel4]] = linecenter - x[sel][line_sel4]
                    point_line_sem[ind[sel][line_sel4]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                ### Set the surface labels here
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, np.mean(x[sel][:,2])])
                    sel_global = ind[sel]
                    point_boundary_mask_z[sel_global] = 1.0
                    point_boundary_sem_z[sel_global] = np.array([center[0], center[1], center[2], xmax - xmin, ymax - ymin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_z[sel_global] = center - x[sel]
                                    
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_upper[:3], 1) + plane_upper[-1])
                mind = np.min(alldist)
                sel = np.abs(alldist - mind) < DIST_THRESH
                ## Get upper four lines
                line_sel1, line_sel2, line_sel3, line_sel4 = get_linesel(x[sel], xmin, xmax, ymin, ymax)
                if np.sum(line_sel1) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = np.mean(x[sel][line_sel1], axis=0)
                    linecenter[1] = (ymin+ymax)/2.0
                    point_line_offset[ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(line_sel2) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = np.mean(x[sel][line_sel2], axis=0)
                    linecenter[1] = (ymin+ymax)/2.0
                    point_line_offset[ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(line_sel3) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel3]] = 1.0
                    linecenter = np.mean(x[sel][line_sel3], axis=0)
                    linecenter[0] = (xmin+xmax)/2.0
                    point_line_offset[ind[sel][line_sel3]] = linecenter - x[sel][line_sel3]
                    point_line_sem[ind[sel][line_sel3]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(line_sel4) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel4]] = 1.0
                    linecenter = np.mean(x[sel][line_sel4], axis=0)
                    linecenter[0] = (xmin+xmax)/2.0
                    point_line_offset[ind[sel][line_sel4]] = linecenter - x[sel][line_sel4]
                    point_line_sem[ind[sel][line_sel4]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, np.mean(x[sel][:,2])])
                    sel_global = ind[sel]
                    point_boundary_mask_z[sel_global] = 1.0
                    point_boundary_sem_z[sel_global] = np.array([center[0], center[1], center[2], xmax - xmin, ymax - ymin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_z[sel_global] = center - x[sel]
                                    
                ## Get left two lines
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
                sel = np.abs(alldist - mind) < DIST_THRESH
                ## Get upper four lines
                line_sel1, line_sel2 = get_linesel2(x[sel], ymin, ymax, zmin, zmax, axis=1)
                if np.sum(line_sel1) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = np.mean(x[sel][line_sel1], axis=0)
                    linecenter[2] = (zmin+zmax)/2.0
                    point_line_offset[ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(line_sel2) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = np.mean(x[sel][line_sel2], axis=0)
                    linecenter[2] = (zmin+zmax)/2.0
                    point_line_offset[ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]
                    
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_right[:3], 1) + plane_right[-1])
                mind = np.min(alldist)
                sel = np.abs(alldist - mind) < DIST_THRESH
                line_sel1, line_sel2 = get_linesel2(x[sel], ymin, ymax,  zmin, zmax, axis=1)
                if np.sum(line_sel1) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = np.mean(x[sel][line_sel1], axis=0)
                    linecenter[2] = (zmin+zmax)/2.0
                    point_line_offset[ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(line_sel2) > NUM_POINT_LINE:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = np.mean(x[sel][line_sel2], axis=0)
                    linecenter[2] = (zmin+zmax)/2.0
                    point_line_offset[ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = np.array([linecenter[0], linecenter[1], linecenter[2], np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]
                                        
                ### Get the boundary points here
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
                    plane_front = plane_front_temp
                    plane_back = np.array([plane_front_temp[0], plane_front_temp[1], plane_front_temp[2], -np.mean(newd)])
                else:
                    import pdb;pdb.set_trace()
                    print ("error with upright")
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_front[:3], 1) + plane_front[-1])
                mind = np.min(alldist)
                sel = np.abs(alldist - mind) < DIST_THRESH
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]
                                    
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_back[:3], 1) + plane_back[-1])
                mind = np.min(alldist)
                sel = np.abs(alldist - mind) < DIST_THRESH
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]
                    
        num_instance = len(obj_meta)
        obj_meta = np.array(obj_meta)
        obj_meta = obj_meta.reshape(-1, 9)

        target_bboxes_mask[0:num_instance] = 1
        target_bboxes[0:num_instance,:6] = obj_meta[:,0:6]
        
        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in obj_meta[:,-1]]   
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:num_instance] = class_ind
        size_residuals[0:num_instance, :] = \
                                            target_bboxes[0:num_instance, 3:6] - DC.mean_size_arr[class_ind,:]
        
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical
        point_sem_label = np.tile(np.expand_dims(point_sem_label, -1), (1, 3)) # make 3 votes identical

        ret_dict = {}
                
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['size_label'] = target_bboxes.astype(np.float32)[:,3:6]
        ret_dict['heading_label'] = angle_label.astype(np.float32)
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)

        if self.use_height:
            ret_dict['floor_height'] = floor_height
        
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls[0:num_instance] = \
            [DC.nyu40id2class[x] for x in obj_meta[:,-1][0:obj_meta.shape[0]]]                
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)

        ret_dict['point_sem_cls_label'] = point_sem_label.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)

        ret_dict['point_boundary_mask_z'] = point_boundary_mask_z.astype(np.float32)
        ret_dict['point_boundary_mask_xy'] = point_boundary_mask_xy.astype(np.float32)
        ret_dict['point_boundary_offset_z'] = point_boundary_offset_z.astype(np.float32)
        ret_dict['point_boundary_offset_xy'] = point_boundary_offset_xy.astype(np.float32)
        ret_dict['point_boundary_sem_z'] = point_boundary_sem_z.astype(np.float32)
        ret_dict['point_boundary_sem_xy'] = point_boundary_sem_xy.astype(np.float32)

        ret_dict['point_line_mask'] = point_line_mask.astype(np.float32)
        ret_dict['point_line_offset'] = point_line_offset.astype(np.float32)
        ret_dict['point_line_sem'] = point_line_sem.astype(np.float32)
        
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['pcl_color'] = pcl_color
        ret_dict['num_instance'] = num_instance
        
        return ret_dict
        
if __name__=='__main__':
    dset = ScannetDetectionDataset(split_set='train', use_height=True, num_points=50000, augment=False, use_angle=False)
    for i_example in range(len(dset.scan_names)):
        example = dset.__getitem__(i_example)
        pc_util.write_ply(example['point_clouds'], 'pc_{}.ply'.format(i_example))
        pc_util.write_ply(example['point_clouds'][example['point_line_mask']==1,0:3], 'pc_obj_line{}.ply'.format(i_example))
        pc_util.write_ply(example['point_clouds'][example['point_boundary_mask_z']==1,0:3], 'pc_obj_boundary_z{}.ply'.format(i_example))
        pc_util.write_ply(example['point_clouds'][example['point_boundary_mask_xy']==1,0:3], 'pc_obj_boundary_xy{}.ply'.format(i_example))
        print (i_example)
