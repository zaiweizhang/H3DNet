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
from scipy.optimize import leastsq

from scipy.cluster.vq import vq, kmeans, whiten
from sklearn import linear_model

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DIST_THRESH = 0.2#0.1
VAR_THRESH = 1e-2
CENTER_THRESH = 0.1
LOWER_THRESH = 1e-6
NUM_POINT = 100
NUM_POINT_LINE = 10
LINE_THRESH = 0.2#0.1
MIND_THRESH = 0.1

def local_regression_plane_ransac(neighborhood):
    """
    Computes parameters for a local regression plane using RANSAC
    """

    XY = neighborhood[:,:2]
    Z  = neighborhood[:,2]
    ransac = linear_model.RANSACRegressor(
                                          linear_model.LinearRegression(),
                                          residual_threshold=0.1
                                         )
    ransac.fit(XY, Z)

    inlier_mask = ransac.inlier_mask_
    inlier_ind = np.where(inlier_mask)[0]
    coeff = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_

    normal = np.concatenate((coeff, [intercept]), 0)
    normal = normal / np.linalg.norm(normal)

    points = neighborhood * np.expand_dims(inlier_mask, -1)
    ori = ((0 - neighborhood)*normal)[inlier_ind,:]

    check = np.mean(np.sum(ori, 1))
    if check < 0:
        normal *= -1
    return normal, inlier_ind
    
def check_upright(para_points):
    return (para_points[0][-1] == para_points[1][-1]) and (para_points[1][-1] == para_points[2][-1]) and (para_points[2][-1] == para_points[3][-1])

def check_z(plane_equ, para_points):
    return np.sum(para_points[:,2] + plane_equ[-1]) / 4.0 < LOWER_THRESH

def ang2cls(ang):
    ang += np.pi / 2
    cls = int(ang / (np.pi/12)) ### Split to 12 cls
    res = (ang / (np.pi/12) - cls)
    return cls,res

def check_xy(plane_equ, para_points):
    organg = (plane_equ[0]+plane_equ[1])*(np.pi/12) - np.pi/2
    slope = np.tan(organg)
    norm = np.linalg.norm([-slope, 1])*plane_equ[2]
    orgplane = np.array([-slope/norm, 1/norm, 0, plane_equ[-1]])
    return np.sum(np.sum(para_points*orgplane[:3], 1) + orgplane[3]) / 4.0 < LOWER_THRESH

def f_min(X,p):
    plane_xyz = p[0:3]
    distance = (plane_xyz*X.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)

def residuals(params, signal, X):
    return f_min(X, params)

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

def get_linesel(points, xmin, xmax, ymin, ymax):
    sel1 = np.abs(points[:,0] - xmin) < LINE_THRESH
    sel2 = np.abs(points[:,0] - xmax) < LINE_THRESH
    sel3 = np.abs(points[:,1] - ymin) < LINE_THRESH
    sel4 = np.abs(points[:,1] - ymax) < LINE_THRESH
    return sel1, sel2, sel3, sel4

### Also remove the redundent points
SWEEP_RATIO = 0.01
SWEEP_RADIUS = 0.02
def sweep(points, taxis, tcons, saxis, range_min, range_max):
    sel = np.zeros((points.shape[0]))
    smin = range_min
    while smin <= range_max:
        tempsel = np.abs(points[:,saxis] - smin) < SWEEP_RADIUS
        tempind = np.argmin(np.abs(points[tempsel,taxis] - tcons))
        if np.abs(points[tempsel,:][tempind,taxis] - tcons) < LINE_THRESH:
            sel[tempsel][tempind] = 1.0
    return sel
    
### Sweep based
def get_linesel_sweep(points, xmin, xmax, ymin, ymax):
    sel1 = sweep(points, 0, xmin, 1, ymin, ymax)
    sel2 = sweep(points, 0, xmax, 1, ymin, ymax)
    sel3 = sweep(points, 1, ymin, 0, xmin, xmax)
    sel4 = sweep(points, 1, ymax, 0, xmin, xmax)
    return sel1, sel2, sel3, sel4

def get_linesel2(points, ymin, ymax, zmin, zmax, axis=0):
    #sel3 = sweep(points, axis, ymax, 2, zmin, zmax)
    #sel4 = sweep(points, axis, ymax, 2, zmin, zmax)
    sel3 = np.abs(points[:,axis] - ymin) < LINE_THRESH
    sel4 = np.abs(points[:,axis] - ymax) < LINE_THRESH
    return sel3, sel4

class ScannetDetectionDataset(Dataset):
       
    def __init__(self, split_set='train', num_points=20000, center_dev=2.0, corner_dev=1.0,
                 use_color=False, use_height=False, augment=False, use_angle=False, vsize=0.06, use_tsdf=0, use_18cls=1):

        # self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data')
        self.data_path = os.path.join('/scratch/cluster/yanght/Dataset/', 'scannet_train_detection_data')
        self.data_path_vox = os.path.join('/scratch/cluster/bosun/data/scannet/', 'scannet_train_detection_data')
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
        plane_vertices = np.load(os.path.join(self.data_path, scan_name)+'_plane.npy')
        #plane_vertices[:,:4] = plane_vertices[:,:4] / np.linalg.norm(plane_vertices[:,:3], axis=1, keepdims=True) ## Normalize the data
        #plane_vertices[:,:4] = np.nan_to_num(plane_vertices[:,:4])
        ### Without ori
        if self.use_angle:
            meta_vertices = np.load(os.path.join(self.data_path, scan_name)+'_all_angle_40cls.npy') ### Need to change the name here
            ### Do not use data with angle for now
            #point_layout_normal = np.load(os.path.join(self.data_path, scan_name)+'_all_noangle_40cls_floor.npy') ### Need to change the name here
        else:
            ### With ori
            meta_vertices = np.load(os.path.join(self.data_path, scan_name)+'_all_noangle_40cls.npy') ### Need to change the name here
            #point_layout_normal = np.load(os.path.join(self.data_path, scan_name)+'_all_noangle_40cls_floor.npy') ### Need to change the name here
        ### Load voxel data
        sem_vox=np.load(os.path.join(self.data_path_vox, scan_name+'_vox_0.06_sem.npy'))
        vox = np.array(sem_vox>0,np.float32)
        if self.use_angle:
            vox_center = np.load(os.path.join(self.data_path_vox, scan_name+'_vox_0.06_center_angle_18.npy'))
            vox_corner = np.load(os.path.join(self.data_path_vox, scan_name+'_vox_0.06_corner_angle_18.npy'))
        else:
            vox_center = np.load(os.path.join(self.data_path_vox, scan_name+'_vox_0.06_center_noangle_18.npy'))
            vox_corner = np.load(os.path.join(self.data_path_vox, scan_name+'_vox_0.06_corner_noangle_18.npy'))

        instance_labels = meta_vertices[:,-2]
        semantic_labels = meta_vertices[:,-1]

        ### Create the dataset here
        '''
        #point_layout_in_mask = np.zeros(self.num_points)
        point_cloud = mesh_vertices[:,0:3] # do not use color for now
        point_layout_normal = np.zeros([point_cloud.shape[0],3])
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            if len(ind) <= 10:
                continue
            # find the layout related parameters
            if semantic_labels[ind[0]] in DC.nyu40ids_room:
                points = point_cloud[ind,:3]
                coeff, inlier_ind = local_regression_plane_ransac(points)
                point_layout_normal[ind,:] = coeff
        np.save(os.path.join(self.data_path, scan_name)+'_all_noangle_40cls_floor.npy', point_layout_normal)
        return
        '''
        #instance_labels = np.load(os.path.join(self.data_path, scan_name)+'_ins_label.npy')
        #semantic_labels = np.load(os.path.join(self.data_path, scan_name)+'_sem_label.npy')
        #support_labels = np.load(os.path.join(self.data_path, scan_name)+'_support_label.npy')
        #support_instance_labels = np.load(os.path.join(self.data_path, scan_name)+'_support_instance_label.npy')
        #instance_bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy')
        
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
        gt_bboxes = np.zeros((MAX_NUM_OBJ, 7))
        pert_bboxes = np.zeros((MAX_NUM_OBJ, 7))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

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
        plane_vertices = plane_vertices[choices]
        meta_vertices = meta_vertices[choices]
        #point_layout_normal = point_layout_normal[choices]
        
        pcl_color = pcl_color[choices]

        #target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        # if False:#self.augment:## Do not use augment for now
        point_yz = -1
        point_xz = -1
        point_rot = np.eye(3).astype(np.float32)
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_yz = 1
                point_cloud[:,0] = -1 * point_cloud[:,0]
                plane_vertices[:,0] = -1 * plane_vertices[:,0]
                # target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                meta_vertices[:, 0] = -1 * meta_vertices[:, 0]                
                meta_vertices[:, 6] = -1 * meta_vertices[:, 6]
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_xz = 1
                point_cloud[:,1] = -1 * point_cloud[:,1]
                plane_vertices[:,1] = -1 * plane_vertices[:,1]
                # target_bboxes[:,1] = -1 * target_bboxes[:,1]
                meta_vertices[:, 1] = -1 * meta_vertices[:, 1]
                meta_vertices[:, 6] = -1 * meta_vertices[:, 6]
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle).astype(np.float32)
            point_rot = rot_mat
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            plane_vertices[:,0:3] = np.transpose(np.dot(rot_mat, np.transpose(plane_vertices[:,0:3])))
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
        point_votes_corner1 = np.zeros([self.num_points, 3])
        point_votes_corner2 = np.zeros([self.num_points, 3])
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
        
        point_center_mask_z = np.zeros(self.num_points)
        point_center_mask_xy = np.zeros(self.num_points)

        point_layout_mask = np.ones(self.num_points)
        point_layout_in_mask = np.zeros(self.num_points)
        point_layout_sem = np.zeros(self.num_points)
        point_layout_normal = np.zeros([self.num_points,3])
        point_sem_label = np.zeros(self.num_points)
        
        ### Plane Patches
        plane_label = np.zeros([self.num_points, 3+4])
        plane_label_mask = np.zeros(self.num_points)
        plane_votes_offset = np.zeros([self.num_points, 1])
        plane_votes_label = np.zeros([self.num_points, 3])
        plane_votes_front = np.zeros([self.num_points, 4])
        plane_votes_back = np.zeros([self.num_points, 4])
        plane_votes_left = np.zeros([self.num_points, 4])
        plane_votes_right = np.zeros([self.num_points, 4])
        plane_votes_upper = np.zeros([self.num_points, 4])
        plane_votes_lower = np.zeros([self.num_points, 4])
        
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
        obj_meta = []
        
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            #if len(ind) <= 10:
            #    continue
            # find the layout related parameters
            '''
            if semantic_labels[ind[0]] in DC.nyu40ids_room:
                meta = meta_vertices[ind[0]]
                #import pdb;pdb.set_trace()
                point_layout_mask[ind] = 1.0
                point_layout_sem[ind] = meta[-1]

                points = point_cloud[ind,:3]
                #coeff, inlier_ind = local_regression_plane_ransac(points)
                #point_layout_normal[ind,:] = coeff
                #point_layout_in_mask[ind[inlier_ind]] = 1.0
            '''
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                ### Meta information here
                meta = meta_vertices[ind[0]]
                obj_meta.append(meta)
                
                ### Get the centroid here
                center = meta[:3]

                ### Corners
                corners, xmin, ymin, zmin, xmax, ymax, zmax = params2bbox(center, meta[3], meta[4], meta[5], meta[6])
                
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
                    #center = np.mean(x[sel], 0)
                    center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, np.mean(x[sel][:,2])])
                    sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                    sel_global = ind[sel]
                    point_boundary_mask_z[sel_global] = 1.0
                    point_boundary_sem_z[sel_global] = np.array([center[0], center[1], center[2], xmax - xmin, ymax - ymin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_z[sel_global] = center - x[sel]
                    sel_center = sel_global[sel_center]
                    point_center_mask_z[sel_center] = 1.0
                    
                ### Check for middle z surfaces
                [count, val] = np.histogram(alldist, bins=20)
                mind_middle = val[np.argmax(count)]
                sel_pre = np.copy(sel)
                sel = np.abs(alldist - mind_middle) < DIST_THRESH
                if np.abs(np.mean(x[sel_pre][:,2]) - np.mean(x[sel][:,2])) > MIND_THRESH:
                    ### Do not use line for middle surfaces
                    """
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
                    """
                    if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                        #center = np.mean(x[sel], 0)
                        center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, np.mean(x[sel][:,2])])
                        sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                        sel_global = ind[sel]
                        point_boundary_mask_z[sel_global] = 1.0
                        point_boundary_sem_z[sel_global] = np.array([center[0], center[1], center[2], xmax - xmin, ymax - ymin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                        point_boundary_offset_z[sel_global] = center - x[sel]
                        sel_center = sel_global[sel_center]
                        point_center_mask_z[sel_center] = 1.0
                    
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_upper[:3], 1) + plane_upper[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,0] >= xmin) & (point_cloud[:,0] <= xmax) & (point_cloud[:,1] >= ymin) & (point_cloud[:,1] <= ymax)

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
                    #center = np.mean(x[sel], 0)
                    center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, np.mean(x[sel][:,2])])
                    sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                    sel_global = ind[sel]
                    point_boundary_mask_z[sel_global] = 1.0
                    point_boundary_sem_z[sel_global] = np.array([center[0], center[1], center[2], xmax - xmin, ymax - ymin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_z[sel_global] = center - x[sel]
                    sel_center = sel_global[sel_center]
                    point_center_mask_z[sel_center] = 1.0
                    
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
                    #center = np.mean(x[sel], 0)
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]
                    #point_boundary_offset_xy[sel] = center - x[sel]
                    sel_center = sel_global[sel_center]
                    point_center_mask_xy[sel_center] = 1.0

                [count, val] = np.histogram(alldist, bins=20)
                mind_middle = val[np.argmax(count)]
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,2] >= zmin) & (point_cloud[:,2] <= zmax) & (point_cloud[:,1] >= ymin) & (point_cloud[:,1] <= ymax)
                ## Get upper four lines
                sel_pre = np.copy(sel)
                sel = np.abs(alldist - mind_middle) < DIST_THRESH
                if np.abs(np.mean(x[sel_pre][:,0]) - np.mean(x[sel][:,0])) > MIND_THRESH:
                    ### Do not use line for middle surfaces
                    """
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
                    """
                    if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                        #center = np.mean(x[sel], 0)
                        center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                        sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                        sel_global = ind[sel]
                        point_boundary_mask_xy[sel_global] = 1.0
                        point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                        point_boundary_offset_xy[sel_global] = center - x[sel]
                        #point_boundary_offset_xy[sel] = center - x[sel]
                        sel_center = sel_global[sel_center]
                        point_center_mask_xy[sel_center] = 1.0
                    
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_right[:3], 1) + plane_right[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,2] >= zmin) & (point_cloud[:,2] <= zmax) & (point_cloud[:,1] >= ymin) & (point_cloud[:,1] <= ymax)
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
                    #center = np.mean(x[sel], 0)
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]
                    sel_center = sel_global[sel_center]
                    point_center_mask_xy[sel_center] = 1.0

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
                    #center = np.mean(x[sel], 0)
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]
                    sel_center = sel_global[sel_center]
                    point_center_mask_xy[sel_center] = 1.0

                [count, val] = np.histogram(alldist, bins=20)
                mind_middle = val[np.argmax(count)]
                sel_pre = np.copy(sel)
                sel = np.abs(alldist - mind_middle) < DIST_THRESH
                if np.abs(np.mean(x[sel_pre][:,1]) - np.mean(x[sel][:,1])) > MIND_THRESH:
                    ### Do not use line for middle surfaces
                    """
                    line_sel1, line_sel2 = get_linesel2(x[sel], xmin, xmax, zmin, zmax, axis=0)
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
                    """
                    if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                        #center = np.mean(x[sel], 0)
                        center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                        sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                        sel_global = ind[sel]
                        point_boundary_mask_xy[sel_global] = 1.0
                        point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                        point_boundary_offset_xy[sel_global] = center - x[sel]
                        sel_center = sel_global[sel_center]
                        point_center_mask_xy[sel_center] = 1.0
                    
                ### Get the boundary points here
                alldist = np.abs(np.sum(x*plane_back[:3], 1) + plane_back[-1])
                mind = np.min(alldist)
                #[count, val] = np.histogram(alldist, bins=20)
                #mind = val[np.argmax(count)]
                sel = np.abs(alldist - mind) < DIST_THRESH
                if np.sum(sel) > NUM_POINT and np.var(alldist[sel]) < VAR_THRESH:
                    #sel = (np.abs(alldist - mind) < DIST_THRESH) & (point_cloud[:,0] >= xmin) & (point_cloud[:,0] <= xmax) & (point_cloud[:,2] >= zmin) & (point_cloud[:,2] <= zmax)
                    #center = np.mean(x[sel], 0)
                    center = np.array([np.mean(x[sel][:,0]), np.mean(x[sel][:,1]), (zmin+zmax)/2.0])
                    #point_boundary_offset_xy[sel] = center - x[sel]
                    sel_center = np.sqrt(np.sum(np.square(x[sel] - center), 1)) < CENTER_THRESH
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = np.array([center[0], center[1], center[2], zmax - zmin, np.where(DC.nyu40ids == meta_vertices[ind[0],-1])[0][0]])
                    point_boundary_offset_xy[sel_global] = center - x[sel]
                    sel_center = sel_global[sel_center]
                    point_center_mask_xy[sel_center] = 1.0

                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
                point_sem_label[ind] = DC.nyu40id2class_sem[meta[-1]]
                
                #xtemp = np.stack([x]*len(corners))
                #dist = np.sum(np.square(xtemp - np.expand_dims(corners, 1)), axis=2)
                #sel_corner = np.argmin(dist, 0)
                #for i in range(len(ind)):
                #    point_votes_corner[ind[i], :] = corners[sel_corner[i]] - x[i,:]
                point_votes_corner1[ind, :] = corners[0] - x
                point_votes_corner2[ind, :] = corners[-1] - x
                #point_votes_corner3[ind, :] = corners[] - x
                
                selected_instances.append(i_instance)
                selected_centers.append(center)

                ### check for planes here
                '''
                @Returns:
                bbox: 8 x 3, order:
                [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
                 [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
                '''
                plane_indicator = plane_vertices[ind,4]
                planes = np.unique(plane_indicator)
                plane_ind = []
                for p in planes:
                    if p > 0:
                        temp_ind = np.where(plane_indicator == p)[0]
                        if len(temp_ind) > 10:
                            plane_ind.append(ind[temp_ind])
                            ### Normalize the vector here
                            ### May need to change later
                            #plane_vertices[ind[temp_ind],:4] = plane_vertices[ind[temp_ind],:4] / np.linalg.norm(plane_vertices[ind[temp_ind][0],:3])
                if len(plane_ind) > 0:
                    planes_org = plane_ind
                    plane_off = []
                    for plane in planes_org:
                        #subx = point_cloud[plane,:3]
                        plane_equ = plane_vertices[plane[0],:][:4]
                        #point_sel = point_cloud[plane,:3]
                        point_sel = np.stack([center]*len(plane), 0)
                        new_off = -np.sum(point_sel * plane_equ[:3], -1)
                        #plane_off.append(new_off - plane_equ[-1])
                        plane_off.append(new_off)
                    ### Just move the plane offset                                        
                    plane_ind = np.concatenate(plane_ind, 0)
                    plane_label_mask[plane_ind] = 1.0
                    plane_votes_offset[plane_ind] = np.expand_dims(np.concatenate(plane_off, 0), -1)
                    plane_votes_label[plane_ind, :] = center - point_cloud[plane_ind,:3]
                    #import pdb;pdb.set_trace()

        num_instance = len(obj_meta)
        obj_meta = np.array(obj_meta)
        obj_meta = obj_meta.reshape(-1, 9)

        target_bboxes_mask[0:num_instance] = 1
        target_bboxes[0:num_instance,:6] = obj_meta[:,0:6]
        #gt_bboxes[0:num_instance,:] = obj_meta[:,0:7]
        gt_bboxes[0:num_instance,:] = np.concatenate((obj_meta[:,0:6], np.expand_dims(obj_meta[:,-1], -1)), 1)

        for i in range(num_instance):
            ### Perturb x y z by 0.5 to 1.0
            if np.random.random() > 0.5:
                pert_xyz = np.random.random((3))*0.2
            else:
                pert_xyz = -np.random.random((3))*0.2
            ### Perturb scale from 0.4 to 0.8
            if np.random.random() > 0.5:
                pert_scale = 1.0 + np.random.random((3))*0.2
            else:
                pert_scale = 1.0 - np.random.random((3))*0.2
            if np.random.random() > 0.5:
                pert_angle = np.random.random()*(np.pi / 4)
            else:
                pert_angle = -np.random.random()*(np.pi / 4)
            pert_bboxes[i,0:3] += pert_xyz
            pert_bboxes[i,3:6] *= pert_scale
            pert_bboxes[i,6] = ((pert_bboxes[i,6] + np.pi / 2.0 + pert_angle) % np.pi) - np.pi / 2.0
        
        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in obj_meta[:,-1]]   
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:num_instance] = class_ind
        size_residuals[0:num_instance, :] = \
                                            target_bboxes[0:num_instance, 3:6] - DC.mean_size_arr[class_ind,:]
        # angle_classes[0:num_instance] = class_ind
        # angle_residuals[0:num_instance] = obj_meta[:,6]
        #for i in range(num_instance):
        #    angle_class, angle_residual = DC.angle2class2(obj_meta[i, 6])
        #    angle_classes[i] = angle_class
        #    angle_residuals[i] = angle_residual
        #    assert np.abs(DC.class2angle2(angle_class, angle_residual) - obj_meta[i, 6]) < 1e-6

        
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical
        point_layout_normal = np.tile(point_layout_normal, (1, 3)) # make 3 votes identical
        point_layout_sem = np.tile(np.expand_dims(point_layout_sem, -1), (1, 3)) # make 3 votes identical
        plane_votes_label = np.tile(plane_votes_label, (1, 3)) # make 3 votes identical
        point_sem_label = np.tile(np.expand_dims(point_sem_label, -1), (1, 3)) # make 3 votes identical
        point_votes_corner1 = np.tile(point_votes_corner1, (1, 3)) # make 3 votes identical
        point_votes_corner2 = np.tile(point_votes_corner2, (1, 3)) # make 3 votes identical

        plane_votes_offset = np.tile(plane_votes_offset, (1, 3)) # make 3 votes identical

        plane_votes_rot_front = np.tile(plane_votes_front[:,:3], (1, 3)) # make 3 votes identical
        plane_votes_off_front = np.tile(np.expand_dims(plane_votes_front[:,3], -1), (1, 3)) # make 3 votes identical

        plane_votes_rot_back = np.tile(plane_votes_back[:,:3], (1, 3)) # make 3 votes identical
        plane_votes_off_back = np.tile(np.expand_dims(plane_votes_back[:,3], -1), (1, 3)) # make 3 votes identical

        plane_votes_rot_lower = np.tile(plane_votes_lower[:,:3], (1, 3)) # make 3 votes identical
        plane_votes_off_lower = np.tile(np.expand_dims(plane_votes_lower[:,3], -1), (1, 3)) # make 3 votes identical

        plane_votes_rot_upper = np.tile(plane_votes_upper[:,:3], (1, 3)) # make 3 votes identical
        plane_votes_off_upper = np.tile(np.expand_dims(plane_votes_upper[:,3], -1), (1, 3)) # make 3 votes identical

        plane_votes_rot_left = np.tile(plane_votes_left[:,:3], (1, 3)) # make 3 votes identical
        plane_votes_off_left = np.tile(np.expand_dims(plane_votes_left[:,3], -1), (1, 3)) # make 3 votes identical

        plane_votes_rot_right = np.tile(plane_votes_right[:,:3], (1, 3)) # make 3 votes identical
        plane_votes_off_right = np.tile(np.expand_dims(plane_votes_right[:,3], -1), (1, 3)) # make 3 votes identical

        ret_dict = {}
                
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)

        ret_dict['gt_bbox'] = gt_bboxes.astype(np.float32)
        ret_dict['pert_bbox'] = pert_bboxes.astype(np.float32)
        
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
        
        ret_dict['point_center_mask_z'] = point_center_mask_z.astype(np.float32)
        ret_dict['point_center_mask_xy'] = point_center_mask_xy.astype(np.float32)
        
        ret_dict['point_layout_mask'] = point_layout_mask.astype(np.int64)
        ret_dict['point_layout_in_mask'] = point_layout_in_mask.astype(np.float32)
        ret_dict['point_layout_normal'] = point_layout_normal.astype(np.float32)
        ret_dict['point_layout_sem'] = point_layout_sem.astype(np.int64)
        
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_corner1'] = point_votes_corner1.astype(np.float32)
        ret_dict['vote_label_corner2'] = point_votes_corner2.astype(np.float32)
        #ret_dict['vote_label_corner'] = ((point_votes_corner1 + point_votes_corner2) / 2.0).astype(np.float32)
        #ret_dict['vote_label_corner2'] = point_votes_corner2.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)

        ret_dict['vote_label_plane'] = plane_votes_label.astype(np.float32)
        
        ret_dict['plane_label'] = np.concatenate([point_cloud, plane_vertices[:,:4]], 1).astype(np.float32)
        ret_dict['plane_label_mask'] = plane_label_mask.astype(np.float32)
        ret_dict['plane_votes_offset'] = plane_votes_offset.astype(np.float32)
        
        ret_dict['plane_votes_y'] = plane_votes_rot_front.astype(np.float32)
        ret_dict['plane_votes_y0'] = plane_votes_off_front.astype(np.float32)
        ret_dict['plane_votes_y1'] = plane_votes_off_back.astype(np.float32)
        
        ret_dict['plane_votes_x'] = plane_votes_rot_left.astype(np.float32)
        ret_dict['plane_votes_x0'] = plane_votes_off_left.astype(np.float32)
        ret_dict['plane_votes_x1'] = plane_votes_off_right.astype(np.float32)
        
        ret_dict['plane_votes_z'] = plane_votes_rot_lower.astype(np.float32)
        ret_dict['plane_votes_z0'] = plane_votes_off_lower.astype(np.float32)
        ret_dict['plane_votes_z1'] = plane_votes_off_upper.astype(np.float32)
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['pcl_color'] = pcl_color
        ret_dict['num_instance'] = num_instance

        ret_dict['voxel'] =np.expand_dims(vox.astype(np.float32), 0)
#         ret_dict['sem_voxel'] =np.array(sem_vox, np.float32)
        ret_dict['vox_center'] = np.expand_dims(np.array(vox_center, np.float32), 0)
        ret_dict['vox_corner'] = np.expand_dims(np.array(vox_corner, np.float32), 0)

        ret_dict['aug_yz'] = point_yz
        ret_dict['aug_xz'] = point_xz
        ret_dict['aug_rot'] = point_rot
        
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
    '''
    dset_train = ScannetDetectionDataset(split_set='train', use_height=True, num_points=40000, augment=False, use_angle=False)
    dset_val = ScannetDetectionDataset(split_set='val', use_height=True, num_points=40000, augment=False, use_angle=False)
    for i in range(len(dset_train.scan_names)):
        dset_train.__getitem__(i)
        print ("finished train"+ str(i))
    for i in range(len(dset_val.scan_names)):
        dset_val.__getitem__(i)
        print ("finished val"+ str(i))
    sys.exit(0)
    '''
    import scipy.io as sio
    dset = ScannetDetectionDataset(split_set='val', use_height=True, num_points=40000, augment=False, use_angle=False)
    for i_example in range(len(dset.scan_names)):
        example = dset.__getitem__(i_example)
        print (i_example)
        
        print (np.unique(example['plane_votes_x'][:,0]))
        print (np.unique(example['plane_votes_x'][:,1]))
        print (np.unique(example['plane_votes_y'][:,0]))
        print (np.unique(example['plane_votes_y'][:,1]))
        #sio.savemat('/scratch/cluster/zaiwei92/scannet_data/data_{}.mat'.format(dset.scan_names[i_example]), {'pc': example['point_clouds'], 'boundary': example['point_clouds'][example['point_boundary_mask']==1,0:3]})
        pc_util.write_ply(example['point_clouds'], '/scratch/cluster/zaiwei92/scannet_data/pc_{}.ply'.format(i_example))
        pc_util.write_ply(example['point_clouds'][example['point_line_mask']==1,0:3], '/scratch/cluster/zaiwei92/scannet_data/pc_new_obj_line{}.ply'.format(i_example))
        pc_util.write_ply(example['point_clouds'][example['point_boundary_mask_z']==1,0:3], '/scratch/cluster/zaiwei92/scannet_data/pc_new_obj_boundary_z{}.ply'.format(i_example))
        pc_util.write_ply(example['point_clouds'][example['point_boundary_mask_xy']==1,0:3], '/scratch/cluster/zaiwei92/scannet_data/pc_new_obj_boundary_xy{}.ply'.format(i_example))
        pc_util.write_ply(example['point_clouds'][example['point_center_mask_z']==1,0:3], '/scratch/cluster/zaiwei92/scannet_data/pc_obj_center_z{}.ply'.format(i_example))
        pc_util.write_ply(example['point_clouds'][example['point_center_mask_xy']==1,0:3], '/scratch/cluster/zaiwei92/scannet_data/pc_obj_center_xy{}.ply'.format(i_example))
        #pc_util.write_ply_color_multi(example['point_clouds'][:,:3], example['point_layout_normal'][:,:3], 'pc_normal_{}.ply'.format(str(i_example)))
        #pc_util.write_ply_label(example['point_clouds'][:,:3], example['point_layout_sem'][:,0], 'pc_sem_room_{}.ply'.format(str(i_example)),  3)
        #pc_util.write_ply_label(example['point_clouds'][:,:3], example['point_sem_cls_label'][:,0]+1, 'pc_sem_{}.ply'.format(str(i_example)),  38)
        import pdb;pdb.set_trace()
        continue
        viz_votes(example['point_clouds'], example['vote_label'],
                  example['vote_label_mask'],name=i_example)
        viz_votes(example['point_clouds'], example['vote_label_corner'],
                  example['vote_label_mask'],name=str(i_example)+'corner')
        viz_plane(example['plane_label'],
                  example['plane_label_mask'],name=str(i_example)+'plane')
        #viz_plane(np.concatenate([example['plane_label'][:,:3], example['plane_votes_rot_upper']], 1),example['plane_label_mask'],name=str(i_example)+'plane_oneside')
        viz_plane_perside(np.concatenate([example['plane_label'][:,:3], np.concatenate([example['plane_votes_rot_upper'][:,:3], np.expand_dims(example['plane_votes_off_upper'][:,0], -1)], 1)], 1),
                  example['plane_label_mask'],name=str(i_example)+'plane_oneside')
        
        #viz_votes(example['point_clouds'], example['vote_label_support_middle'],example['vote_label_mask_support'],name=str(i_example)+'support_middle')
        #viz_votes(example['point_clouds'], example['vote_label_bsupport_middle'],example['vote_label_mask_bsupport'],name=str(i_example)+'bsupport_middle')
        #viz_votes(example['point_clouds'], example['vote_label_support_offset'],example['vote_label_mask_support'],name=str(i_example)+'support_offset')
        #viz_votes(example['point_clouds'], example['vote_label_bsupport_offset'],example['vote_label_mask_bsupport'],name=str(i_example)+'bsupport_offset')

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
        
