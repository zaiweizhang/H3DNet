# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from box_util import get_3d_box

class ScannetDatasetConfig(object):
    def __init__(self):
        self.dataset = 'scannet'
        self.num_class = 18
        self.num_heading_bin = 24 # angle: -pi/2~pi/2, so divide 0~2*pi into 24 bin
        self.num_size_cluster = 18

        self.type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5, 'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11, 'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17}
        #self.type2class = {'wall':0, 'floor':1, 'cabinet':2, 'bed':3, 'chair':4, 'sofa':5, 'table':6, 'door':7,'window':8,'bookshelf':9,'picture':10, 'counter':11, 'blinds':12, 'desk':13, 'shelves':14, 'curtain':15, 'dresser':16, 'pillow':17, 'mirror':18, 'floormat':19, 'clothes':20, 'ceiling':21, 'books':22, 'refrigerator':23, 'television':24, 'paper':25, 'towel':26, 'showercurtrain':27, 'box':28, 'whiteboard':29, 'person':30, 'nightstand':31, 'toilet':32, 'sink':33, 'lamp':34, 'bathtub':35, 'bag':36}
        self.type2class_room = {'other':0, 'wall':1, 'floor':2}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.class2type_room = {self.type2class_room[t]:t for t in self.type2class_room}
        self.nyu40ids = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        #self.nyu40ids = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        self.nyu40ids_room = np.array([1,2])
        
        self.nyu40id2class = {nyu40id: i for i,nyu40id in enumerate(list(self.nyu40ids))}
        self.nyu40id2class_sem = {nyu40id: i for i,nyu40id in enumerate(list(self.nyu40ids))}
        self.mean_size_arr = np.load(os.path.join(ROOT_DIR,'scannet/meta_data/scannet_means.npz'))['arr_0']
        #self.mean_size_arr = np.load(os.path.join(ROOT_DIR,'scannet/meta_data/scannet_means_v2.npz.npy'))[:self.num_class,:]
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i,:]


    def class2angle(self, pred_cls, residual, to_label_format=True):
        return 0
            
    '''    
    def angle2class(self, angle):
        # assert(False)
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    def class2angle(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle
    '''
    def angle2class2(self, angle):
        ''' modify according to sunrgbd         
            scannet_angle: angle: -pi/2 ~ pi/2       
            1: angle += pi/2 -> 0~pi                  
            2: class*(2pi/N) + number = angle + pi/2  
        '''   
        class_id, residual_angle = self.angle2class(angle + np.pi / 2)
        return class_id, residual_angle
            
    def class2angle2(self, pred_cls, residual, to_label_format=True):
        angle = self.class2angle(pred_cls, residual)
        angle = angle - np.pi / 2
        return angle

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''        
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

    def param2obb2(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle
        return obb

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)
