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
import torch
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = '/home/bo/projects/cvpr2020/detection/votenet'
sys.path.append(ROOT_DIR)
# import pc_util
from utils.pc_util import volume_to_point_cloud,volume_to_point_cloud_color, write_ply, write_ply_color

class ScannetDetectionDataset(Dataset):
    def __init__(self, data_root, data_path, split_set='train', vsize=0.05,center_dev=2,
                 corner_dev=1,use_tsdf=1, center_reduce=0, corner_reduce=1):
        
        self.data_root = data_root
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
        # print(self.scan_names[816])
        # self.scan_names = self.scan_names[816:]
        self.scan_names = sorted(self.scan_names)
        self.vsize = vsize
        self.center_dev = center_dev
        self.corner_dev = corner_dev
        self.center_reduce = center_reduce
        self.corner_reduce = corner_reduce
        self.use_tsdf = use_tsdf
        
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        ret_dict = {}
        voxels = np.load(os.path.join(self.data_root, scan_name+'_vox.npy'))
        center = np.load(os.path.join(self.data_root, scan_name+'_center_%d_reduce%d.npy'%(self.center_dev, self.center_reduce)))
        corners = np.load(os.path.join(self.data_root, scan_name+'_corner_%d_reduce%d.npy'%(self.corner_dev, self.corner_reduce)))
        center = np.clip(center, 0.0,1.0)
        corners = np.clip(corners, 0.0,1.0)

        ret_dict['voxel'] =voxels.astype(np.float32)
        ret_dict['center'] = center.astype(np.float32)
        ret_dict['corners'] = corners.astype(np.float32)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['scan_name'] = scan_name
        return ret_dict


if __name__=='__main__':
    data_path = '/home/bo/data/scannet/new/scannet_train_detection_data'
    data_root = '/home/bo/data/scannet/new/training_data/vs0.05_tsdf0'
    dset = ScannetDetectionDataset(data_root=data_root, data_path = data_path, vsize=0.05, 
                                   center_dev=2.0, corner_dev=1.0, center_reduce=False, 
                                   corner_reduce=True,split_set='train',use_tsdf=0)
    out_path = 'dataset_v_test'
    prefix = 'tsdf0_'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
      
    for i_example in range(20):
        example = dset.__getitem__(i_example)
        name = example['scan_name']
        print(name)
        print(example['corners'].shape, example['center'].shape)
       
        pt_vox = volume_to_point_cloud(example['voxel'][0])
        pt_center, center_label = volume_to_point_cloud_color(example['center'][0])
        num_pt_vox = pt_vox.shape[0]
        num_pt_center = pt_center.shape[0]
        pt_vox_center = np.concatenate((pt_vox, pt_center),axis=0)
        label_center = np.zeros(num_pt_center+num_pt_vox)
        label_center[-num_pt_center:-1]=1
        
        pt_corner, corner_label = volume_to_point_cloud_color(example['corners'][0])
        num_pt_vox = pt_vox.shape[0]
        num_pt_corner = pt_corner.shape[0]
        pt_vox_corner = np.concatenate((pt_vox, pt_corner),axis=0)
        label_corner = np.zeros(num_pt_corner+num_pt_vox)
        label_corner[-num_pt_corner:-1]=1
        
        # write_ply(pt_vox, os.path.join(out_path, 'vox_{}.ply'.format(i_example)))
        write_ply_color(pt_vox_center, label_center, os.path.join(out_path, prefix+'center_{}.ply'.format(i_example)))
        write_ply_color(pt_vox_corner, label_corner, os.path.join(out_path, prefix+'corner_{}.ply'.format(i_example)))
        # write_ply_color(pt_vox_angle, label2, os.path.join(out_path, 'angle_{}.ply'.format(i_example)), num_classes=21)
        # write_ply_color(pt_center, center_label, os.path.join(out_path, 'center_color_{}.ply'.format(i_example)))
