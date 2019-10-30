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
from utils.pc_util import process_bbx, read_tsdf, get_corner
from utils.pc_util import point_to_volume_gaussion,center_to_volume_gaussion, volume_to_point_cloud,volume_to_point_cloud_color
from utils.pc_util import write_ply, write_ply_color, get_oriented_corners
from utils.pc_util import multichannel_volume_to_point_cloud, center_to_volume_gaussion_multiclass, point_to_volume_gaussion_multiclass
ROOT_DIR = '/home/bo/projects/cvpr2020/detection/votenet'
sys.path.append(ROOT_DIR)
# import pc_util

abandon_classes = [38,39,40]
reduce_classes = [1, 2, 22] 
  
class ScannetDetectionDataset(Dataset):
    def __init__(self, data_path, split_set='train', vsize=0.06, use_tsdf=0, use_18cls=1):
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
        self.use_tsdf = use_tsdf
        self.use_18cls = use_18cls
        
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        ret_dict = {}
        sem_vox=np.load(os.path.join(self.data_path, scan_name+'_vox_0.06_sem.npy'))
        vox = np.array(sem_vox>0,np.float32)
        if self.use_tsdf:
            vox =  read_tsdf(os.path.join(self.data_path, scan_name+'.TDF.bin'))
        if self.use_18cls:
            center = np.load(os.path.join(self.data_path, scan_name+'_vox_0.06_center_noangle_18.npy'))
            corner = np.load(os.path.join(self.data_path, scan_name+'_vox_0.06_corner_noangle_18.npy'))
            sem_vox=np.load(os.path.join(self.data_path, scan_name+'_vox_0.06_sem_18cls.npy'))
        else:
            center = np.load(os.path.join(self.data_path, scan_name+'_vox_0.06_center.npy'))
            corner = np.load(os.path.join(self.data_path, scan_name+'_vox_0.06_corner.npy'))
        ret_dict['voxel'] =np.expand_dims(vox.astype(np.float32), 0)
        ret_dict['sem_voxel'] =np.array(sem_vox, np.float32)
        ret_dict['center'] = np.expand_dims(np.array(center, np.float32), 0)
        ret_dict['corner'] = np.expand_dims(np.array(corner, np.float32), 0)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['scan_name'] = scan_name
        return ret_dict

class ScannetDetectionDataset_online(Dataset):
    def __init__(self, data_root, split_set='train', vsize=0.05,center_dev=2,
                 corner_dev=1, num_sem_classes=37, use_tsdf=0, center_reduce=0, select_corners=0,
                 corner_reduce=0, train_corner=False, train_center=False):
        self.data_root = data_root
        self.data_path = os.path.join(self.data_root, 'scannet_train_detection_data')
        self.tsdf_path = os.path.join(self.data_root, 'tsdf_'+str(vsize))
        
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
        self.scan_names = sorted(self.scan_names)
        self.vsize = vsize
        self.center_dev = center_dev
        self.corner_dev = corner_dev
        self.num_sem_classes = num_sem_classes        
        self.center_reduce = center_reduce
        self.corner_reduce = corner_reduce
        self.use_tsdf = use_tsdf
        self.select_corners = select_corners
        self.train_center = train_center
        self.train_corner = train_corner

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        ret_dict = {}
        if not self.use_tsdf:
            if self.vsize==0.025:
                voxels = np.load(os.path.join(self.data_path, scan_name+'_trivox_%s_new.npy'%(str(self.vsize))))
            else:
                voxels = np.load(os.path.join(self.data_path, scan_name+'_trivox_%s.npy'%(str(self.vsize))))
            voxels = np.array(voxels>0.0, np.float32)
        else:
            voxels = read_tsdf(os.path.join(self.tsdf_path, scan_name+'.TDF.bin'))
            
        # instance_bboxes = np.load(os.path.join(self.data_path, scan_name+'_bbox.npy'))
        oriented_bboxes = np.load(os.path.join(self.data_path, scan_name+'_all.npy'))
        oriented_bboxes = np.unique(oriented_bboxes, axis=0)
        oriented_bboxes =self.reduce_class(oriented_bboxes, abandon_classes)
        instance_bboxes = oriented_bboxes
        # if self.corner_reduce: 
        #     oriented_bboxes = self.reduce_class(oriented_bboxes, reduce_classes)
        # if self.center_reduce:
        #     instance_bboxes = self.reduce_class(instance_bboxes, reduce_classes)
    
        instance_bboxes = process_bbx(instance_bboxes, vs=self.vsize)
        if self.train_center:
            center, angle = center_to_volume_gaussion(instance_bboxes, dev=self.center_dev, vs=self.vsize)
            center_class = center_to_volume_gaussion_multiclass(instance_bboxes, self.num_sem_classes, dev=self.center_dev, vs=self.vsize)
            mask=(center>0.05).astype(np.float32)
            center = center*mask
            mask=(center_class>0.05).astype(np.float32)
            center_class = center_class*mask
            ret_dict['center'] = np.expand_dims(center.astype(np.float32), 0)
            ret_dict['center_class'] = center_class.astype(np.float32)
            ret_dict['angle'] = np.expand_dims(angle.astype(np.float32), 0)
        else: 
            center=0
            ret_dict['center']=0
        if self.train_corner:
            corners_pt, corners_sem_labels = get_oriented_corners(oriented_bboxes, multiclass=True, select=self.select_corners, vs=self.vsize)
            corners = point_to_volume_gaussion(corners_pt, dev=self.corner_dev, vs=self.vsize)  
            corners_class = point_to_volume_gaussion_multiclass(corners_pt, corners_sem_labels, num_classes=self.num_sem_classes)
            # corner_class = 
            mask2=(corners>0.05).astype(np.float32)
            corners = corners*mask2
            ret_dict['corners'] = np.expand_dims(corners.astype(np.float32), 0)
            ret_dict['corners_class'] = corners_class.astype(np.float32)
        else:
            corners =0
            ret_dict['corners']=0
        
        # print(voxels.shape, center.shape)

        ret_dict['voxel'] = np.expand_dims(voxels.astype(np.float32), 0)
        # ret_dict['bbx'] = instance_bboxes
        # ret_dict['center'] = np.expand_dims(center.astype(np.float32), 0)
        # ret_dict['corners'] = np.expand_dims(corners.astype(np.float32), 0)
        # ret_dict['scale'] = scale.astype(np.float32)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['scan_name'] = scan_name
        return ret_dict
    
    def reduce_class(self, bbx, classes):
        idx=[]
        for i in range(bbx.shape[0]):
            if bbx[i,-1] not in classes:
                idx.append(i)
        new_bbx = bbx[idx]
        return new_bbx

if __name__=='__main__':
    data_path = '/home/bo/data/scannet/new2/scannet_train_detection_data'
    data_root = '/home/bo/data/scannet/new/training_data/new/vs0.05_tsdf0'
    data_root_online = '/home/bo/data/scannet/new'

    # dset = ScannetDetectionDataset(data_root=data_root, data_path = data_path, vsize=0.05, 
    #                                center_dev=2.0, corner_dev=1.0, center_reduce=0, 
    #                                corner_reduce=0,split_set='train',use_tsdf=0)
    # dset = ScannetDetectionDataset_online(data_root=data_root_online, vsize=0.05, center_dev=2, corner_dev=1,
    #                                center_reduce=0, corner_reduce=0, split_set='all', 
    #                                select_corners=0, use_tsdf=0, train_corner=1, train_center=1)
    dset = ScannetSemanticDataset(data_path=data_path)
    dataloader = torch.utils.data.DataLoader(
    dset,
    batch_size=1,
    shuffle=False,
    num_workers=2)
    
    out_path = 'dataset_test'
    prefix = '0.06_'
    # out_path = '/home/bo/data/scannet/new/training_data/new/vs0.05_tsdf1'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
  
    # for i, data in enumerate(dataloader):
    #     scene_id = data['scan_idx']
    #     scan_name = data['scan_name'][0]
    #     vox = data['voxel'][0]
    #     center = data['center'][0]
    #     corner = data['corners'][0] 
    #     center_class = data['center_class'][0]
    #     print (scene_id.item(), scan_name)
    #     print(center.shape, center_class.shape) 
    #     np.save(os.path.join(out_path, scan_name+'_vox.npy'), vox)
    #     np.save(os.path.join(out_path, scan_name+'_center_dev2_reduce0.npy'), center)
    #     np.save(os.path.join(out_path, scan_name+'_corner_dev1_reduce0_select0.npy'), corner)
    #     np.save(os.path.join(out_path, scan_name+'_center_multiclass_dev2_reduce0.npy'), center_class)
        
    for i_example in range(30):
        example = dset.__getitem__(i_example)
        name = example['scan_name']
        print(name)
        print(example['sem_voxel'].shape, example['center'].shape)
        
        pt_vox = volume_to_point_cloud(example['voxel'][0])
        pt_center, center_label = volume_to_point_cloud_color(example['center'][0])
        num_pt_vox = pt_vox.shape[0]
        num_pt_center = pt_center.shape[0]
        pt_vox_center = np.concatenate((pt_vox, pt_center),axis=0)
        label_center = np.zeros(num_pt_center+num_pt_vox)
        label_center[-num_pt_center:]=1
        
        pt_corner, corner_label = volume_to_point_cloud_color(example['corner'][0])
        num_pt_vox = pt_vox.shape[0]
        num_pt_corner = pt_corner.shape[0]
        pt_vox_corner = np.concatenate((pt_vox, pt_corner),axis=0)
        label_corner = np.zeros(num_pt_corner+num_pt_vox)
        label_corner[-num_pt_corner:]=1
        
        # pt_center_class, label_center_class=multichannel_volume_to_point_cloud(example['center_class'])
        # num_pt_vox = pt_vox.shape[0]
        # num_pt_center_class = pt_center_class.shape[0]
        # pt_vox_center_class = np.concatenate((pt_vox, pt_center_class),axis=0)
        # label_center_class_vox = np.zeros(num_pt_center_class+num_pt_vox)
        # label_center_class_vox[-num_pt_center_class:]=label_center_class
        
        # pt_corner_class, label_corner_class=multichannel_volume_to_point_cloud(example['corners_class'])
        # num_pt_vox = pt_vox.shape[0]
        # num_pt_corner_class = pt_corner_class.shape[0]
        # pt_vox_corner_class = np.concatenate((pt_vox, pt_corner_class),axis=0)
        # label_corner_class_vox = np.zeros(num_pt_corner_class+num_pt_vox)
        # label_corner_class_vox[-num_pt_corner_class:]=label_corner_class
        
        # write_ply(pt_vox, os.path.join(out_path, 'vox_{}.ply'.format(i_example)))
        write_ply_color(pt_vox_center, label_center, os.path.join(out_path, prefix+'center_{}.ply'.format(i_example)))
        write_ply_color(pt_vox_corner, label_corner, os.path.join(out_path, prefix+'corner_{}.ply'.format(i_example)))
        # write_ply_color(pt_vox_center_class, label_center_class_vox, os.path.join(out_path, prefix+'center_class_{}.ply'.format(i_example)), num_classes=38)
        # write_ply_color(pt_vox_corner_class, label_corner_class_vox, os.path.join(out_path, prefix+'corner_class_{}.ply'.format(i_example)), num_classes=38)
        # write_ply_color(pt_vox_angle, label2, os.path.join(out_path, 'angle_{}.ply'.format(i_example)), num_classes=21)
        # write_ply_color(pt_center, center_label, os.path.join(out_path, 'center_color_{}.ply'.format(i_example)))
