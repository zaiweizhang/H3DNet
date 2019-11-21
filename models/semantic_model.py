# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule, PointnetPlaneVotes
from resnet_autoencoder import *

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
N_CLASSES = len(CLASS_LABELS)

def log_string(out_str, LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)

def point_seg_iou(pred, labels, LOG_FOUT):
    # pred: B, C, N, labels: B, N
    _, classes = torch.max(pred, 1)
    classes = classes.view(-1)
    labels = labels.view(-1)
    pred_ids = classes.detach().cpu().numpy().astype(np.int32)
    gt_ids = labels.cpu().numpy().astype(np.int32)
    idxs= gt_ids>=0
    if np.sum(pred_ids[idxs]<0)>0 or np.sum(gt_ids[idxs]<0)>0:
        print('error')
        return
    #pred_ids[pred_ids<0]=0
    confusion = np.bincount(pred_ids[idxs]*20+gt_ids[idxs],minlength=400).reshape((20,20)).astype(np.ulonglong)
    class_ious = {}
    mean_iou = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        #print(class_ious[label_name])
        mean_iou+=class_ious[label_name][0]/20
    log_string('classes          IoU', LOG_FOUT)
    log_string('----------------------------', LOG_FOUT)
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        log_string('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]), LOG_FOUT)
    log_string('mean IOU: %f'%(mean_iou), LOG_FOUT)

class Pointnet2_Voxel_SemSegNet(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, num_classes=20, num_filters=16,
                 block=BasicBlock,
                 layers=[1,1,1,1],
                 c_in=1,
                 c_out=21,
                 shortcut_type='B'):
        super().__init__()
        self.inplanes = c_in
        self.outplanes = c_out
        self.sa1 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_feature_dim,32, 32, 64],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64,64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=64,
                radius=0.3,
                nsample=32,
                mlp=[128,128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
                normalize_xyz=True
            )

        ''' 
        self.fp1 = PointnetFPModule(mlp=[512+256, 256,256])
        self.fp2 = PointnetFPModule(mlp=[256+128, 256,256])
        self.fp3 = PointnetFPModule(mlp=[256+64,256,128])
        self.fp4 = PointnetFPModule(mlp=[128+input_feature_dim,128,128,128])
        '''
        self.fp1 = PointnetFPModule(mlp=[512+256, 256,256])
        self.fp2 = PointnetFPModule(mlp=[256+128+64, 256,256])
        self.fp3 = PointnetFPModule(mlp=[256+64+32,256,128])
        self.fp4 = PointnetFPModule(mlp=[128+input_feature_dim+16,128,128,128])
        
        self.fc1 = torch.nn.Conv1d(128+8, 128, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Conv1d(128, num_classes, 1, stride=1)
        
        self.conv1 = conv3d_block(self.inplanes, num_filters, kernel_size=4,stride=2)
        self.layer1 = self._make_layer(block, num_filters, num_filters, layers[0], shortcut_type)
        
        self.conv2 = conv3d_block(num_filters, num_filters*2, kernel_size=4,stride=2)
        self.layer2 = self._make_layer(block, num_filters*2, num_filters*2, layers[1], shortcut_type)
        
        self.conv3 = conv3d_block(num_filters*2, num_filters*4, kernel_size=4,stride=2)
        self.layer3 = self._make_layer(block, num_filters*4, num_filters*4, layers[2], shortcut_type)
        
        self.conv4 = conv3d_block(num_filters*4, num_filters*8, kernel_size=4,stride=2)
        self.layer4 = self._make_layer(block, num_filters*8, num_filters*8, layers[3], shortcut_type)

        self.up1 = conv_trans_block_3d(num_filters*8, num_filters*4, kernel_size=4, stride=2)
        self.up_layer1 = self._make_layer(block, num_filters*4, num_filters*4, layers[3], shortcut_type)
        
        self.up2 = conv_trans_block_3d(num_filters*4, num_filters*2, kernel_size=4, stride=2)
        self.up_layer2 = self._make_layer(block, num_filters*2, num_filters*2, layers[2], shortcut_type)
    
        self.up3 = conv_trans_block_3d(num_filters*2, num_filters*1, kernel_size=4, stride=2)
        self.up_layer3 = self._make_layer(block, num_filters*1, num_filters*1, layers[1], shortcut_type)
        
        self.up4 = conv_trans_block_3d(num_filters*1, num_filters//2, kernel_size=4, stride=2)
        self.up_layer4 = self._make_layer(block, num_filters//2, num_filters//2, layers[0], shortcut_type)

        self.pred_layer = conv3d(num_filters//2, self.outplanes, kernel_size=3, stride=1)
        
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features
    def _make_layer(self, block, inplanes, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, inputs, end_points=None, mode=''):
        if not end_points: end_points = {}
        pointcloud = inputs['pointcloud']
        voxel = inputs['voxel']
        
        xyz, features = self._break_up_pc(pointcloud)
        # print('sa0', xyz.shape, features.shape)

        end_points['sa0_xyz'+mode] = xyz
        end_points['sa0_features'+mode] = features
        
        # --------- 4 SET ABSTRACTION LAYERS ---------
        if mode != '':
            ### Reuse inds from point
            xyz, features, fps_inds = self.sa1(xyz, features, inds=end_points['sa1_inds'])
        else:
            xyz, features, fps_inds = self.sa1(xyz, features)
        # print('sa1', xyz.shape, features.shape)
        end_points['sa1_inds'+mode] = fps_inds
        end_points['sa1_xyz'+mode] = xyz
        end_points['sa1_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa2(xyz, features, inds=end_points['sa2_inds']) # this fps_inds is just 0,1,...,1023
        else:
            xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        # print('sa2', xyz.shape, features.shape)
        end_points['sa2_inds'+mode] = fps_inds
        end_points['sa2_xyz'+mode] = xyz
        end_points['sa2_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa3(xyz, features, inds=end_points['sa3_inds']) # this fps_inds is just 0,1,...,511
        else:
            xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,1023
        # print('sa3', xyz.shape, features.shape)
        end_points['sa3_inds'+mode] = fps_inds
        end_points['sa3_xyz'+mode] = xyz
        end_points['sa3_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa4(xyz, features, inds=end_points['sa4_inds']) # this fps_inds is just 0,1,...,255
        else:
            xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        # print('sa4', xyz.shape, features.shape)
        end_points['sa4_inds'+mode] = fps_inds
        end_points['sa4_xyz'+mode] = xyz
        end_points['sa4_features'+mode] = features
        
        x = self.conv1(voxel)
        # print('conv1', x.shape)
        x = self.layer1(x)
        x = self.conv2(x)
        # print('conv2', x.shape)
        x = self.layer2(x)
        x = self.conv3(x)
        # print('conv3', x.shape)
        x = self.layer3(x)
        x = self.conv4(x)
        # print('conv4', x.shape)
        x = self.layer4(x)
        x = self.up1(x)
        end_points['voxel_feature_up1'] = x
        # print('up1', x.shape)
        x = self.up_layer1(x)
        x = self.up2(x)
        end_points['voxel_feature_up2'] = x

        # print('up2', x.shape)
        x = self.up_layer2(x)
        x = self.up3(x)
        end_points['voxel_feature_up3'] = x
        # print('up3', x.shape)
        x = self.up_layer3(x)
        x = self.up4(x)
        end_points['voxel_feature_up4'] = x
        # print('up4' ,x.shape)
        x = self.up_layer4(x)
        x = self.pred_layer(x)
        end_points['voxel_pred_sem'] = x
        '''
        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'+mode], end_points['sa4_xyz'+mode], end_points['sa3_features'+mode], end_points['sa4_features'+mode])
        # print('f1', features.shape)
        features = self.fp2(end_points['sa2_xyz'+mode], end_points['sa3_xyz'+mode], end_points['sa2_features'+mode], features)
        # print('f2', features.shape)
        features = self.fp3(end_points['sa1_xyz'+mode], end_points['sa2_xyz'+mode], end_points['sa1_features'+mode], features)
        # print('f3', features.shape)
        features = self.fp4(end_points['sa0_xyz'+mode], end_points['sa1_xyz'+mode], end_points['sa0_features'+mode], features)
        end_points['pt_feature_fp4']=features
        # print('f4', features.shape)
        end_points['fp4_features'+mode] = features
        end_points['fp4_xyz'+mode] = end_points['sa0_xyz'+mode]
        
        features_vox = pc_util.voxel_to_pt_feature_batch(end_points['voxel_feature_up4'], end_points['fp4_xyz'+mode])
        features_vox = features_vox.contiguous().transpose(2,1)
        end_points['vox_feature_up4']=features_vox
        features_combine_point = torch.cat((features, features_vox), 1)
        # print(features_combine_point.shape)
        '''
        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        
        features = self.fp1(end_points['sa3_xyz'+mode], end_points['sa4_xyz'+mode], end_points['sa3_features'+mode], end_points['sa4_features'+mode])
        end_points['pt_feature_fp1']=features
        #print('f1', features.shape)
        features_vox = pc_util.voxel_to_pt_feature_batch(end_points['voxel_feature_up1'], end_points['sa3_xyz'+mode], reduce_factor=8)
        features_vox = features_vox.contiguous().transpose(2,1)
        end_points['vox_feature_up1']=features_vox
        features_combine_point = torch.cat((features, features_vox), 1)
        #print(features_combine_point.shape)
        features = self.fp2(end_points['sa2_xyz'+mode], end_points['sa3_xyz'+mode], end_points['sa2_features'+mode], features_combine_point)
        end_points['pt_feature_fp2']=features

        features_vox = pc_util.voxel_to_pt_feature_batch(end_points['voxel_feature_up2'], end_points['sa2_xyz'+mode], reduce_factor=4)
        features_vox = features_vox.contiguous().transpose(2,1)
        end_points['vox_feature_up2']=features_vox
        features_combine_point = torch.cat((features, features_vox), 1)
        #print('f2', features.shape)
        features = self.fp3(end_points['sa1_xyz'+mode], end_points['sa2_xyz'+mode], end_points['sa1_features'+mode], features_combine_point)
        end_points['pt_feature_fp3']=features
        #print('f3', features.shape)

        features_vox = pc_util.voxel_to_pt_feature_batch(end_points['voxel_feature_up3'], end_points['sa1_xyz'+mode], reduce_factor=2)
        features_vox = features_vox.contiguous().transpose(2,1)
        end_points['vox_feature_up3']=features_vox
        features_combine_point = torch.cat((features, features_vox), 1)
        features = self.fp4(end_points['sa0_xyz'+mode], end_points['sa1_xyz'+mode], end_points['sa0_features'+mode], features_combine_point)
        end_points['pt_feature_fp4']=features
        #print('f4', features.shape)
        
        end_points['fp4_features'+mode] = features
        end_points['fp4_xyz'+mode] = end_points['sa0_xyz'+mode]

        features_vox = pc_util.voxel_to_pt_feature_batch(end_points['voxel_feature_up4'], end_points['fp4_xyz'+mode], reduce_factor=1)
        features_vox = features_vox.contiguous().transpose(2,1)
        end_points['vox_feature_up4']=features_vox
        features_combine_point = torch.cat((features, features_vox), 1)
        # print(features_combine_point.shape)
        
        f1 = nn.ReLU()(self.bn1(self.fc1(features_combine_point)))
        f1 = self.dropout(f1)
        f2 = self.fc2(f1)
        end_points['pred_sem'] = f2

        return end_points
    
def get_sem_loss_vox(pred, target, w_classes):
    criterion = torch.nn.CrossEntropyLoss(weight=w_classes)
    loss = criterion(pred, target.long())
    return loss

def get_sem_acc_vox(pred, target):
    acc=0
    # _, classes = torch.max(pred, 1)
    # acc = (classes == target).float().sum() / target.numel()
    for i in range(pred.shape[0]):
        _,p = torch.topk(pred[i], k=1,dim=0)
        mask1 = (target[i]==0).type(torch.FloatTensor)
        mask2= (p[0]==0).type(torch.FloatTensor)
        mask= 1 - mask1*mask2
        correct = (1-mask1)*(p[0].eq(target[i].long())).type(torch.FloatTensor)
        h,w,d=target[i].size()
        # print(torch.sum(correct))
        acc_i=torch.sum(correct)/(h*w*d-torch.sum(mask1))
        acc+=acc_i
    acc = acc/pred.shape[0]
    return acc
  
class Pointnet2SemSeg_new(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, num_classes=20):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_feature_dim,32, 32, 64],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64,64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=64,
                radius=0.3,
                nsample=32,
                mlp=[128,128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
                normalize_xyz=True
            )

        #self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        #self.fp2 = PointnetFPModule(mlp=[256+256,256,128])
        self.fp1 = PointnetFPModule(mlp=[512+256, 256,256])
        self.fp2 = PointnetFPModule(mlp=[256+128, 256,256])
        self.fp3 = PointnetFPModule(mlp=[256+64,256,128])
        self.fp4 = PointnetFPModule(mlp=[128+input_feature_dim,128,128,128])

        self.fc1 = torch.nn.Conv1d(128, 128, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Conv1d(128, num_classes, 1, stride=1)
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None, mode=''):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}

        xyz, features = self._break_up_pc(pointcloud)
        # print('sa0', xyz.shape, features.shape)

        end_points['sa0_xyz'+mode] = xyz
        end_points['sa0_features'+mode] = features
        
        # --------- 4 SET ABSTRACTION LAYERS ---------
        if mode != '':
            ### Reuse inds from point
            xyz, features, fps_inds = self.sa1(xyz, features, inds=end_points['sa1_inds'])
        else:
            xyz, features, fps_inds = self.sa1(xyz, features)
        # print('sa1', xyz.shape, features.shape)
        end_points['sa1_inds'+mode] = fps_inds
        end_points['sa1_xyz'+mode] = xyz
        end_points['sa1_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa2(xyz, features, inds=end_points['sa2_inds']) # this fps_inds is just 0,1,...,1023
        else:
            xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        # print('sa2', xyz.shape, features.shape)
        end_points['sa2_inds'+mode] = fps_inds
        end_points['sa2_xyz'+mode] = xyz
        end_points['sa2_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa3(xyz, features, inds=end_points['sa3_inds']) # this fps_inds is just 0,1,...,511
        else:
            xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,1023
        # print('sa3', xyz.shape, features.shape)
        end_points['sa3_inds'+mode] = fps_inds
        end_points['sa3_xyz'+mode] = xyz
        end_points['sa3_features'+mode] = features

        if mode != '':
            xyz, features, fps_inds = self.sa4(xyz, features, inds=end_points['sa4_inds']) # this fps_inds is just 0,1,...,255
        else:
            xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        # print('sa4', xyz.shape, features.shape)
        end_points['sa4_inds'+mode] = fps_inds
        end_points['sa4_xyz'+mode] = xyz
        end_points['sa4_features'+mode] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'+mode], end_points['sa4_xyz'+mode], end_points['sa3_features'+mode], end_points['sa4_features'+mode])
        # print('f1', features.shape)
        features = self.fp2(end_points['sa2_xyz'+mode], end_points['sa3_xyz'+mode], end_points['sa2_features'+mode], features)
        # print('f2', features.shape)
        features = self.fp3(end_points['sa1_xyz'+mode], end_points['sa2_xyz'+mode], end_points['sa1_features'+mode], features)
        # print('f3', features.shape)
        features = self.fp4(end_points['sa0_xyz'+mode], end_points['sa1_xyz'+mode], end_points['sa0_features'+mode], features)
        # print('f4', features.shape)
        end_points['fp4_features'+mode] = features
        end_points['fp4_xyz'+mode] = end_points['sa0_xyz'+mode]
        
        f1 = nn.ReLU()(self.bn1(self.fc1(features)))
        f1 = self.dropout(f1)
        f2 = self.fc2(f1)
        end_points['pred_sem'] = f2

        return end_points

if __name__=='__main__':
    backbone_net = Pointnet2_Voxel_SemSegNet(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    point = torch.rand(4,8192,6).cuda()
    vox = torch.rand(4,1,128,128,48).cuda()
    inputs = {'pointcloud': point, 'voxel': vox}
    out = backbone_net(inputs)
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
