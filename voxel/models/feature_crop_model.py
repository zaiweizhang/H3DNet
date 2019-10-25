import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import os
from model_utils import *

class ResNet_Autoencoder(nn.Module):
    '''
    Input: Voxel [b, 1, vx, vy, vz]
    Output: Center voxel [b, 1, vx, vy, vz]
    '''
    def __init__(self,
                 num_filters=12,
                 block=BasicBlock,
                 layers=[1,1,1,1],
                 c_in=1,
                 c_out=1,
                 shortcut_type='B'):
        self.inplanes = c_in
        self.outplanes = c_out
        super(ResNet_Autoencoder, self).__init__()
        
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

    def forward(self, x):
        x = self.conv1(x)
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
        # print('up1', x.shape)

        x = self.up_layer1(x)
        x = self.up2(x)
        # print('up2', x.shape)

        x = self.up_layer2(x)
        x = self.up3(x)
        # print('up3', x.shape)

        x = self.up_layer3(x)
        x = self.up4(x)
        # print('up4' ,x.shape)
        x = self.up_layer4(x)
        x = self.pred_layer(x)
        # x= nn.Sigmoid()(x)
        # print('pred', x.shape)
        return x

def get_loss(pred, target, w95, w8, w5, w3):
    # y1 = nn.Sigmoid()(pred)
    # bce_loss = nn.BCELoss()(pred, target)
    # l1_loss = nn.L1Loss()(pred, target)
    l2_loss = nn.MSELoss()(pred, target)
    
    # p = pred.view(pred.shape[0], -1)
    # t = pred.view(target.shape[0], -1)
    mask = (target>0.9).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w95*nn.MSELoss(reduction='mean')(p1, t1)
    
    mask = (target>0.8).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w8*nn.MSELoss(reduction='mean')(p1, t1)
    mask = (target>0.5).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w5*nn.MSELoss(reduction='mean')(p1, t1)
    mask = (target>0.3).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w3*nn.MSELoss(reduction='mean')(p1, t1)
    
    # crossentropy_loss = nn.CrossEntropyLoss()(p, t.long())
    # loss = crossentropy_loss+bce_weight*bce_loss
    # loss = bce_loss +l1_weight*l1_loss+l2_weight*l2_loss
    loss = l2_loss
    return loss

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.Tensor(1, 1, 240, 240, 96)
    x.to(device)
    print("x size: {}".format(x.size()))
    model = ResNet_Autoencoder()
    out = model(x)
    print("out size: {}".format(out.size()))
