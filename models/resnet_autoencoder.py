import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import os


# os.environ['CUDA_VISIBLE_DEVICES']='1'
def conv3d(in_planes, out_planes, kernel_size=3, stride=1):
    return  nn.Conv3d(in_planes,out_planes,kernel_size=kernel_size,stride=stride, padding=1,bias=False)

def conv3d_block(in_planes, out_planes, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True)):
    # 3x3x3 convolution with padding
    return nn.Sequential(
        nn.Conv3d(in_planes,out_planes,kernel_size=kernel_size,stride=stride, padding=1,bias=False),
         nn.BatchNorm3d(out_planes),
          activation,)

def conv_trans_block_3d(in_planes, out_planes, kernel_size=3, stride=1,activation=nn.ReLU(inplace=True)):
    return nn.Sequential(
        nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=0),
        nn.BatchNorm3d(out_planes),
        activation,)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3d(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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
        latent_feature = x

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
        pred={}
        pred['pred'] = x
        pred['latent_feature']=latent_feature
        return pred

class TwoStreamNet(nn.Module):
    '''
    Input: Voxel [b, 1, vx, vy, vz]
    Output: Center voxel [b, 1, vx, vy, vz]
    '''
    def __init__(self,
                 num_filters=16,
                 encoder_filters=32,
                 block=BasicBlock,
                 layers=[1,1,1,1],
                 c_in=1,
                 c_out_1=1,
                 c_out_2=1,
                 shortcut_type='B'):
        self.inplanes = c_in
        self.outplanes1 = c_out_1
        self.outplanes2 = c_out_2
        
        super(TwoStreamNet, self).__init__()
        self.encoder_filters = encoder_filters
        self.conv1 = conv3d_block(self.inplanes, self.encoder_filters, kernel_size=4,stride=2)
        self.layer1 = self._make_layer(block, self.encoder_filters, self.encoder_filters, layers[0], shortcut_type)
        
        self.conv2 = conv3d_block(self.encoder_filters, self.encoder_filters*2, kernel_size=4,stride=2)
        self.layer2 = self._make_layer(block, self.encoder_filters*2, self.encoder_filters*2, layers[1], shortcut_type)
        
        self.conv3 = conv3d_block(self.encoder_filters*2, self.encoder_filters*4, kernel_size=4,stride=2)
        self.layer3 = self._make_layer(block, self.encoder_filters*4, self.encoder_filters*4, layers[2], shortcut_type)
        
        self.conv4 = conv3d_block(self.encoder_filters*4, self.encoder_filters*8, kernel_size=4,stride=2)
        self.layer4 = self._make_layer(block, self.encoder_filters*8, self.encoder_filters*8, layers[3], shortcut_type)

        self.up1 = conv_trans_block_3d(self.encoder_filters*8, num_filters*4, kernel_size=4, stride=2)
        self.up_layer1 = self._make_layer(block, num_filters*4, num_filters*4, layers[3], shortcut_type)
        
        self.up2 = conv_trans_block_3d(num_filters*4, num_filters*2, kernel_size=4, stride=2)
        self.up_layer2 = self._make_layer(block, num_filters*2, num_filters*2, layers[2], shortcut_type)
    
        self.up3 = conv_trans_block_3d(num_filters*2, num_filters*1, kernel_size=4, stride=2)
        self.up_layer3 = self._make_layer(block, num_filters*1, num_filters*1, layers[1], shortcut_type)
        
        self.up4 = conv_trans_block_3d(num_filters*1, num_filters//2, kernel_size=4, stride=2)
        self.up_layer4 = self._make_layer(block, num_filters//2, num_filters//2, layers[0], shortcut_type)

        self.pred_layer = conv3d(num_filters//2, self.outplanes1, kernel_size=3, stride=1)
        
        self.up12 = conv_trans_block_3d(self.encoder_filters*8, num_filters*4, kernel_size=4, stride=2)
        self.up_layer12 = self._make_layer(block, num_filters*4, num_filters*4, layers[3], shortcut_type)
        
        self.up22 = conv_trans_block_3d(num_filters*4, num_filters*2, kernel_size=4, stride=2)
        self.up_layer22 = self._make_layer(block, num_filters*2, num_filters*2, layers[2], shortcut_type)
    
        self.up32 = conv_trans_block_3d(num_filters*2, num_filters*1, kernel_size=4, stride=2)
        self.up_layer32 = self._make_layer(block, num_filters*1, num_filters*1, layers[1], shortcut_type)
        
        self.up42 = conv_trans_block_3d(num_filters*1, num_filters//2, kernel_size=4, stride=2)
        self.up_layer42 = self._make_layer(block, num_filters//2, num_filters//2, layers[0], shortcut_type)

        self.pred_layer2 = conv3d(num_filters//2, self.outplanes2, kernel_size=3, stride=1)

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

    def forward(self, x, end_points, inputs):
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
        latent_feature = x
        
        x = self.up1(latent_feature)
        latent_feature1 = x
        # print('up1', x.shape)
        x = self.up_layer1(x)
        x = self.up2(x)
        latent_feature2 = x
        # print('up2', x.shape)
        x = self.up_layer2(x)
        x = self.up3(x)
        latent_feature3 = x
        # print('up3', x.shape)
        x = self.up_layer3(x)
        x = self.up4(x)
        latent_feature4 = x
        # print('up4' ,x.shape)
        x = self.up_layer4(x)
        pred1 = self.pred_layer(x)
        
        x = self.up12(latent_feature)
        # print('up1', x.shape)
        x = self.up_layer12(x)
        x = self.up22(x)
        # print('up2', x.shape)
        x = self.up_layer22(x)
        x = self.up32(x)
        # print('up3', x.shape)
        x = self.up_layer32(x)
        x = self.up42(x)
        # print('up4' ,x.shape)
        x = self.up_layer42(x)
        pred2 = self.pred_layer2(x)
        #if inputs['sunrgbd'] == False:
        #    pred1 = torch.nn.Sigmoid()(pred1)
        #    pred2 = torch.nn.Sigmoid()(pred2)
        end_points['vox_pred1'] = pred1
        end_points['vox_pred2'] = pred2
        end_points['vox_latent_feature0'] = latent_feature
        end_points['vox_latent_feature1'] = latent_feature1
        end_points['vox_latent_feature2'] = latent_feature2
        end_points['vox_latent_feature3'] = latent_feature3
        end_points['vox_latent_feature4'] = latent_feature4
        return end_points

class TwoStreamNetEncoder(nn.Module):
    '''
    Input: Voxel [b, 1, vx, vy, vz]
    Output: Center voxel [b, 1, vx, vy, vz]
    '''
    def __init__(self,
                 num_filters=16,
                 encoder_filters=32,
                 block=BasicBlock,
                 layers=[1,1,1,1],
                 c_in=1,
                 c_out_1=1,
                 c_out_2=1,
                 shortcut_type='B'):
        self.inplanes = c_in
        self.outplanes1 = c_out_1
        self.outplanes2 = c_out_2
        
        super(TwoStreamNetEncoder, self).__init__()
        self.encoder_filters = encoder_filters
        self.conv1 = conv3d_block(self.inplanes, self.encoder_filters, kernel_size=4,stride=2)
        self.layer1 = self._make_layer(block, self.encoder_filters, self.encoder_filters, layers[0], shortcut_type)
        
        self.conv2 = conv3d_block(self.encoder_filters, self.encoder_filters*2, kernel_size=4,stride=2)
        self.layer2 = self._make_layer(block, self.encoder_filters*2, self.encoder_filters*2, layers[1], shortcut_type)
        
        self.conv3 = conv3d_block(self.encoder_filters*2, self.encoder_filters*4, kernel_size=4,stride=2)
        self.layer3 = self._make_layer(block, self.encoder_filters*4, self.encoder_filters*4, layers[2], shortcut_type)
        
        self.conv4 = conv3d_block(self.encoder_filters*4, self.encoder_filters*8, kernel_size=4,stride=2)
        self.layer4 = self._make_layer(block, self.encoder_filters*8, self.encoder_filters*8, layers[3], shortcut_type)

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

    def forward(self, x, end_points):
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
        latent_feature = x

        end_points['vox_latent_feature'] = latent_feature
        return end_points

class TwoStreamNetDecoder(nn.Module):
    '''
    Input: Voxel [b, 1, vx, vy, vz]
    Output: Center voxel [b, 1, vx, vy, vz]
    '''
    def __init__(self,
                 num_filters=16,
                 encoder_filters=32,
                 block=BasicBlock,
                 layers=[1,1,1,1],
                 c_in=1,
                 c_out_1=1,
                 c_out_2=1,
                 shortcut_type='B'):
        self.inplanes = c_in
        self.outplanes1 = c_out_1
        self.outplanes2 = c_out_2
        
        super(TwoStreamNetDecoder, self).__init__()
        self.encoder_filters = encoder_filters

        self.up1 = conv_trans_block_3d(self.encoder_filters*8*2, num_filters*4, kernel_size=4, stride=2)
        self.up_layer1 = self._make_layer(block, num_filters*4, num_filters*4, layers[3], shortcut_type)
        
        self.up2 = conv_trans_block_3d(num_filters*4, num_filters*2, kernel_size=4, stride=2)
        self.up_layer2 = self._make_layer(block, num_filters*2, num_filters*2, layers[2], shortcut_type)
    
        self.up3 = conv_trans_block_3d(num_filters*2, num_filters*1, kernel_size=4, stride=2)
        self.up_layer3 = self._make_layer(block, num_filters*1, num_filters*1, layers[1], shortcut_type)
        
        self.up4 = conv_trans_block_3d(num_filters*1, num_filters//2, kernel_size=4, stride=2)
        self.up_layer4 = self._make_layer(block, num_filters//2, num_filters//2, layers[0], shortcut_type)

        self.pred_layer = conv3d(num_filters//2, self.outplanes1, kernel_size=3, stride=1)
        
        self.up12 = conv_trans_block_3d(self.encoder_filters*8*2, num_filters*4, kernel_size=4, stride=2)
        self.up_layer12 = self._make_layer(block, num_filters*4, num_filters*4, layers[3], shortcut_type)
        
        self.up22 = conv_trans_block_3d(num_filters*4, num_filters*2, kernel_size=4, stride=2)
        self.up_layer22 = self._make_layer(block, num_filters*2, num_filters*2, layers[2], shortcut_type)
    
        self.up32 = conv_trans_block_3d(num_filters*2, num_filters*1, kernel_size=4, stride=2)
        self.up_layer32 = self._make_layer(block, num_filters*1, num_filters*1, layers[1], shortcut_type)
        
        self.up42 = conv_trans_block_3d(num_filters*1, num_filters//2, kernel_size=4, stride=2)
        self.up_layer42 = self._make_layer(block, num_filters//2, num_filters//2, layers[0], shortcut_type)

        self.pred_layer2 = conv3d(num_filters//2, self.outplanes2, kernel_size=3, stride=1)

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

    def forward(self, x, end_points, inputs):
        latent_feature = x
        
        x = self.up1(latent_feature)
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
        pred1 = self.pred_layer(x)
        
        x = self.up12(latent_feature)
        # print('up1', x.shape)
        x = self.up_layer12(x)
        x = self.up22(x)
        # print('up2', x.shape)
        x = self.up_layer22(x)
        x = self.up32(x)
        # print('up3', x.shape)
        x = self.up_layer32(x)
        x = self.up42(x)
        # print('up4' ,x.shape)
        x = self.up_layer42(x)
        pred2 = self.pred_layer2(x)
        if inputs['sunrgbd'] == False:
            pred1 = torch.nn.Sigmoid()(pred1)
            pred2 = torch.nn.Sigmoid()(pred2)
        end_points['vox_pred1'] = pred1
        end_points['vox_pred2'] = pred2
        return end_points

def get_loss(pred, target, w95, w8, w5):
    # y1 = nn.Sigmoid()(pred)
    # bce_loss = nn.BCELoss()(pred, target)
    # l1_loss = nn.L1Loss()(pred, target)
    l2_loss = nn.MSELoss()(pred, target)
    
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
    loss = l2_loss
    return loss

def focal_loss(pred, gt, w10=1.0, w8=0.0):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  pos_inds8 = (gt>0.8).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)
  loss = 0
  
  pos_loss = w10*torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  pos_loss8 = w8*torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds8
  
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  pos_loss8 = pos_loss8.sum()
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss + pos_loss8) / num_pos
  return loss


def get_sem_loss(pred, target, w_class):
    criterion = torch.nn.CrossEntropyLoss(weight=w_class)
    loss = criterion(pred, target)
    return loss

def get_sem_acc(pred, target):
    acc=0
    for i in range(pred.shape[0]):
        _,p = torch.topk(pred[0], k=1,dim=0)
        correct = (p[0].eq(target[0])).type(torch.FloatTensor)
        h,w,d=target[0].size()
        # print(torch.sum(correct))
        
        acc_i=torch.sum(correct)/(h*w*d)
        acc+=acc_i
    acc=acc/pred.shape[0]
    return acc

def get_angle_loss(pred_angle, target_center, target_angle, w95, w8, w5, w3):
    # y1 = nn.Sigmoid()(pred)
    # bce_loss = nn.BCELoss()(pred, target)
    # l1_loss = nn.L1Loss()(pred, target)
    target = target_angle
    pred = pred_angle
    l2_loss = nn.MSELoss()(pred, target)
    
    # p = pred.view(pred.shape[0], -1)
    # t = pred.view(target.shape[0], -1)
    mask = (target_center>0.9).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w95*nn.MSELoss(reduction='mean')(p1, t1)
    
    mask = (target_center>0.8).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w8*nn.MSELoss(reduction='mean')(p1, t1)
    mask = (target_center>0.5).float()
    p1 = pred*mask
    t1 = target*mask
    # bce_loss += nn.BCELoss()(p1, t1)
    # l1_loss += nn.L1Loss()(p1, t1)
    l2_loss += w5*nn.MSELoss(reduction='mean')(p1, t1)
        
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
