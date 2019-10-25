import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import sys 
sys.path.append('/home/bo/projects/cvpr2020/retrievel/voxel')
from utils.utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class autoencoder(nn.Module):
    def __init__(self, num_filters=16):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            conv3x3x3(1, num_filters, 1),   # 32 32 32 32
            nn.BatchNorm3d(num_filters),
            nn.Relu(True),
            conv3x3x3(num_filters, num_filters*2, 2),   # 16 16 16  64
            nn.BatchNorm3d(num_filters*2),
            nn.Relu(True),
            conv3x3x3(num_filters*2, num_filters*4, 2),  # 8 8 8 128
            nn.BatchNorm3d(num_filters*4),
            nn.Relu(True),
            conv3x3x3(num_filters*4, num_filters*8, 2),  # 4 4 4 256
            nn.BatchNorm3d(num_filters*8),
            nn.Relu(True)
        )
        self.decoder = nn.Sequential(
            upconv3x3x3(num_filters*8, num_filters*4, 2),
            nn.BatchNorm3d(num_filters*4),
            nn.Relu(True),
            upconv3x3x3(num_filters*4, num_filters*2, 2),
            nn.BatchNorm3d(num_filters*2),
            nn.Relu(True),
            upconv3x3x3(num_filters*2, num_filters, 2),
            nn.BatchNorm3d(num_filters),
            nn.Relu(True),
            upconv3x3x3(num_filters, 1, 1),
            nn.BatchNorm3d(1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_filters, in_dim=1, out_dim=1):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim # 1
        self.out_dim = out_dim # 1
        self.num_filters = num_filters # 16
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        # self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        # self.pool_4 = max_pooling_3d()
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation,stride=1)
        
        # Up sampling
        # self.trans_1 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        # self.up_1 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation=nn.Sigmoid())
    
    def forward(self, x):
        # Down sampling
        #         # print('down1')
        down_1 = self.down_1(x) # -> [1, 16, 32, 32, 32]
        pool_1 = self.pool_1(down_1) # -> [1, 16, 16, 16, 16]
        
        down_2 = self.down_2(pool_1) # -> [1, 32, 16, 16, 16]
        pool_2 = self.pool_2(down_2) # -> [1, 32, 8, 8, 8]
        
        down_3 = self.down_3(pool_2) # -> [1, 64, 8, 8, 8]
        pool_3 = self.pool_3(down_3) # -> [1, 64, 4, 4, 4]
        
     
        # Bridge
        bridge = self.bridge(pool_3) # -> [1, 128, 4, 4, 4]
        # print('brideg', bridge.shape)
        
        # Up sampling
        # print('up1')
        trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        # print(trans_1.shape)
        concat_1 = torch.cat([trans_1, down_3], dim=1) # -> [1, 128+64, 8, 8, 8]
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        # print(up_1.shape)

        # print('up2')
        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        # print(trans_2.shape)
        concat_2 = torch.cat([trans_2, down_2], dim=1) # -> [1, 64+32, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        # print(up_2.shape)
        
        # print('up3')
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_1], dim=1) # -> [1, 16+32, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        
        # trans_4 = self.trans_4(up_3) # -> [1, 64, 32, 32, 32]
        # concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 64+32, 32, 32, 32]
        # up_4 = self.up_4(concat_4) # -> [1, 32, 32, 32, 32]
        
        # Output
        out = self.out(up_3) # -> [1, 1, 32, 32, 32]
        return out

def compute_loss(target, pred, l1_weight, l2_weight):
    # l1 loss and crossentropy loss
    l1_loss = nn.L1Loss()(target, pred)
    l2_loss = nn.MSELoss()(target, pred)
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    crossentropy_loss = 0.0
    # crossentropy_loss = nn.CrossEntropyLoss()(pred_flat, target_flat.long())
    loss = crossentropy_loss+l1_weight*l1_loss+l2_weight*l2_loss
    return loss, crossentropy_loss, l1_loss, l2_loss
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  vsize =32
  x = torch.Tensor(16, 1, vsize, vsize, vsize)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = UNet(in_dim=1, out_dim=1, num_filters=32)
  out = model(x)
  print("out size: {}".format(out.size()))
        

