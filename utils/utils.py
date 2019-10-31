import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1)
def upconv3x3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(
        in_planes, 
        out_planes, 
        kernel_size=3, 
        stride=1,
        padding=1,
        output_padding=1)
    
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation, stride=1):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm3d(out_dim),)

