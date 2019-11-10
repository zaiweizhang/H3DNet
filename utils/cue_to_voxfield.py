import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
 
import numpy as np
import random
import torch
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm
import matplotlib.pyplot as pyplot
from pc_util import *

def trilinear_interpolation(points, vs_x=0.1, vs_y=0.1, vs_z=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84, zmin=-0.2, zmax=2.68):
    '''
    Mainly for center and corner. 
    '''
    points[:,0] = torch.clamp(points[:,0], xmin, xmax)
    points[:,1] = torch.clamp(points[:,1], ymin, ymax)
    points[:,2] = torch.clamp(points[:,2], zmin, zmax)
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    points[:,2] = (points[:,2]-zmin)/vs_z
    
    
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    vz = int((zmax-zmin)/vs_z)
    vox = torch.zeros((vx,vy,vz)).float()
  
    low = points.int()
    high = low+1
    low[:,0] = torch.clamp(low[:,0], 0, vx-1)
    low[:,1] = torch.clamp(low[:,1], 0, vy-1)
    low[:,2] = torch.clamp(low[:,2], 0, vz-1)
    high[:,0] = torch.clamp(high[:,0], 0, vx-1)
    high[:,1] = torch.clamp(high[:,1], 0, vy-1)
    high[:,2] = torch.clamp(high[:,2], 0, vz-1)
    
    f=points-low.float()
    
    low=low.long()
    high = high.long()
    vox[low[:,0], low[:,1], low[:,2]]+=(1-f[:,0])*(1-f[:,1])*(1-f[:,2])
    vox[low[:,0], low[:,1], high[:,2]]+=(1-f[:,0])*(1-f[:,1])*f[:,2]
    vox[low[:,0], high[:,1], low[:,2]]+=(1-f[:,0])*f[:,1]*(1-f[:,2])
    vox[low[:,0], high[:,1], high[:,2]]+=(1-f[:,0])*f[:,1]*f[:,2]
    vox[high[:,0], low[:,1], low[:,2]]+=f[:,0]*(1-f[:,1])*(1-f[:,2])
    vox[high[:,0], high[:,1], low[:,2]]+=f[:,0]*f[:,1]*(1-f[:,2])
    vox[high[:,0], low[:,1], high[:,2]]+=f[:,0]*(1-f[:,1])*f[:,2]
    vox[high[:,0], high[:,1], high[:,2]]+=f[:,0]*f[:,1]*f[:,2]
    return vox

def bilinear_interpolation(points, vs_x=0.1, vs_y=10, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84):
    '''
    Mainly for plane (d and theta).
    d: the same range of x or y
    theta: 0 - 180
    '''
    points[:,0] = torch.clamp(points[:,0], xmin, xmax)
    points[:,1] = torch.clamp(points[:,1], ymin, ymax)
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y

    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    vox=torch.zeros((vx,vy)).float()
  
    low = points.int().float()
    high = low+1
    low[:,0] = torch.clamp(low[:,0], 0, vx-1)
    low[:,1] = torch.clamp(low[:,1], 0, vy-1)
    high[:,0] = torch.clamp(high[:,0], 0, vx-1)
    high[:,1] = torch.clamp(high[:,1], 0, vy-1)
    
    f=points-low
    low=low.long()
    high = high.long()
    vox[low[:,0], low[:,1]]+=(1-f[:,0])*(1-f[:,1])
    vox[low[:,0], high[:,1]]+=(1-f[:,0])*f[:,1]
    vox[high[:,0], low[:,1]]+=f[:,0]*(1-f[:,1])
    vox[high[:,0], low[:,1]]+=f[:,0]*(1-f[:,1])
    return vox
    

def linear_interpolation(points, vs_x=0.1,xmin=-3.84, xmax=3.84):
    '''
    Mainly for plane (d and theta).
    d: the same range of x or y
    theta: 0 - 180
    '''
    points = torch.clamp(points, xmin, xmax)
    points = (points-xmin)/vs_x

    vx = int((xmax-xmin)/vs_x)
    vox=torch.zeros(vx).float()
  
    low = points.int().float()
    high = low+1
    low = torch.clamp(low, 0, vx-1)
    high = torch.clamp(high, 0, vx-1)
    
    f=points-low
    low=low.long()
    high = high.long()
    vox[low[:], low[:]]+=(1-f[:])
    vox[high[:], low[:]]+=f[:]
    return vox



if __name__=="__main__":
    pt = np.loadtxt('rm.xyz')
    print(pt.shape)
    print(pt.max(0), pt.min(0))
    pt = torch.from_numpy(pt).float()
    vox1 = trilinear_interpolation(pt, 0.05, 0.05, 0.05, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    
    pt2, label2 = volume_to_point_cloud_color(vox1.numpy())
    write_ply_color(pt2, label2*5, '00.ply')
    print(label2.max())
