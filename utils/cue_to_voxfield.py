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
    vox = torch.zeros((vx,vy,vz)).float().cuda()
  
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

def bilinear_interpolation(points, vs_x=12, vs_y=0.1, xmin=0, xmax=12, ymin=-3.84, ymax=3.84):
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
    vox=torch.zeros((vx,vy)).float().cuda()
  
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
    vox=torch.zeros(vx).float().cuda()
  
    low = points.int().float()
    high = low+1
    low = torch.clamp(low, 0, vx-1)
    high = torch.clamp(high, 0, vx-1)
    
    f=points-low
    low=low.long()
    high = high.long()
    vox[low]+=(1-f[:])
    vox[high]+=f[:]
    return vox
   

def get_center_or_corner_potential_function(points, field, vs_x=0.1, vs_y=0.1, vs_z=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84, zmin=-0.2, zmax=2.68):    
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    points[:,2] = (points[:,2]-zmin)/vs_z
    
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    vz = int((zmax-zmin)/vs_z)
    
    points[:.0] = torch.clamp(points[:,0], 0, vx-1)
    points[:,1] = torch.clamp(points[:,1], 0, vy-1)
    points[:,2] = torch.clamp(points[:,2], 0, vz-1)

    low = points.int()
    high = low + 1
    f = points - low.float()
    low=low.long()
    high = high.long()
    
    potential = torch.zeros(points.shape[0])
    potential += field[low[:,0], low[:,1], low[:,2]]*(1-f[:,0])*(1-f[:,1])*(1-f[:,2])
    potential += field[low[:,0], low[:,1], high[:,2]]*(1-f[:,0])*(1-f[:,1])*f[:,2]
    potential += field[low[:,0], high[:,1], low[:,2]]*(1-f[:,0])*f[:,1]*(1-f[:,2])
    potential += field[low[:,0], high[:,1], high[:,2]]*(1-f[:,0])*f[:,1]*f[:,2]
    potential += field[high[:,0], low[:,1], low[:,2]]*f[:,0]*(1-f[:,1])*(1-f[:,2])
    potential += field[high[:,0], high[:,1], low[:,2]]*f[:,0]*f[:,]*(1-f[:,2])
    potential += field[high[:,0], low[:,1], high[:,2]]*f[:,0]*(1-f[:,1])*f[:,2] 
    potential += field[high[:,0], high[:,1], high[:,2]]*f[:,0]*f[:,1]*f[:,2]
    return torch.sum(potential)
    
def get_xy_plane_potential_function(points, field, vs_x=0.1, vs_y=10, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84):
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    
    points[:.0] = torch.clamp(points[:,0], 0, vx-1)
    points[:,1] = torch.clamp(points[:,1], 0, vy-1)

    low = points.int()
    high = low + 1
    f = points - low.float()
    low=low.long()
    high = high.long()
    
    potential = torch.zeros(points.shape[0])
    potential += field[low[:,0], low[:,1]]*(1-f[:,0])*(1-f[:,1])
    potential += field[low[:,0], high[:,1]]*(1-f[:,0])*f[:,1]
    potential += field[high[:,0], low[:,1]]*f[:,0]*(1-f[:,1])
    potential += field[high[:,0], low[:,1]]*f[:,0]*(1-f[:,1])
    return torch.sum(potential)


def get_z_plane_potential_function(points, field, vs_x=0.1, xmin=-3.84, xmax=3.84):
    points = (points-xmin)/vs_x
    vx = int((xmax-xmin)/vs_x)
    points = torch.clamp(points, 0, vx-1)

    low = points.int()
    high = low + 1
    f = points - low.float()
    low=low.long()
    high = high.long()
    
    potential = 0.0
    potential += field[low[0], low[1]]*(1-f[0])*(1-f[1])
    potential += field[high[0], low[1]]*f[0]*(1-f[1])
    return potential
   
def trilinear_interpolation_window(points, window_size=5, sigma=0.1, vs_x=0.1, vs_y=0.1, vs_z=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84, zmin=-0.2, zmax=2.68):
    '''
    Mainly for center and corner. 
    '''
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    points[:,2] = (points[:,2]-zmin)/vs_z
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    vz = int((zmax-zmin)/vs_z)
    vox = torch.zeros((vx,vy,vz)).float().cuda()
    locations = points.int()
    k2 = int((window_size - 1)/2)
    x,y,z = torch.meshgrid(torch.arange(window_size), torch.arange(window_size),torch.arange(window_size))
    
    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations[:, 0] - k2)
    xmax = torch.min(m*(vx - 1), locations[:, 0] + k2)
    ymin = torch.max(m*0, locations[:, 1] - k2)
    ymax = torch.min(m*(vy - 1), locations[:, 1] + k2)
    zmin = torch.max(m*0, locations[:, 2] - k2)
    zmax = torch.min(m*(vz - 1), locations[:, 2] + k2)
    for i in range(points.shape[0]):
        window = torch.cat((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), dim=1).float()
        window[:,0] += locations[i,0]-k2
        window[:,1] += locations[i,1]-k2
        window[:,2] += locations[i,2]-k2
        dis_3d = torch.abs(points[i] - window)

        dis = dis_3d[:,0]*dis_3d[:,1]*dis_3d[:,2]
        dis = torch.reshape(dis, (window_size, window_size, window_size))
        dis = sigma**2/(sigma**2+dis**2)
        vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1, zmin[i]:zmax[i]+1] = torch.max(vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1, zmin[i]:zmax[i]+1], \
                                                dis[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
                                                  k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1,\
                                                  k2-locations[i, 2]+zmin[i] : k2-locations[i, 2]+zmax[i]+1])
    
    return vox

def bilinear_interpolation_window(points, window_size=3, sigma=0.1, vs_x=0.1, vs_y=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84):
    '''
    Mainly for center and corner. 
    '''
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    vox = torch.zeros((vx,vy)).float().cuda()
    locations = points.int()
    k2 = int((window_size - 1)/2)
    x,y = torch.meshgrid(torch.arange(window_size), torch.arange(window_size))
      
    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations[:, 0] - k2)
    xmax = torch.min(m*(vx - 1), locations[:, 0] + k2)
    ymin = torch.max(m*0, locations[:, 1] - k2)
    ymax = torch.min(m*(vy - 1), locations[:, 1] + k2)
   
    for i in range(points.shape[0]):
        window = torch.cat((x.reshape(-1,1), y.reshape(-1,1)), dim=1).float()
        window[:,0] += locations[i,0]-k2
        window[:,1] += locations[i,1]-k2
        dis_2d = torch.abs(points[i] - window)
        dis = dis_2d[:,0]*dis_2d[:,1]
        dis = torch.reshape(dis, (window_size, window_size))
        dis = sigma**2/(sigma**2+dis**2)
        
        vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1] = torch.max(vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1], \
                                                dis[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
                                                  k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1])
    
    return vox

def linear_interpolation_window(points, window_size=3, sigma=0.1, vs_x=0.1, xmin=-3.84, xmax=3.84):
    '''
    Mainly for center and corner. 
    '''
    points = (points-xmin)/vs_x
    vx = int((xmax-xmin)/vs_x)
    vox = torch.zeros(vx).float().cuda()
    locations = points.int()
    k2 = int((window_size - 1)/2)
    x = torch.arange(window_size)

    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations - k2)
    xmax = torch.min(m*(vx - 1), locations + k2)
   
    for i in range(points.shape[0]):
        window = x.reshape(-1,1).float()
        window += locations-k2
       
        dis_1d = torch.abs(points[i] - window)
        dis = dis_1d
        dis = sigma**2/(sigma**2+dis**2)
        
        vox[xmin[i]:xmax[i]+1] = torch.max(vox[xmin[i]:xmax[i]+1], dis[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1])
    
    return vox  
 
def get_potential_function_trilinear_window(points, field, window_size=5, sigma=0.1, vs_x=0.1, vs_y=0.1, vs_z=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84, zmin=-0.2, zmax=2.68):
    '''
    Mainly for center and corner. 
    '''
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    points[:,2] = (points[:,2]-zmin)/vs_z
    
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    vz = int((zmax-zmin)/vs_z)
    
    points[:,0] = torch.clamp(points[:,0], 0, vx-1)
    points[:,1] = torch.clamp(points[:,1], 0, vy-1)
    points[:,2] = torch.clamp(points[:,2], 0, vz-1)
    locations = points.int()
    k2 = int((window_size - 1)/2)
    x,y,z = torch.meshgrid(torch.arange(window_size), torch.arange(window_size),torch.arange(window_size))
    
    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations[:, 0] - k2)
    xmax = torch.min(m*(vx - 1), locations[:, 0] + k2)
    ymin = torch.max(m*0, locations[:, 1] - k2)
    ymax = torch.min(m*(vy - 1), locations[:, 1] + k2)
    zmin = torch.max(m*0, locations[:, 2] - k2)
    zmax = torch.min(m*(vz - 1), locations[:, 2] + k2)
    
    potential = 0.0 
    
    for i in range(points.shape[0]):
        window = torch.cat((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), dim=1).float()
        window[:,0] += locations[i,0]-k2
        window[:,1] += locations[i,1]-k2
        window[:,2] += locations[i,2]-k2
        dis_3d = torch.abs(points[i] - window)
        dis = dis_3d[:,0]*dis_3d[:,1]*dis_3d[:,2]
        dis = torch.reshape(dis, (window_size, window_size, window_size))
        weight = sigma**2/(sigma**2+dis**2)
        potential += torch.sum(field[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1, zmin[i]:zmax[i]+1]*
                                                weight[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
                                                  k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1,\
                                                  k2-locations[i, 2]+zmin[i] : k2-locations[i, 2]+zmax[i]+1])
    
    return potential


def get_potential_function_bilinear_window(points, field, window_size=5, sigma=0.1, vs_x=0.1, vs_y=0.1, xmin=-3.84, xmax=3.84, ymin=-3.84, ymax=3.84):
    '''
    Mainly for center and corner. 
    '''
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    
    points[:,0] = torch.clamp(points[:,0], 0, vx-1)
    points[:,1] = torch.clamp(points[:,1], 0, vy-1)
    
    locations = points.int()
    k2 = int((window_size - 1)/2)
    x,y= torch.meshgrid(torch.arange(window_size), torch.arange(window_size))
    
    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations[:, 0] - k2)
    xmax = torch.min(m*(vx - 1), locations[:, 0] + k2)
    ymin = torch.max(m*0, locations[:, 1] - k2)
    ymax = torch.min(m*(vy - 1), locations[:, 1] + k2)
    
    potential = 0.0 
    
    for i in range(points.shape[0]):
        window = torch.cat((x.reshape(-1,1), y.reshape(-1,1)), dim=1).float()
        window[:,0] += locations[i,0]-k2
        window[:,1] += locations[i,1]-k2
        
        dis_2d = torch.abs(points[i] - window)
        dis = dis_2d[:,0]*dis_2d[:,1]
        dis = torch.reshape(dis, (window_size, window_size))
        weight = sigma**2/(sigma**2+dis**2)
        
        potential += torch.sum(field[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1,]*
                                                weight[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
                                                  k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1])
    
    return potential


def get_potential_function_linear_window(points, field, window_size=5, sigma=0.1, vs_x=0.1, xmin=-3.84, xmax=3.84):
    '''
    Mainly for center and corner. 
    '''
    points= (points-xmin)/vs_x
    vx = int((xmax-xmin)/vs_x)
    points = torch.clamp(points, 0, vx-1)
    
    locations = points.int()
    k2 = int((window_size - 1)/2)
    x= torch.arange(window_size)
    
    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations - k2)
    xmax = torch.min(m*(vx - 1), locations + k2)
    
    potential = 0.0 
    
    for i in range(points.shape[0]):
        window =x.reshape(-1,1).float()
        window += locations-k2
        dis_1d = torch.abs(points[i] - window)
        dis = dis_1d
        weight = sigma**2/(sigma**2+dis**2)
        potential += torch.sum(field[xmin[i]:xmax[i]+1]*weight[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1])
    
    return potential
def get_oriented_corners_torch(bbx):
    center = bbx[0:3]
    xsize = bbx[3]
    ysize = bbx[4]
    zsize = bbx[5]
    angle = bbx[6]
    vx = torch.tensor([np.cos(angle), np.sin(angle), 0])
    vy = torch.tensor([-np.sin(angle), np.cos(angle), 0])
    vx = vx * torch.abs(xsize) / 2
    vy = vy * torch.abs(ysize) / 2
    vz = torch.tensor([0, 0, torch.abs(zsize) / 2])
    corners = torch.zeros((8,3)).cuda()
    corners[0] = center + vx + vy + vz
    corners[1] = center - vx + vy + vz
    corners[2] = center + vx - vy + vz
    corners[3] = center - vx - vy + vz
    corners[4] = center + vx + vy - vz
    corners[5] = center - vx + vy - vz
    corners[6] = center + vx - vy - vz
    corners[7] = center - vx - vy - vz
    
    return corners

def get_oriented_cues_torch(bbx):
    center = bbx[0:3]
    xsize = bbx[3]
    ysize = bbx[4]
    zsize = bbx[5]
    angle = bbx[6]
    vx = torch.tensor([np.cos(angle), np.sin(angle), 0])
    vy = torch.tensor([-np.sin(angle), np.cos(angle), 0])
    vx = vx * torch.abs(xsize) / 2
    vy = vy * torch.abs(ysize) / 2
    vz = torch.tensor([0, 0, torch.abs(zsize) / 2])
    corners = torch.zeros((8,3))
    corners[0] = center + vx + vy + vz
    corners[1] = center - vx + vy + vz
    corners[2] = center + vx - vy + vz
    corners[3] = center - vx - vy + vz
    corners[4] = center + vx + vy - vz
    corners[5] = center - vx + vy - vz
    corners[6] = center + vx - vy - vz
    corners[7] = center - vx - vy - vz

    return center, corners

def get_oriented_cues_batch_torch(bbx, end_points, batch_data_label):
    center = bbx[:,0:3]
    xsize = bbx[:,3]
    ysize = bbx[:,4]
    zsize = bbx[:,5]
    angle = bbx[:,6]
    vx = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros(angle.shape).cuda()], 1)
    vy = torch.stack([-torch.sin(angle), torch.cos(angle), torch.zeros(angle.shape).cuda()], 1)
    vx = vx * torch.abs(xsize).unsqueeze(-1) / 2
    vy = vy * torch.abs(ysize).unsqueeze(-1) / 2
    vz = torch.stack([torch.zeros(zsize.shape).cuda(), torch.zeros(zsize.shape).cuda(), torch.abs(zsize) / 2],1)
    corners = torch.zeros((bbx.shape[0],8,3)).cuda()
    corners[:,0] += center + vx + vy + vz
    corners[:,1] += center - vx + vy + vz
    corners[:,2] += center + vx - vy + vz
    corners[:,3] += center - vx - vy + vz
    corners[:,4] += center + vx + vy - vz
    corners[:,5] += center - vx + vy - vz
    corners[:,6] += center + vx - vy - vz
    corners[:,7] += center - vx - vy - vz

    planez0 = torch.zeros((bbx.shape[0],1)).cuda()
    planez0[:,0] += -(corners[:,4,-1])
    
    planez1 = torch.zeros((bbx.shape[0],1)).cuda()
    planez1[:,0] += -(corners[:,0,-1])

    v1 = corners[:,1] - corners[:,5]
    v2 = corners[:,5] - corners[:,7]
    cp = torch.cross(v1, v2)
    d = torch.sum(cp*corners[:,7], 1)
    d = - d / torch.norm(cp, dim=1)
    a = (cp / torch.norm(cp, dim=1).unsqueeze(1))[:,0]
    b = (cp / torch.norm(cp, dim=1).unsqueeze(1))[:,1]
    xcls = (torch.atan(-a/(b+1e-16)) + np.pi / 2) / (np.pi/12)
    planex0 = torch.zeros((bbx.shape[0],2)).cuda()
    planex0[:,0] += xcls
    planex0[:,1] += d
    v1 = corners[:,0] - corners[:,4]
    v2 = corners[:,4] - corners[:,6]
    cp = torch.cross(v1, v2)
    d = torch.sum(cp*corners[:,6], 1)
    d = -d / torch.norm(cp, dim=1)
    planex1 = torch.zeros((bbx.shape[0],2)).cuda()
    planex1[:,0] += xcls
    planex1[:,1] += d

    v1 = corners[:,7] - corners[:,6]
    v2 = corners[:,6] - corners[:,2]
    cp = torch.cross(v1, v2)
    d = torch.sum(cp*corners[:,2], 1)
    d = -d / torch.norm(cp, dim=1)
    a = (cp / torch.norm(cp, dim=1).unsqueeze(1))[:,0]
    b = (cp / torch.norm(cp, dim=1).unsqueeze(1))[:,1]
    xcls = (torch.atan(-a/(b+1e-16)) + np.pi / 2) / (np.pi/12)
    planey0 = torch.zeros((bbx.shape[0],2)).cuda()
    planey0[:,0] += xcls
    planey0[:,1] += d
    v1 = corners[:,5] - corners[:,4]
    v2 = corners[:,4] - corners[:,0]
    cp = torch.cross(v1, v2)
    d = torch.sum(cp*corners[:,0], 1)
    d = -d / torch.norm(cp, dim=1)
    planey1 = torch.zeros((bbx.shape[0],2)).cuda()
    planey1[:,0] += xcls
    planey1[:,1] += d

    return center[0:batch_data_label['num_instance'][0],...], corners[0:batch_data_label['num_instance'][0],...], planex0[0:batch_data_label['num_instance'][0],...], planex1[0:batch_data_label['num_instance'][0],...], planey0[0:batch_data_label['num_instance'][0],...], planey1[0:batch_data_label['num_instance'][0],...], planez0[0:batch_data_label['num_instance'][0],...], planez1[0:batch_data_label['num_instance'][0],...]

def gaussian_3d_torch(x_mean, y_mean, z_mean,  ksize, dev=0.5):
  x, y, z = torch.meshgrid(torch.arange(ksize), torch.arange(ksize),torch.arange(ksize))
  x,y,z=x.float(), y.float(), z.float()
  m = torch.exp(-((x-x_mean)**2 + (y-y_mean)**2+(z-z_mean)**2)/(2.0*dev**2))
  return m

def gaussian_2d_torch(x_mean, y_mean, ksize, dev=0.5):
  x, y = torch.meshgrid(torch.arange(ksize), torch.arange(ksize))
  x,y=x.float(), y.float()
  m = torch.exp(-((x-x_mean)**2 + (y-y_mean)**2)/(2.0*dev**2))
  return m

def gaussian_1d_torch(x_mean, ksize, dev=0.5):
  x= torch.arange(ksize).float()
  m = torch.exp(-((x-x_mean)**2)/(2.0*dev**2))
  return m

def get_3d_field(points, vs_x=0.1, vs_y=0.1, vs_z=0.1, dev=0.5, ksize=3, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0, zmin=-0.5, zmax=3.0):
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    points[:,2] = (points[:,2]-zmin)/vs_z

    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    vz = int((zmax-zmin)/vs_z)
    points[:,0] = torch.clamp(points[:,0], 0, vx-1)
    points[:,1] = torch.clamp(points[:,1], 0, vy-1)
    points[:,2] = torch.clamp(points[:,2], 0, vz-1)
    
    vox = torch.zeros((vx,vy,vz)).float().cuda()
    k2 = int((ksize - 1)/2)
    gauss = gaussian_3d_torch(k2, k2, k2, ksize, dev=dev)
    locations = points.int()
    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations[:, 0] - k2)
    xmax = torch.min(m*(vx - 1), locations[:, 0] + k2)
    ymin = torch.max(m*0, locations[:, 1] - k2)
    ymax = torch.min(m*(vy - 1), locations[:, 1] + k2)
    zmin = torch.max(m*0, locations[:, 2] - k2)
    zmax = torch.min(m*(vz - 1), locations[:, 2] + k2)
    for i in range(points.shape[0]):
#         vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1, zmin[i]:zmax[i]+1] += gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
#                                                   k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1,\
#                                                   k2-locations[i, 2]+zmin[i] : k2-locations[i, 2]+zmax[i]+1] 
        vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1, zmin[i]:zmax[i]+1] = torch.max(vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1, zmin[i]:zmax[i]+1], \
                                                gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
                                                  k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1,\
                                                  k2-locations[i, 2]+zmin[i] : k2-locations[i, 2]+zmax[i]+1])
    return vox
    

def get_2d_field(points, vs_x=0.1, vs_y=0.1,  dev=0.5, ksize=3, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    points[:,0] = torch.clamp(points[:,0], 0, vx-1)
    points[:,1] = torch.clamp(points[:,1], 0, vy-1)
    vox = torch.zeros((vx,vy)).float().cuda()
    k2 = int((ksize - 1)/2)
    gauss = gaussian_2d_torch(k2, k2, ksize, dev=dev)
    locations = points.int()
    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations[:, 0] - k2)
    xmax = torch.min(m*(vx - 1), locations[:, 0] + k2)
    ymin = torch.max(m*0, locations[:, 1] - k2)
    ymax = torch.min(m*(vy - 1), locations[:, 1] + k2)
    for i in range(points.shape[0]):
#         vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1] += gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
#                                                   k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1]
        vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1] += torch.max(vox[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1], \
                                                gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
                                                  k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1])
    return vox

def get_1d_field(points, vs_x=0.1,  dev=0.5, ksize=3, xmin=-5.0, xmax=5.0):
    points= (points-xmin)/vs_x
    vx = int((xmax-xmin)/vs_x)
    points = torch.clamp(points, 0, vx-1)
    vox = torch.zeros(vx).float().cuda()
    k2 = int((ksize - 1)/2)
    gauss = gaussian_1d_torch(k2, ksize, dev=dev)
    locations = points.int()
    m = torch.ones(points.shape[0]).int().cuda()
    xmin = torch.max(m*0, locations-k2)
    xmax = torch.min(m*(vx - 1), locations + k2)
    for i in range(points.shape[0]):
#         vox[xmin[i]:xmax[i]+1] += gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1]
        vox[xmin[i]:xmax[i]+1] += torch.max(vox[xmin[i]:xmax[i]+1], \
            gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1])
    return vox
   
def get_3d_potential_function(points, field, vs_x=0.1, vs_y=0.1, vs_z=0.1, dev=0.5, ksize=3, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0, zmin=-0.5, zmax=3.0):   
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    points[:,2] = (points[:,2]-zmin)/vs_z
    
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    vz = int((zmax-zmin)/vs_z)
    
    points[:,0] = torch.clamp(points[:,0], 0, vx-1)
    points[:,1] = torch.clamp(points[:,1], 0, vy-1)
    points[:,2] = torch.clamp(points[:,2], 0, vz-1)
    
    k2 = int((ksize - 1)/2)
    gauss = gaussian_3d_torch(k2, k2, k2, ksize, dev=dev)
    locations = points.int()
    m = torch.ones(points.shape[0]).int()
    xmin = torch.max(m*0, locations[:, 0] - k2)
    xmax = torch.min(m*(vx - 1), locations[:, 0] + k2)
    ymin = torch.max(m*0, locations[:, 1] - k2)
    ymax = torch.min(m*(vy - 1), locations[:, 1] + k2)
    zmin = torch.max(m*0, locations[:, 2] - k2)
    zmax = torch.min(m*(vz - 1), locations[:, 2] + k2)
    
    potential = 0.0 
    for i in range(points.shape[0]):
        potential += torch.sum(field[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1, zmin[i]:zmax[i]+1]*gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
                                                  k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1,\
                                                  k2-locations[i, 2]+zmin[i] : k2-locations[i, 2]+zmax[i]+1])
    return potential
 

def get_2d_potential_function(points, field, vs_x=0.1, vs_y=0.1, dev=0.5, ksize=3, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):   
    points[:,0] = (points[:,0]-xmin)/vs_x
    points[:,1] = (points[:,1]-ymin)/vs_y
    vx = int((xmax-xmin)/vs_x)
    vy = int((ymax-ymin)/vs_y)
    points[:,0] = torch.clamp(points[:,0], 0, vx-1)
    points[:,1] = torch.clamp(points[:,1], 0, vy-1)
    
    k2 = int((ksize - 1)/2)
    gauss = gaussian_2d_torch(k2, k2, ksize, dev=dev)
    locations = points.int()
    m = torch.ones(points.shape[0]).int()
    xmin = torch.max(m*0, locations[:, 0] - k2)
    xmax = torch.min(m*(vx - 1), locations[:, 0] + k2)
    ymin = torch.max(m*0, locations[:, 1] - k2)
    ymax = torch.min(m*(vy - 1), locations[:, 1] + k2)
    
    potential = 0.0 
    for i in range(points.shape[0]):
        potential += torch.sum(field[xmin[i]:xmax[i]+1, ymin[i]:ymax[i]+1]*gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1,\
                                                  k2-locations[i, 1]+ymin[i] : k2-locations[i, 1]+ymax[i]+1])
    return potential
    
def get_1d_potential_function(points, field, vs_x=0.1, dev=0.5, ksize=3, xmin=-5.0, xmax=5.0):   
    points = (points-xmin)/vs_x
    vx = int((xmax-xmin)/vs_x)
    points = torch.clamp(points, 0, vx-1)
    
    k2 = int((ksize - 1)/2)
    gauss = gaussian_1d_torch(k2, ksize, dev=dev)
    locations = points.int()
    m = torch.ones(points.shape[0]).int()
    xmin = torch.max(m*0, locations - k2)
    xmax = torch.min(m*(vx - 1), locations + k2)
    potential = 0.0 
    for i in range(points.shape[0]):
        potential += torch.sum(field[xmin[i]:xmax[i]+1]*gauss[k2-locations[i, 0]+xmin[i] : k2-locations[i, 0]+xmax[i]+1])
    return potential


if __name__=="__main__":
    pt = np.loadtxt('rm.xyz')
    print(pt.shape)
    print(pt.max(0), pt.min(0))
    pt = torch.from_numpy(pt).float()
    vox1 = trilinear_interpolation(pt, 0.05, 0.05, 0.05, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    
    pt2, label2 = volume_to_point_cloud_color(vox1.numpy())
    write_ply_color(pt2, label2*5, '00.ply')
    print(label2.max())
