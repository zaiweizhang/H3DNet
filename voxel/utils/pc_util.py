import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Point cloud IO
import numpy as np
import random
import torch
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm
import matplotlib.pyplot as pyplot
import struct

def save_results(points, pred, target, outf, epoch, i, names=None):
    points = points.cpu().numpy()
    batch_size = points.shape[0]
    pred = pred.detach().cpu().numpy()
    print(pred.shape)
    target = target.detach().cpu().numpy()
    out_dir = os.path.join(outf, "%d-%d"%(epoch, i))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if names==None:
        for k in range(batch_size//8):
            pt_pred = volume_to_point_cloud(pred[k*4,0]) 
            write_ply(points[k*4], os.path.join(out_dir, "%d_points.ply"%(k)))
            write_ply(pt_pred, os.path.join(out_dir, "%d_pred.ply"%(k)))
            write_ply(target[k*4], os.path.join(out_dir, "%d_target.ply"%(k)))
    else:
        for k in range(batch_size//8):
            pt_pred = volume_to_point_cloud(pred[k*4,0])
            write_ply(points[k*4], os.path.join(out_dir, "%d_%s_points.ply"%(k,names[k])))
            write_ply(pt_pred, os.path.join(out_dir, "%d_%s_pred.ply"%(k,names[k])))
            write_ply(target[k*4], os.path.join(out_dir, "%d_%s_target.ply"%(k,names[k])))
    print("save results")
    
def save_vox_results(vox, center, pred, outfolder, epoch, i, mode):
    vox = vox.cpu().numpy()[0,0]
    center = center.cpu().numpy()[0,0]
    pred = pred.detach().cpu().numpy()[0,0]
    
    pt_vox = volume_to_point_cloud(vox)
    pt_center = volume_to_point_cloud(center, thres=0.6)
    pred_center = volume_to_point_cloud(pred, thres=0.6)
    pred_center2 = volume_to_point_cloud(pred, thres=0.8)

    num_pt_vox = pt_vox.shape[0]
    num_pt_center = pt_center.shape[0]
    num_pred_center = pred_center.shape[0]
    num_pred_center2 = pred_center2.shape[0]
    print ('num_pred'+mode,num_pred_center)
    print ('num_pred2'+mode, num_pred_center2)
    print ('num_gt'+mode, num_pt_center)
    
    pt_vox_center = np.concatenate((pt_vox, pt_center),axis=0)
    pred_vox_center = np.concatenate((pt_vox, pred_center),axis=0)
    label = np.zeros(num_pt_center+num_pt_vox)
    label[-num_pt_center:]=1
    label_pred = np.zeros(num_pred_center+num_pt_vox)
    label_pred[-num_pred_center:]=1
    
   
    # write_ply(pt_vox, os.path.join(outfolder, '{}_{}_{}_vox.ply'.format(epoch, i, mode)))        
    # write_ply(pred_center, os.path.join(outfolder, '{}_{}_pred_center.ply'.format(epoch, i)))        
    write_ply_color(pt_vox_center, label, os.path.join(outfolder,'{}_{}_{}.ply'.format(epoch, i, mode)))  
    write_ply_color(pred_vox_center, label_pred, os.path.join(outfolder,'{}_{}_{}_pred.ply'.format(epoch, i, mode)))  
    # write_ply_color(pred_vox_center2, label_pred2, os.path.join(outfolder,'{}_{}_{}_pred2.ply'.format(epoch, i, mode)))  
  
def save_angle_results(vox, center, pred_center, angle, pred_angle, outfolder, epoch, i, mode):
    vox = vox.cpu().numpy()[0,0]
    center = center.cpu().numpy()[0,0]
    angle = angle.cpu().numpy()[0,0]
    pred_center = pred_center.detach().cpu().numpy()[0,0]
    pred_angle = pred_angle.detach().cpu().numpy()[0,0]
    
    pt_vox = volume_to_point_cloud(vox)
    pt_center = volume_to_point_cloud(center, thres=0.6)
    pt_angle, label_angle_0 = volume_to_point_cloud_color(angle, mini=-1.0)
    pt_pred_center = volume_to_point_cloud(pred_center, thres=0.6)
    pt_pred_angle, label_pred_angle_0 = volume_to_point_cloud_color(pred_angle, mini=-1.0)
    
    num_pt_vox = pt_vox.shape[0]
    num_pt_center = pt_center.shape[0]
    num_pt_angle = pt_angle.shape[0]
    num_pred_center = pt_pred_center.shape[0]
    num_pred_angle = pt_pred_angle.shape[0]
    print ('num pred center', num_pred_center)
    print ('num gt center', num_pt_center)
    print ('num pred angle', num_pred_angle)
    print ('num gt angle', num_pt_angle)
    
    pt_vox_center = np.concatenate((pt_vox, pt_center),axis=0)
    pt_vox_angle = np.concatenate((pt_vox, pt_angle),axis=0)
    pred_vox_center = np.concatenate((pt_vox, pt_pred_center),axis=0)
    pred_vox_angle = np.concatenate((pt_vox, pt_pred_angle),axis=0)
    
    label_center = np.zeros(num_pt_center+num_pt_vox)
    label_center[-num_pt_center:]=1
    label_pred_center = np.zeros(num_pred_center+num_pt_vox)
    label_pred_center[-num_pred_center:]=1
    label_angle = np.zeros(num_pt_angle+num_pt_vox)
    label_angle[-num_pt_angle:]=label_angle_0+1
    label_pred_angle = np.zeros(num_pred_angle+num_pt_vox)
    label_pred_angle[-num_pred_angle:]=label_pred_angle_0+1

    # write_ply(pt_vox, os.path.join(outfolder, '{}_{}_{}_vox.ply'.format(epoch, i, mode)))        
    write_ply_color(pt_vox_center, label_center, os.path.join(outfolder,'{}_{}_{}_center.ply'.format(epoch, i, mode)))  
    write_ply_color(pred_vox_center, label_pred_center, os.path.join(outfolder,'{}_{}_{}_pred_center.ply'.format(epoch, i, mode)))  
    write_ply_color(pt_vox_angle, label_angle, os.path.join(outfolder,'{}_{}_{}_angle.ply'.format(epoch, i, mode)),num_classes=21)  
    write_ply_color(pred_vox_angle, label_pred_angle, os.path.join(outfolder,'{}_{}_{}_pred_angle.ply'.format(epoch, i, mode)), num_classes=21)  
 
def process_results(vox, center, scene_name, out_path, pt_thres=0.8):
  pt_vox = volume_to_point_cloud(vox)
  pt_center = volume_to_point_cloud(center, pt_thres)
  num_pt_vox = pt_vox.shape[0]
  num_pt_center = pt_center.shape[0]
  pt_vox_center = np.concatenate((pt_vox, pt_center),axis=0)
  label = np.zeros(num_pt_center+num_pt_vox)
  label[-num_pt_center:-1]=1
  
def compute_iou(v1, v2, thres=0.5):
  bv1 = (v1>thres).float()
  bv2 = (v2>thres).float()
  iou=0.0
  for i in range(v1.shape[0]):
      iou += torch.sum(bv1[i]*bv2[i]).float()/torch.sum(bv1[i]+bv2[i]-bv1[i]*bv2[i]).float()
  iou = iou/v1.shape[0]
  return iou

def compute_iou_pc(pc1, pc2):
    v1 = point_cloud_to_volume_batch(pc1, 48)
    v2 = point_cloud_to_volume_batch(pc2, 48)
    iou = 0.0
    for i in range(pc1.shape[0]):
        iou += torch.sum(v1[i]*v2[i]).float()/torch.sum(v1[i]+v2[i]-v1[i]*v2[i]).float()
    iou = iou/pc1.shape[0]
    return iou

def point_cloud_to_volume_batch(point_clouds, vsize=32, radius=1.0, flatten=True):
  """ Input is BxNx3 batch of point cloud
      Output is Bx(vsize^3)
  """
  vol_list = []
  for b in range(point_clouds.shape[0]):
    vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
    if flatten:
      vol_list.append(vol.flatten())
    else:
      vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
  if flatten:
    return np.vstack(vol_list)
  else:
    return np.concatenate(vol_list, 0)

def point_cloud_to_volume(points, vsize=32, radius=1.0):
  """ input is Nx3 points.
      output is vsize*vsize*vsize in range 0 or 1
      assumes points are in range [-radius, radius]
  """
  vol = np.zeros((vsize, vsize, vsize)).astype(np.int32)
  voxel = 2 * radius / float(vsize)
  locations = (points + radius) / voxel
  # print 'localcations', locations
  locations = np.clip(locations, 0, vsize-1).astype(int)
  vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
  return vol

def volume_to_point_cloud(vol, thres =0.1):
  """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
      return Nx3 numpy array.
  """
  vx = vol.shape[0]
  vy = vol.shape[1]
  vz = vol.shape[2]
  points = []

  for a in range(vx):
    for b in range(vy):
      for c in range(vz):
        if vol[a, b, c]>thres:
          points.append(np.array([a, b, c]))
          # points.append(np.array([a+random.random(), b+random.random(), c+random.random()]))
  if len(points) == 0:
    return np.zeros((0, 3))
  points = np.vstack(points)
  return points

def volume_to_point_cloud_color(vol, mini=0):
  vx = vol.shape[0]
  vy = vol.shape[1]
  vz = vol.shape[2]
  points = []
  labels = []
  for a in range(vx):
    for b in range(vy):
      for c in range(vz):
        if vol[a, b, c]!=0.0:
          points.append(np.array([a, b, c]))
          labels.append(vol[a,b,c])
  labels = (np.array(labels, np.float32)-mini)*10
  
  if len(points) == 0:
    return np.zeros((0, 3))
  points = np.vstack(points)
  
  return points, labels

def volume_pt_to_pt(vpt, vs=0.025, xymin=-3.2, xymax=3.2, zmin=-0.1, zmax=2.32):
  pt = vpt*vs
  pt[:,0] = pt[:,0]+xymin
  pt[:,1] = pt[:,1]+xymin
  pt[:,2] = pt[:,2]+zmin
  return pt
  
def process_bbx(bbx, vs=0.025, xymin=-3.2, xymax=3.2, zmin=-0.1, zmax=2.32):
  center = bbx[:,0:3]
  #clip to min max range
  crop_idxs = []
  for i in range(bbx.shape[0]):
    if center[i, 0]>xymin and center[i, 0]<xymax:
      if center[i, 1]>xymin and center[i, 1]<xymax:
        if center[i, 2]>zmin and center[i, 2]<zmax:
          crop_idxs.append(i)
  new_bbx = bbx[crop_idxs]
  new_center = new_bbx[:,0:3]
  new_scale = new_bbx[:,3:6]
  new_center[:,0] = new_center[:,0]-xymin
  new_center[:,1] = new_center[:,1]-xymin
  new_center[:,2] = new_center[:,2]-zmin
  new_center = new_center/vs
  new_scale = new_scale/vs
  new_bbx[:,0:3] = new_center
  new_bbx[:,3:6] = new_scale
  return new_bbx

def get_corner(bbx, vox, select=True, select_thres=3):
  corners = np.zeros((bbx.shape[0], 8 ,3))
  for i in range(bbx.shape[0]):
    center = bbx[i, 0:3]
    scale = bbx[i, 3:6]/2.0
    corners[i,0] = [center[0]-scale[0], center[1]-scale[1], center[2]-scale[2]]
    corners[i,1] = [center[0]-scale[0], center[1]-scale[1], center[2]+scale[2]]
    corners[i,2] = [center[0]-scale[0], center[1]+scale[1], center[2]-scale[2]]
    corners[i,3] = [center[0]-scale[0], center[1]+scale[1], center[2]+scale[2]]
    corners[i,4] = [center[0]+scale[0], center[1]-scale[1], center[2]-scale[2]]
    corners[i,5] = [center[0]+scale[0], center[1]-scale[1], center[2]+scale[2]]
    corners[i,6] = [center[0]+scale[0], center[1]+scale[1], center[2]-scale[2]]
    corners[i,7] = [center[0]+scale[0], center[1]+scale[1], center[2]+scale[2]]
  corners = np.reshape(corners, (-1, 3))
  corners = np.clip(corners, 0, vox.shape[0]-1)
  corners[:,2] = np.clip(corners[:,2], 0, vox.shape[2]-1)
  
  if select:
    vox = (vox>0.05).astype(np.float32)
    max_xy = vox.shape[0]-1
    max_z = vox.shape[2]-1
    select_ids = []
    th = select_thres
    for j in range(corners.shape[0]):
      x1 = int(np.maximum(0, corners[j,0]-th))
      y1 = int(np.maximum(0, corners[j,1]-th))
      z1 = int(np.maximum(0, corners[j,2]-th))
      x2 = int(np.minimum(max_xy, corners[j,0]+th))
      y2 = int(np.minimum(max_xy, corners[j,1]+th))
      z2 = int(np.minimum(max_z, corners[j,2]+th))
      crop_vox = vox[x1:x2, y1:y2, z1:z2]
      if np.sum(crop_vox)>2:
        select_ids.append(j)
    corners = corners[select_ids]
  return corners

def params2bbox(bbx):
    ''' from bbox_center, angle and size to bbox
    @Args:
        center: (3)
        x/y/zsize: scalar
        angle: -pi ~ pi
    @Returns:
        bbox: 8 x 3, order:
         [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
          [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    '''
    center = bbx[0:3]
    xsize = bbx[3]
    ysize = bbx[4]
    zsize = bbx[5]
    angle = bbx[6]
    vx = np.array([np.cos(angle), np.sin(angle), 0])
    vy = np.array([-np.sin(angle), np.cos(angle), 0])
    vx = vx * np.abs(xsize) / 2
    vy = vy * np.abs(ysize) / 2
    vz = np.array([0, 0, np.abs(zsize) / 2])
    bbox = np.array([\
        center - vx - vy - vz, center - vx - vy + vz,
        center - vx + vy - vz, center - vx + vy + vz,
        center + vx - vy - vz, center + vx - vy + vz,
        center + vx + vy - vz, center + vx + vy + vz])
    return bbox

def get_oriented_corners(oriented_bbx, vox, select=True, select_thres=5,vs=0.025, xymin=-3.2, xymax=3.2, zmin=-0.1, zmax=2.32):
  # corners in origin scale
  corners = np.zeros((oriented_bbx.shape[0], 8,3))
  for i in range(oriented_bbx.shape[0]):
    corners[i] = params2bbox(oriented_bbx[i])
  
  # rescale to voxel space
  corners = np.reshape(corners, (-1, 3)) 
  corners = np.clip(corners, xymin, xymax)
  corners[:,2] = np.clip(corners[:,2], zmin, zmax)
  corners[:,0] = corners[:,0]-xymin
  corners[:,1] = corners[:,1]-xymin
  corners[:,2] = corners[:,2]-zmin
  corners = corners/vs
  vxy = int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  
  if select:
    vox = (vox>0.05).astype(np.float32)
    max_xy = vox.shape[0]-1
    max_z = vox.shape[2]-1
    select_ids = []
    th = select_thres
    for j in range(corners.shape[0]):
      x1 = int(np.maximum(0, corners[j,0]-th))
      y1 = int(np.maximum(0, corners[j,1]-th))
      z1 = int(np.maximum(0, corners[j,2]-th))
      x2 = int(np.minimum(max_xy, corners[j,0]+th))
      y2 = int(np.minimum(max_xy, corners[j,1]+th))
      z2 = int(np.minimum(max_z, corners[j,2]+th))
      crop_vox = vox[x1:x2, y1:y2, z1:z2]
      if np.sum(crop_vox)>2:
        select_ids.append(j)
    corners = corners[select_ids]
  return corners

def gaussian_3d(x_mean, y_mean, z_mean, vxy, vz, dev=4.0):
  x, y, z = np.meshgrid(np.arange(vxy), np.arange(vxy), np.arange(vz))
  #z=(1.0/(2.0*np.pi*dev*dev))*np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
  m=np.exp(-((x-x_mean)**2 + (y-y_mean)**2+(z-z_mean)**2)/(2.0*dev**2))
  return m
  
def point_to_volume_gaussion(points, dev=4.0, vs=0.025, xymin=-3.2, xymax=3.2, zmin=-0.1, zmax=2.32):
  vxy = int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  vol=np.zeros((vxy,vxy,vz), dtype=np.float32)
  # print 'localcations', locations
  locations = points.astype(int)
  for i in range(points.shape[0]):
    if locations[i,0]==64 and locations[i,1]==64:
      continue
    vol+=gaussian_3d(locations[i, 1], locations[i, 0], locations[i, 2], vxy, vz, dev=dev)
  # vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
  return vol

def center_to_volume_gaussion(bbx, dev=4.0, vs=0.025, xymin=-3.2, xymax=3.2, zmin=-0.1, zmax=2.32):
  points = bbx[:,0:3]
  angle = bbx[:, 6]
  vxy = int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  vol=np.zeros((vxy,vxy,vz), dtype=np.float32)
  v_angle = np.zeros((vxy,vxy,vz), dtype=np.float32)
  th = dev*3
  # print 'localcations', locations
  locations = points.astype(int)
  # locations = np.clip(locations, xymin/vs, xymax/vs)
  # locations[:,2] = np.clip(locations[:,2], zmin/vs, zmax/vs)
  
  for i in range(points.shape[0]):
    if locations[i,0]==64:
      print(locations[i])
    vol+=gaussian_3d(locations[i, 1], locations[i, 0], locations[i, 2], vxy, vz, dev=dev)
    #v_angle+=cube_3d(locations[i, 1], locations[i, 0], locations[i, 2], vxy, vz, dev=dev)
    x1 = int(np.maximum(0, locations[i,0]-th))
    y1 = int(np.maximum(0, locations[i,1]-th))
    z1 = int(np.maximum(0, locations[i,2]-th))
    x2 = int(np.minimum(vxy-1, locations[i,0]+th))
    y2 = int(np.minimum(vxy-1, locations[i,1]+th))
    z2 = int(np.minimum(vz-1, locations[i,2]+th))
    v_angle[x1:x2, y1:y2, z1:z2] = angle[i]
    
  # vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
  return vol, v_angle

def point_cloud_to_value_scene(pred, vs=0.025, xymin=-3.2, xymax=3.2, zmin=-0.1, zmax=2.32):
  pred = np.clip(pred, xymin, xymax)
  pred[:,2] = np.clip(pred[:,2], zmin, zmax)
  pred[:,0] = pred[:,0]-xymin
  pred[:,1] = pred[:,1]-xymin
  pred[:,2] = pred[:,2]-zmin
  pred = pred/vs
  vxy = int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  low=pred.astype(int)
  high=low+1
  low=np.clip(low, 0, vxy-1)
  high=np.clip(high, 0, vxy-1)
  low[:,2] = np.clip(low[:,2], 0, vz-1)
  high[:,2] = np.clip(high[:,2], 0, vz-1)
  f=pred-low
  vox=np.zeros((vxy,vxy,vz), dtype=np.float32)
  for i in range(pred.shape[0]):
    vox[low[i,0], low[i,1], low[i,2]]+=(1-f[i,0])*(1-f[i,1])*(1-f[i,2])
    vox[low[i,0], low[i,1], high[i,2]]+=(1-f[i,0])*(1-f[i,1])*f[i,2]
    vox[low[i,0], high[i,1], low[i,2]]+=(1-f[i,0])*f[i,1]*(1-f[i,2])
    vox[low[i,0], high[i,1], high[i,2]]+=(1-f[i,0])*f[i,1]*f[i,2]
    vox[high[i,0], low[i,1], low[i,2]]+=f[i,0]*(1-f[i,1])*(1-f[i,2])
    vox[high[i,0], high[i,1], low[i,2]]+=f[i,0]*f[i,1]*(1-f[i,2])
    vox[high[i,0], low[i,1], high[i,2]]+=f[i,0]*(1-f[i,1])*f[i,2]
    vox[high[i,0], high[i,1], high[i,2]]+=f[i,0]*f[i,1]*f[i,2]
  return vox

# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_tsdf(filename):
  f = open(filename, 'rb')
  fileContent = f.read()

  A = struct.unpack("<3f", fileContent[:12])
  A = np.array(A, np.int32)
  # print(A)
  s1 = A[2]
  s2 = A[0]
  s3 = A[1]
  size = int(A[0]*A[1]*A[2])
  s = "<%df"%(int(size))
  TDF = struct.unpack(s, fileContent[12:12+size*4])

  out = np.zeros((s1, s2, s3))
  for i in range(s1):
    t = np.transpose(np.reshape(TDF[i*s2*s3: (i+1)*s2*s3], (s3, s2)))
    out[i] = t

  out = np.transpose(out, (1,2,0))

  out = out[20:-20, 20:-20, 20:-20]
  return out 

def read_ply(filename):
  """ read XYZ point cloud from filename PLY file """
  plydata = PlyData.read(filename)
  pc = plydata['vertex'].data
  pc_array = np.array([[x, y, z] for x, y, z in pc])
  return pc_array

def read_ply_color(filename):
  """ read XYZRGB point cloud from PLY file
  """
  print('ply reading ...', filename)
  plydata = PlyData.read(filename)
  pc = plydata['vertex'].data
  pc_array = np.arrary([x, y, z, r, g, b] for x, y, z, r, g, b in pc)
  return pc_array

def write_ply(points, filename, text=True):
  """ input: Nx3, write points to filename as PLY format. """
  points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
  vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
  el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
  PlyData([el], text=text).write(filename)
  
def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    # print(labels.max(), labels.min())
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        labels = np.clip(labels, 0, num_classes-1)
        assert(num_classes>np.max(labels))
    
    vertex = []
    #colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]    
    colors = [colormap(i/float(num_classes)) for i in range(num_classes)]    
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        vertex.append( (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)
    
def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()
