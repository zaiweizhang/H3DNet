# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Utility functions for processing point clouds.

Author: Charles R. Qi and Or Litany
"""

import os
import sys
import torch
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from sklearn import metrics
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Point cloud IO
import numpy as np
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)


# Mesh IO
import trimesh

import matplotlib.pyplot as pyplot

def pc2obj(pc, filepath='test.obj'):
    pc = pc.T
    nverts = pc.shape[1]
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in range(nverts):
            f.write("v %.4f %.4f %.4f\n" % (pc[0,v],pc[1,v],pc[2,v]))

def compute_iou(v1, v2, thres=0.3):
    bv1 = (v1>thres).float()
    bv2 = (v2>thres).float()
    iou=0.0
    for i in range(v1.shape[0]):
        iou += torch.sum(bv1[i]*bv2[i]).float()/torch.sum(bv1[i]+bv2[i]-bv1[i]*bv2[i]).float()
    iou = iou/v1.shape[0]
    return iou

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
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
def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b,:,:], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)

def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize,vsize,vsize,num_sample,3))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i,j,k) not in loc2pc:
                    vol[i,j,k,:,:] = np.zeros((num_sample,3))
                else:
                    pc = loc2pc[(i,j,k)] # a list of (3,) arrays
                    pc = np.vstack(pc) # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0]>num_sample:
                        pc = random_sampling(pc, num_sample, False)
                    elif pc.shape[0]<num_sample:
                        pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i,j,k])+0.5)*voxel - radius
                    pc = (pc - pc_center) / voxel # shift and scale
                    vol[i,j,k,:,:] = pc 
    return vol

def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b,:,:], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)


def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2*radius/float(imgsize)
    locations = (points[:,0:2] + radius)/pixel # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i,j) not in loc2pc:
                img[i,j,:,:] = np.zeros((num_sample,3))
            else:
                pc = loc2pc[(i,j)]
                pc = np.vstack(pc)
                if pc.shape[0]>num_sample:
                    pc = random_sampling(pc, num_sample, False)
                elif pc.shape[0]<num_sample:
                    pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                pc_center = (np.array([i,j])+0.5)*pixel - radius
                pc[:,0:2] = (pc[:,0:2] - pc_center)/pixel
                img[i,j,:,:] = pc
    return img
# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
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

def hashlist(l):
    temps = ''
    for i in l:
        temps += str(i)
    return temps

def construct_dict(labels, predict=None):
    if predict == None:
        d = {}
        counter = 0
    else:
        d = predict
        counter = len(predict)
    for i in range(len(labels)):
        label = hashlist(labels[i,:])
        if label not in d:
            d[label] = counter
            counter += 1
    return d
    
def get_correct(pred, gt):
    total = 0
    for i in range(len(pred)):
        if hashlist(pred[i,:]) == hashlist(gt[i,:]):
            total += 1
    return total

def write_ply_color_multi(points, labels, filename, pre_dict=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    #labels = labels.astype(int)
    N = points.shape[0]
    
    cdict = construct_dict(labels, predict=pre_dict)
            
    num_classes = len(cdict.keys())
    vertex = []
    #colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]    
    colors = [colormap(i/float(num_classes)) for i in range(num_classes)]    
    for i in range(N):
        c = colors[cdict[hashlist(labels[i,:])]]
        c = [int(x*255) for x in c]
        vertex.append( (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)
    return cdict
    
def write_ply_label(points, labels, filename, num_classes, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    #cdict = construct_dict(labels, predict=pre_dict)
            
    #num_classes = len(cdict.keys())
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

# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #savefig(output_filename)

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)

# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def rotate_point_cloud(points, rotation_matrix=None):
    """ Input: (n,3), Output: (n,3) """
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)     
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points-ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix

def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape)+[3,3]))
    c = np.cos(t)
    s = np.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


# ----------------------------------------
# BBox
# ----------------------------------------
def bbox_corner_dist_measure(crnr1, crnr2):
    """ compute distance between box corners to replace iou
    Args:
        crnr1, crnr2: Nx3 points of box corners in camera axis (y points down)
        output is a scalar between 0 and 1        
    """
    
    dist = sys.maxsize
    for y in range(4):
        rows = ([(x+y)%4 for x in range(4)] + [4+(x+y)%4 for x in range(4)])
        d_ = np.linalg.norm(crnr2[rows, :] - crnr1, axis=1).sum() / 8.0            
        if d_ < dist:
            dist = d_

    u = sum([np.linalg.norm(x[0,:] - x[6,:]) for x in [crnr1, crnr2]])/2.0

    measure = max(1.0 - dist/u, 0)
    print(measure)
    
    
    return measure


def point_cloud_to_bbox(points):
    """ Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths        
    """
    which_dim = len(points.shape) - 2 # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5*(mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)

def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """
    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    #trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    mesh_list.export(out_filename, file_type='ply')
    return

def write_oriented_bbox_camera_coord(scene_bbox, out_filename):
    """Export oriented (around Y axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Y axis.
            Z forward, X rightward, Y downward. heading angle of positive X is 0,
            heading angle of negative Z is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[1,1] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0,:] = np.array([cosval, 0, sinval])
        rotmat[2,:] = np.array([-sinval, 0, cosval])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def write_lines_as_cylinders(pcl, filename, rad=0.005, res=64):
    """Create lines represented as cylinders connecting pairs of 3D points
    Args:
        pcl: (N x 2 x 3 numpy array): N pairs of xyz pos             
        filename: (string) filename for the output mesh (ply) file
        rad: radius for the cylinder
        res: number of sections used to create the cylinder
    """
    scene = trimesh.scene.Scene()
    for src,tgt in pcl:
        # compute line
        vec = tgt - src
        M = trimesh.geometry.align_vectors([0,0,1],vec, False)
        vec = tgt - src # compute again since align_vectors modifies vec in-place!
        M[:3,3] = 0.5*src + 0.5*tgt
        height = np.sqrt(np.dot(vec, vec))
        scene.add_geometry(trimesh.creation.cylinder(radius=rad, height=height, sections=res, transform=M))
    mesh_list = trimesh.util.concatenate(scene.dump())
    trimesh.io.export.export_mesh(mesh_list, f'{filename}.ply', file_type='ply')

    

def compute_iou_pc(pc1, pc2):
    v1 = point_cloud_to_volume_batch(pc1, 48)
    v2 = point_cloud_to_volume_batch(pc2, 48)
    iou = 0.0
    for i in range(pc1.shape[0]):
        iou += torch.sum(v1[i]*v2[i]).float()/torch.sum(v1[i]+v2[i]-v1[i]*v2[i]).float()
    iou = iou/pc1.shape[0]
    return iou

def volume_to_point_cloud_color(vol, thres=0.1, mini=0):
  vx = vol.shape[0]
  vy = vol.shape[1]
  vz = vol.shape[2]
  points = []
  labels = []
  for a in range(vx):
    for b in range(vy):
      for c in range(vz):
        if vol[a, b, c]>thres:
          points.append(np.array([a, b, c]))
          labels.append(vol[a,b,c])
  labels = (np.array(labels, np.float32)-mini)
  
  if len(points) == 0:
    return np.zeros((0, 3))
  points = np.vstack(points)
  
  return points, labels

def multichannel_volume_to_point_cloud(vol):
  num_classes = vol.shape[0]
  pts=np.zeros((0,3))
  pt_labels=np.zeros(0)
  
  for i in range(1, num_classes):
    pt_i = volume_to_point_cloud(vol[i])
    pts = np.concatenate((pts, pt_i), axis=0)
    pt_labels = np.concatenate((pt_labels, np.ones(pt_i.shape[0])*i))
  pts = np.array(pts, np.float32)
  pt_labels=np.array(pt_labels, np.int32)
  return pts, pt_labels   

def volume_pt_to_pt(vpt, vs=0.025, xmin=-3.2, xmax=3.2, ymin=-3.2, ymax=3.2, zmin=-0.1, zmax=2.32):
  if vpt.shape[0]==0:
      return np.zeros((0,3))
  else:
    pt = vpt*vs
    pt[:,0] = pt[:,0]+xmin
    pt[:,1] = pt[:,1]+ymin
    pt[:,2] = pt[:,2]+zmin
    return pt

  
def process_bbx(bbx, vs=0.06, xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68):
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

def get_corner(bbx,  vs=0.06,xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68):
  corners = np.zeros((bbx.shape[0], 8 ,3))
  vxy = int(xymax-xymin)/vs
  vz = int(zmax-zmin)/vs
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
  crop_ids = []
  for i in range(corners.shape[0]):
    if corners[i,0]>-1 and corners[i,0]<vxy and corners[i,1]>-1 and corners[i,1]<vxy and corners[i,2]>-1 and corners[i,2]<vz:
      crop_ids.append(i)
  corners = corners[crop_ids]      
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

def get_oriented_corners(oriented_bbx, vs=0.06, xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68):
  # corners in origin scale
  corners = np.zeros((oriented_bbx.shape[0], 8,3))
  for i in range(oriented_bbx.shape[0]):
    corners[i] = params2bbox(oriented_bbx[i])
  # rescale to voxel space
  corners = np.reshape(corners, (-1, 3)) 
  crop_ids= []
  for i in range(corners.shape[0]):
    if corners[i,0]>xymin and corners[i,0]<xymax and corners[i,1]>xymin and corners[i,1]<xymax and corners[i,2]>zmin and corners[i,2]<zmax:
      crop_ids.append(i)
  corners=corners[crop_ids]
  corners[:,0] = corners[:,0]-xymin
  corners[:,1] = corners[:,1]-xymin
  corners[:,2] = corners[:,2]-zmin
  corners = corners/vs
  return corners

def gaussian_3d(x_mean, y_mean, z_mean, vxy, vz, dev=2.0):
  x, y, z = np.meshgrid(np.arange(vxy), np.arange(vxy), np.arange(vz))
  #z=(1.0/(2.0*np.pi*dev*dev))*np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
  m=np.exp(-((x-x_mean)**2 + (y-y_mean)**2+(z-z_mean)**2)/(2.0*dev**2))
  return m

def point_to_volume_gaussion(points, dev=2.0, vs=0.06, xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68, ksize=11):
  vxy = int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  vol=np.zeros((vxy,vxy,vz), dtype=np.float32)
  locations = points.astype(int)
  locations = np.clip(locations, 0, vxy-1)
  locations[:,2] = np.clip(locations[:,2], 0, vz-1)

  vxyc = int(vxy/2)
  vzc = int(vz/2)
  k2 = int((ksize - 1)/2)
  gauss = gaussian_3d(k2, k2, k2, ksize, ksize, dev=dev)

  for i in range(points.shape[0]):
    if locations[i,0]==64 and locations[i,1]==64:
      continue
    xmin = np.maximum(0, locations[i, 0] - k2)
    xmax = np.minimum(vxy - 1, locations[i, 0] + k2)
    ymin = np.maximum(0, locations[i, 1] - k2)
    ymax = np.minimum(vxy - 1, locations[i, 1] + k2)
    zmin = np.maximum(0, locations[i, 2] - k2)
    zmax = np.minimum(vz - 1, locations[i, 2] + k2)
    vol[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] += gauss[k2-locations[i, 0]+xmin : k2-locations[i, 0]+xmax+1,\
                                                  k2-locations[i, 1]+ymin : k2-locations[i, 1]+ymax+1,\
                                                  k2-locations[i, 2]+zmin : k2-locations[i, 2]+zmax+1] 
  return vol


def point_to_volume_gaussion_dep(points, dev=2.0, vs=0.06, xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68):
  vxy = int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  vol=np.zeros((vxy,vxy,vz), dtype=np.float32)
  locations = points.astype(int)
  locations = np.clip(locations, 0, vxy-1)
  locations[:,2] = np.clip(locations[:,2], 0, vz-1)
  for i in range(points.shape[0]):
    if locations[i,0]==64 and locations[i,1]==64:
      continue
    vol+=gaussian_3d(locations[i, 1], locations[i, 0], locations[i, 2], vxy, vz, dev=dev)
  # vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
  return vol

def center_to_volume_gaussion(bbx, dev=2.0, vs=0.06, xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68):
  points = bbx[:,0:3]
  vxy = int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  vol=np.zeros((vxy,vxy,vz), dtype=np.float32)
  locations = points.astype(int)
  locations = np.clip(locations, 0, vxy-1)
  locations[:,2] = np.clip(locations[:,2], 0, vz-1)
  for i in range(points.shape[0]):
    if locations[i,0]==64 and locations[i,1]==64:
      continue
    vol+=gaussian_3d(locations[i, 1], locations[i, 0], locations[i, 2], vxy, vz, dev=dev)
  return vol

def point_cloud_to_voxel_scene(pt, vs=0.06, xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68):
  pt, crop_ids = crop_point_cloud(pt)
  pt[:,0] = pt[:,0]-xymin
  pt[:,1] = pt[:,1]-xymin
  pt[:,2] = pt[:,2]-zmin
  pt = pt/vs
  locations = pt.astype(np.int32)
  vxy = int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  locations = np.clip(locations, 0, vxy-1)
  locations[:,2] = np.clip(locations[:,2], 0, vz-1)
  vox = np.zeros((vxy, vxy, vz))
  vox[locations[:,0], locations[:,1], locations[:,2]]=1.0
  return vox

def crop_point_cloud(pt,xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68):
  crop_ids=[]
  for i in range(pt.shape[0]):
    if pt[i,0]>xymin and pt[i,0]<xymax and pt[i,1]>xymin and pt[i,1]<xymax and pt[i,2]>zmin and pt[i,2]<zmax:
      crop_ids.append(i)
  pt = pt[crop_ids]
  return pt, crop_ids

def point_cloud_to_sem_vox(pt, sem_label, vs=0.06,xymin=-3.84, xymax=3.84, zmin=-0.2, zmax=2.68):
  pt[:,0]=pt[:,0]-xymin
  pt[:,1]=pt[:,1]-xymin
  pt[:,2]=pt[:,2]-zmin
  pt=pt/vs
  vxy=int((xymax-xymin)/vs)
  vz = int((zmax-zmin)/vs)
  pt = np.clip(pt, 0,vxy-1)
  pt[:,2] = np.clip(pt[:,2], 0,vz-1)
  vol=np.zeros((vxy,vxy,vz), np.float32)
  pt = pt.astype(np.int32)
  for i in range(pt.shape[0]):
    if sem_label[i] not in choose_classes:
      continue
    vol[pt[i,0], pt[i,1], pt[i,2]]=np.argwhere(choose_classes==sem_label[i])[0,0]+1
  return vol
     
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
  
def point_add_sem_label(pt, sem, k=10):
    sem_pt = sem[:, 0:3]
    sem_label = sem[:,3]
    pt_label = np.zeros(pt.shape[0])
    if pt.shape[0]==0:
        return pt_label
    else:
        nbrs = NearestNeighbors(n_neighbors=k,algorithm='ball_tree').fit(sem_pt)
        distances, indices = nbrs.kneighbors(pt)
        for i in range(pt.shape[0]):
            labels = sem_label[indices[i]]
            l, count = stats.mode(labels, axis=None)
            pt_label[i] = l
        return pt_label


    
# ----------------------------------------
# Testing
# ----------------------------------------
if __name__ == '__main__':
    print('running some tests')
    
    ############
    ## Test "write_lines_as_cylinders"
    ############
    pcl = np.random.rand(32, 2, 3)
    write_lines_as_cylinders(pcl, 'point_connectors')
    input()
    
   
    scene_bbox = np.zeros((1,7))
    scene_bbox[0,3:6] = np.array([1,2,3]) # dx,dy,dz
    scene_bbox[0,6] = np.pi/4 # 45 degrees 
    write_oriented_bbox(scene_bbox, 'single_obb_45degree.ply')
    ############
    ## Test point_cloud_to_bbox 
    ############
    pcl = np.random.rand(32, 16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)
    assert pcl_bbox.shape == (32, 6)
    
    pcl = np.random.rand(16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)    
    assert pcl_bbox.shape == (6,)
    
    ############
    ## Test corner distance
    ############
    crnr1 = np.array([[2.59038660e+00, 8.96107932e-01, 4.73305349e+00],
 [4.12281644e-01, 8.96107932e-01, 4.48046631e+00],
 [2.97129656e-01, 8.96107932e-01, 5.47344275e+00],
 [2.47523462e+00, 8.96107932e-01, 5.72602993e+00],
 [2.59038660e+00, 4.41155793e-03, 4.73305349e+00],
 [4.12281644e-01, 4.41155793e-03, 4.48046631e+00],
 [2.97129656e-01, 4.41155793e-03, 5.47344275e+00],
 [2.47523462e+00, 4.41155793e-03, 5.72602993e+00]])
    crnr2 = crnr1

    print(bbox_corner_dist_measure(crnr1, crnr2))
    
    
    
    print('tests PASSED')
    
    
