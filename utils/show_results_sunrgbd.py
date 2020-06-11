# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""
import os
import sys
import datetime
import numpy as np
import pdb
import matplotlib.pyplot as pyplot
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff
import json
import pickle
import random
import scipy.io as sio
from pc_util import params2bbox, write_ply_rgb

THRESH = 0
THRESH2 = -0.1
DATA_DIR = os.path.join('/home/bo/data/sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_val') # path of sunrgbd dataset 
VAL_SCAN_NAMES = sorted(list(set([os.path.basename(x)[0:6] for x in os.listdir(DATA_DIR)])))
PRED_PATH= '/home/bo/projects/cvpr2020/detection/new/new/sunrgbd/code_sunrgbd/indoor_scene_understanding/dump_sunrgbd/result' # path of predictions

DONOTCARE_CLASS_IDS = np.array([])
MAX_NUM_POINT = 40000
mode = sys.argv[1]

color_mapping = {1:[30,144,255], 2:[255,69,0], 3:[255,215,0], 4:[50,205,50], 5:[255,127,80],
        6:[255,20,147], 7:[100,149,237], 8:[255,127,80],9:[210,105,30], 10:[221,160,221],11:[95,158,  160]}

def create_lineset_old(bbox, colors=[1, 0, 0]):
    ''' create bounding box
    '''
    xmin = bbox[0] - bbox[3] / 2
    xmax = bbox[0] + bbox[3] / 2
    ymin = bbox[1] - bbox[4] / 2
    ymax = bbox[1] + bbox[4] / 2
    zmin = bbox[2] - bbox[5] / 2
    zmax = bbox[2] + bbox[5] / 2
    points = [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
              [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    lines = [[0, 1], [0, 2], [2, 3], [1, 3], [0, 4], [1, 5], [3, 7], [2, 6],
             [4, 5], [5, 7], [6, 7], [4, 6]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(colors, [12, 1]))
    return line_set


def create_lineset(bbox, colors=[1, 0, 0]):
    ''' create bounding box
    '''
    points = params2bbox(bbox)
    lines = [[0, 1], [0, 2], [2, 3], [1, 3], [0, 4], [1, 5], [3, 7], [2, 6],
             [4, 5], [5, 7], [6, 7], [4, 6]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(colors, [12, 1]))
    return line_set


def load_view_point(pcd, filename, window_name):
    if mode=='pred':
        left = 50
        top=50
    elif mode=='gt':
        left = 1000
        top=730
    else:
        print("model must be gt or pred")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=880, height=680, left=left, top=top)
    for part in pcd:
        vis.add_geometry(part)
    ctr = vis.get_view_control()
    current_param = ctr.convert_to_pinhole_camera_parameters()
    trajectory = o3d.io.read_pinhole_camera_trajectory(filename)
    f = 983.80485869912241
    cx = current_param.intrinsic.width / 2 - 0.5
    cy = current_param.intrinsic.height / 2 - 0.5
    trajectory.parameters[0].intrinsic.set_intrinsics(current_param.intrinsic.width, current_param.intrinsic.height, f, f, cx, cy)

    ctr.convert_from_pinhole_camera_parameters(trajectory.parameters[0])
    vis.run()
    vis.destroy_window()

def select_bbox(bboxes):
    choose_ids = []
    for i in range(bboxes.shape[0]):
        if bboxes[i,-1] in OBJ_CLASS_IDS:
            choose_ids.append(i)
    bboxes = bboxes[choose_ids]
    return bboxes

def export_one_scan(scan_name):
    pt = np.load(os.path.join(DATA_DIR, scan_name+'_pc.npz'))['pc']
    np.savetxt(mode+'tmp.xyz', pt)
    os.system("mv {}tmp.xyz {}tmp.xyzrgb".format(mode, mode))
    point_cloud = o3d.io.read_point_cloud(mode+'tmp.xyzrgb')

    pred_proposals = np.load(os.path.join(PRED_PATH, 'center'+scan_name+'_nms.npy'))
    gt_bbox = sio.loadmat(os.path.join(PRED_PATH, 'center'+scan_name+'_gt.mat'))['gt']
    bb =[]
    if mode=='gt':
        boundingboxes = gt_bbox
    elif mode =='pred':
        boundingboxes = pred_proposals
    else:
        print("model must be gt or pred")
        return
    for i in range(boundingboxes.shape[0]):
        c = np.array(color_mapping[int(boundingboxes[i,-1])])/255.0
        for _ in range(2):
            bb.append(create_lineset(boundingboxes[i]+0.005*(np.random.rand()-0.5)*2, colors=c))
    load_view_point([point_cloud] + bb, './viewpoint.json', window_name=scan_name+'_'+mode)


def batch_export():
    for i, scan_name in enumerate(VAL_SCAN_NAMES):
        if not scan_name.endswith('10'):
            continue
        print('-'*20+'begin')
        print(scan_name)
        export_one_scan(scan_name)
        print('-'*20+'done')

if __name__=='__main__':
    batch_export()
