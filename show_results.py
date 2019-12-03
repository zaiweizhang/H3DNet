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
# sys.path.append('/home/bo/projects/cvpr2020/detection/votenet')
sys.path.append('/home/bo/projects/cvpr2020/scannet_annotate')
import datetime
import numpy as np
from load_scannet_data import export
import pdb
import matplotlib.pyplot as pyplot
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff
import json
import pickle
import random
import scipy.io as sio


THRESH = 0
THRESH2 = -0.1
TRAIN_SCAN_NAMES = [line.rstrip() for line in open('/home/bo/projects/cvpr2020/detection/votenet/scannet/meta_data/scannet_train.txt')]
VAL_SCAN_NAMES = [line.rstrip() for line in open('/home/bo/projects/cvpr2020/detection/votenet/scannet/meta_data/scannetv2_val.txt')]
SCANNET_DIR = '/home/bo/data/scannet/scans/'
LABEL_MAP_FILE = '/home/bo/projects/cvpr2020/detection/votenet/scannet/meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
MAX_NUM_POINT = 40000
PRED_PATH = '/home/bo/data/scannet/new2/scannet_train_detection_data'
# our_init_path = '/home/bo/projects/cvpr2020/detection/indoor_scene_understanding/aggregation_module/v10/our_proposals'
init_path = '/home/bo/projects/cvpr2020/detection/indoor_scene_understanding/3results/result'
mode = sys.argv[1]


def create_lineset(bbox, colors=[1, 0, 0]):
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

def load_view_point(pcd, filename, window_name):
    if mode=='gt':
        left = 50
    else:
        left = 1000
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=880, height=680, left=left, top=50,)
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
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.
    mesh, mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE)
    gt_bbox = np.load(os.path.join(PRED_PATH, scan_name+'_all_angle_40cls.npy'))
    gt_bbox = select_bbox(np.unique(gt_bbox,axis=0))
    init_proposals = np.load(os.path.join(init_path, 'plane'+scan_name+'_nms.npy'))
    
    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]
    #generate_obj(mesh, mesh_vertices, instance_labels, semantic_labels, np.unique(instance_labels), preds, scan_name)
    bb =[]
    if mode=='gt':
        boundingboxes = gt_bbox
    elif mode =='init':
        boundingboxes = init_proposals
    for i in range(boundingboxes.shape[0]):
        c = [random.random(), random.random(),random.random() ]
        for _ in range(1):
            bb.append(create_lineset(boundingboxes[i]+0.005*(np.random.rand()-0.5)*2, colors=c))

    load_view_point([mesh] + bb, './viewpoint.json', window_name=scan_name+'_'+mode)


def batch_export():
    val_names = ['scene0568_00', 'scene0050_01', 'scene0575_00', 'scene0426_03', 'scene0700_00', 
                      'scene0549_01', 'scene0164_00', 'scene0606_00', 'scene0616_00', 'scene0084_00']
    train_names = ['scene0000_00', 'scene0069_00', 'scene0007_00', 'scene0469_00', 'scene0673_03', 
                   'scene0035_00', 'scene0569_00']
    # for scan_name in sorted(TRAIN_SCAN_NAMES)[:10]:
    for scan_name in val_names:
        # if not scan_name.endswith('_00'):
        #     continue
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        export_one_scan(scan_name)
        print('-'*20+'done')

if __name__=='__main__':
    batch_export()
