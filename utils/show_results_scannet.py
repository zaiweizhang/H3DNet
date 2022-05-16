import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
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


THRESH = 0
THRESH2 = -0.1
VAL_SCAN_NAMES = [line.rstrip() for line in open('scannet/meta_data/scannetv2_val.txt')] 
SCANNET_DIR = '/home/bo/data/scannet/scans/' # path of scannet dataset 
LABEL_MAP_FILE = 'scannet/meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
MAX_NUM_POINT = 40000
GT_PATH = '/home/bo/data/scannet/scannet_train_detection_data' # path of data dumped with scripts in scannet folder 
PRED_PATH = '/home/bo/data/scannet/dump/supp/result' # path of predictions 
mode = sys.argv[1] # gt or pred
color_mapping = {3: [255,140,0], 4:[30,144,255], 5:[50,205,50], 6:[255,215,0], 7:[255,69,0], 8:[138,43,226],9:[0,255,255],10:[210,105,30],11:[255,0,255], 12:[255,255,0], 14:[255,20,147], 16:[165,42,42], 24:[100,149,237], 28:[0,128,0], 33:[255,127,80],34:[221,160,221], 36:[95,158,160], 39:[119,136,153]}

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
    if mode=='pred':
        left = 50
        top=50
    elif mode=='gt':
        left = 1000
        top=50
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
    pt = np.load(os.path.join(GT_PATH, scan_name+'_vert.npy'))
    x = np.array([1 , 1, 1,255, 255, 255]) # Divide each column. x,y,z,r,g,b
    np.savetxt('tmp.xyz', pt[:,:6]/x)
    #np.savetxt('tmp.xyz', pt)
    os.system("mv tmp.xyz tmp.xyzrgb")
    pcd  = o3d.io.read_point_cloud('tmp.xyzrgb')

    #gt_bbox = np.load(os.path.join(GT_PATH, scan_name+'_all_angle_40cls.npy'))
    gt_bbox = np.load(os.path.join(GT_PATH, scan_name+'_all_noangle_40cls.npy')) # The file name in scannet_train_detection_data is scene0011_00_all_noangle_40cls.npy
    gt_bbox = select_bbox(np.unique(gt_bbox,axis=0))
    semantic_labels = gt_bbox[:,-1]
    pred_proposals = np.load(os.path.join(PRED_PATH, 'opt'+scan_name+'_nms.npy'))

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    semantic_labels = semantic_labels[mask]

    bb =[]
    if mode=='gt':
        boundingboxes = gt_bbox
    elif mode =='pred':
        boundingboxes = pred_proposals
    else:
          print("model must be gt or pred")
          return

    for i in range(boundingboxes.shape[0]):
        if mode =='gt':
            c = np.array(color_mapping[int(boundingboxes[i,-1])])/255.0
        else:
            c = np.array(color_mapping[int(OBJ_CLASS_IDS[int(boundingboxes[i,-1])-1])])/255.0
        for _ in range(2):
            bb.append(create_lineset(boundingboxes[i]+0.005*(np.random.rand()-0.5)*2, colors=c))
    load_view_point([pcd] + bb, './viewpoint.json', window_name=scan_name+'_'+mode)


def batch_export():
    for i, scan_name in enumerate(sorted(VAL_SCAN_NAMES)):
        #if not scan_name.endswith('_00'):
        #    continue
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        export_one_scan(scan_name)
        print('-'*20+'done')

if __name__=='__main__':
    batch_export()
