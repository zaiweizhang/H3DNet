#!/usr/bin/env python
# coding=utf-8
import os
import sys
import argparse
import numpy as np
from map_helper import APCalculator, parse_predictions, parse_groundtruths

''' --------------------------------  Configuration  --------------------------------
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--data_dir', default=None, help='Dir that stores predicted results [default: None]') # 'votenet_data_clsSem_rmEmpty' 
parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
FLAGS = parser.parse_args()
if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)

# Global variables
BATCH_SIZE = FLAGS.batch_size # make sure 312 % batch_size == 0
DUMP_DIR = FLAGS.dump_dir
DATA_DIR = FLAGS.data_dir
ROOT_DIR = os.path.dirname(DATA_DIR)
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)

# Config dataset
if FLAGS.dataset == 'sunrgbd':
    type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
elif FLAGS.dataset == 'scannet':
    type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
        'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
        'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17} 
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset)); exit(-1)

class DatasetConfig(object):
    def __init__(self):
        self.type2class = type2class
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.num_class = len(self.type2class)
DATASET_CONFIG = DatasetConfig()

# AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms, 'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh, 'dataset_config':DATASET_CONFIG}

ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
    for iou_thresh in AP_IOU_THRESHOLDS]


''' --------------------------------  Load data  --------------------------------
File Format of Files in DATA_DIR:
    pred_[sceneID].npy         (num_proposal, 9): center, size, angle, semantic, objectness_scores
    gt_[sceneID].npy           (num_max_gt, 9): center, size, angle, semantic, gt_bbox_mask(1 valid, 0 invalid)
    predNeMask_[sceneID].npy   (num_proposal,): nonempty_box_mask. 0 means almost no points in the bounding box
    predSemProb_[sceneID].npy  (num_proposal, num_class): sem_cls_probs. np.sum(sem_cls_probs, 1) == 1
'''

scene_id_list = []
for filename in os.listdir(DATA_DIR):
    scene = filename.split('_')[1] + '_' + filename.split('_')[2]
    scene = scene[:-4]
    scene_id_list.append(scene)

scene_id_list = list(set(scene_id_list))
num_total_sample = len(scene_id_list)

assert(num_total_sample % BATCH_SIZE == 0)
steps = int(num_total_sample / BATCH_SIZE)
for i in range(steps):
    pred_bbox_list = []
    objectness_scores_list = []
    sem_cls_probs_list = []
    nonempty_box_mask_list = []
    gt_bbox_list = []
    gt_bbox_mask_list = []
    for scene in scene_id_list[i*BATCH_SIZE : (i+1)*BATCH_SIZE]:
        pred = np.load(os.path.join(DATA_DIR, 'pred_' + scene + '.npy'))
        gt = np.load(os.path.join(DATA_DIR, 'gt_' + scene + '.npy'))
        pred_bbox_list.append(pred[:, :8])
        objectness_scores_list.append(pred[:, 8])
        gt_bbox_list.append(gt[:, :8])
        gt_bbox_mask_list.append(gt[:, 8])

        if FLAGS.per_class_proposal:
            sem_cls_probs = np.load(os.path.join(DATA_DIR, 'predSemProb_' + scene + '.npy'))
            sem_cls_probs_list.append(sem_cls_probs)
        if not FLAGS.faster_eval:
            nonempty_box_mask = np.load(os.path.join(DATA_DIR, 'predNeMask_' + scene + '.npy'))
            nonempty_box_mask_list.append(nonempty_box_mask)

    end_points = dict()
    end_points['pred_bbox'] = np.array(pred_bbox_list)
    end_points['objectness_scores'] = np.array(objectness_scores_list)
    end_points['gt_bbox'] = np.array(gt_bbox_list)
    end_points['gt_bbox_mask'] = np.array(gt_bbox_mask_list)
    if FLAGS.per_class_proposal:
        end_points['sem_cls_probs'] = np.array(sem_cls_probs_list)
    if not FLAGS.faster_eval:
        end_points['nonempty_box_mask'] = np.array(nonempty_box_mask_list)

    batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
    batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
    for ap_calculator in ap_calculator_list:
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    
# Evaluate average precision
for i, ap_calculator in enumerate(ap_calculator_list):
    print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f'%(key, metrics_dict[key]))


