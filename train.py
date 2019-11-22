# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from pc_util import compute_iou
from dump_helper import dump_results, dump_planes, dump_objcue

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_angle', action='store_true', help='Use angle for input in scannet.')
parser.add_argument('--use_objcue', action='store_true', help='Use support relation in input.')
parser.add_argument('--opt_proposal', action='store_true', help='Use support relation in input.')
parser.add_argument('--use_plane', action='store_true', help='Use support relation in input.')
parser.add_argument('--get_data', action='store_true', help='Use support relation in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
    TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
elif FLAGS.dataset == 'scannet_hd':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset_hd import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
                                            augment=False, use_angle=FLAGS.use_angle,
                                            use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
                                           augment=False, use_angle=FLAGS.use_angle,
                                           use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
elif FLAGS.model == 'planenet':
    Detector = MODEL.PlaneNet
    num_input_channel += 4
elif FLAGS.model == 'hdnet':
    Detector = MODEL.HDNet
    if FLAGS.opt_proposal:
        Obj_opt = MODEL.OBJNet
    num_input_channel += 4
else:
    Detector = MODEL.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

if FLAGS.opt_proposal:
   net_obj = Obj_opt(num_class=DATASET_CONFIG.num_class,
                  num_heading_bin=DATASET_CONFIG.num_heading_bin,
                  num_size_cluster=DATASET_CONFIG.num_size_cluster,
                  mean_size_arr=DATASET_CONFIG.mean_size_arr,
                  num_proposal=FLAGS.num_target,
                  input_feature_dim=num_input_channel,
                  vote_factor=FLAGS.vote_factor,
                  sampling=FLAGS.cluster_sampling)
   w_cons = MODEL.weightConstraint()
   net_obj._modules['weight'].apply(w_cons)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    '''
    # 1. filter out unnecessary keys
    model_dict = net.state_dict()
    pretrained_dict = checkpoint['model_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (('pnet' not in k))}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    '''
    net.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

if torch.cuda.device_count() > 1:
  log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)
net.to(device)
# Load the Adam optimizer

criterion = MODEL.get_loss
if FLAGS.opt_proposal:
    net_obj.to(device)

optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
if FLAGS.opt_proposal:
    optimizer_obj = optim.Adam(net_obj.parameters(), lr=0.001, weight_decay=FLAGS.weight_decay)
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    #net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG}

EPOCH_THRESH = 400

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    if FLAGS.get_data == True:
        net.eval() # set model to training mode
    else:
        adjust_learning_rate(optimizer, EPOCH_CNT)
        bnm_scheduler.step() # decay BN momentum
        net.train() # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        #for i in range(len(batch_data_label['num_instance'])):
        #    if batch_data_label['num_instance'][i] == 0:
        #        print (batch_data_label['scan_name'][i])
        end_points = {}
        for key in batch_data_label:
            #if 'name' not in key:
            batch_data_label[key] = batch_data_label[key].to(device)
            if 'aug' in key:
                end_points[key] = batch_data_label[key].to(device)
                
        # Forward pass
        optimizer.zero_grad()
        if FLAGS.dataset == 'sunrgbd':
            inputs = {'point_clouds': batch_data_label['point_clouds'], 'plane_label': batch_data_label['plane_label'], 'voxel_label': batch_data_label['voxel'], 'sunrgbd': True, 'epoch': EPOCH_CNT}
        else:
            inputs = {'point_clouds': batch_data_label['point_clouds'], 'plane_label': batch_data_label['plane_label'], 'voxel_label': batch_data_label['voxel'], 'sunrgbd': False,  'epoch': EPOCH_CNT}
        if FLAGS.opt_proposal:
            inputs['opt_proposal'] = True
        else:
            inputs['opt_proposal'] = False
        if FLAGS.opt_proposal:
            inputs['gt_bboxes'] = batch_data_label['gt_bbox']
            inputs['pert_bboxes'] = batch_data_label['pert_bbox']
            ### Also need to get the initial proposal
        if FLAGS.get_data == True:
            with torch.no_grad():
                end_points = net(inputs, end_points)
        else:
            end_points = net(inputs, end_points)

        ### Train the obj proposal
        if FLAGS.opt_proposal:
            end_points = net_obj(inputs, end_points, batch_data_label)
            
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            if "aug" in key:
                continue
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        if FLAGS.get_data == True:
            #optimize_proposal(inputs, end_points)
            dump_objcue(inputs, end_points, DUMP_DIR+'/objcue', DATASET_CONFIG, TRAIN_DATASET)
        if FLAGS.opt_proposal:
            loss, end_points = criterion(inputs, end_points, DATASET_CONFIG, net=net_obj)
        else:
            loss, end_points = criterion(inputs, end_points, DATASET_CONFIG)
        if FLAGS.opt_proposal == False and FLAGS.get_data == False:
            loss.backward()
            optimizer.step()
        if FLAGS.opt_proposal:
            loss.backward()
            optimizer_obj.step()
            for p in net_obj.parameters():
                p.data.clamp_(min=0.0,max=1.00001)
            
        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        ### For voxel
        #center_iou = compute_iou(end_points['vox_pred1'], end_points['vox_center'])
        #corner_iou = compute_iou(end_points['vox_pred2'], end_points['vox_corner'])
        #if i % 100 == 0 and i!=0: 
        #    save_vox_results(vox, corners, pred_corner, opt.outf, epoch, i, 'train_corner')
        #    save_vox_results(vox, center, pred_center, opt.outf, epoch, i, 'train_center')
            
        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
                (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0
            #log_string('cen iou: %f cor iou: %f' % (center_iou.cpu().numpy(), corner_iou.cpu().numpy()))

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    ap_calculator_center = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type)
    ap_calculator_corner = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type)
    ap_calculator_plane = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type)
    net.eval() # set model to eval mode (for bn and dp)
    total_correct_sem = 0
    total_sem = 0
    total_correct_sem_plane = 0
    total_sem_plane = 0
    total_correct_angle_plane = 0
    total_angle_plane = 0
    correct_cls = {}
    total_cls = {}
    correct_cls_plane = {}
    total_cls_plane = {}
    correct_cls_angle_plane = {}
    total_cls_angle_plane = {}
    
    for cls in DATASET_CONFIG.class2type:
        correct_cls[cls] = 0
        total_cls[cls] = 0
        correct_cls_plane[cls] = 0
        total_cls_plane[cls] = 0
    for cls in ['x', 'y', 'z']:
        correct_cls_angle_plane[cls] = 0
        total_cls_angle_plane[cls] = 0
        
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        #for i in range(len(batch_data_label['num_instance'])):
        #    if batch_data_label['num_instance'][i] == 0:
        #        print (batch_data_label['scan_name'][i])
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        end_points = {}
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
            if 'aug' in key:
                end_points[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds'], 'plane_label': batch_data_label['plane_label'], 'voxel_label': batch_data_label['voxel']}
        if FLAGS.dataset == 'sunrgbd':
            inputs = {'point_clouds': batch_data_label['point_clouds'], 'plane_label': batch_data_label['plane_label'], 'voxel_label': batch_data_label['voxel'], 'sunrgbd': True,  'epoch': EPOCH_CNT}
        else:
            inputs = {'point_clouds': batch_data_label['point_clouds'], 'plane_label': batch_data_label['plane_label'], 'voxel_label': batch_data_label['voxel'], 'sunrgbd': False,  'epoch': EPOCH_CNT}
        if FLAGS.opt_proposal:
            inputs['opt_proposal'] = True
        else:
            inputs['opt_proposal'] = False
        #inputs = {'point_clouds': batch_data_label['point_clouds'], 'plane_label': batch_data_label['plane_label']}
        with torch.no_grad():
            end_points = net(inputs, end_points)

        # Compute loss
        for key in batch_data_label:
            if "aug" in key:
                continue
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        #if FLAGS.get_data == True:
        #    dump_objcue(inputs, end_points, DUMP_DIR+'/objcue', DATASET_CONFIG, TEST_DATASET)
        loss, end_points = criterion(inputs, end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        ### For voxel
        #center_iou = compute_iou(end_points['vox_pred1'], end_points['vox_center'])
        #corner_iou = compute_iou(end_points['vox_pred2'], end_points['vox_corner'])
        #log_string('cen iou: %f cor iou: %f' % (center_iou.cpu().numpy(), corner_iou.cpu().numpy()))

        '''
        for i in range(len(batch_data_label['num_instance'])):
            ### For point
            pre_sem = end_points['pred_sem_class'][i,0,...].detach().cpu().numpy()
            sem_idx = np.where(end_points['sem_mask'][i,...].detach().cpu().numpy() > 0)[0]
            pre_sem = np.expand_dims(pre_sem[sem_idx], -1)
            gt_sem = np.squeeze(end_points['sub_point_sem_cls_label'][i,:].cpu().numpy())[sem_idx]
            for cls in DATASET_CONFIG.class2type.keys():
                correct_cls[cls] += np.sum((np.sum(pre_sem == gt_sem, 1) > 0) & (np.sum(gt_sem == cls, 1) > 0))
                total_cls[cls] += np.sum(np.sum(gt_sem == cls, 1) > 0)
            total_correct_sem += np.sum((np.sum(pre_sem == gt_sem, 1) > 0))
            total_sem += len(gt_sem)
            ### For plane
            pre_sem = end_points['pred_sem_class'][i,2,...].detach().cpu().numpy()
            sem_idx = np.where(end_points['sem_mask_plane'][i,...].detach().cpu().numpy() > 0)[0]
            pre_sem = np.expand_dims(pre_sem[sem_idx], -1)
            gt_sem = np.squeeze(end_points['sub_point_sem_cls_label'][i,:].cpu().numpy())[sem_idx]
            for cls in DATASET_CONFIG.class2type.keys():
                correct_cls_plane[cls] += np.sum((np.sum(pre_sem == gt_sem, 1) > 0) & (np.sum(gt_sem == cls, 1) > 0))
                total_cls_plane[cls] += np.sum(np.sum(gt_sem == cls, 1) > 0)
            total_correct_sem_plane += np.sum((np.sum(pre_sem == gt_sem, 1) > 0))
            total_sem_plane += len(gt_sem)
            ### For angle plane
            sem_idx = np.where(end_points['sem_mask_plane'][i,...].detach().cpu().numpy() > 0)[0]
            for cls in ['x', 'y', 'z']:
                gt_sem = (end_points[cls+'_gt'][i,...].detach().cpu().numpy())[sem_idx,:]
                pre_sem = (end_points[cls+'_angle'][i,...].detach().cpu().numpy().T)[sem_idx]
                pre_sem = np.argmax(pre_sem, 1)

                correct_cls_angle_plane[cls] += np.sum(np.minimum((pre_sem == gt_sem[:,0]) + (pre_sem == gt_sem[:,1]) + (pre_sem == gt_sem[:,2]), 1))
                total_cls_angle_plane[cls] += len(gt_sem)
        '''
        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, mode='center')
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        ap_calculator_center.step(batch_pred_map_cls, batch_gt_map_cls)

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, mode='corner')
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        ap_calculator_corner.step(batch_pred_map_cls, batch_gt_map_cls)

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, mode='plane')
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        ap_calculator_plane.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if FLAGS.dump_results:# and EPOCH_CNT %10 == 0:
            #if FLAGS.use_plane:
            #    dump_planes(end_points, DUMP_DIR, DATASET_CONFIG)
            #else:
            #dump_results(end_points, DUMP_DIR, DATASET_CONFIG)
            dump_results(end_points, DUMP_DIR+'/result/', DATASET_CONFIG, TEST_DATASET, mode='center')
            dump_results(end_points, DUMP_DIR+'/result/', DATASET_CONFIG, TEST_DATASET, mode='corner')
            dump_results(end_points, DUMP_DIR+'/result/', DATASET_CONFIG, TEST_DATASET, mode='plane')
            
    # Log statistics
    TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
        (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))
    #for cls in ['x', 'y', 'z']:
    #    log_string("For plane class %s: %f"%(cls, correct_cls_angle_plane[cls] / float(total_cls_angle_plane[cls])))
    # Evaluate average precision
    log_string('Using center')
    metrics_dict = ap_calculator_center.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f'%(key, metrics_dict[key]))
    log_string('Using corner')
    metrics_dict = ap_calculator_corner.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f'%(key, metrics_dict[key]))
    log_string('Using plane')
    metrics_dict = ap_calculator_plane.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss

def train(start_epoch):
    global EPOCH_CNT
    min_loss = 1e10
    loss = 0
    local_epoch = MAX_EPOCH 
    if FLAGS.get_data == True:
        local_epoch = start_epoch + 1
        #import pdb;pdb.set_trace()
    for epoch in range(start_epoch, local_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        if (EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9 or FLAGS.get_data == True) and FLAGS.opt_proposal == False: # Eval every 10 epochs
            loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

if __name__=='__main__':
    train(start_epoch)
