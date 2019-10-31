from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset_voxel import ScannetDetectionDataset
from models.resnet_autoencoder import TwoStreamNet, get_loss 
from utils.pc_util import compute_iou, save_vox_results, write_ply_color, volume_to_point_cloud, get_pred_pts
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import scipy.io as sio

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# python train_center_corner.py --
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_filters', type=int, default=16, help='network channel unit')
parser.add_argument('--encoder_filters', type=int, default=32, help='network channel unit')
parser.add_argument('--outf', type=str, default='out_bin', help='output folder')
# parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--model', type=str, default='', help='model path')
# parser.add_argument('--dataroot', type=str, default='/home/bo/data/scannet/new', help="dataset path")
parser.add_argument('--datapath', type=str, default='/home/bo/data/scannet/new2/scannet_train_detection_data', help="dataset path")

# 0.025 model: /home/bo/projects/cvpr2020/detection/voxel/out_corners/out_0.025000_tsdf1_select1_reduce1/dev_2.000000_9_0.000000_8_100_5_0/seg_model_0.pth
parser.add_argument('--center_reduce', type=int, default=0)
parser.add_argument('--corner_reduce', type=int, default=0)
parser.add_argument('--use_tsdf', type=int, default=0)
parser.add_argument('--select_corners', type=int, default=0)
parser.add_argument('--train_corner', type=int, default=1)
parser.add_argument('--train_center', type=int, default=1)

parser.add_argument('--vsize', type=float, default=0.06, help='size of one voxel')
parser.add_argument('--center_dev', type=float, default=2.0, help='gaussian deviation')
parser.add_argument('--corner_dev', type=float, default=1.0, help='gaussian deviation')
parser.add_argument('--wcenter', type=float, default=3, help='weight of center loss')
parser.add_argument('--wcorner', type=float, default=2, help='weight of corner loss')
parser.add_argument('--w9_cen', type=float, default=0, help='weight of center>0.95')
parser.add_argument('--w8_cen', type=float, default=250, help='weight of center>0.8')
parser.add_argument('--w5_cen', type=float, default=5, help='weight of center>0.5')
parser.add_argument('--w9_cor', type=float, default=60, help='weight of corner>0.95')
parser.add_argument('--w8_cor', type=float, default=250, help='weight of corner>0.8')
parser.add_argument('--w5_cor', type=float, default=5, help='weight of corner>0.5')
parser.add_argument('--xymin', type=float, default=-3.85)
parser.add_argument('--xymax', type=float, default=3.85)
parser.add_argument('--zmin', type=float, default=-0.2)
parser.add_argument('--zmax', type=float, default=2.69)
parser.add_argument('--eps', type=float, default=2)
# parser.add_argument('--class_choice', type=str, default=None, help="class_choice")
# parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
opt = parser.parse_args()
print(opt)
opt.outf = 'results/center+corner_18cls/vs%s_tsdf%d_cenreduce%d+correduce%d/wcenter%d_wcorner%d_cendev%d_9_%d_8_%d_5_%d_cordev%d_9_%d_8_%d_5_%d'%(
    opt.vsize, opt.use_tsdf, opt.center_reduce, opt.corner_reduce, int(opt.wcenter), int(opt.wcorner), int(opt.center_dev), int(opt.w9_cen), int(opt.w8_cen), 
    int(opt.w5_cen), int(opt.corner_dev), int(opt.w9_cor), int(opt.w8_cor), int(opt.w5_cor))
vis_out = opt.outf.replace('results', 'vis_out')
if not os.path.exists(vis_out):
    os.makedirs(vis_out)
opt.model = '/home/bo/projects/cvpr2020/detection/voxel/out_center+corner/out_0.05_tsdf0_18cls1/wcenter3_wcorner2_cendev2_9_0_8_250_5_5_cordev1_9_60_8_250_5_5/seg_model_101.pth'
# opt.outf = 'out_dev'+str(opt.dev)+'_'+str(opt.w95)+'_'+str(opt.w8)+'_'+str(opt.w5)+'_'+str(opt.w3)

if opt.gpu!=None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

dataset = ScannetDetectionDataset(
    data_path=opt.datapath,
    split_set='all',
    vsize=opt.vsize,
    use_tsdf=opt.use_tsdf,
    )
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)


model = TwoStreamNet(num_filters=opt.num_filters, encoder_filters=opt.encoder_filters, c_out_1=1, c_out_2=1)
if torch.cuda.device_count() > 1:
  print("Let's use %d GPUs!" % (torch.cuda.device_count()))
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = torch.nn.DataParallel(model)

if opt.model != '':
    model.load_state_dict(torch.load(opt.model))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

num_batch = len(dataset) / opt.batchSize


for i, data in enumerate(dataloader):
    vox = data['voxel']
    corners = data['corner']
    center = data['center'] 
    scene_name = data['scan_name'][0]
    vox, center, corners = vox.cuda(), center.cuda(), corners.cuda()
    out_path = opt.outf
    model = model.eval()
    pred = model(vox)
    pred_center = pred['pred1']
    pred_corner = pred['pred2']
    feature = pred['latent_feature']
    center_iou = compute_iou(pred_center, center)
    corner_iou = compute_iou(pred_corner, corners)
    # np.save(os.path.join(out_path, scene_name+'_feature_l4.npy'), feature.detach().cpu().numpy()[0])
    # np.save(os.path.join(out_path, scene_name+'_vox.npy'), vox.cpu().numpy()[0, 0])
    # np.save(os.path.join(out_path, 'corner.npy'), corners.cpu().numpy()[0,0])
    # np.save(os.path.join(out_path, scene_name+'_pred_center.npy'), pred_center.detach().cpu().numpy()[0,0])
    # np.save(os.path.join(out_path, scene_name+'_pred_corner.npy'), pred_corner.detach().cpu().numpy()[0,0])
    # print('[%d: %d/%d] loss: %f, crossentropy: %f, bce: %f, iou: %f' % (epoch, i, num_batch, loss.item(), crossentropy_loss.item(), bce_loss.item(), iou.cpu().numpy()))
    print('[%d/%d] [cen iou: %f cor iou: %f]' % (i, num_batch, center_iou.cpu().numpy(), corner_iou.cpu().numpy()))
    center = volume_to_point_cloud(pred_center.detach().cpu().numpy()[0,0])
    corner = volume_to_point_cloud(pred_corner.detach().cpu().numpy()[0,0])
    if center.shape[0]==0:
        print('*****')
        print(scene_name, '0 points')
        continue
    pt_center = get_pred_pts(center, vsize=opt.vsize, eps=opt.eps)
    pt_corner = get_pred_pts(corner, vsize=opt.vsize, eps=opt.eps)
    # remove nan
    print(pt_center.shape, pt_corner.shape)
    crop_ids = []
    for i in range(pt_center.shape[0]):
        if pt_center[i,0]<100:
          crop_ids.append(i)
    pt_center = pt_center[crop_ids]  
    # sio.savemat(os.path.join(pred_path, name+'_center_0.06_vox.mat'), {'center_vox': center})
    crop_ids=[]
    for i in range(pt_corner.shape[0]):
        if pt_corner[i,0]<100:
            crop_ids.append(i)
    pt_corner=pt_corner[crop_ids]       
    print(pt_center.shape, pt_corner.shape)
    sio.savemat(os.path.join(out_path, scene_name+'_center_%s_vox.mat'%(str(opt.vsize))), {'center_vox': pt_center})
    sio.savemat(os.path.join(out_path, scene_name+'_corner_%s_vox.mat'%(str(opt.vsize))), {'corner_vox': pt_corner})
       
    if i%100==0: # optional, visulize for debugging
        pt = np.load(os.path.join(opt.datapath, scene_name+'_vert.npy'))[:,0:3]
        num_pt = pt.shape[0]
        num_pt_center = pt_center.shape[0]
        pt_and_center = np.concatenate((pt, pt_center),axis=0)
        label = np.zeros(num_pt_center+num_pt)
        label[-num_pt_center:]=1
        write_ply_color(pt_and_center, label, os.path.join(vis_out,scene_name+'pt_and_pred_center.ply'))
   
        num_pt_corner = pt_corner.shape[0]
        pt_and_corner = np.concatenate((pt, pt_corner),axis=0)
        label = np.zeros(num_pt_corner+num_pt)
        label[-num_pt_corner:]=1
        write_ply_color(pt_and_corner, label, os.path.join(vis_out,scene_name+'pt_and_pred_corner.ply'))
   