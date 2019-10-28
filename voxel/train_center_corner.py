from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset2 import ScannetDetectionDataset
from models.resnet_autoencoder import CenterCornerNet, get_loss 
from utils.pc_util import compute_iou, save_vox_results
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# python train_center_corner.py --
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_filters', type=int, default=16, help='network channel unit')
parser.add_argument('--encoder_filters', type=int, default=32, help='network channel unit')
parser.add_argument('--outf', type=str, default='out_bin', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
# parser.add_argument('--dataroot', type=str, default='/home/bo/data/scannet/new', help="dataset path")
parser.add_argument('--datapath', type=str, default='/home/bo/data/scannet/new/scannet_train_detection_data', help="dataset path")

# 0.025 model: /home/bo/projects/cvpr2020/detection/voxel/out_corners/out_0.025000_tsdf1_select1_reduce1/dev_2.000000_9_0.000000_8_100_5_0/seg_model_0.pth
parser.add_argument('--center_reduce', type=int, default=0)
parser.add_argument('--corner_reduce', type=int, default=1)
parser.add_argument('--use_tsdf', type=int, default=1)
parser.add_argument('--select_corners', type=int, default=1)
parser.add_argument('--train_corner', type=int, default=1)
parser.add_argument('--train_center', type=int, default=1)
parser.add_argument('--train_angle', type=int, default=0)

parser.add_argument('--vsize', type=float, default=0.05, help='size of one voxel')
parser.add_argument('--center_dev', type=float, default=2.0, help='gaussian deviation')
parser.add_argument('--corner_dev', type=float, default=1.0, help='gaussian deviation')
parser.add_argument('--wcenter', type=float, default=1, help='weight of center loss')
parser.add_argument('--wcorner', type=float, default=1, help='weight of corner loss')
parser.add_argument('--w9_cen', type=float, default=0, help='weight of center>0.95')
parser.add_argument('--w8_cen', type=float, default=180, help='weight of center>0.8')
parser.add_argument('--w5_cen', type=float, default=20, help='weight of center>0.5')
parser.add_argument('--w9_cor', type=float, default=60, help='weight of corner>0.95')
parser.add_argument('--w8_cor', type=float, default=80, help='weight of corner>0.8')
parser.add_argument('--w5_cor', type=float, default=5, help='weight of corner>0.5')
parser.add_argument('--xymin', type=float, default=-3.2)
parser.add_argument('--xymax', type=float, default=3.2)
parser.add_argument('--zmin', type=float, default=-0.1)
parser.add_argument('--zmax', type=float, default=2.32)
# parser.add_argument('--class_choice', type=str, default=None, help="class_choice")
# parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
opt = parser.parse_args()
print(opt)
opt.dataroot = '/home/bo/data/scannet/new/training_data/vs%s_tsdf%d'%(str(opt.vsize), opt.use_tsdf)
print(opt.dataroot)
opt.outf = 'out_center+corner/out_%s_tsdf%d_cenreduce%d+correduce%d/wcenter%d_wcorner%d_cendev%d_9_%d_8_%d_5_%d_cordev%d_9_%d_8_%d_5_%d'%(
    opt.vsize, opt.use_tsdf, opt.center_reduce, opt.corner_reduce, int(opt.wcenter), int(opt.wcorner), int(opt.center_dev), int(opt.w9_cen), int(opt.w8_cen), 
    int(opt.w5_cen), int(opt.corner_dev), int(opt.w9_cor), int(opt.w8_cor), int(opt.w5_cor))

# opt.outf = 'out_dev'+str(opt.dev)+'_'+str(opt.w95)+'_'+str(opt.w8)+'_'+str(opt.w5)+'_'+str(opt.w3)

if opt.gpu!=None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
# opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

dataset = ScannetDetectionDataset(
    data_root=opt.dataroot,
    data_path=opt.datapath,
    split_set='train',
    vsize=opt.vsize,
    center_reduce=opt.center_reduce,
    corner_reduce=opt.corner_reduce,
    center_dev=opt.center_dev,
    corner_dev=opt.corner_dev,
    select_corners=opt.select_corners,
    use_tsdf=opt.use_tsdf,
 
    )
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ScannetDetectionDataset(
    data_root=opt.dataroot,
    data_path=opt.datapath,
    split_set='val',
    center_reduce=opt.center_reduce,
    corner_reduce=opt.corner_reduce,
    vsize=opt.vsize,
    center_dev=opt.center_dev,
    corner_dev=opt.corner_dev,
    select_corners=opt.select_corners,
    augment=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))
print(len(dataset), len(test_dataset))

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

blue = lambda x: '\033[94m' + x + '\033[0m'

model = CenterCornerNet(num_filters=opt.num_filters)
if torch.cuda.device_count() > 1:
  print("Let's use %d GPUs!" % (torch.cuda.device_count()))
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = torch.nn.DataParallel(model)

if opt.model != '':
    model.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader):
        vox = data['voxel']
        corners = data['corners']
        center = data['center'] 
        scene_id = data['scan_idx']
        vox, center, corners = vox.cuda(), center.cuda(), corners.cuda()
        optimizer.zero_grad()
        model = model.train()
        pred = model(vox)
        pred_center = pred['pred_center']
        pred_corner = pred['pred_corner']
        
        center_loss = get_loss(pred_center, center, opt.w9_cen, opt.w8_cen, opt.w5_cen)
        corner_loss = get_loss(pred_corner, corners, opt.w9_cor, opt.w8_cor, opt.w5_cor)
        loss = opt.wcenter*center_loss +opt.wcorner*corner_loss
        loss.backward()
        optimizer.step()
        center_iou = compute_iou(pred_center, center)
        corner_iou = compute_iou(pred_corner, corners)
        if i % 100 == 0 and i!=0: 
            save_vox_results(vox, corners, pred_corner, opt.outf, epoch, i, 'train_corner')
            save_vox_results(vox, center, pred_center, opt.outf, epoch, i, 'train_center')
        # print('[%d: %d/%d] loss: %f, crossentropy: %f, bce: %f, iou: %f' % (epoch, i, num_batch, loss.item(), crossentropy_loss.item(), bce_loss.item(), iou.cpu().numpy()))
        print('[%d: %d/%d] [loss: %f, center: %f, corner: %f cen iou: %f cor iou: %f]' % (epoch, i, num_batch, loss.item(), center_loss.item(), corner_loss.item(), center_iou.cpu().numpy(), corner_iou.cpu().numpy()))

        if i % 20 == 0:
            j, data = next(enumerate(testdataloader))
            vox = data['voxel']
            corners = data['corners'] 
            center = data['center'] 
            scene_id = data['scan_idx']
            vox, center, corners = vox.cuda(), center.cuda(), corners.cuda()
            model = model.eval()
            pred = model(vox)
            pred_center = pred['pred_center']
            pred_corner = pred['pred_corner']
            center_loss = get_loss(pred_center, center, opt.w9_cen, opt.w8_cen, opt.w5_cen)
            corner_loss = get_loss(pred_corner, corners, opt.w9_cor, opt.w8_cor, opt.w5_cor)
            loss = center_loss + corner_loss
            center_iou = compute_iou(pred_center, center)
            corner_iou = compute_iou(pred_corner, corners)
            
            if i % 60 == 0 and i!=0: 
                save_vox_results(vox, corners, pred_corner, opt.outf, epoch, i, 'val_corner')
                save_vox_results(vox, center, pred_center, opt.outf, epoch, i, 'val_center')

            print('****test****')
            # print('[%d: %d/%d] loss: %f, crossentropy: %f, bce: %f, iou: %f' % (epoch, i, num_batch, loss.item(), crossentropy_loss.item(), bce_loss.item(), iou.cpu().numpy()))
            print('[center+corner--vsize: %f, cen_dev: %d, cor_dev: %d, tsdf: %d, center reduce: %d]'%(opt.vsize, opt.center_dev, opt.corner_dev, opt.use_tsdf,  opt.center_reduce ))
            print('cen: %d, w9:%d, w8: %d; cor: %d, w9: %d, w8: %d'%(int(opt.wcenter),int(opt.w9_cen), int(opt.w8_cen), int(opt.wcorner), int(opt.w9_cor), int(opt.w8_cor)))
            print('[%d: %d/%d] [loss: %f, center: %f, corner: %f cen iou: %f cor iou: %f]' % (epoch, i, num_batch, loss.item(), center_loss.item(), corner_loss.item(), center_iou.cpu().numpy(), corner_iou.cpu().numpy()))

    torch.save(model.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))

    