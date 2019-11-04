import numpy as np 
import torch 
import os 
import sys 
import scipy.io as sio

def energy_object2cues(cues, weight, cur_object):
    center, corners = extract_center_corners(cur_object)
    # center 
    
    if cues['center_vox'].shape[0]!=0:
        dif = cues['center_vox'] - center # (K, 3)
        e = torch.sum(torch.sum(dif**2, 1)*weight['center_vox'])
    else:
        e=0
    dif = cues['center_pn'] - center # (K, 3)
    e = e + torch.sum(torch.sum(dif**2, 1)*weight['center_pn'])
    # corner
    if cues['corner_vox'].shape[0]!=0:
        for i in range(8):
            dif = cues['corner_vox'] - corners[i]
            e = e + torch.sum(torch.sum(dif**2, 1)*weight['corner_vox'][i])
            dif = cues['corner_pn'] - corners[i]
            e = e + torch.sum(torch.sum(dif**2, 1)*weight['corner_pn'][i])
    else:
        for i in range(8):
            dif = cues['corner_pn'] - corners[i]
            e = e + torch.sum(torch.sum(dif**2, 1)*weight['corner_pn'][i])
        
    # plane 
    sqrDis = face_2_plane_sqrDis(corners[[0,1,2,3]], cues['pred_upper'])
    e = e + torch.sum(sqrDis*weight['face_upper'])
    sqrDis = face_2_plane_sqrDis(corners[[4,5,6,7]], cues['pred_lower'])
    e = e + torch.sum(sqrDis*weight['face_lower'])
    sqrDis = face_2_plane_sqrDis(corners[[0,2,4,6]], cues['pred_right'])
    e = e + torch.sum(sqrDis*weight['face_right'])
    sqrDis = face_2_plane_sqrDis(corners[[1,3,5,7]], cues['pred_left'])
    e = e + torch.sum(sqrDis*weight['face_left'])
    sqrDis = face_2_plane_sqrDis(corners[[2,3,6,7]], cues['pred_front'])
    e = e + torch.sum(sqrDis*weight['face_front'])
    sqrDis = face_2_plane_sqrDis(corners[[0,1,4,5]], cues['pred_back'])
    e = e + torch.sum(sqrDis*weight['face_back'])
    return e

def load_data(data_path, idx):
    cues = {}
    f = sio.loadmat(data_path)
    cues['center_vox'] = torch.from_numpy(f['PFunc'][0,idx]['center_vox'][0,0]).transpose(0,1).float()
    cues['center_pn'] = torch.from_numpy(f['PFunc'][0,idx]['center_pn'][0,0]).transpose(0,1).float()
    cues['corner_vox'] = torch.from_numpy(f['PFunc'][0,idx]['corner_vox'][0,0]).transpose(0,1).float()
    cues['corner_pn'] = torch.from_numpy(f['PFunc'][0,idx]['corner_pn'][0,0]).transpose(0,1).float()
    cues['pred_upper'] = torch.from_numpy(f['PFunc'][0,idx]['pred_upper2'][0,0]).transpose(0,1).float()
    cues['pred_lower'] = torch.from_numpy(f['PFunc'][0,idx]['pred_lower2'][0,0]).transpose(0,1).float()
    cues['pred_right'] = torch.from_numpy(f['PFunc'][0,idx]['pred_right2'][0,0]).transpose(0,1).float()
    cues['pred_left'] = torch.from_numpy(f['PFunc'][0,idx]['pred_left2'][0,0]).transpose(0,1).float()
    cues['pred_front'] = torch.from_numpy(f['PFunc'][0,idx]['pred_front2'][0,0]).transpose(0,1).float()
    cues['pred_back'] = torch.from_numpy(f['PFunc'][0,idx]['pred_back2'][0,0]).transpose(0,1).float()
    gt_objects =  torch.from_numpy(f['PFunc'][0,idx]['gt_objects'][0,0]).float()
    return cues, gt_objects

def load_data_v2(data_path, idx):
    cues = {}
    f = sio.loadmat(data_path)
    cues['center_vox'] = torch.from_numpy(f['PFunc_small'][0,idx]['center_vox'][0,0]).transpose(0,1).float()
    cues['center_pn'] = torch.from_numpy(f['PFunc_small'][0,idx]['center_pn'][0,0]).transpose(0,1).float()
    cues['corner_vox'] = torch.from_numpy(f['PFunc_small'][0,idx]['corner_vox'][0,0]).transpose(0,1).float()
    cues['corner_pn'] = torch.from_numpy(f['PFunc_small'][0,idx]['corner_pn'][0,0]).transpose(0,1).float()
    cues['pred_upper'] = torch.from_numpy(f['PFunc_small'][0,idx]['pred_upper'][0,0]).transpose(0,1).float()
    cues['pred_lower'] = torch.from_numpy(f['PFunc_small'][0,idx]['pred_lower'][0,0]).transpose(0,1).float()
    cues['pred_right'] = torch.from_numpy(f['PFunc_small'][0,idx]['pred_right'][0,0]).transpose(0,1).float()
    cues['pred_left'] = torch.from_numpy(f['PFunc_small'][0,idx]['pred_left'][0,0]).transpose(0,1).float()
    cues['pred_front'] = torch.from_numpy(f['PFunc_small'][0,idx]['pred_front'][0,0]).transpose(0,1).float()
    cues['pred_back'] = torch.from_numpy(f['PFunc_small'][0,idx]['pred_back'][0,0]).transpose(0,1).float()
    init_proposals = torch.from_numpy(f['Data_small'][0,idx]['init_proposals'][0,0]).transpose(0,1).float()
    return cues, init_proposals

def extract_center_corners(cur_object, oriented=True): 
    cur_object = cur_object[:,0]
    centers = cur_object[0:3].unsqueeze(0)
    if oriented:
        corners= get_oriented_corners(cur_object)
    else:
        corners = get_corners(cur_object)
    return centers, corners
    
def face_2_plane_sqrDis(corners, planes): # (4, 3), (M, 4)
    dis=torch.zeros((4, planes.shape[0]))
    for i in range(4):
        dis[i] = (corners[i,0]*planes[:,0]+corners[i,1]*planes[:,1]+corners[i,2]*planes[:,2]+planes[:,3])
    dis = dis**2
    dis = torch.sum(dis, dim=0)/4

    return dis

def cue_reweighting(cues, cur_object, paras, flag=0, thres=0.05, oriented=False):
    weight={}
    center, corners = extract_center_corners(cur_object, oriented=oriented)
    print('extract', center.shape,center.dtype,corners.shape, corners.dtype)
    # center 
    if cues['center_vox'].shape[0]!=0:    
        diff_cen = cues['center_vox']-center # (k, 3)
        weight['center_vox'] = paras['sigma_center_vox']**2/(paras['sigma_center_vox']**2+torch.sum(diff_cen**2, dim=1))
        if flag==1:
            weight['center_vox'] = ((weight['center_vox']>thres).float())*weight['center_vox']
    else:
        weight['center_vox'] =0
        
    diff_cen = cues['center_pn']-center # (k, 3)
    weight['center_pn'] = paras['sigma_center_pn']**2/(paras['sigma_center_pn']**2+torch.sum(diff_cen**2, dim=1))
    if flag==1:
        weight['center_pn'] = ((weight['center_pn']>thres).float())*weight['center_pn']
    
    # corner
    weight['corner_vox']=torch.zeros((8,cues['corner_vox'].shape[0]))
    weight['corner_pn']=torch.zeros((8,cues['corner_pn'].shape[0]))
    if cues['corner_vox'].shape[0]!=0: 
        for i in range(8):
            diff = cues['corner_vox']-corners[i]
            weight['corner_vox'][i] = paras['sigma_corner_vox']**2/(paras['sigma_corner_vox']**2+torch.sum(diff**2, dim=1))
            if flag==1:
                weight['corner_vox'][i] = ((weight['corner_vox'][i]>thres).float())*weight['corner_vox'][i]

            diff = cues['corner_pn']-corners[i]
            weight['corner_pn'][i] = paras['sigma_corner_pn']**2/(paras['sigma_corner_pn']**2+torch.sum(diff**2, dim=1))
            if flag==1:
                weight['corner_pn'][i] = ((weight['corner_pn'][i]>thres).float())*weight['corner_pn'][i]
    else:
        for i in range(8):
            weight['corner_vox'][i] = 0
            diff = cues['corner_pn']-corners[i]
            weight['corner_pn'][i] = paras['sigma_corner_pn']**2/(paras['sigma_corner_pn']**2+torch.sum(diff**2, dim=1))
            if flag==1:
                weight['corner_pn'][i] = ((weight['corner_pn'][i]>thres).float())*weight['corner_pn'][i]
        
    # plane                               
    diff = face_2_plane_sqrDis(corners[[0,1,2,3], :], cues['pred_upper'])
    weight['face_upper'] = paras['sigma_face_upper']**2/(paras['sigma_face_upper']**2+diff)
    if flag==1:
        weight['face_upper'] = ((weight['face_upper']>thres).float())*weight['face_upper']
    diff = face_2_plane_sqrDis(corners[[4,5,6,7], :], cues['pred_lower'])
    weight['face_lower'] = paras['sigma_face_lower']**2/(paras['sigma_face_lower']**2+diff)
    if flag==1:
        weight['face_lower'] = ((weight['face_lower']>thres).float())*weight['face_lower']
    diff = face_2_plane_sqrDis(corners[[0,2,4,6], :], cues['pred_right'])
    weight['face_right'] = paras['sigma_face_right']**2/(paras['sigma_face_right']**2+diff)
    if flag==1:
        weight['face_right'] = ((weight['face_right']>thres).float())*weight['face_right']
    diff = face_2_plane_sqrDis(corners[[1,3,5,7], :], cues['pred_left'])
    weight['face_left'] = paras['sigma_face_left']**2/(paras['sigma_face_left']**2+diff)
    if flag==1:
        weight['face_left'] = ((weight['face_left']>thres).float())*weight['face_left']
    diff = face_2_plane_sqrDis(corners[[2,3,6,7], :], cues['pred_front'])
    weight['face_front'] = paras['sigma_face_front']**2/(paras['sigma_face_front']**2+diff)
    if flag==1:
        weight['face_front'] = ((weight['face_front']>thres).float())*weight['face_front']
    diff = face_2_plane_sqrDis(corners[[0,1,4,5], :], cues['pred_back'])
    weight['face_back'] = paras['sigma_face_back']**2/(paras['sigma_face_back']**2+diff)
    if flag==1:
        weight['face_back'] = ((weight['face_back']>thres).float())*weight['face_back']
    return weight

def adjust_wegiht(weight, paras):
    if weight['center_vox']!=0:
        weight['center_vox'] = paras['lambda_center_vox']*weight['center_vox']/len(weight['center_vox'])
    weight['center_pn'] = paras['lambda_center_pn']*weight['center_pn']/len(weight['center_pn'])
    if weight['corner_vox'][0]!=0:
        weight['corner_vox'] = paras['lambda_corner_vox']*weight['corner_vox']/(weight['corner_vox'].shape[1])
    weight['corner_pn'] = paras['lambda_corner_pn']*weight['corner_pn']/(weight['corner_pn'].shape[1])
    weight['face_upper'] =  paras['lambda_face_upper']*weight['face_upper']/len(weight['face_upper'])
    weight['face_lower'] =  paras['lambda_face_upper']*weight['face_lower']/len(weight['face_lower'])
    weight['face_left'] =  paras['lambda_face_upper']*weight['face_left']/len(weight['face_left'])
    weight['face_right'] =  paras['lambda_face_upper']*weight['face_right']/len(weight['face_right'])
    weight['face_front'] =  paras['lambda_face_upper']*weight['face_front']/len(weight['face_front'])
    weight['face_back'] =  paras['lambda_face_upper']*weight['face_back']/len(weight['face_back'])
    return weight

def relevant_cues(cues, cur_object):
    label = cur_object[7]
    cues_pruned = {}
    ids = (cues['center_vox'][:,3]==label)
    cues_pruned['center_vox'] = cues['center_vox'][ids,0:3]
    ids = (cues['center_pn'][:,3]==label)
    cues_pruned['center_pn'] = cues['center_pn'][ids, 0:3]
    ids = (cues['corner_vox'][:,3]==label)
    cues_pruned['corner_vox'] = cues['corner_vox'][ids, 0:3]
    ids = (cues['corner_pn'][:,3]==label)
    cues_pruned['corner_pn'] = cues['corner_pn'][ids,0:3]
    ids = (cues['pred_back'][:,4]==label)
    cues_pruned['pred_back'] = cues['pred_back'][ids, 0:4]
    ids = (cues['pred_front'][:,4]==label)
    cues_pruned['pred_front'] = cues['pred_front'][ids, 0:4]
    ids = (cues['pred_left'][:,4]==label)
    cues_pruned['pred_left'] = cues['pred_left'][ids, 0:4]
    ids = (cues['pred_right'][:,4]==label)
    cues_pruned['pred_right'] = cues['pred_right'][ids, 0:4]
    ids = (cues['pred_upper'][:,4]==label)
    cues_pruned['pred_upper'] = cues['pred_upper'][ids, 0:4]
    ids = (cues['pred_lower'][:,4]==label)
    cues_pruned['pred_lower'] = cues['pred_lower'][ids, 0:4]
    return cues_pruned

def relevant_cues_2(cues, cur_object):
    label = cur_object[7]
    cues_pruned = {}
    ids = (cues['center_vox'][:,3]==label)
    cues_pruned['center_vox'] = cues['center_vox'][:,0:3]
    ids = (cues['center_pn'][:,3]==label)
    cues_pruned['center_pn'] = cues['center_pn'][:, 0:3]
    ids = (cues['corner_vox'][:,3]==label)
    cues_pruned['corner_vox'] = cues['corner_vox'][:, 0:3]
    ids = (cues['corner_pn'][:,3]==label)
    cues_pruned['corner_pn'] = cues['corner_pn'][:,0:3]
    ids = (cues['pred_back'][:,4]==label)
    cues_pruned['pred_back'] = cues['pred_back'][:, 0:4]
    ids = (cues['pred_front'][:,4]==label)
    cues_pruned['pred_front'] = cues['pred_front'][:, 0:4]
    ids = (cues['pred_left'][:,4]==label)
    cues_pruned['pred_left'] = cues['pred_left'][:, 0:4]
    ids = (cues['pred_right'][:,4]==label)
    cues_pruned['pred_right'] = cues['pred_right'][:, 0:4]
    ids = (cues['pred_upper'][:,4]==label)
    cues_pruned['pred_upper'] = cues['pred_upper'][:, 0:4]
    ids = (cues['pred_lower'][:,4]==label)
    cues_pruned['pred_lower'] = cues['pred_lower'][:, 0:4]
    return cues_pruned


def  Jacobi_corners(cur_object):
    center= cur_object[0:3]
    xsize = cur_object[3]
    ysize = cur_object[4]
    zsize = cur_object[5]
    theta = cur_object[6]
    nx = torch.tensor([torch.cos(theta), torch.sin(theta), 0])
    ny = torch.tensor([-torch.sin(theta), torch.cos(theta), 0])
    nz = torch.tensor([0,0,1]).float()
    
    signs = torch.zeros((8,3))
    signs[0] = torch.tensor([1,1,1])
    signs[1] = torch.tensor([-1,1,1])
    signs[2] = torch.tensor([1,-1,1])
    signs[3] = torch.tensor([-1,-1,1])
    signs[4] = torch.tensor([1,1,-1])
    signs[5] = torch.tensor([-1,1,-1])
    signs[6] = torch.tensor([1,-1,-1])
    signs[7] = torch.tensor([-1,-1,-1])

    J_corners = torch.zeros((24, 7))
    for i in range(8):
        J_corners[3*i:3*(i+1), 0:3] = torch.eye(3)
        J_corners[3*i:3*(i+1), 3] = signs[i,0]*nx/2
        J_corners[3*i:3*(i+1), 4] = signs[i,1]*ny/2
        J_corners[3*i:3*(i+1), 5] = signs[i,2]*nz/2
        J_corners[3*i:3*(i+1), 6] = (signs[i,0]*xsize*ny-signs[i,1]*ysize*nx)/2
    return J_corners

def gn_approx_face2plane(corners, J_corners, ids, planes, weights):
    A = torch.zeros((7,7))
    b = torch.zeros((7,1))
    e = 0
    for i in ids:
        c =corners[i].unsqueeze(0)
        J = J_corners[i*3:(i+1)*3] # (3, 7)
        # print(i)
        # print(c)
        # print(J)
        dis = torch.sum(c*planes[:,0:3], dim=1)+planes[:,3] # K
        
        mat_J = torch.mm(J.transpose(0,1), planes[:, 0:3].transpose(0,1)) # (7, K)
        mat_Jw = mat_J*weights.reshape(1,dis.shape[0]) # (7, K)
        b = b - torch.mm(mat_Jw, dis.reshape(dis.shape[0],1))
        A = A + torch.mm(mat_Jw, mat_J.transpose(0, 1))
        e = e + torch.sum((dis**2)*weights)
        # b = b - torch.sum(dis.reshape(dis.shape[0], 1)*weights.reshape(dis.shape[0], 1)*torch.ones((dis.shape[0], 7)), dim=0)
        # A = A + torch.mm(mat_J, torch.mm(torch.eye(len(weights))*weights.reshape(len(weights, 1)),  mat_J.transpose()))
    b = b/4
    A = A/4
    e = e/4
    return A, b, e    

    
def get_corners(bbx): 
    corners = bbx.new_zeros((8 ,3))
    center = bbx[0:3]
    scale = bbx[3:6]/2.0
    corners[0] = torch.tensor([center[0]+scale[0], center[1]+scale[1], center[2]+scale[2]])
    corners[1] = torch.tensor([center[0]-scale[0], center[1]+scale[1], center[2]+scale[2]])
    corners[2] = torch.tensor([center[0]+scale[0], center[1]-scale[1], center[2]+scale[2]])
    corners[3] = torch.tensor([center[0]-scale[0], center[1]-scale[1], center[2]+scale[2]])
    corners[4] = torch.tensor([center[0]+scale[0], center[1]+scale[1], center[2]-scale[2]])
    corners[5] = torch.tensor([center[0]-scale[0], center[1]+scale[1], center[2]-scale[2]])
    corners[6] = torch.tensor([center[0]+scale[0], center[1]-scale[1], center[2]-scale[2]])
    corners[7] = torch.tensor([center[0]-scale[0], center[1]-scale[1], center[2]-scale[2]])
    return corners

def get_oriented_corners(bbx):
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
    
    # corners = torch.tensor([\
    #     center + vx + vy + vz, center - vx + vy + vz,
    #     center + vx - vy + vz, center - vx - vy + vz,
    #     center + vx + vy - vz, center - vx + vy - vz,
    #     center + vx - vy - vz, center - vx - vy - vz])
    return corners

def set_paras():
    paras={}
    paras['sigma_face_back']=0.1
    paras['sigma_face_front']=0.1
    paras['sigma_face_left']=0.1
    paras['sigma_face_right']=0.1
    paras['sigma_face_upper']=0.1
    paras['sigma_face_lower']=0.1
    paras['sigma_corner_vox']=0.1
    paras['sigma_corner_pn']=0.1
    paras['sigma_center_vox']=0.1
    paras['sigma_center_pn']=0.1

    paras['lambda_face_back']=0.0625
    paras['lambda_face_front']=0.0625
    paras['lambda_face_left']=0.0625
    paras['lambda_face_right']=0.0625
    paras['lambda_face_upper']=0.0625
    paras['lambda_face_lower']=0.0625
    paras['lambda_corner_vox']=0.0625
    paras['lambda_corner_pn']=0.0625
    paras['lambda_center_vox']=1
    paras['lambda_center_pn']=1

    return paras