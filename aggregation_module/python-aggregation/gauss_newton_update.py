import numpy as np 
import torch 
import os 
import sys 
from utils import *
import time

def one_proposal_opt(cues, init_object, paras):
    opt_object = torch.cat((init_object.reshape(init_object.shape[0], 1), torch.zeros((10,1))), dim=0)
    cues = relevant_cues(cues, opt_object)
    
    for outer in range(1):
        weight = cue_reweighting(cues, opt_object, paras)
        # print(weight['face_upper'][:100])
        weight = adjust_wegiht(weight, paras)
        for inner in range(1):
            A, b, e_cur = gaussian_newton_step(cues, opt_object, weight)
            # print (A)
            # print(b)
            # print(e_cur)
            eigvals, _ = torch.eig(A)
            eigvals = eigvals[:,0] # real part 
        
            if torch.min(eigvals)<1e-12:
                return
        x_dif = torch.mm(A.inverse(), b)
        next_object = opt_object
        next_object[0:7] = next_object[0:7] + x_dif
        # print(next_object)
        flag = 0
        e_next = energy_object2cues(cues, weight,next_object)
        # print(e_next)
        if e_next > e_cur:
            lmbda = torch.max(eigvals)/1e4
            for lsId in range(10):
                x_dif = torch.mm((A+lmbda*torch.eye(7)).inverse(), b)
                next_object = opt_object
                next_object[0:7] = next_object[0:7] + x_dif
                e_next = energy_object2cues(cues, weight,next_object)
                if e_next < e_cur:
                    opt_object = next_object
                    flag=1
                    break
                lmbda = lmbda*4
        else:
            opt_object = next_object
            flag=1
        if flag==0:
            break
    return opt_object

"""
def object_opt_v1(cues, init_object, paras):
    opt_object = init_object
    for outer in range(8):
        weight = cue_reweighting(cues, opt_object, paras)
        weight = adjust_wegiht(weight, paras)
        for inner in range(20):
            para_diff = gaussian_newton_step(cues, opt_object, weight)
            e_cur = energy_object2cues(cues, weight, opt_object)
            alpha = 1.0
            flag = 0
            for searchID in range(10):
                next_object = opt_object
                next_object = opt_object+para_diff*alpha
                e_next = energy_object2cues(cues, weight, next_object)
                if e_next < e_cur:
                    opt_object = next_object
                    flag=1
                    break
                alpha = alpha/2
            if flag==0:
                break
    return opt_object
"""        
def gaussian_newton_step(cues, cur_object, weight):
    
    A = torch.zeros((7,7))
    b = torch.zeros((7,1))
    e = 0
    
    # center 
    center = cur_object[0:3].unsqueeze(0)[:,:,0]
    if cues['center_vox'].shape[0]>0:
        dif = cues['center_vox']-center # (K, 3)
        A[0:3, 0:3] = A[0:3, 0:3] + torch.sum(weight['center_vox'])*torch.eye(3)
        b[0:3] = b[0:3] +  torch.sum(dif*weight['center_vox'].reshape((dif.shape[0], 1)), dim=0).unsqueeze(1)
        e = e + torch.sum(torch.sum(dif**2, dim=1)*weight['center_vox'])
    if cues['center_pn'].shape[0]>0:
        dif = cues['center_pn']-center # (K, 3)
        A[0:3, 0:3] = A[0:3, 0:3] + torch.sum(weight['center_pn'])*torch.eye(3)
        b[0:3] = b[0:3] +  torch.sum(dif*weight['center_pn'].reshape((dif.shape[0], 1)), dim=0).unsqueeze(1)
        e = e + torch.sum(torch.sum(dif**2, dim=1)*weight['center_pn'])
    # corner
    center, corners = extract_center_corners(cur_object)
    J_corners = Jacobi_corners(cur_object)
    for i in range(8):
        J = J_corners[i*3:i*3+3] # (3, 7)
        dif = cues['corner_vox']-corners[i] # (K, 3)
        w = weight['corner_vox'][i]
        A = A + torch.mm(J.transpose(0,1), J)*torch.sum(w)
        b = b + torch.sum(torch.mm(J.transpose(0,1), dif.transpose(0,1))*w.reshape((1, dif.shape[0])), 1).unsqueeze(1)
        e = e + torch.sum(torch.sum(dif**2, dim=1)*weight['corner_vox'][i])
        dif = cues['corner_pn']-corners[i] # (K, 3)
        w = weight['corner_pn'][i]
        A = A + torch.mm(J.transpose(0,1), J)*torch.sum(w)
        b = b + torch.sum(torch.mm(J.transpose(0,1), dif.transpose(0,1))*w.reshape((1, dif.shape[0])), 1).unsqueeze(1)
        e = e + torch.sum(torch.sum(dif**2, dim=1)*weight['corner_pn'][i])
    # face 
    A_dif, b_dif, e_dif = gn_approx_face2plane(corners, J_corners, [0,1,2,3], cues['pred_upper'], weight['face_upper'])

    b = b + b_dif
    A = A + A_dif
    e = e + e_dif
    
    A_dif, b_dif, e_dif = gn_approx_face2plane(corners, J_corners, [4,5,6,7], cues['pred_lower'], weight['face_lower'])
    b = b + b_dif
    A = A + A_dif
    e = e + e_dif
    
    A_dif, b_dif, e_dif = gn_approx_face2plane(corners, J_corners, [0,2,4,6], cues['pred_right'], weight['face_right'])
    b = b + b_dif
    A = A + A_dif
    e = e + e_dif
    
    A_dif, b_dif, e_dif = gn_approx_face2plane(corners, J_corners, [1,3,5,7], cues['pred_left'], weight['face_left'])
    b = b + b_dif
    A = A + A_dif
    e = e + e_dif
    
    A_dif, b_dif, e_dif = gn_approx_face2plane(corners, J_corners, [2,3,6,7], cues['pred_front'], weight['face_front'])
    b = b + b_dif
    A = A + A_dif
    e = e + e_dif
    
    A_dif, b_dif, e_dif = gn_approx_face2plane(corners, J_corners, [0,1,4,5], cues['pred_back'], weight['face_back'])
    b = b + b_dif
    A = A + A_dif
    e = e + e_dif
    
    A = (A+A.transpose(0,1))/2
    # eigvals, _ = torch.eig(A)
    # eigvals = eigvals[:,0] # real part 
    # if torch.min(eigvals)<torch.max(eigvals)/10:
    #     A = A + torch.max(eigvals)*torch.eye(7)/10
    # para_diff = torch.mm(A.inverse(), b)
    return A, b, e

if __name__=="__main__":
    data_path = 'data_v7.mat'
    # cues, gt_objects = load_data(data_path, 100)
    cues, init_proposals = load_data_v2(data_path, 1)
    paras = set_paras()
    print (time.asctime( time.localtime(time.time()) ))
    for i in range(256):
        print(i)
        cur_object = init_proposals[i]
    # print(cur_object.shape)
    
    # perturbed_obj = gt_objects[10]+(torch.rand(9)*0.4).double()
    # perturbed_obj = torch.tensor([1.3359,  1.2984,  1.7791,  0.4415,  3.4382,  1.7772,  0.0403, 28.3084, 1.2495])
        opt_object = one_proposal_opt(cues, cur_object, paras)
        # print(opt_object)
    print (time.asctime( time.localtime(time.time()) ))
