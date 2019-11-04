import torch 
import h5py
import numpy as np
import scipy.io as sio

a = torch.ones(23)
# a = torch.tensor([1,2,3,4,5]).float()
b = torch.ones(23)
c=a-b
print(c)
# b = torch.rand(5,5)
# eigvals, _ = torch.eig(b)
# eigvals = eigvals[:,0] # real part 
# print(eigvals)

# # b = torch.eye(5)
# # # c=1/torch.sum(b, dim=1)
# # # c = b.transpose()*b
# ids = (a>3)
# c = b[ids]
# print(c)
# print(c**2)
# data_path = '/home/bo/projects/cvpr2020/detection/indoor_scene_understanding/python-aggregation/data_v7.mat'
# f = sio.loadmat(data_path)
# print(f['P'][0,1]['gt_center'][0,0].shape)