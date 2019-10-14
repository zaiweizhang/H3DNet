import glob
import sys
import numpy as np

mean_size = {i:[] for i in range(1,41)}

orig = np.load(sys.argv[2])#['arr_0']

def list2str(l):
    st = ''
    for i in l:
        st += str(i)
    return st

flist = glob.glob('scannet_train_detection_data/*_bbox.npy')
num_inst = []
num_inst_in = []
inter = []
for fin in flist:
    data = np.load(fin)
    data_support = np.load(fin.replace('bbox', 'support_instance_label'))
    #import pdb;pdb.set_trace()
    num_inst_in.append(len(np.unique(data_support)))
    count = 0
    temp = []
    #import pdb;pdb.set_trace()
    for obj in data_support:
        temps = list2str(obj)
        if temps not in temp:
            temp.append(temps)
    inter.append(len(temp))
    for obj in data:
        mean_size[obj[-1]].append(obj[3:6])
        if obj[-1] not in [38, 39, 40]:
            count += 1
    #import pdb;pdb.set_trace()
    num_inst.append(count)
    avg = 0
    cot = 0
    for i in range(len(num_inst)):
        avg += num_inst_in[i] / float(num_inst[i])
        cot += inter[i]
    print (avg / float(len(num_inst)))
    print (cot / float(len(num_inst)))
import pdb;pdb.set_trace()
        
mean_shape = np.zeros((40,3))

for cls in mean_size:
    mean_shape[cls-1,:] = np.mean(mean_size[cls], axis=0)

#np.save('meta_data/scannet_means_v2.npz', mean_shape)
#import pdb;pdb.set_trace()
