import os
import numpy as np
import open3d as o3d 

def create_lineset_from_points(points, colors=[0, 0, 0]):
    ''' create bounding box from points (order matters):
    points = [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
              [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    '''
    lines = [[0, 1], [0, 2], [2, 3], [1, 3], [0, 4], [1, 5], [3, 7], [2, 6],
             [4, 5], [5, 7], [6, 7], [4, 6]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(colors, [12, 1]))
    return line_set


def params2bbox(center, xsize, ysize, zsize, angle):
    ''' from bbox_center, angle and size to bbox
    @Args:
        center: (3)
        x/y/zsize: scalar
        angle: -pi ~ pi
    @Returns:
        bbox: 8 x 3, order:
         [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
          [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]

    '''
    vx = np.array([np.cos(angle), np.sin(angle), 0])
    vy = np.array([-np.sin(angle), np.cos(angle), 0])
    vx = vx * np.abs(xsize) / 2
    vy = vy * np.abs(ysize) / 2
    vz = np.array([0, 0, np.abs(zsize) / 2])
    bbox = np.array([\
        center - vx - vy - vz, center - vx - vy + vz,
        center - vx + vy - vz, center - vx + vy + vz,
        center + vx - vy - vz, center + vx - vy + vz,
        center + vx + vy - vz, center + vx + vy + vz])
    return bbox

if __name__ == '__main__':
    '''
    _all.npy: 500000 * 9
    each line: 
        center_x, center_y, center_z, size_x, size_y, size_z, angle, instance_label, semantic label
    '''
    dataset_dir = '/home/yanghaitao/Dataset/scannet_train_detection_data/'

    filelist = os.listdir(dataset_dir) 
    filelist = [f[:12] for f in filelist] 
    files = np.unique(filelist) 

    for sceneid in files:
        label = np.load(dataset_dir + sceneid + '_all.npy') 
        obbox = label[:, :7]
        inst  = label[:, 7]
        sem   = label[:, 8]
        xyz   = np.load(dataset_dir + sceneid + '_vert.npy')                                                                                       

        obbox_list = [] 
        for i in np.unique(inst): 
            segid = np.where(inst == i)
            semi = np.unique(sem[segid])[0] 
            if semi > 0 and semi < 38: 
                obbox_list.append(obbox[segid][0]) 
        obboxvis = np.vstack(obbox_list)                                                                                           

        lineset_box_list = [] 
        for b in obboxvis: 
            box = params2bbox(b[:3], b[3], b[4], b[5], b[6]) 
            lineset_box_list.append(create_lineset_from_points(box)) 

        pcd = o3d.geometry.PointCloud(); 
        pcd.points = o3d.utility.Vector3dVector(xyz[:, :3]); 
        o3d.visualization.draw_geometries([pcd] + lineset_box_list)    


