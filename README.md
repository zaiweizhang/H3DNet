# indoor_scene_understanding
cvpr2020

Object Cues:
1. center
2. bounding box center(6 points)
3. corner (8 points)
4. support surface points (4 points)

Command:
CUDA_VISIBLE_DEVICES=0 python train.py --dataset scannet_hd --log_dir log_objcue --num_point 40000 --use_plane --model hdnet --max_epoch 360 --dump_results --use_objcue --batch_size 4

CUDA_VISIBLE_DEVICES=3,4,5 python train.py --dataset scannet_hd --log_dir /scratch/cluster/zaiwei92/scannet_log/log_scannet_batch8_normal_withplane --num_point 40000 --model hdnet --max_epoch 360 --batch_size 9
