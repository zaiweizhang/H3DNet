# indoor_scene_understanding
cvpr2020

Object Cues:
1. center
2. bounding box center(6 points)
3. corner (8 points)
4. support surface points (4 points)

Command:
CUDA_VISIBLE_DEVICES=0 python train.py --dataset scannet --log_dir log_obj_detection_objcue_nojoint --num_point 40000 --use_color --use_support --max_epoch 360
