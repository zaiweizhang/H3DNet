# indoor_scene_understanding
cvpr2020

Object Cues:
1. center
2. boundary points (8 points)
3. contact plane (1 point or dense points)
4. front center, boundary points, surface

Command:
CUDA_VISIBLE_DEVICES=0 python train.py --dataset scannet --log_dir log_obj_detection_objcue_nojoint --num_point 40000 --use_color --use_support --max_epoch 360
