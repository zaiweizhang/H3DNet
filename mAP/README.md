# README

## 1 Baseline 

### 1.1  The Paper's Results

The Paper's results are 

`eval mAP@0.25: 0.581901, eval mAP@0.5: 0.330735 `

### 1.2 The Worse Results

VoteNet has two tricks when calculating the mAP. 

- Remove bounding box with few points in it (less than a threshold).
- Predict `sem_cls_prob` for each semantic class.

If we do not use these tricks, that is the setup is `no_remove_empty_box` and `do_not_use_sem_cls_prob`.

The results are a little worse, which is:

`eval mAP@0.25: 0.553758, eval mAP@0.5: 0.313452`



## 2 Evaluate with Our Code

If `sem_cls_prob` is predicted, then add `--per_class_proposal`

If `nonempty_box_mask` is **not** calculated, then add `--faster_eval`

### 2.0 Preparation

- Download VoteNet's data from https://drive.google.com/drive/folders/1hDGK3h52VFubNmfB8KK4nAshsFC0RR-X?usp=sharing
- Put the `votenet_data` in the root directory `./mAP`

### 2.1 Reproduce the Paper's Results

`eval mAP@0.25: 0.581901, eval mAP@0.5: 0.330734 `

```bash
python demo.py --dataset scannet --dump_dir ./ --data_dir ./votenet_data --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
```

### 2.2 Reproduce the Worse Results

`eval mAP@0.25: 0.553759, eval mAP@0.5: 0.313452 `

```bash
python demo.py --dataset scannet --dump_dir ./ --data_dir ./votenet_data --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --faster_eval
```

### 2.3  What Is Our Setup

In our case, we don't have `sem_cls_prob`, but we can calculate the `nonempty_box_mask` ourselves. Therefore, we could use the following command:

```bash
python demo.py --dataset scannet --dump_dir ./ --data_dir ./votenet_data --cluster_sampling seed_fps --use_3d_nms --use_cls_nms
```

If both `sem_cls_prob` and `nonempty_box_mask` are **not** calculated, then:

```bash
python demo.py --dataset scannet --dump_dir ./ --data_dir ./votenet_data --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --faster_eval
```












