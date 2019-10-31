function [PFunc] = qixing_debug(Data)
% Generate the potential function that fits 
PFunc.scene_name = Data.scene_name;
% use Data.gt_center to filter Data.gt_objects
[IDX,d] = knnsearch(Data.gt_objects(:,1:3), Data.gt_center', 'k', 1);
PFunc.gt_objects = Data.gt_objects(IDX,:);
%
PFunc.center_vox = Data.center_vox;
PFunc.center_vox2 = Data.center_vox2;
PFunc.center_pn = meanshift_cluster(Data.pred_center, 0.1);
PFunc.center_pn2 = Data.pred_center;
PFunc.corner_pn = meanshift_cluster(Data.pred_corner, 0.1);
PFunc.corner_pn2 = Data.pred_corner;
PFunc.corner_pn3 = Data.gt_corner;
PFunc.corner_vox = Data.corner_vox;
%
%PFunc.pred_back = meanshift_cluster(Data.pred_back, 0.2);
PFunc.pred_back2 = Data.pred_back;
%PFunc.pred_front = meanshift_cluster(Data.pred_front, 0.2);
PFunc.pred_front2 = Data.pred_front;
%PFunc.pred_left = meanshift_cluster(Data.pred_left, 0.2);
PFunc.pred_left2 = Data.pred_left;
%PFunc.pred_right = meanshift_cluster(Data.pred_right, 0.2);
PFunc.pred_right2 = Data.pred_right;
%PFunc.pred_lower = meanshift_cluster(Data.pred_lower, 0.2);
PFunc.pred_lower2 = Data.pred_lower;
%PFunc.pred_upper = meanshift_cluster(Data.pred_upper, 0.2);
PFunc.pred_upper2 = Data.pred_upper;
