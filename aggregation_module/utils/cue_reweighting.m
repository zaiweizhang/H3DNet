
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Weight] = cue_reweighting(PFunc, cur_object, Para)
%
flag = 1;
thre = 0.05;
[cen, corners] = extract_center_and_corners(cur_object);
%
dif = PFunc.center_vox - cen*ones(1, size(PFunc.center_vox,2));
Weight.vec_center_vox =...
    Para.sigma_center_vox^2./(Para.sigma_center_vox^2 + sum(dif.*dif));
if flag == 1
    Weight.vec_center_vox(find(Weight.vec_center_vox < thre)) = 0;
end
dif = PFunc.center_vox2 - cen*ones(1, size(PFunc.center_vox2,2));
Weight.vec_center_vox2 =...
    Para.sigma_center_vox2^2./(Para.sigma_center_vox2^2+sum(dif.*dif));
if flag == 1
    Weight.vec_center_vox2(find(Weight.vec_center_vox2 < thre)) = 0;
end
dif = PFunc.center_pn - cen*ones(1, size(PFunc.center_pn,2));
Weight.vec_center_pn =...
    Para.sigma_center_pn^2./(Para.sigma_center_pn^2+sum(dif.*dif));
if flag == 1
    Weight.vec_center_pn(find(Weight.vec_center_pn < thre)) = 0;
end
for id = 1 : 8
    dif = PFunc.corner_vox - corners(:,id)*ones(1, size(PFunc.corner_vox,2));
    Weight.vec_corner_vox(id,:) =...
        Para.sigma_corner_vox^2./(Para.sigma_corner_vox^2+sum(dif.*dif));
    if flag == 1
        Weight.vec_corner_vox(id,find(Weight.vec_corner_vox(id,:) < thre)) = 0;
    end
    dif = PFunc.corner_pn - corners(:,id)*ones(1, size(PFunc.corner_pn,2));
    Weight.vec_corner_pn(id,:) =...
        Para.sigma_corner_pn^2./(Para.sigma_corner_pn^2+sum(dif.*dif));
    if flag == 1
        Weight.vec_corner_pn(id,find(Weight.vec_corner_pn(id,:) < thre)) = 0;
    end
end
% upper : [1,2,3,4]
sqrDis = face_2_plane_sqrDis(corners(:,[1,2,3,4]), PFunc.pred_upper2);
Weight.vec_face_upper = ...
    Para.sigma_face_upper^2./(Para.sigma_face_upper^2+sqrDis);
if flag == 1
    Weight.vec_face_upper(find(Weight.vec_face_upper < thre)) = 0;
end
% lower : [5,6,7,8]
sqrDis = face_2_plane_sqrDis(corners(:,[5,6,7,8]), PFunc.pred_lower2);
Weight.vec_face_lower = ...
    Para.sigma_face_lower^2./(Para.sigma_face_lower^2+sqrDis);
if flag == 1
    Weight.vec_face_lower(find(Weight.vec_face_lower < thre)) = 0;
end
% right : [1,3,5,7]
sqrDis = face_2_plane_sqrDis(corners(:,[1,3,5,7]), PFunc.pred_right2);
Weight.vec_face_right = ...
    Para.sigma_face_right^2./(Para.sigma_face_right^2+sqrDis);
if flag == 1
    Weight.vec_face_right(find(Weight.vec_face_right < thre)) = 0;
end
% left : [2,4,6,8]
sqrDis = face_2_plane_sqrDis(corners(:,[2,4,6,8]), PFunc.pred_left2);
Weight.vec_face_left = ...
    Para.sigma_face_left^2./(Para.sigma_face_left^2+sqrDis);
if flag == 1
    Weight.vec_face_left(find(Weight.vec_face_left < thre)) = 0;
end
% front : [3,4,7,8]
sqrDis = face_2_plane_sqrDis(corners(:,[3,4,7,8]), PFunc.pred_front2);
Weight.vec_face_front = ...
    Para.sigma_face_front^2./(Para.sigma_face_front^2+sqrDis);
if flag == 1
    Weight.vec_face_front(find(Weight.vec_face_front < thre)) = 0;
end
% back : [1,2,5,6]
sqrDis = face_2_plane_sqrDis(corners(:,[1,2,5,6]), PFunc.pred_back2);
Weight.vec_face_back = ...
    Para.sigma_face_back^2./(Para.sigma_face_back^2+sqrDis);
if flag == 1
    Weight.vec_face_back(find(Weight.vec_face_back < thre)) = 0;
end